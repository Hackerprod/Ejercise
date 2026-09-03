#include "mrdl/graph.hpp"

namespace mrdl {

GraphStore::GraphStore(std::shared_ptr<IGraphJournal> journal) : journal_(std::move(journal)) {}

RelationId GraphStore::allocate_id() noexcept {
    return next_id_.fetch_add(1U, std::memory_order_relaxed);
}

void GraphStore::add_unique(std::vector<RelationId>& index, RelationId id) {
    if (std::find(index.begin(), index.end(), id) == index.end()) index.push_back(id);
}

void GraphStore::remove_id(std::vector<RelationId>& index, RelationId id) {
    index.erase(std::remove(index.begin(), index.end(), id), index.end());
}

std::uint64_t GraphStore::pair_key(NodeId source, NodeId destination) noexcept {
    return (static_cast<std::uint64_t>(source) << 32U) | static_cast<std::uint64_t>(destination);
}

void GraphStore::index_relation_locked(const RelationRecord& relation) {
    require(relation.lanes.participates_in_full, "relation must participate in FULL");
    add_unique(full_index_[relation.source], relation.id);
    if (relation.lanes.participates_in_clean) add_unique(clean_index_[relation.source], relation.id);
    add_unique(pair_index_[pair_key(relation.source, relation.destination)], relation.id);
    keys_[RelationKey{relation.source, relation.destination, relation.prototype}] = relation.id;
}

void GraphStore::unindex_relation_locked(const RelationRecord& relation) {
    if (auto it = full_index_.find(relation.source); it != full_index_.end()) {
        remove_id(it->second, relation.id);
        if (it->second.empty()) full_index_.erase(it);
    }
    if (auto it = clean_index_.find(relation.source); it != clean_index_.end()) {
        remove_id(it->second, relation.id);
        if (it->second.empty()) clean_index_.erase(it);
    }
    if (auto it = pair_index_.find(pair_key(relation.source, relation.destination)); it != pair_index_.end()) {
        remove_id(it->second, relation.id);
        if (it->second.empty()) pair_index_.erase(it);
    }
    const RelationKey key{relation.source, relation.destination, relation.prototype};
    if (auto it = keys_.find(key); it != keys_.end() && it->second == relation.id) keys_.erase(it);
}

void GraphStore::sort_source_indexes_locked(NodeId source) {
    auto sorter = [&](std::vector<RelationId>& ids) {
        std::sort(ids.begin(), ids.end(), [&](RelationId lhs, RelationId rhs) {
            const auto left = relations_.find(lhs);
            const auto right = relations_.find(rhs);
            if (left == relations_.end()) return false;
            if (right == relations_.end()) return true;
            const float left_priority = retrieval_priority(*left->second);
            const float right_priority = retrieval_priority(*right->second);
            if (left_priority != right_priority) return left_priority > right_priority;
            return lhs < rhs;
        });
    };
    if (auto it = full_index_.find(source); it != full_index_.end()) sorter(it->second);
    if (auto it = clean_index_.find(source); it != clean_index_.end()) sorter(it->second);
}

void GraphStore::load_relation(RelationRecord relation) {
    require(relation.id != 0U, "loaded relation id cannot be zero");
    relation.lanes = LaneMask::from_level(relation.level);
    auto record = std::make_shared<const RelationRecord>(std::move(relation));
    std::unique_lock lock(mutex_);
    if (const auto existing = relations_.find(record->id); existing != relations_.end()) {
        unindex_relation_locked(*existing->second);
    }
    relations_[record->id] = record;
    index_relation_locked(*record);
    sort_source_indexes_locked(record->source);
    next_id_.store(std::max(next_id_.load(std::memory_order_relaxed), record->id + 1U), std::memory_order_relaxed);
    generation_.fetch_add(1U, std::memory_order_release);
}

void GraphStore::upsert(RelationRecord relation) {
    if (relation.id == 0U) relation.id = allocate_id();
    relation.lanes = LaneMask::from_level(relation.level);
    relation.updated_at_ms = unix_millis();
    if (relation.created_at_ms == 0) relation.created_at_ms = relation.updated_at_ms;
    require(relation.transform.dimension() > 0U, "relation transform is empty");
    require(relation.relation.dimension() > 0U, "relation vector is empty");
    {
        std::shared_lock lock(mutex_);
        const RelationKey key{relation.source, relation.destination, relation.prototype};
        if (const auto duplicate = keys_.find(key); duplicate != keys_.end() && duplicate->second != relation.id) {
            throw Error("relation key already belongs to a different id");
        }
    }
    if (journal_) journal_->persist_relation(relation);

    auto record = std::make_shared<const RelationRecord>(std::move(relation));
    std::unique_lock lock(mutex_);
    if (const auto existing = relations_.find(record->id); existing != relations_.end()) {
        unindex_relation_locked(*existing->second);
    }
    relations_[record->id] = record;
    index_relation_locked(*record);
    sort_source_indexes_locked(record->source);
    generation_.fetch_add(1U, std::memory_order_release);
}

bool GraphStore::promote(RelationId id, std::uint64_t expected_version) {
    std::unique_lock lock(mutex_);
    const auto it = relations_.find(id);
    if (it == relations_.end()) return false;
    if (expected_version != 0U && it->second->version != expected_version) return false;
    if (it->second->level == MemoryLevel::M2) return true;

    RelationRecord updated = *it->second;
    updated.level = MemoryLevel::M2;
    updated.lanes = LaneMask::from_level(MemoryLevel::M2);
    updated.escrow_state = EscrowState::Promoted;
    updated.expires_at_ms = 0;
    updated.updated_at_ms = unix_millis();
    ++updated.version;
    if (journal_) journal_->persist_relation(updated);

    auto replacement = std::make_shared<const RelationRecord>(std::move(updated));
    it->second = replacement;
    add_unique(clean_index_[replacement->source], replacement->id);
    sort_source_indexes_locked(replacement->source);
    generation_.fetch_add(1U, std::memory_order_release);
    return true;
}

bool GraphStore::mark_state(RelationId id, EscrowState state, std::uint64_t expected_version) {
    std::unique_lock lock(mutex_);
    const auto it = relations_.find(id);
    if (it == relations_.end()) return false;
    if (expected_version != 0U && it->second->version != expected_version) return false;
    RelationRecord updated = *it->second;
    updated.escrow_state = state;
    updated.updated_at_ms = unix_millis();
    ++updated.version;
    if (journal_) journal_->persist_relation(updated);
    it->second = std::make_shared<const RelationRecord>(std::move(updated));
    sort_source_indexes_locked(it->second->source);
    generation_.fetch_add(1U, std::memory_order_release);
    return true;
}

bool GraphStore::erase(RelationId id) {
    std::unique_lock lock(mutex_);
    const auto it = relations_.find(id);
    if (it == relations_.end()) return false;
    if (journal_) journal_->delete_relation(id);
    unindex_relation_locked(*it->second);
    relations_.erase(it);
    generation_.fetch_add(1U, std::memory_order_release);
    return true;
}

std::size_t GraphStore::invalidate_derived_from(RelationId source_relation) {
    std::vector<RelationId> doomed;
    {
        std::shared_lock lock(mutex_);
        for (const auto& [id, relation] : relations_) {
            if (relation->derived && std::find(relation->derived_from.begin(), relation->derived_from.end(), source_relation) != relation->derived_from.end()) {
                doomed.push_back(id);
            }
        }
    }
    std::size_t removed = 0;
    for (const auto id : doomed) removed += erase(id) ? 1U : 0U;
    return removed;
}

float GraphStore::retrieval_priority(const RelationRecord& relation) noexcept {
    const float support = std::log1p(static_cast<float>(relation.support));
    const float confidence = safe_logit(relation.confidence);
    const float state_penalty = relation.escrow_state == EscrowState::Rejected ||
                                relation.escrow_state == EscrowState::Expired ||
                                relation.escrow_state == EscrowState::Unreplayable ? -100.0F : 0.0F;
    return confidence + 0.15F * support + state_penalty;
}

std::vector<std::shared_ptr<const RelationRecord>> GraphStore::outgoing(
    Lane lane, NodeId source, std::size_t limit) const {
    std::vector<std::shared_ptr<const RelationRecord>> result;
    std::shared_lock lock(mutex_);
    const auto& index = lane == Lane::Full ? full_index_ : clean_index_;
    const auto adjacency = index.find(source);
    if (adjacency == index.end()) return result;
    result.reserve(limit == 0U ? adjacency->second.size() : std::min(limit, adjacency->second.size()));
    for (const RelationId id : adjacency->second) {
        const auto it = relations_.find(id);
        if (it == relations_.end()) continue;
        // This assertion catches any accidental post-index lane filtering design drift.
        require(it->second->eligible(lane), "lane index contains ineligible relation");
        if (it->second->escrow_state == EscrowState::Expired || it->second->escrow_state == EscrowState::Rejected ||
            it->second->escrow_state == EscrowState::Unreplayable) continue;
        result.push_back(it->second);
        if (limit != 0U && result.size() >= limit) break;
    }
    return result;
}

std::shared_ptr<const RelationRecord> GraphStore::get(RelationId id) const {
    std::shared_lock lock(mutex_);
    const auto it = relations_.find(id);
    return it == relations_.end() ? std::shared_ptr<const RelationRecord>{} : it->second;
}

std::vector<std::shared_ptr<const RelationRecord>> GraphStore::between(
    NodeId source, NodeId destination) const {
    std::vector<std::shared_ptr<const RelationRecord>> result;
    std::shared_lock lock(mutex_);
    const auto pair = pair_index_.find(pair_key(source, destination));
    if (pair != pair_index_.end()) {
        result.reserve(pair->second.size());
        for (const RelationId id : pair->second) {
            const auto relation = relations_.find(id);
            if (relation != relations_.end()) result.push_back(relation->second);
        }
    }
    std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) { return lhs->prototype < rhs->prototype; });
    return result;
}

GraphStats GraphStore::stats() const {
    std::shared_lock lock(mutex_);
    GraphStats result;
    result.relations_total = relations_.size();
    result.nodes_with_full_edges = full_index_.size();
    result.nodes_with_clean_edges = clean_index_.size();
    for (const auto& [_, relation] : relations_) {
        if (relation->level == MemoryLevel::M2) ++result.relations_m2;
        else if (relation->level == MemoryLevel::M1) ++result.relations_m1;
    }
    for (const auto& [_, ids] : full_index_) result.full_index_entries += ids.size();
    for (const auto& [_, ids] : clean_index_) result.clean_index_entries += ids.size();
    return result;
}

}  // namespace mrdl

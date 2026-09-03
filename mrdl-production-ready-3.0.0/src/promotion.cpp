#include "mrdl/promotion.hpp"

namespace mrdl {

std::vector<std::byte> EscrowObservation::serialize() const {
    BinaryWriter writer;
    writer.pod(contextual_key);
    writer.pod(observed_content);
    writer.vector<TokenId>(context_tokens);
    writer.vector<float>(bound_frame);
    writer.vector<ReplayId>(active_trace);
    const auto replay_payload = replay_closure.serialize();
    writer.vector<std::byte>(replay_payload);
    writer.string(source);
    writer.pod(timestamp_ms);
    writer.pod(support);
    writer.pod(contradiction);
    writer.pod(source_position);  // Appended so pre-field payloads remain decodable.
    return writer.take();
}

EscrowObservation EscrowObservation::deserialize(std::span<const std::byte> bytes) {
    BinaryReader reader(bytes);
    EscrowObservation observation;
    observation.contextual_key = reader.pod<std::uint64_t>();
    observation.observed_content = reader.pod<TokenId>();
    observation.context_tokens = reader.vector<TokenId>();
    observation.bound_frame = reader.vector<float>();
    observation.active_trace = reader.vector<ReplayId>();
    observation.replay_closure = ReplayClosure::deserialize(reader.vector<std::byte>());
    observation.source = reader.string();
    observation.timestamp_ms = reader.pod<std::int64_t>();
    observation.support = reader.pod<float>();
    observation.contradiction = reader.pod<float>();
    if (!reader.empty()) observation.source_position = reader.pod<std::uint32_t>();
    require(reader.empty(), "trailing escrow observation payload");
    return observation;
}

std::size_t EscrowRecord::unique_contexts() const {
    std::unordered_set<std::uint64_t> contexts;
    for (const auto& observation : observations) contexts.insert(observation.contextual_key);
    return contexts.size();
}

std::vector<std::byte> EscrowRecord::serialize() const {
    BinaryWriter writer;
    writer.pod(id);
    writer.pod(relation_id);
    writer.pod<std::uint64_t>(observations.size());
    for (const auto& observation : observations) {
        const auto payload = observation.serialize();
        writer.vector<std::byte>(payload);
    }
    writer.pod(support);
    writer.pod(confidence_cap);
    writer.pod(created_at_ms);
    writer.pod(expires_at_ms);
    writer.pod(state);
    writer.pod(pin_count);
    writer.pod(expiry_pending);
    const auto closure_payload = closure.serialize();
    writer.vector<std::byte>(closure_payload);
    return writer.take();
}

EscrowRecord EscrowRecord::deserialize(std::span<const std::byte> bytes) {
    BinaryReader reader(bytes);
    EscrowRecord record;
    record.id = reader.pod<std::uint64_t>();
    record.relation_id = reader.pod<RelationId>();
    const auto count = reader.pod<std::uint64_t>();
    record.observations.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        record.observations.push_back(EscrowObservation::deserialize(reader.vector<std::byte>()));
    }
    record.support = reader.pod<std::uint64_t>();
    record.confidence_cap = reader.pod<float>();
    record.created_at_ms = reader.pod<std::int64_t>();
    record.expires_at_ms = reader.pod<std::int64_t>();
    record.state = reader.pod<EscrowState>();
    record.pin_count = reader.pod<std::int32_t>();
    record.expiry_pending = reader.pod<bool>();
    record.closure = ReplayClosure::deserialize(reader.vector<std::byte>());
    require(reader.empty(), "trailing escrow record payload");
    return record;
}

PromotionManager::PromotionManager(GraphStore& graph,
                                   ReplayRecorder& replay,
                                   Controller& controller,
                                   RoleInducer& role_inducer,
                                   std::shared_ptr<SqliteModelStore> persistence)
    : graph_(graph), replay_(replay), controller_(controller), role_inducer_(role_inducer),
      persistence_(std::move(persistence)) {}

void PromotionManager::load() {
    if (!persistence_) return;
    std::lock_guard lock(mutex_);
    records_.clear();
    for (const auto& row : persistence_->load_escrows()) {
        auto record = EscrowRecord::deserialize(row.payload);
        record.id = row.id;
        record.relation_id = row.relation_id;
        record.state = row.state;
        record.pin_count = row.pin_count;
        record.expiry_pending = row.expiry_pending;
        record.expires_at_ms = row.expires_at_ms;
        if (!row.closure.empty()) record.closure = ReplayClosure::deserialize(row.closure);
        records_[record.relation_id] = std::move(record);
    }
}

void PromotionManager::persist_locked(const EscrowRecord& record) {
    if (!persistence_) return;
    EscrowRow row;
    row.id = record.id;
    row.relation_id = record.relation_id;
    row.state = record.state;
    row.pin_count = record.pin_count;
    row.expiry_pending = record.expiry_pending;
    row.expires_at_ms = record.expires_at_ms;
    row.payload = record.serialize();
    row.closure = record.closure.serialize();
    persistence_->save_escrow(row);
}

void PromotionManager::remember(RelationId relation,
                                EscrowObservation observation,
                                ReplayClosure closure,
                                float confidence_cap,
                                std::int64_t ttl_seconds) {
    require(relation != 0U, "cannot escrow relation zero");
    require(ttl_seconds > 0, "escrow TTL must be positive");
    if (observation.timestamp_ms == 0) observation.timestamp_ms = unix_millis();
    observation.replay_closure = closure;
    std::lock_guard lock(mutex_);
    auto& record = records_[relation];
    if (record.id == 0U) {
        record.id = relation;
        record.relation_id = relation;
        record.created_at_ms = observation.timestamp_ms;
        record.state = EscrowState::Active;
        record.confidence_cap = confidence_cap;
    }
    if (record.state != EscrowState::Active) return;
    record.support += 1U;
    record.expires_at_ms = std::max(record.expires_at_ms,
                                    observation.timestamp_ms + ttl_seconds * 1000);
    record.confidence_cap = std::min(record.confidence_cap, confidence_cap);
    if (record.observations.size() < 32U) record.observations.push_back(std::move(observation));
    else record.observations[record.support % record.observations.size()] = std::move(observation);

    // Replay closure is replaced only by an equal-or-more-complete closure rooted at the same edge.
    if (closure.root_relation == relation && closure.replay_steps.size() >= record.closure.replay_steps.size()) {
        record.closure = std::move(closure);
    }
    persist_locked(record);
}

std::optional<EscrowRecord> PromotionManager::get(RelationId relation) const {
    std::lock_guard lock(mutex_);
    const auto it = records_.find(relation);
    return it == records_.end() ? std::optional<EscrowRecord>{} : std::optional<EscrowRecord>{it->second};
}

std::vector<RelationId> PromotionManager::promotion_candidates(std::uint32_t min_support,
                                                               std::uint32_t min_contexts) const {
    std::lock_guard lock(mutex_);
    std::vector<RelationId> result;
    for (const auto& [relation, record] : records_) {
        if (record.state == EscrowState::Active && record.support >= min_support &&
            record.unique_contexts() >= min_contexts) result.push_back(relation);
    }
    std::sort(result.begin(), result.end());
    return result;
}

EscrowStats PromotionManager::stats() const {
    std::lock_guard lock(mutex_);
    EscrowStats result;
    result.total = records_.size();
    for (const auto& [_, record] : records_) {
        const auto state = static_cast<std::size_t>(record.state);
        if (state < result.by_state.size()) ++result.by_state[state];
        if (record.pin_count > 0) ++result.pinned;
        result.observations += record.observations.size();
    }
    return result;
}

bool PromotionManager::closure_valid_locked(const EscrowRecord& record) const {
    auto valid_closure = [&](const ReplayClosure& closure) {
        if (!closure.complete() || closure.root_relation != record.relation_id) return false;
        if (!replay_.closure_available(closure)) return false;
        const auto decoded_controller = Controller::decode(closure.controller_snapshot);
        if (decoded_controller.version != closure.controller_version) return false;
        if (closure.snapshot_hashes.size() != closure.relation_snapshots.size()) return false;
        for (std::size_t index = 0; index < closure.relation_snapshots.size(); ++index) {
            if (hash_bytes(closure.relation_snapshots[index].payload) != closure.snapshot_hashes[index]) return false;
        }
        for (const auto& [relation_id, version] : closure.relation_versions) {
            const auto snapshot = std::find_if(closure.relation_snapshots.begin(), closure.relation_snapshots.end(),
                [&](const auto& item) { return item.relation == relation_id && item.version == version; });
            if (snapshot == closure.relation_snapshots.end()) return false;
            const auto relation = RelationRecord::deserialize(snapshot->payload);
            if (relation.id != relation_id || relation.version != version) return false;
        }
        return true;
    };
    if (!valid_closure(record.closure)) return false;
    for (const auto& observation : record.observations) {
        if (!valid_closure(observation.replay_closure)) return false;
    }
    return true;
}

bool PromotionManager::reserve(RelationId relation) {
    std::lock_guard lock(mutex_);
    const auto it = records_.find(relation);
    if (it == records_.end() || it->second.state != EscrowState::Active) return false;
    auto& record = it->second;
    if (!closure_valid_locked(record)) {
        record.state = EscrowState::Unreplayable;
        graph_.mark_state(relation, EscrowState::Unreplayable);
        persist_locked(record);
        return false;
    }
    if (persistence_ && !persistence_->compare_exchange_escrow_state(record.id,
                                                                     EscrowState::Active,
                                                                     EscrowState::AuditReserved,
                                                                     +1)) return false;
    record.state = EscrowState::AuditReserved;
    ++record.pin_count;
    if (!persistence_) persist_locked(record);
    return true;
}

bool PromotionManager::begin_audit(RelationId relation) {
    std::lock_guard lock(mutex_);
    const auto it = records_.find(relation);
    if (it == records_.end() || it->second.state != EscrowState::AuditReserved) return false;
    if (persistence_ && !persistence_->compare_exchange_escrow_state(it->second.id,
                                                                     EscrowState::AuditReserved,
                                                                     EscrowState::Auditing,
                                                                     0)) return false;
    it->second.state = EscrowState::Auditing;
    if (!persistence_) persist_locked(it->second);
    return true;
}

std::optional<PromotionPermit> PromotionManager::complete(RelationId relation,
                                                          const AuditOutcome& outcome,
                                                          float controller_learning_rate) {
    std::lock_guard lock(mutex_);
    const auto it = records_.find(relation);
    if (it == records_.end() || it->second.state != EscrowState::Auditing) return std::nullopt;
    if (!outcome.accepted || !outcome.stable) {
        (void)reject(relation, outcome.reason);
        return std::nullopt;
    }
    const auto current = graph_.get(relation);
    if (!current || current->level != MemoryLevel::M1) return std::nullopt;

    RelationRecord promoted = *current;
    promoted.level = MemoryLevel::M2;
    promoted.lanes = LaneMask::from_level(MemoryLevel::M2);
    promoted.escrow_state = EscrowState::Promoted;
    promoted.expires_at_ms = 0;
    promoted.confidence = std::clamp(std::max(promoted.confidence, sigmoid(outcome.causal_influence)), 0.51F, 0.995F);
    ++promoted.version;
    promoted.updated_at_ms = unix_millis();

    if (persistence_) persistence_->promote_atomic(promoted, it->second.id);
    graph_.load_relation(promoted);  // Automatic integration missing in the old partial implementation is closed here.
    graph_.invalidate_derived_from(relation);

    it->second.state = EscrowState::Promoted;
    it->second.pin_count = std::max(0, it->second.pin_count - 1);
    it->second.expiry_pending = false;
    if (!persistence_) persist_locked(it->second);

    PromotionPermit permit(promoted.id, promoted.version);
    controller_.update_from_promoted(permit, outcome.positive_features,
                                     outcome.negative_features, controller_learning_rate);
    for (const auto& observation : outcome.role_observations) role_inducer_.observe_promoted(permit, observation);
    if (persistence_) {
        const auto controller = controller_.serialize();
        persistence_->save_controller(controller, controller_.snapshot().version);
        const auto roles = role_inducer_.serialize();
        persistence_->save_role_inducer(roles);
    }
    return permit;
}

bool PromotionManager::release_to_active(RelationId relation, std::string_view /*reason*/) {
    std::lock_guard lock(mutex_);
    const auto it = records_.find(relation);
    if (it == records_.end()) return false;
    auto& record = it->second;
    if (record.state != EscrowState::Auditing && record.state != EscrowState::AuditReserved) return false;
    const auto previous = record.state;
    if (persistence_ && !persistence_->compare_exchange_escrow_state(record.id, previous, EscrowState::Active, -1)) return false;
    record.state = EscrowState::Active;
    record.pin_count = std::max(0, record.pin_count - 1);
    if (record.expiry_pending || record.expires_at_ms <= unix_millis()) return expire_record_locked(relation, record);
    if (!persistence_) persist_locked(record);
    return true;
}

bool PromotionManager::reject(RelationId relation, std::string_view /*reason*/) {
    std::lock_guard lock(mutex_);
    const auto it = records_.find(relation);
    if (it == records_.end()) return false;
    auto& record = it->second;
    if (record.state != EscrowState::Auditing && record.state != EscrowState::AuditReserved) return false;
    const auto previous = record.state;
    if (persistence_ && !persistence_->compare_exchange_escrow_state(record.id, previous, EscrowState::Rejected, -1)) return false;
    record.state = EscrowState::Rejected;
    record.pin_count = std::max(0, record.pin_count - 1);
    graph_.mark_state(relation, EscrowState::Rejected);
    if (!persistence_) persist_locked(record);
    if (record.expiry_pending || record.expires_at_ms <= unix_millis()) {
        delete_record_locked(relation);
    }
    return true;
}

bool PromotionManager::mark_unreplayable(RelationId relation, std::string_view /*reason*/) {
    std::lock_guard lock(mutex_);
    const auto it = records_.find(relation);
    if (it == records_.end()) return false;
    auto& record = it->second;
    if (record.state != EscrowState::Auditing && record.state != EscrowState::AuditReserved &&
        record.state != EscrowState::Active) return false;
    const auto previous = record.state;
    const std::int32_t pin_delta = record.pin_count > 0 ? -1 : 0;
    if (persistence_ && !persistence_->compare_exchange_escrow_state(record.id, previous,
                                                                     EscrowState::Unreplayable,
                                                                     pin_delta)) return false;
    record.state = EscrowState::Unreplayable;
    record.pin_count = std::max(0, record.pin_count + pin_delta);
    graph_.mark_state(relation, EscrowState::Unreplayable);
    if (!persistence_) persist_locked(record);
    return true;
}

bool PromotionManager::expire_record_locked(RelationId relation, EscrowRecord& record) {
    if (record.pin_count > 0 || record.state == EscrowState::AuditReserved || record.state == EscrowState::Auditing) {
        record.expiry_pending = true;
        if (persistence_) persistence_->set_escrow_expiry_pending(record.id, true);
        else persist_locked(record);
        return false;
    }
    if (record.state != EscrowState::Active && record.state != EscrowState::Rejected &&
        record.state != EscrowState::Unreplayable) return false;
    record.state = EscrowState::Expired;
    graph_.erase(relation);
    std::unordered_set<ReplayId> replay_ids(record.closure.replay_steps.begin(), record.closure.replay_steps.end());
    for (const auto& observation : record.observations) {
        replay_ids.insert(observation.replay_closure.replay_steps.begin(), observation.replay_closure.replay_steps.end());
    }
    for (const ReplayId step : replay_ids) replay_.erase(step);
    if (persistence_) persistence_->delete_escrow(record.id);
    return true;
}

void PromotionManager::delete_record_locked(RelationId relation) {
    const auto it = records_.find(relation);
    if (it == records_.end()) return;
    std::unordered_set<ReplayId> replay_ids(it->second.closure.replay_steps.begin(), it->second.closure.replay_steps.end());
    for (const auto& observation : it->second.observations) {
        replay_ids.insert(observation.replay_closure.replay_steps.begin(), observation.replay_closure.replay_steps.end());
    }
    for (const ReplayId step : replay_ids) replay_.erase(step);
    graph_.erase(relation);
    if (persistence_) persistence_->delete_escrow(it->second.id);
    records_.erase(it);
}

std::size_t PromotionManager::expire_due(std::int64_t now_ms) {
    std::lock_guard lock(mutex_);
    std::vector<RelationId> erase_from_map;
    std::size_t expired = 0;
    for (auto& [relation, record] : records_) {
        if (record.expires_at_ms > now_ms || record.state == EscrowState::Promoted) continue;
        if (expire_record_locked(relation, record)) {
            erase_from_map.push_back(relation);
            ++expired;
        }
    }
    for (const auto relation : erase_from_map) records_.erase(relation);
    return expired;
}

std::size_t PromotionManager::collect_rejected_and_unreplayable(std::int64_t older_than_ms) {
    std::lock_guard lock(mutex_);
    std::vector<RelationId> doomed;
    for (const auto& [relation, record] : records_) {
        if ((record.state == EscrowState::Rejected || record.state == EscrowState::Unreplayable) &&
            record.created_at_ms <= older_than_ms && record.pin_count == 0) doomed.push_back(relation);
    }
    for (const auto relation : doomed) delete_record_locked(relation);
    return doomed.size();
}

}  // namespace mrdl

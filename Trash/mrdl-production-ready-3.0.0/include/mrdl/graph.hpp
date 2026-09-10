#pragma once

#include "mrdl/common.hpp"
#include "mrdl/relation.hpp"

namespace mrdl {

class IGraphJournal {
public:
    virtual ~IGraphJournal() = default;
    virtual void persist_relation(const RelationRecord& relation) = 0;
    virtual void delete_relation(RelationId id) = 0;
};

struct GraphStats {
    std::uint64_t relations_total{0};
    std::uint64_t relations_m1{0};
    std::uint64_t relations_m2{0};
    std::uint64_t full_index_entries{0};
    std::uint64_t clean_index_entries{0};
    std::uint64_t nodes_with_full_edges{0};
    std::uint64_t nodes_with_clean_edges{0};
};

class IRelationStore {
public:
    virtual ~IRelationStore() = default;
    [[nodiscard]] virtual std::vector<std::shared_ptr<const RelationRecord>> outgoing(
        Lane lane, NodeId source, std::size_t limit = 0) const = 0;
    [[nodiscard]] virtual std::shared_ptr<const RelationRecord> get(RelationId id) const = 0;
    [[nodiscard]] virtual std::vector<std::shared_ptr<const RelationRecord>> between(
        NodeId source, NodeId destination) const = 0;
    [[nodiscard]] virtual GraphStats stats() const = 0;
};

class GraphStore final : public IRelationStore {
public:
    explicit GraphStore(std::shared_ptr<IGraphJournal> journal = {});

    RelationId allocate_id() noexcept;
    void load_relation(RelationRecord relation);
    void upsert(RelationRecord relation);
    bool promote(RelationId id, std::uint64_t expected_version);
    bool mark_state(RelationId id, EscrowState state, std::uint64_t expected_version = 0);
    bool erase(RelationId id);
    std::size_t invalidate_derived_from(RelationId source_relation);

    [[nodiscard]] std::vector<std::shared_ptr<const RelationRecord>> outgoing(
        Lane lane, NodeId source, std::size_t limit = 0) const override;
    [[nodiscard]] std::shared_ptr<const RelationRecord> get(RelationId id) const override;
    [[nodiscard]] std::vector<std::shared_ptr<const RelationRecord>> between(
        NodeId source, NodeId destination) const override;
    [[nodiscard]] GraphStats stats() const override;

    [[nodiscard]] std::uint64_t generation() const noexcept { return generation_.load(std::memory_order_acquire); }

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<RelationId, std::shared_ptr<const RelationRecord>> relations_;
    std::unordered_map<RelationKey, RelationId, RelationKeyHash> keys_;
    std::unordered_map<std::uint64_t, std::vector<RelationId>> pair_index_;
    std::unordered_map<NodeId, std::vector<RelationId>> full_index_;
    std::unordered_map<NodeId, std::vector<RelationId>> clean_index_;
    std::shared_ptr<IGraphJournal> journal_;
    std::atomic<RelationId> next_id_{1};
    std::atomic<std::uint64_t> generation_{1};

    static float retrieval_priority(const RelationRecord& relation) noexcept;
    static void add_unique(std::vector<RelationId>& index, RelationId id);
    static void remove_id(std::vector<RelationId>& index, RelationId id);
    static std::uint64_t pair_key(NodeId source, NodeId destination) noexcept;
    void index_relation_locked(const RelationRecord& relation);
    void unindex_relation_locked(const RelationRecord& relation);
    void sort_source_indexes_locked(NodeId source);
};

}  // namespace mrdl

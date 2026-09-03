#pragma once

#include "mrdl/common.hpp"

namespace mrdl {

struct LaneReplayData {
    std::vector<GateDecision> gate_decisions;
    std::uint32_t fold_budget{0};
    std::vector<BranchId> survivor_ids;
    float shadow_upper_bound{0.0F};
    std::uint64_t candidate_set_hash{0};
};

struct ReplayStep {
    ReplayId id{0};
    std::uint64_t operation_id{0};
    std::uint64_t controller_version{0};
    std::vector<std::pair<RelationId, std::uint64_t>> relation_versions;
    std::vector<BranchId> parent_branch_ids;
    std::array<LaneReplayData, 2> lanes;
    std::uint64_t deterministic_seed{0};
    std::uint32_t depth{0};

    [[nodiscard]] std::vector<std::byte> serialize() const;
    static ReplayStep deserialize(std::span<const std::byte> bytes);
};

struct ExecutionTraceEntry {
    BranchId branch{0};
    RelationId relation{0};
    TokenId candidate{0};
    float score{0.0F};
    std::uint32_t depth{0};
};

struct ExecutionTrace {
    std::vector<ExecutionTraceEntry> entries;
    std::vector<ReplayStep> replay_steps;
    std::uint64_t operation_id{0};
};

struct ShadowBranch {
    BranchId branch{0};
    RelationId relation{0};
    TokenId candidate{0};
    float score{0.0F};
    float influence_upper_bound{0.0F};
};

struct ShadowFrontier {
    std::vector<ShadowBranch> discarded;
    float maximum_influence{0.0F};

    [[nodiscard]] bool certified(float prediction_margin, float epsilon) const noexcept {
        return prediction_margin > maximum_influence + epsilon;
    }
};

struct RelationSnapshot {
    RelationId relation{0};
    std::uint64_t version{0};
    std::vector<std::byte> payload;
};

struct ReplayClosure {
    RelationId root_relation{0};
    std::uint64_t root_version{0};
    std::uint64_t operation_id{0};
    std::uint64_t base_seed{0};
    std::vector<ReplayId> replay_steps;
    std::vector<std::pair<RelationId, std::uint64_t>> relation_versions;
    std::vector<RelationSnapshot> relation_snapshots;
    std::uint64_t controller_version{0};
    std::vector<std::byte> controller_snapshot;
    std::vector<std::byte> role_snapshot;
    std::vector<std::uint64_t> deterministic_seeds;
    std::vector<std::uint64_t> snapshot_hashes;
    std::vector<std::uint64_t> binding_hashes;

    [[nodiscard]] bool complete() const noexcept;
    [[nodiscard]] std::vector<std::byte> serialize() const;
    static ReplayClosure deserialize(std::span<const std::byte> bytes);
};

class IReplayRepository {
public:
    virtual ~IReplayRepository() = default;
    virtual void save_step(const ReplayStep& step) = 0;
    [[nodiscard]] virtual std::optional<ReplayStep> load_step(ReplayId id) const = 0;
    [[nodiscard]] virtual ReplayId max_step_id() const = 0;
    virtual void delete_step(ReplayId id) = 0;
};

class ReplayRecorder final {
public:
    explicit ReplayRecorder(std::shared_ptr<IReplayRepository> repository = {});

    ReplayId record(ReplayStep step);
    [[nodiscard]] std::optional<ReplayStep> get(ReplayId id) const;
    bool erase(ReplayId id);
    [[nodiscard]] bool closure_available(const ReplayClosure& closure) const;

private:
    mutable std::shared_mutex mutex_;
    std::unordered_map<ReplayId, ReplayStep> steps_;
    std::shared_ptr<IReplayRepository> repository_;
    std::atomic<ReplayId> next_id_{1};
};

}  // namespace mrdl

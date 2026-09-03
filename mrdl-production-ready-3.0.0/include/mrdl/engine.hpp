#pragma once

#include "mrdl/common.hpp"
#include "mrdl/config.hpp"
#include "mrdl/controller.hpp"
#include "mrdl/embeddings.hpp"
#include "mrdl/graph.hpp"
#include "mrdl/metrics.hpp"
#include "mrdl/replay.hpp"
#include "mrdl/routing.hpp"

namespace mrdl {

class LaneWorker;

struct CandidateScore {
    TokenId token{0};
    float score{0.0F};
    RelationId relation_id{0};
    ScoreFeatures features{};
};

struct LaneRoundReplay {
    std::uint32_t depth{0};
    std::uint32_t fold_budget{0};
    std::vector<GateDecision> gate_decisions;
    std::vector<BranchId> parent_branch_ids;
    std::vector<BranchId> survivor_ids;
    std::vector<std::pair<RelationId, std::uint64_t>> relation_versions;
    float shadow_upper_bound{0.0F};
    std::uint64_t candidate_set_hash{0};
    std::uint64_t deterministic_seed{0};
};

struct LanePrediction {
    Lane lane{Lane::Full};
    TokenId selected{kEosToken};
    std::vector<CandidateScore> candidates;
    float margin{0.0F};
    ExecutionTrace execution;
    ShadowFrontier shadow;
    LaneMetrics metrics;
    std::vector<RouteCapsule> final_capsules;
    std::vector<LaneRoundReplay> rounds;

    [[nodiscard]] float score_of(TokenId token, float missing = -20.0F) const noexcept;
};

struct DualPrediction {
    LanePrediction full;
    LanePrediction clean;
    Certification certification{Certification::Empty};
    DualMetrics metrics;
    std::vector<ReplayId> replay_ids;
    std::uint64_t operation_id{0};
    std::uint64_t base_seed{0};
};

struct PureComputeKey {
    RelationId relation{0};
    std::uint64_t relation_version{0};
    std::uint64_t controller_version{0};
    std::uint64_t input_hash{0};
    std::uint64_t operator_hash{0};

    bool operator==(const PureComputeKey&) const = default;
};

struct PureComputeKeyHash {
    std::size_t operator()(const PureComputeKey& key) const noexcept {
        std::uint64_t hash = hash_combine(key.relation, key.relation_version);
        hash = hash_combine(hash, key.controller_version);
        hash = hash_combine(hash, key.input_hash);
        hash = hash_combine(hash, key.operator_hash);
        return static_cast<std::size_t>(hash);
    }
};

class PureComputeCache final {
public:
    [[nodiscard]] std::optional<std::vector<float>> get(const PureComputeKey& key) const;
    void put(PureComputeKey key, std::vector<float> value);
    void clear();
private:
    mutable std::mutex mutex_;
    std::unordered_map<PureComputeKey, std::vector<float>, PureComputeKeyHash> values_;
};

class LaneEngine final {
public:
    LaneEngine(Lane lane,
               const IRelationStore& graph,
               const IEmbeddingStore& embeddings,
               const Controller& controller,
               const RoleInducer& roles,
               EngineConfig config,
               std::shared_ptr<IFoldPolicy> fold_policy = {});

    [[nodiscard]] LanePrediction predict(std::span<const TokenId> context,
                                         const RelationRecord* temporary_relation = nullptr,
                                         PureComputeCache* pure_cache = nullptr,
                                         std::uint64_t operation_id = 0,
                                         std::uint64_t deterministic_seed = 0) const;

private:
    Lane lane_;
    const IRelationStore& graph_;
    const IEmbeddingStore& embeddings_;
    const Controller& controller_;
    const RoleInducer& roles_;
    EngineConfig config_;
    std::shared_ptr<IFoldPolicy> fold_policy_;

    [[nodiscard]] std::vector<RouteCapsule> initialize_capsules(std::span<const TokenId> context) const;
    [[nodiscard]] std::vector<float> apply_relation(const RelationRecord& relation,
                                                    std::span<const float> input,
                                                    PureComputeCache* cache,
                                                    std::uint64_t controller_version) const;
};

class DualLaneEngine final {
public:
    DualLaneEngine(const IRelationStore& graph,
                   const IEmbeddingStore& embeddings,
                   const Controller& controller,
                   const RoleInducer& roles,
                   EngineConfig config,
                   std::shared_ptr<ReplayRecorder> replay = {},
                   std::shared_ptr<MetricsRegistry> metrics = {});
    ~DualLaneEngine();

    [[nodiscard]] DualPrediction predict(std::span<const TokenId> context,
                                         bool persist_replay = false,
                                         std::uint64_t deterministic_seed = 0) const;

    [[nodiscard]] LanePrediction clean_only(std::span<const TokenId> context,
                                              std::uint64_t deterministic_seed = 0) const;
    [[nodiscard]] LanePrediction clean_with_temporary(std::span<const TokenId> context,
                                                      const RelationRecord& temporary,
                                                      std::uint64_t deterministic_seed = 0) const;
    [[nodiscard]] LanePrediction clean_replay(std::span<const TokenId> context,
                                              const RelationRecord* temporary,
                                              std::uint64_t operation_id,
                                              std::uint64_t deterministic_seed) const;

private:
    LaneEngine full_;
    LaneEngine clean_;
    EngineConfig config_;
    const Controller& controller_;
    std::shared_ptr<ReplayRecorder> replay_;
    std::shared_ptr<MetricsRegistry> metrics_;
    std::unique_ptr<LaneWorker> worker_;
    mutable std::atomic<std::uint64_t> next_operation_{1};
};

}  // namespace mrdl

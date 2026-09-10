#pragma once

#include "mrdl/common.hpp"
#include "mrdl/relation.hpp"

namespace mrdl {

class PromotionManager;

class PromotionPermit final {
public:
    PromotionPermit(const PromotionPermit&) = default;
    [[nodiscard]] RelationId relation_id() const noexcept { return relation_id_; }
    [[nodiscard]] std::uint64_t relation_version() const noexcept { return relation_version_; }

private:
    friend class PromotionManager;
    PromotionPermit(RelationId relation_id, std::uint64_t relation_version)
        : relation_id_(relation_id), relation_version_(relation_version) {}
    RelationId relation_id_{0};
    std::uint64_t relation_version_{0};
};

enum class ScoreFeature : std::size_t {
    Bias = 0,
    OperatorFit,
    RelationContext,
    Confidence,
    Support,
    Energy,
    Coverage,
    Continuation,
    Closure,
    Repetition,
    Cycle,
    PathConfidence,
    Length,
    Count
};

constexpr std::size_t kScoreFeatureCount = static_cast<std::size_t>(ScoreFeature::Count);
using ScoreFeatures = std::array<float, kScoreFeatureCount>;

struct ControllerSnapshot {
    std::uint64_t version{1};
    std::array<float, kScoreFeatureCount> score_weights{};
    float gate_alpha{2.0F};
    float gate_beta{1.0F};
    float gate_rho{2.0F};
    float gate_delta{0.2F};
    float gate_compose_threshold{0.65F};
    float gate_reject_threshold{0.35F};
};

class Controller final {
public:
    Controller();
    explicit Controller(ControllerSnapshot snapshot);

    [[nodiscard]] ControllerSnapshot snapshot() const;
    [[nodiscard]] float score(const ScoreFeatures& features) const;
    [[nodiscard]] GateDecision composition_gate(float key_compatibility,
                                                 float binding_compatibility,
                                                 float conflict) const;

    // Only PromotionManager can mint PromotionPermit. This prevents M1 writes from
    // updating controller parameters through an accidental call path.
    void update_from_promoted(const PromotionPermit& permit,
                              const ScoreFeatures& positive,
                              std::span<const ScoreFeatures> negatives,
                              float learning_rate);

    void update_gate_from_promoted(const PromotionPermit& permit,
                                   float key_compatibility,
                                   float binding_compatibility,
                                   float conflict,
                                   bool valid,
                                   float learning_rate);

    [[nodiscard]] std::vector<std::byte> serialize() const;
    void restore(std::span<const std::byte> bytes);
    static ControllerSnapshot decode(std::span<const std::byte> bytes);

private:
    mutable std::shared_mutex mutex_;
    ControllerSnapshot state_;
};

struct RoleObservation {
    std::uint64_t structural_slot{0};
    TokenId entity{0};
    std::uint64_t structural_variant{0};
};

class RoleInducer final {
public:
    struct Config {
        std::uint32_t min_support{16};
        float variable_score_threshold{1.5F};
        std::uint32_t max_identity_bins{256};
        std::uint32_t max_structure_bins{64};
    };

    RoleInducer();
    explicit RoleInducer(Config config);

    void observe_promoted(const PromotionPermit& permit, const RoleObservation& observation);
    [[nodiscard]] std::optional<RoleId> role_for(std::uint64_t structural_slot) const;
    [[nodiscard]] float variable_score(std::uint64_t structural_slot) const;

    [[nodiscard]] std::vector<std::byte> serialize() const;
    void restore(std::span<const std::byte> bytes);

private:
    struct Statistics {
        std::uint64_t support{0};
        std::unordered_map<TokenId, std::uint64_t> identities;
        std::unordered_map<std::uint64_t, std::uint64_t> structures;
        std::optional<RoleId> role;
    };

    Config config_;
    mutable std::shared_mutex mutex_;
    std::unordered_map<std::uint64_t, Statistics> statistics_;
    RoleId next_role_{1};

    static float entropy(const auto& histogram, std::uint64_t support);
};

}  // namespace mrdl

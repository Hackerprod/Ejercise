#include "mrdl/controller.hpp"

namespace mrdl {
namespace {

constexpr std::size_t index(ScoreFeature feature) noexcept {
    return static_cast<std::size_t>(feature);
}

float dot_features(const std::array<float, kScoreFeatureCount>& weights,
                   const ScoreFeatures& features) noexcept {
    float result = 0.0F;
    for (std::size_t i = 0; i < weights.size(); ++i) result = std::fma(weights[i], features[i], result);
    return result;
}

}  // namespace

Controller::Controller() {
    state_.score_weights.fill(0.0F);
    state_.score_weights[index(ScoreFeature::Bias)] = 0.0F;
    state_.score_weights[index(ScoreFeature::OperatorFit)] = 1.4F;
    state_.score_weights[index(ScoreFeature::RelationContext)] = 1.1F;
    state_.score_weights[index(ScoreFeature::Confidence)] = 0.7F;
    state_.score_weights[index(ScoreFeature::Support)] = 0.15F;
    state_.score_weights[index(ScoreFeature::Energy)] = 0.8F;
    state_.score_weights[index(ScoreFeature::Coverage)] = 0.4F;
    state_.score_weights[index(ScoreFeature::Continuation)] = 0.2F;
    state_.score_weights[index(ScoreFeature::Closure)] = 0.2F;
    state_.score_weights[index(ScoreFeature::Repetition)] = -1.0F;
    state_.score_weights[index(ScoreFeature::Cycle)] = -1.2F;
    state_.score_weights[index(ScoreFeature::PathConfidence)] = 0.8F;
    state_.score_weights[index(ScoreFeature::Length)] = -0.15F;
}

Controller::Controller(ControllerSnapshot snapshot) : state_(std::move(snapshot)) {}

ControllerSnapshot Controller::snapshot() const {
    std::shared_lock lock(mutex_);
    return state_;
}

float Controller::score(const ScoreFeatures& features) const {
    std::shared_lock lock(mutex_);
    return dot_features(state_.score_weights, features);
}

GateDecision Controller::composition_gate(float key_compatibility,
                                           float binding_compatibility,
                                           float conflict) const {
    std::shared_lock lock(mutex_);
    const float probability = sigmoid(state_.gate_alpha * key_compatibility +
                                      state_.gate_beta * binding_compatibility -
                                      state_.gate_rho * conflict - state_.gate_delta);
    if (probability >= state_.gate_compose_threshold) return GateDecision::Compose;
    if (probability <= state_.gate_reject_threshold) return GateDecision::Reject;
    return GateDecision::Defer;
}

void Controller::update_from_promoted(const PromotionPermit& permit,
                                      const ScoreFeatures& positive,
                                      std::span<const ScoreFeatures> negatives,
                                      float learning_rate) {
    require(permit.relation_id() != 0U, "invalid promotion permit");
    if (negatives.empty()) return;
    std::unique_lock lock(mutex_);
    const float rate = std::clamp(learning_rate, 0.0F, 0.1F);
    for (const auto& negative : negatives) {
        const float positive_score = dot_features(state_.score_weights, positive);
        const float negative_score = dot_features(state_.score_weights, negative);
        const float gradient = 1.0F - sigmoid(positive_score - negative_score);
        for (std::size_t i = 0; i < state_.score_weights.size(); ++i) {
            const float delta = rate * gradient * (positive[i] - negative[i]);
            state_.score_weights[i] = clamp_finite(state_.score_weights[i] + delta, -8.0F, 8.0F);
        }
    }
    ++state_.version;
}

void Controller::update_gate_from_promoted(const PromotionPermit& permit,
                                           float key_compatibility,
                                           float binding_compatibility,
                                           float conflict,
                                           bool valid,
                                           float learning_rate) {
    require(permit.relation_id() != 0U, "invalid promotion permit");
    std::unique_lock lock(mutex_);
    const float logit = state_.gate_alpha * key_compatibility + state_.gate_beta * binding_compatibility -
                        state_.gate_rho * conflict - state_.gate_delta;
    const float error = (valid ? 1.0F : 0.0F) - sigmoid(logit);
    const float rate = std::clamp(learning_rate, 0.0F, 0.1F);
    state_.gate_alpha = clamp_finite(state_.gate_alpha + rate * error * key_compatibility, -8.0F, 8.0F);
    state_.gate_beta = clamp_finite(state_.gate_beta + rate * error * binding_compatibility, -8.0F, 8.0F);
    state_.gate_rho = clamp_finite(state_.gate_rho - rate * error * conflict, 0.0F, 8.0F);
    state_.gate_delta = clamp_finite(state_.gate_delta - rate * error, -4.0F, 4.0F);
    ++state_.version;
}

std::vector<std::byte> Controller::serialize() const {
    std::shared_lock lock(mutex_);
    BinaryWriter writer;
    writer.pod(state_.version);
    writer.vector<float>(state_.score_weights);
    writer.pod(state_.gate_alpha);
    writer.pod(state_.gate_beta);
    writer.pod(state_.gate_rho);
    writer.pod(state_.gate_delta);
    writer.pod(state_.gate_compose_threshold);
    writer.pod(state_.gate_reject_threshold);
    return writer.take();
}

ControllerSnapshot Controller::decode(std::span<const std::byte> bytes) {
    BinaryReader reader(bytes);
    ControllerSnapshot snapshot;
    snapshot.version = reader.pod<std::uint64_t>();
    const auto weights = reader.vector<float>();
    require(weights.size() == kScoreFeatureCount, "controller score weight count mismatch");
    std::copy(weights.begin(), weights.end(), snapshot.score_weights.begin());
    snapshot.gate_alpha = reader.pod<float>();
    snapshot.gate_beta = reader.pod<float>();
    snapshot.gate_rho = reader.pod<float>();
    snapshot.gate_delta = reader.pod<float>();
    snapshot.gate_compose_threshold = reader.pod<float>();
    snapshot.gate_reject_threshold = reader.pod<float>();
    require(reader.empty(), "trailing controller payload");
    return snapshot;
}

void Controller::restore(std::span<const std::byte> bytes) {
    auto decoded = decode(bytes);
    std::unique_lock lock(mutex_);
    state_ = std::move(decoded);
}


RoleInducer::RoleInducer() = default;
RoleInducer::RoleInducer(Config config) : config_(config) {}

template <typename Histogram>
float entropy_impl(const Histogram& histogram, std::uint64_t support) {
    if (support == 0U) return 0.0F;
    float result = 0.0F;
    for (const auto& [_, count] : histogram) {
        const float probability = static_cast<float>(count) / static_cast<float>(support);
        if (probability > 0.0F) result -= probability * std::log(probability);
    }
    return result;
}

float RoleInducer::entropy(const auto& histogram, std::uint64_t support) {
    return entropy_impl(histogram, support);
}

void RoleInducer::observe_promoted(const PromotionPermit& permit, const RoleObservation& observation) {
    require(permit.relation_id() != 0U, "invalid promotion permit");
    std::unique_lock lock(mutex_);
    auto& stats = statistics_[observation.structural_slot];
    ++stats.support;
    if (stats.identities.size() < config_.max_identity_bins || stats.identities.contains(observation.entity)) {
        ++stats.identities[observation.entity];
    }
    if (stats.structures.size() < config_.max_structure_bins || stats.structures.contains(observation.structural_variant)) {
        ++stats.structures[observation.structural_variant];
    }
    if (!stats.role && stats.support >= config_.min_support) {
        const float identity_entropy = entropy(stats.identities, stats.support);
        const float structural_entropy = entropy(stats.structures, stats.support);
        const float score = identity_entropy / std::max(structural_entropy, 1.0e-3F);
        if (score >= config_.variable_score_threshold) stats.role = next_role_++;
    }
}

std::optional<RoleId> RoleInducer::role_for(std::uint64_t structural_slot) const {
    std::shared_lock lock(mutex_);
    const auto it = statistics_.find(structural_slot);
    return it == statistics_.end() ? std::optional<RoleId>{} : it->second.role;
}

float RoleInducer::variable_score(std::uint64_t structural_slot) const {
    std::shared_lock lock(mutex_);
    const auto it = statistics_.find(structural_slot);
    if (it == statistics_.end()) return 0.0F;
    const float identity_entropy = entropy(it->second.identities, it->second.support);
    const float structural_entropy = entropy(it->second.structures, it->second.support);
    return identity_entropy / std::max(structural_entropy, 1.0e-3F);
}

std::vector<std::byte> RoleInducer::serialize() const {
    std::shared_lock lock(mutex_);
    BinaryWriter writer;
    writer.pod(config_.min_support);
    writer.pod(config_.variable_score_threshold);
    writer.pod(config_.max_identity_bins);
    writer.pod(config_.max_structure_bins);
    writer.pod(next_role_);
    writer.pod<std::uint64_t>(statistics_.size());
    for (const auto& [slot, stats] : statistics_) {
        writer.pod(slot);
        writer.pod(stats.support);
        writer.pod(stats.role.has_value());
        if (stats.role) writer.pod(*stats.role);
        writer.pod<std::uint64_t>(stats.identities.size());
        for (const auto& [entity, count] : stats.identities) { writer.pod(entity); writer.pod(count); }
        writer.pod<std::uint64_t>(stats.structures.size());
        for (const auto& [structure, count] : stats.structures) { writer.pod(structure); writer.pod(count); }
    }
    return writer.take();
}

void RoleInducer::restore(std::span<const std::byte> bytes) {
    BinaryReader reader(bytes);
    Config config;
    config.min_support = reader.pod<std::uint32_t>();
    config.variable_score_threshold = reader.pod<float>();
    config.max_identity_bins = reader.pod<std::uint32_t>();
    config.max_structure_bins = reader.pod<std::uint32_t>();
    RoleId next_role = reader.pod<RoleId>();
    std::unordered_map<std::uint64_t, Statistics> restored;
    const auto count = reader.pod<std::uint64_t>();
    for (std::uint64_t i = 0; i < count; ++i) {
        const auto slot = reader.pod<std::uint64_t>();
        Statistics stats;
        stats.support = reader.pod<std::uint64_t>();
        if (reader.pod<bool>()) stats.role = reader.pod<RoleId>();
        const auto identity_count = reader.pod<std::uint64_t>();
        for (std::uint64_t j = 0; j < identity_count; ++j) {
            const auto entity = reader.pod<TokenId>();
            stats.identities[entity] = reader.pod<std::uint64_t>();
        }
        const auto structure_count = reader.pod<std::uint64_t>();
        for (std::uint64_t j = 0; j < structure_count; ++j) {
            const auto structure = reader.pod<std::uint64_t>();
            stats.structures[structure] = reader.pod<std::uint64_t>();
        }
        restored.emplace(slot, std::move(stats));
    }
    require(reader.empty(), "trailing role inducer payload");
    std::unique_lock lock(mutex_);
    config_ = config;
    next_role_ = next_role;
    statistics_ = std::move(restored);
}


}  // namespace mrdl

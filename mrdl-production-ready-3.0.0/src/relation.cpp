#include "mrdl/relation.hpp"

#include <numeric>

namespace mrdl {
namespace {

void fill_hashed(std::span<float> target, std::uint64_t signature) {
    if (target.empty()) return;
    std::fill(target.begin(), target.end(), 0.0F);
    const std::size_t count = std::min<std::size_t>(4U, target.size());
    for (std::size_t i = 0; i < count; ++i) {
        signature = mix64(signature + i);
        const auto index = static_cast<std::size_t>(signature % target.size());
        target[index] += (signature >> 63U) != 0U ? 1.0F : -1.0F;
    }
    normalize_in_place(target);
}

void ema(std::span<float> destination, std::span<const float> observation, float rate) {
    require(destination.size() == observation.size(), "EMA dimension mismatch");
    const float alpha = std::clamp(rate, 0.0F, 1.0F);
    for (std::size_t i = 0; i < destination.size(); ++i) {
        destination[i] = std::fma(alpha, observation[i] - destination[i], destination[i]);
    }
}

std::span<const float> slice(const std::vector<float>& values, std::size_t begin, std::size_t size) {
    return std::span<const float>(values).subspan(begin, size);
}

std::span<float> slice(std::vector<float>& values, std::size_t begin, std::size_t size) {
    return std::span<float>(values).subspan(begin, size);
}

}  // namespace

MonomialOperator::MonomialOperator(std::size_t dimension)
    : permutation_(dimension), signs_(dimension, 1), scales_(dimension, 1.0F), biases_(dimension, 0.0F) {
    require(dimension <= std::numeric_limits<std::uint16_t>::max(), "monomial dimension exceeds uint16 permutation range");
    std::iota(permutation_.begin(), permutation_.end(), static_cast<std::uint16_t>(0));
}

MonomialOperator MonomialOperator::identity(std::size_t dimension) {
    return MonomialOperator(dimension);
}

MonomialOperator MonomialOperator::seeded(std::size_t dimension, std::uint64_t seed) {
    MonomialOperator result(dimension);
    std::mt19937_64 rng(seed);
    std::shuffle(result.permutation_.begin(), result.permutation_.end(), rng);
    for (auto& sign : result.signs_) sign = (rng() & 1ULL) != 0ULL ? 1 : -1;
    return result;
}

void MonomialOperator::apply(std::span<const float> input, std::span<float> output) const {
    require(input.size() == dimension() && output.size() == dimension(), "monomial apply dimension mismatch");
    for (std::size_t i = 0; i < dimension(); ++i) {
        const float permuted = static_cast<float>(signs_[i]) * input[permutation_[i]];
        output[i] = std::fma(scales_[i], permuted, biases_[i]);
    }
}

std::vector<float> MonomialOperator::apply(std::span<const float> input) const {
    std::vector<float> output(dimension());
    apply(input, output);
    return output;
}

MonomialOperator MonomialOperator::compose(const MonomialOperator& after,
                                           const MonomialOperator& before) {
    require(after.dimension() == before.dimension(), "monomial compose dimension mismatch");
    MonomialOperator result(after.dimension());
    for (std::size_t i = 0; i < result.dimension(); ++i) {
        const std::size_t middle = after.permutation_[i];
        result.permutation_[i] = before.permutation_[middle];
        result.signs_[i] = static_cast<std::int8_t>(after.signs_[i] * before.signs_[middle]);
        result.scales_[i] = clamp_finite(after.scales_[i] * before.scales_[middle], -8.0F, 8.0F, 1.0F);
        const float signed_bias = static_cast<float>(after.signs_[i]) * before.biases_[middle];
        result.biases_[i] = clamp_finite(std::fma(after.scales_[i], signed_bias, after.biases_[i]), -8.0F, 8.0F);
    }
    return result;
}

void MonomialOperator::update_delta(std::span<const float> input,
                                    std::span<const float> target,
                                    float learning_rate,
                                    float weight_decay) {
    require(input.size() == dimension() && target.size() == dimension(), "monomial update dimension mismatch");
    const float rate = std::clamp(learning_rate, 0.0F, 1.0F);
    for (std::size_t i = 0; i < dimension(); ++i) {
        const float x = static_cast<float>(signs_[i]) * input[permutation_[i]];
        const float predicted = std::fma(scales_[i], x, biases_[i]);
        const float error = target[i] - predicted;
        scales_[i] = clamp_finite(scales_[i] + rate * (error * x - weight_decay * scales_[i]), -4.0F, 4.0F, 1.0F);
        biases_[i] = clamp_finite(biases_[i] + rate * (error - weight_decay * biases_[i]), -4.0F, 4.0F);
    }
}

float MonomialOperator::normalized_error(std::span<const float> input,
                                         std::span<const float> target) const {
    const auto predicted = apply(input);
    float numerator = 0.0F;
    float denominator = 0.0F;
    for (std::size_t i = 0; i < target.size(); ++i) {
        const float error = predicted[i] - target[i];
        numerator = std::fma(error, error, numerator);
        denominator = std::fma(target[i], target[i], denominator);
    }
    return std::sqrt(numerator / std::max(denominator, 1.0e-12F));
}

bool MonomialOperator::same_structure(const MonomialOperator& other) const noexcept {
    return permutation_ == other.permutation_ && signs_ == other.signs_;
}

std::uint64_t MonomialOperator::structure_hash() const noexcept {
    std::uint64_t hash = hash_bytes(std::as_bytes(std::span(permutation_)));
    return hash_bytes(std::as_bytes(std::span(signs_)), hash);
}

std::uint64_t MonomialOperator::full_hash() const noexcept {
    std::uint64_t hash = structure_hash();
    hash = hash_floats(scales_, hash);
    return hash_floats(biases_, hash);
}

MonomialOperator MonomialOperator::blend_same_structure(std::span<const MonomialOperator> operators,
                                                        std::span<const float> weights) {
    require(!operators.empty() && operators.size() == weights.size(), "invalid monomial blend");
    MonomialOperator result = operators.front();
    float total = 0.0F;
    std::fill(result.scales_.begin(), result.scales_.end(), 0.0F);
    std::fill(result.biases_.begin(), result.biases_.end(), 0.0F);
    for (std::size_t n = 0; n < operators.size(); ++n) {
        require(operators.front().same_structure(operators[n]), "cannot blend different monomial structures");
        const float weight = std::max(weights[n], 0.0F);
        total += weight;
        for (std::size_t i = 0; i < result.dimension(); ++i) {
            result.scales_[i] = std::fma(weight, operators[n].scales_[i], result.scales_[i]);
            result.biases_[i] = std::fma(weight, operators[n].biases_[i], result.biases_[i]);
        }
    }
    const float denominator = total > 1.0e-12F ? total : static_cast<float>(operators.size());
    for (std::size_t i = 0; i < result.dimension(); ++i) {
        result.scales_[i] /= denominator;
        result.biases_[i] /= denominator;
    }
    return result;
}

std::vector<std::byte> MonomialOperator::serialize() const {
    BinaryWriter writer;
    writer.vector<std::uint16_t>(permutation_);
    writer.vector<std::int8_t>(signs_);
    writer.vector<float>(scales_);
    writer.vector<float>(biases_);
    return writer.take();
}

MonomialOperator MonomialOperator::deserialize(std::span<const std::byte> bytes) {
    BinaryReader reader(bytes);
    MonomialOperator result;
    result.permutation_ = reader.vector<std::uint16_t>();
    result.signs_ = reader.vector<std::int8_t>();
    result.scales_ = reader.vector<float>();
    result.biases_ = reader.vector<float>();
    require(reader.empty(), "trailing monomial payload");
    require(!result.permutation_.empty(), "empty monomial operator");
    require(result.signs_.size() == result.dimension() && result.scales_.size() == result.dimension() &&
            result.biases_.size() == result.dimension(), "corrupt monomial dimensions");
    std::vector<bool> seen(result.dimension(), false);
    for (const auto index : result.permutation_) {
        require(index < result.dimension() && !seen[index], "invalid monomial permutation");
        seen[index] = true;
    }
    for (const auto sign : result.signs_) require(sign == 1 || sign == -1, "invalid monomial sign");
    return result;
}

RelationChannelLayout RelationChannelLayout::for_dimension(std::size_t dimension) {
    require(dimension >= 8U, "relation vector dimension must be at least 8");
    // Proportions preserve all seven functional channels even at small dimensions.
    std::array<std::size_t, 7> sizes{dimension * 25U / 100U,
                                     dimension * 12U / 100U,
                                     dimension * 12U / 100U,
                                     dimension * 25U / 100U,
                                     1U, 1U, 1U};
    for (auto& size : sizes) size = std::max<std::size_t>(size, 1U);
    std::size_t used = std::accumulate(sizes.begin(), sizes.end(), std::size_t{0});
    while (used > dimension) {
        auto it = std::max_element(sizes.begin(), sizes.begin() + 4);
        require(*it > 1U, "relation channel allocation failed");
        --(*it);
        --used;
    }
    sizes[6] += dimension - used;

    RelationChannelLayout layout;
    std::size_t offset = 0;
    layout.semantic_begin = offset; layout.semantic_size = sizes[0]; offset += sizes[0];
    layout.role_begin = offset; layout.role_size = sizes[1]; offset += sizes[1];
    layout.temporal_begin = offset; layout.temporal_size = sizes[2]; offset += sizes[2];
    layout.composition_begin = offset; layout.composition_size = sizes[3]; offset += sizes[3];
    layout.continuation_begin = offset; layout.continuation_size = sizes[4]; offset += sizes[4];
    layout.closure_begin = offset; layout.closure_size = sizes[5]; offset += sizes[5];
    layout.confidence_begin = offset; layout.confidence_size = sizes[6];
    return layout;
}

RelationVector::RelationVector(std::size_t dimension) : values_(dimension, 0.0F) {
    (void)RelationChannelLayout::for_dimension(dimension);
}

std::span<const float> RelationVector::semantic() const {
    const auto l = layout(); return slice(values_, l.semantic_begin, l.semantic_size);
}
std::span<const float> RelationVector::role() const {
    const auto l = layout(); return slice(values_, l.role_begin, l.role_size);
}
std::span<const float> RelationVector::temporal() const {
    const auto l = layout(); return slice(values_, l.temporal_begin, l.temporal_size);
}
std::span<const float> RelationVector::composition() const {
    const auto l = layout(); return slice(values_, l.composition_begin, l.composition_size);
}
float RelationVector::continuation_signal() const noexcept {
    if (values_.empty()) return 0.0F;
    const auto l = RelationChannelLayout::for_dimension(values_.size());
    return values_[l.continuation_begin];
}
float RelationVector::closure_signal() const noexcept {
    if (values_.empty()) return 0.0F;
    const auto l = RelationChannelLayout::for_dimension(values_.size());
    return values_[l.closure_begin];
}
float RelationVector::confidence_state() const noexcept {
    if (values_.empty()) return 0.0F;
    const auto l = RelationChannelLayout::for_dimension(values_.size());
    float sum = 0.0F;
    for (std::size_t i = 0; i < l.confidence_size; ++i) sum += values_[l.confidence_begin + i];
    return sum / static_cast<float>(l.confidence_size);
}

void RelationVector::update_observation(std::span<const float> source,
                                        std::span<const float> target,
                                        std::uint64_t role_signature,
                                        std::uint32_t temporal_distance,
                                        std::uint64_t composition_signature,
                                        bool closes,
                                        float confidence,
                                        float learning_rate) {
    require(source.size() == target.size(), "relation observation embedding mismatch");
    const auto l = layout();
    std::vector<float> observation(values_.size(), 0.0F);

    auto semantic_target = slice(observation, l.semantic_begin, l.semantic_size);
    for (std::size_t i = 0; i < semantic_target.size(); ++i) {
        const std::size_t source_index = (i * source.size()) / semantic_target.size();
        semantic_target[i] = target[source_index] - source[source_index];
    }
    normalize_in_place(semantic_target);

    fill_hashed(slice(observation, l.role_begin, l.role_size), role_signature);
    auto temporal_target = slice(observation, l.temporal_begin, l.temporal_size);
    fill_hashed(temporal_target, temporal_distance);
    if (!temporal_target.empty()) temporal_target.front() = 1.0F / (1.0F + static_cast<float>(temporal_distance));
    fill_hashed(slice(observation, l.composition_begin, l.composition_size), composition_signature);
    observation[l.continuation_begin] = closes ? -1.0F : 1.0F;
    observation[l.closure_begin] = closes ? 1.0F : -1.0F;
    for (std::size_t i = 0; i < l.confidence_size; ++i) observation[l.confidence_begin + i] = safe_logit(confidence) / 8.0F;
    ema(values_, observation, learning_rate);
    for (float& value : values_) value = clamp_finite(value, -1.0F, 1.0F);
}

float RelationVector::context_compatibility(std::span<const float> contextual_state,
                                            std::uint64_t role_signature,
                                            std::uint32_t temporal_distance,
                                            std::uint64_t composition_signature) const {
    const auto l = layout();
    float semantic_score = 0.0F;
    const auto sem = semantic();
    if (!sem.empty() && !contextual_state.empty()) {
        std::vector<float> projected(sem.size());
        for (std::size_t i = 0; i < sem.size(); ++i) {
            projected[i] = contextual_state[(i * contextual_state.size()) / sem.size()];
        }
        normalize_in_place(projected);
        semantic_score = cosine(sem, projected);
    }

    std::vector<float> role_key(l.role_size);
    fill_hashed(role_key, role_signature);
    std::vector<float> composition_key(l.composition_size);
    fill_hashed(composition_key, composition_signature);
    std::vector<float> temporal_key(l.temporal_size);
    fill_hashed(temporal_key, temporal_distance);
    if (!temporal_key.empty()) temporal_key.front() = 1.0F / (1.0F + static_cast<float>(temporal_distance));

    const float role_score = l.role_size != 0U ? cosine(role(), role_key) : 0.0F;
    const float temporal_score = l.temporal_size != 0U ? cosine(temporal(), temporal_key) : 0.0F;
    const float composition_score = l.composition_size != 0U ? cosine(composition(), composition_key) : 0.0F;
    return 0.40F * semantic_score + 0.20F * role_score + 0.15F * temporal_score + 0.25F * composition_score;
}

std::vector<std::byte> RelationVector::serialize_quantized() const {
    require(!values_.empty(), "cannot serialize empty relation vector");
    float maximum = 0.0F;
    for (const float value : values_) maximum = std::max(maximum, std::abs(value));
    const float scale = maximum > 1.0e-12F ? maximum / 127.0F : 1.0F;
    BinaryWriter writer;
    writer.pod<std::uint32_t>(static_cast<std::uint32_t>(values_.size()));
    writer.pod(scale);
    std::vector<std::int8_t> quantized(values_.size());
    for (std::size_t i = 0; i < values_.size(); ++i) {
        quantized[i] = static_cast<std::int8_t>(std::clamp(static_cast<int>(std::nearbyint(values_[i] / scale)), -127, 127));
    }
    writer.vector<std::int8_t>(quantized);
    return writer.take();
}

RelationVector RelationVector::deserialize_quantized(std::span<const std::byte> bytes) {
    BinaryReader reader(bytes);
    const auto dimension = reader.pod<std::uint32_t>();
    const float scale = reader.pod<float>();
    const auto quantized = reader.vector<std::int8_t>();
    require(reader.empty() && quantized.size() == dimension, "corrupt relation vector");
    RelationVector result(dimension);
    for (std::size_t i = 0; i < quantized.size(); ++i) result.values_[i] = scale * static_cast<float>(quantized[i]);
    return result;
}

std::uint64_t RelationRecord::key_hash() const noexcept {
    std::uint64_t hash = hash_combine(source, destination);
    return hash_combine(hash, prototype);
}

std::vector<std::byte> RelationRecord::serialize() const {
    BinaryWriter writer;
    writer.pod(id);
    writer.pod(source);
    writer.pod(destination);
    writer.pod(prototype);
    writer.pod(level);
    writer.pod(lanes.participates_in_full);
    writer.pod(lanes.participates_in_clean);
    writer.pod(support);
    writer.pod(confidence);
    writer.pod(version);
    writer.pod(created_at_ms);
    writer.pod(updated_at_ms);
    writer.pod(expires_at_ms);
    writer.pod(escrow_state);
    writer.pod(derived);
    writer.vector<RelationId>(derived_from);
    const auto transform_bytes = transform.serialize();
    writer.vector<std::byte>(transform_bytes);
    const auto relation_bytes = relation.serialize_quantized();
    writer.vector<std::byte>(relation_bytes);
    return writer.take();
}

RelationRecord RelationRecord::deserialize(std::span<const std::byte> bytes) {
    BinaryReader reader(bytes);
    RelationRecord result;
    result.id = reader.pod<RelationId>();
    result.source = reader.pod<NodeId>();
    result.destination = reader.pod<NodeId>();
    result.prototype = reader.pod<std::uint8_t>();
    result.level = reader.pod<MemoryLevel>();
    result.lanes.participates_in_full = reader.pod<bool>();
    result.lanes.participates_in_clean = reader.pod<bool>();
    result.support = reader.pod<std::uint64_t>();
    result.confidence = reader.pod<float>();
    result.version = reader.pod<std::uint64_t>();
    result.created_at_ms = reader.pod<std::int64_t>();
    result.updated_at_ms = reader.pod<std::int64_t>();
    result.expires_at_ms = reader.pod<std::int64_t>();
    result.escrow_state = reader.pod<EscrowState>();
    result.derived = reader.pod<bool>();
    result.derived_from = reader.vector<RelationId>();
    const auto transform_bytes = reader.vector<std::byte>();
    result.transform = MonomialOperator::deserialize(transform_bytes);
    const auto relation_bytes = reader.vector<std::byte>();
    result.relation = RelationVector::deserialize_quantized(relation_bytes);
    require(reader.empty(), "trailing relation record payload");
    require(result.lanes.participates_in_full, "relations must participate in FULL");
    require(result.lanes.participates_in_clean == (result.level == MemoryLevel::M2),
            "relation lane mask does not match memory level");
    return result;
}

float path_confidence(std::span<const float> edge_confidences,
                      float epsilon,
                      float length_penalty) {
    if (edge_confidences.empty()) return 0.0F;
    float minimum = 1.0F;
    float mean_logit = 0.0F;
    for (const float confidence : edge_confidences) {
        minimum = std::min(minimum, confidence);
        mean_logit += safe_logit(confidence);
    }
    mean_logit /= static_cast<float>(edge_confidences.size());
    const float score = safe_logit(minimum) + epsilon * mean_logit -
                        length_penalty * std::log1p(static_cast<float>(edge_confidences.size()));
    return sigmoid(score);
}

}  // namespace mrdl

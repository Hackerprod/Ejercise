#pragma once

#include "mrdl/common.hpp"

namespace mrdl {

class MonomialOperator final {
public:
    MonomialOperator() = default;
    explicit MonomialOperator(std::size_t dimension);

    static MonomialOperator identity(std::size_t dimension);
    static MonomialOperator seeded(std::size_t dimension, std::uint64_t seed);

    [[nodiscard]] std::size_t dimension() const noexcept { return permutation_.size(); }
    [[nodiscard]] const std::vector<std::uint16_t>& permutation() const noexcept { return permutation_; }
    [[nodiscard]] const std::vector<std::int8_t>& signs() const noexcept { return signs_; }
    [[nodiscard]] const std::vector<float>& scales() const noexcept { return scales_; }
    [[nodiscard]] const std::vector<float>& biases() const noexcept { return biases_; }

    void apply(std::span<const float> input, std::span<float> output) const;
    [[nodiscard]] std::vector<float> apply(std::span<const float> input) const;

    // Returns after(this(input)).
    [[nodiscard]] static MonomialOperator compose(const MonomialOperator& after,
                                                   const MonomialOperator& before);

    void update_delta(std::span<const float> input,
                      std::span<const float> target,
                      float learning_rate,
                      float weight_decay);

    [[nodiscard]] float normalized_error(std::span<const float> input,
                                         std::span<const float> target) const;
    [[nodiscard]] bool same_structure(const MonomialOperator& other) const noexcept;
    [[nodiscard]] std::uint64_t structure_hash() const noexcept;
    [[nodiscard]] std::uint64_t full_hash() const noexcept;

    static MonomialOperator blend_same_structure(std::span<const MonomialOperator> operators,
                                                  std::span<const float> weights);

    [[nodiscard]] std::vector<std::byte> serialize() const;
    static MonomialOperator deserialize(std::span<const std::byte> bytes);

private:
    std::vector<std::uint16_t> permutation_;
    std::vector<std::int8_t> signs_;
    std::vector<float> scales_;
    std::vector<float> biases_;
};

struct RelationChannelLayout {
    std::size_t semantic_begin{0};
    std::size_t semantic_size{0};
    std::size_t role_begin{0};
    std::size_t role_size{0};
    std::size_t temporal_begin{0};
    std::size_t temporal_size{0};
    std::size_t composition_begin{0};
    std::size_t composition_size{0};
    std::size_t continuation_begin{0};
    std::size_t continuation_size{0};
    std::size_t closure_begin{0};
    std::size_t closure_size{0};
    std::size_t confidence_begin{0};
    std::size_t confidence_size{0};

    static RelationChannelLayout for_dimension(std::size_t dimension);
};

class RelationVector final {
public:
    RelationVector() = default;
    explicit RelationVector(std::size_t dimension);

    [[nodiscard]] std::size_t dimension() const noexcept { return values_.size(); }
    [[nodiscard]] const std::vector<float>& values() const noexcept { return values_; }
    [[nodiscard]] std::vector<float>& values() noexcept { return values_; }
    [[nodiscard]] RelationChannelLayout layout() const { return RelationChannelLayout::for_dimension(values_.size()); }

    [[nodiscard]] std::span<const float> semantic() const;
    [[nodiscard]] std::span<const float> role() const;
    [[nodiscard]] std::span<const float> temporal() const;
    [[nodiscard]] std::span<const float> composition() const;
    [[nodiscard]] float continuation_signal() const noexcept;
    [[nodiscard]] float closure_signal() const noexcept;
    [[nodiscard]] float confidence_state() const noexcept;

    void update_observation(std::span<const float> source,
                            std::span<const float> target,
                            std::uint64_t role_signature,
                            std::uint32_t temporal_distance,
                            std::uint64_t composition_signature,
                            bool closes,
                            float confidence,
                            float learning_rate);

    [[nodiscard]] float context_compatibility(std::span<const float> contextual_state,
                                              std::uint64_t role_signature,
                                              std::uint32_t temporal_distance,
                                              std::uint64_t composition_signature) const;

    [[nodiscard]] std::vector<std::byte> serialize_quantized() const;
    static RelationVector deserialize_quantized(std::span<const std::byte> bytes);

private:
    std::vector<float> values_;
};

struct RelationRecord {
    RelationId id{0};
    NodeId source{0};
    NodeId destination{0};
    std::uint8_t prototype{0};
    MemoryLevel level{MemoryLevel::M1};
    LaneMask lanes{LaneMask::from_level(MemoryLevel::M1)};
    std::uint64_t support{0};
    float confidence{0.1F};
    std::uint64_t version{1};
    std::int64_t created_at_ms{0};
    std::int64_t updated_at_ms{0};
    std::int64_t expires_at_ms{0};
    EscrowState escrow_state{EscrowState::Active};
    bool derived{false};
    std::vector<RelationId> derived_from;
    MonomialOperator transform;
    RelationVector relation;

    [[nodiscard]] bool eligible(Lane lane) const noexcept { return lanes.participates(lane); }
    [[nodiscard]] std::uint64_t key_hash() const noexcept;
    [[nodiscard]] std::vector<std::byte> serialize() const;
    static RelationRecord deserialize(std::span<const std::byte> bytes);
};

struct RelationKey {
    NodeId source{0};
    NodeId destination{0};
    std::uint8_t prototype{0};

    bool operator==(const RelationKey&) const = default;
};

struct RelationKeyHash {
    std::size_t operator()(const RelationKey& key) const noexcept {
        std::uint64_t hash = hash_combine(key.source, key.destination);
        hash = hash_combine(hash, key.prototype);
        return static_cast<std::size_t>(hash);
    }
};

[[nodiscard]] float path_confidence(std::span<const float> edge_confidences,
                                    float epsilon,
                                    float length_penalty);

}  // namespace mrdl

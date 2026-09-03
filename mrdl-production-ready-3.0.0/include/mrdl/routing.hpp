#pragma once

#include "mrdl/common.hpp"
#include "mrdl/config.hpp"
#include "mrdl/controller.hpp"
#include "mrdl/relation.hpp"

namespace mrdl {

struct RoleBinding {
    RoleId role{0};
    TokenId entity{0};
    std::vector<float> bound_vector;
};

struct OpenExpectation {
    std::uint64_t key{0};
    float strength{0.0F};
};

struct RouteCapsule {
    BranchId id{0};
    NodeId current_node{0};
    std::vector<float> contextual_state;
    std::vector<RoleBinding> role_bindings;
    std::vector<OpenExpectation> open_expectations;
    float energy{1.0F};
    float accumulated_score{0.0F};
    std::uint64_t route_signature{0};
    std::vector<BranchId> parent_references;
    std::vector<RelationId> local_contributions;
    MonomialOperator composed_transform;
    std::vector<float> edge_confidences;
    std::uint32_t depth{0};
    std::uint32_t port_id{0};
};

struct ContextPort {
    std::uint32_t id{0};
    std::vector<float> frozen_key;
    std::uint32_t capacity{0};
    std::uint32_t current_load{0};
    float utility{0.0F};
    std::vector<float> accumulated_key;
    float accumulated_energy{0.0F};
};

struct PortAssignment {
    std::array<std::uint32_t, 2> ports{0, 0};
    std::array<float, 2> energies{0.0F, 0.0F};
    std::uint8_t count{0};
    bool overflow{false};
};

class IPortRouter {
public:
    virtual ~IPortRouter() = default;
    virtual void begin_round() = 0;
    virtual PortAssignment route(NodeId node, std::span<const float> key, float energy) = 0;
    virtual void end_round() = 0;
};

class OnePassPortRouter final : public IPortRouter {
public:
    explicit OnePassPortRouter(const EngineConfig& config);

    void begin_round() override;
    PortAssignment route(NodeId node, std::span<const float> key, float energy) override;
    void end_round() override;

    [[nodiscard]] std::size_t port_count(NodeId node) const;
    [[nodiscard]] std::uint64_t assignment_count() const noexcept { return assignment_count_; }

private:
    EngineConfig config_;
    std::unordered_map<NodeId, std::vector<ContextPort>> ports_;
    std::unordered_map<NodeId, std::unordered_map<std::uint64_t, std::uint32_t>> key_histograms_;
    std::uint32_t next_port_id_{1};
    std::uint64_t assignment_count_{0};

    [[nodiscard]] float pressure(NodeId node) const;
    static std::uint64_t quantized_key_hash(std::span<const float> key);
    ContextPort& create_port(NodeId node, std::span<const float> key);
};

struct EphemeralTunnel {
    NodeId source{0};
    NodeId destination{0};
    MonomialOperator composed_relation;
    std::vector<RelationId> provenance_path;
    std::uint64_t context_key{0};
    std::uint32_t ttl_rounds{1};
};

struct Branch {
    RouteCapsule capsule;
    TokenId candidate{0};
    float score{0.0F};
    ScoreFeatures features{};
    GateDecision gate_decision{GateDecision::Defer};
    RelationId relation_id{0};
};

class IFoldPolicy {
public:
    virtual ~IFoldPolicy() = default;
    [[nodiscard]] virtual std::vector<Branch> fold(std::vector<Branch> branches,
                                                   std::size_t budget) const = 0;
};

class DiverseBeamFold final : public IFoldPolicy {
public:
    explicit DiverseBeamFold(float duplicate_penalty = 0.10F) : duplicate_penalty_(duplicate_penalty) {}
    [[nodiscard]] std::vector<Branch> fold(std::vector<Branch> branches,
                                           std::size_t budget) const override;
private:
    float duplicate_penalty_{0.10F};
};

class ContextHistory final {
public:
    explicit ContextHistory(std::size_t capacity = 64U) : capacity_(std::max<std::size_t>(capacity, 4U)) {}

    void observe_token(TokenId token);
    void observe_route(std::uint64_t signature);
    void observe_edge(RelationId relation);

    [[nodiscard]] float repetition_penalty(TokenId token) const;
    [[nodiscard]] float cycle_penalty(std::uint64_t next_signature) const;
    [[nodiscard]] float saturation(RelationId relation) const;
    [[nodiscard]] bool short_loop_detected(TokenId candidate) const;
    [[nodiscard]] std::uint64_t compressed_hash() const noexcept;

private:
    std::size_t capacity_;
    std::vector<TokenId> tokens_;
    std::vector<std::uint64_t> routes_;
    std::unordered_map<RelationId, std::uint32_t> edge_use_;
};

[[nodiscard]] std::vector<float> make_port_key(const RouteCapsule& capsule,
                                               std::size_t dimension = 16U);
[[nodiscard]] std::uint64_t structural_slot_key(std::span<const TokenId> context,
                                                std::size_t position,
                                                std::uint64_t route_signature);

}  // namespace mrdl

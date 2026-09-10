#pragma once

#include "mrdl/common.hpp"
#include "mrdl/controller.hpp"
#include "mrdl/graph.hpp"
#include "mrdl/persistence.hpp"
#include "mrdl/replay.hpp"

namespace mrdl {

struct EscrowObservation {
    std::uint64_t contextual_key{0};
    TokenId observed_content{0};
    std::vector<TokenId> context_tokens;
    std::vector<float> bound_frame;
    std::vector<ReplayId> active_trace;
    ReplayClosure replay_closure;
    std::string source;
    std::int64_t timestamp_ms{0};
    float support{1.0F};
    float contradiction{0.0F};
    std::uint32_t source_position{0};

    [[nodiscard]] std::vector<std::byte> serialize() const;
    static EscrowObservation deserialize(std::span<const std::byte> bytes);
};

struct EscrowRecord {
    std::uint64_t id{0};
    RelationId relation_id{0};
    std::vector<EscrowObservation> observations;
    std::uint64_t support{0};
    float confidence_cap{0.45F};
    std::int64_t created_at_ms{0};
    std::int64_t expires_at_ms{0};
    EscrowState state{EscrowState::Active};
    std::int32_t pin_count{0};
    bool expiry_pending{false};
    ReplayClosure closure;

    [[nodiscard]] std::size_t unique_contexts() const;
    [[nodiscard]] std::vector<std::byte> serialize() const;
    static EscrowRecord deserialize(std::span<const std::byte> bytes);
};

struct AuditOutcome {
    bool accepted{false};
    bool stable{false};
    bool replay_exact{true};
    float causal_influence{0.0F};
    float stability_ratio{0.0F};
    std::string reason;
    ScoreFeatures positive_features{};
    std::vector<ScoreFeatures> negative_features;
    std::vector<RoleObservation> role_observations;
};

struct EscrowStats {
    std::uint64_t total{0};
    std::array<std::uint64_t, 7> by_state{};
    std::uint64_t pinned{0};
    std::uint64_t observations{0};
};

class PromotionManager final {
public:
    PromotionManager(GraphStore& graph,
                     ReplayRecorder& replay,
                     Controller& controller,
                     RoleInducer& role_inducer,
                     std::shared_ptr<SqliteModelStore> persistence);

    void load();
    void remember(RelationId relation,
                  EscrowObservation observation,
                  ReplayClosure closure,
                  float confidence_cap,
                  std::int64_t ttl_seconds);

    [[nodiscard]] std::optional<EscrowRecord> get(RelationId relation) const;
    [[nodiscard]] std::vector<RelationId> promotion_candidates(std::uint32_t min_support,
                                                                std::uint32_t min_contexts) const;
    [[nodiscard]] EscrowStats stats() const;

    bool reserve(RelationId relation);
    bool begin_audit(RelationId relation);
    [[nodiscard]] std::optional<PromotionPermit> complete(RelationId relation,
                                                          const AuditOutcome& outcome,
                                                          float controller_learning_rate);
    bool release_to_active(RelationId relation, std::string_view reason);
    bool reject(RelationId relation, std::string_view reason);
    bool mark_unreplayable(RelationId relation, std::string_view reason);

    std::size_t expire_due(std::int64_t now_ms = unix_millis());
    std::size_t collect_rejected_and_unreplayable(std::int64_t older_than_ms);

private:
    GraphStore& graph_;
    ReplayRecorder& replay_;
    Controller& controller_;
    RoleInducer& role_inducer_;
    std::shared_ptr<SqliteModelStore> persistence_;
    mutable std::recursive_mutex mutex_;
    std::unordered_map<RelationId, EscrowRecord> records_;

    [[nodiscard]] bool closure_valid_locked(const EscrowRecord& record) const;
    void persist_locked(const EscrowRecord& record);
    void delete_record_locked(RelationId relation);
    bool expire_record_locked(RelationId relation, EscrowRecord& record);
};

}  // namespace mrdl

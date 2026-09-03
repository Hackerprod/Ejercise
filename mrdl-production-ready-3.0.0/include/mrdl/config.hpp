#pragma once

#include "mrdl/common.hpp"

namespace mrdl {

struct ModelConfig {
    std::uint32_t embedding_dim{96};
    std::uint32_t relation_dim{32};
    std::uint32_t max_relation_prototypes{4};
    std::uint64_t seed{0x4d52444cULL};
};

struct TokenizerConfig {
    std::uint32_t vocab_size{20000};
    std::uint32_t heavy_hitter_multiplier{4};
    bool lowercase{false};
};

struct EngineConfig {
    std::uint32_t top_k_full{16};
    std::uint32_t top_k_clean{16};
    std::uint32_t beam_full{32};
    std::uint32_t beam_clean{32};
    std::uint32_t max_rounds{4};
    std::uint32_t max_ports_per_node{8};
    std::uint32_t port_capacity{8};
    float port_similarity_threshold{0.68F};
    float port_pressure_threshold{2.0F};
    float branch_energy_floor{0.015F};
    float clean_margin{0.30F};
    float clean_health_threshold{0.50F};
    float repetition_penalty{0.35F};
    float cycle_penalty{0.50F};
    float saturation_penalty{0.20F};
    float length_log_penalty{0.08F};
    float confidence_epsilon{0.05F};
    bool exact_pure_reuse{true};
    bool parallel_lanes{true};
};

struct MemoryConfig {
    std::int64_t m1_ttl_seconds{604800};
    float m1_confidence_cap{0.45F};
    std::uint32_t promotion_min_support{4};
    std::uint32_t promotion_min_contexts{3};
    float promotion_min_influence{0.08F};
    float promotion_stability_ratio{0.75F};
    std::uint32_t audit_top_m{8};
};

struct TrainingConfig {
    std::string mode{"B"};
    std::uint32_t context_tokens{16};
    std::uint32_t max_source_capsules{4};
    std::uint32_t epochs{1};
    std::uint32_t batch_tokens{4096};
    std::uint32_t checkpoint_every_tokens{100000};
    float fast_learning_rate{0.04F};
    float controller_learning_rate{0.001F};
    float relation_weight_decay{0.0001F};
    std::uint32_t negative_samples{8};
    bool auto_audit{true};
    bool trusted_source{false};
};

struct PersistenceConfig {
    std::filesystem::path model_dir{"model"};
    std::filesystem::path database{"model/mrdl.db"};
    std::filesystem::path tokenizer{"model/tokenizer.mrdltok"};
    std::filesystem::path embeddings{"model/embeddings.mrdlemb"};
    std::uint32_t sqlite_busy_timeout_ms{10000};
    bool synchronous_full{true};
};

struct RuntimeConfig {
    std::uint32_t threads{4};
    std::uint32_t max_generation_tokens{128};
    float temperature{0.0F};
    std::uint32_t top_p_candidates{32};
};

struct AppConfig {
    ModelConfig model;
    TokenizerConfig tokenizer;
    EngineConfig engine;
    MemoryConfig memory;
    TrainingConfig training;
    PersistenceConfig persistence;
    RuntimeConfig runtime;

    static AppConfig load(const std::filesystem::path& path);
    void save(const std::filesystem::path& path) const;
    void validate() const;
};

class IniDocument {
public:
    static IniDocument load(const std::filesystem::path& path);

    [[nodiscard]] std::optional<std::string> get(std::string_view section, std::string_view key) const;
    [[nodiscard]] std::string get_string(std::string_view section, std::string_view key, std::string fallback) const;
    [[nodiscard]] std::uint32_t get_u32(std::string_view section, std::string_view key, std::uint32_t fallback) const;
    [[nodiscard]] std::uint64_t get_u64(std::string_view section, std::string_view key, std::uint64_t fallback) const;
    [[nodiscard]] std::int64_t get_i64(std::string_view section, std::string_view key, std::int64_t fallback) const;
    [[nodiscard]] float get_float(std::string_view section, std::string_view key, float fallback) const;
    [[nodiscard]] bool get_bool(std::string_view section, std::string_view key, bool fallback) const;
    void validate_known_keys(std::span<const std::string_view> allowed) const;

private:
    std::unordered_map<std::string, std::string> values_;
};

}  // namespace mrdl

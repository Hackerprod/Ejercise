#pragma once

#include "rlm/status.hpp"
#include "rlm/types.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace rlm {

struct SearchConfig final {
  std::size_t beam_width{16};
  std::size_t candidate_k{32};
  std::size_t max_depth{4};
  std::size_t replay_reopen_limit{4096};
  float state_decay{0.65F};
  float relation_mix{0.65F};
  float target_mix{0.35F};
};

struct ScoringConfig final {
  float confidence_weight{1.40F};
  float support_weight{0.25F};
  float relation_weight{0.90F};
  float target_weight{1.10F};
  float context_weight{0.35F};
  float repetition_penalty{0.30F};
};

struct AuditConfig final {
  std::size_t min_exact_cases{3};
  std::size_t max_cases{8};
  std::size_t max_unknown_cases{2};
  float causal_margin{0.05F};
  float min_pass_fraction{0.67F};
  bool allow_empty_clean_bootstrap{true};
};

struct TrainingConfig final {
  std::size_t batch_tokens{4096};
  std::size_t context_radius{4};
  std::size_t evidence_cases_per_edge{4};
  std::size_t auto_promote_per_batch{128};
  std::uint64_t min_support_for_promotion{4};
  float min_confidence_for_promotion{0.55F};
  std::uint64_t m1_ttl_seconds{604800};
  std::size_t max_m1_edges{5'000'000};
  std::size_t checkpoint_every_batches{10};
  bool reject_unknown_tokens{false};
};

struct StorageConfig final {
  std::filesystem::path state_dir{"state"};
  std::filesystem::path embedding_file{"embeddings.rle"};
  std::size_t shards{64};
  Durability durability{Durability::data};
};

struct ReplayConfig final {
  std::size_t max_records{250'000};
};

struct RuntimeConfig final {
  bool parallel_lanes{true};
  std::size_t worker_threads{4};
  std::size_t queue_capacity{1024};
  std::uint16_t service_port{9087};
};

struct EngineConfig final {
  SearchConfig search;
  ScoringConfig scoring;
  AuditConfig audit;
  TrainingConfig training;
  StorageConfig storage;
  ReplayConfig replay;
  RuntimeConfig runtime;

  [[nodiscard]] Status validate() const;
  [[nodiscard]] static Result<EngineConfig> load(const std::filesystem::path& path);
};

}  // namespace rlm

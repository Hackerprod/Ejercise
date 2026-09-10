#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cnrl {

enum class WeightVariant { shared, clone, untied, cold };
enum class KernelKind { scalar, avx2_repeat, avx2_fused };
enum class TransitionKind { frozen, fixed_point, group_rms, global_rms };
enum class GateKind { t0r, t0m, t0rm, calibrate };
enum class TimingScope { full_repetition, round_window };

struct Shape {
  std::uint32_t dimension = 512;  // input width D
  std::uint32_t slots = 1;        // latent slots S
  std::uint32_t depth = 16;       // recurrent rounds R
};

struct ShardSpec {
  std::uint32_t worker_index = 0;
  std::uint32_t logical_cpu = 0;
  std::uint32_t row_offset = 0;
  std::uint32_t rows = 0;
};

struct TransitionConfig {
  TransitionKind kind = TransitionKind::frozen;
  std::uint32_t projection_shift = 12;
  std::int32_t state_multiplier = 1;
  std::int32_t output_multiplier = 1;
  std::uint32_t final_shift = 0;
  double target_rms = 32.0;
  double epsilon = 1.0e-6;
};

struct RunConfig {
  GateKind gate = GateKind::t0r;
  Shape shape{};
  WeightVariant variant = WeightVariant::shared;
  KernelKind kernel = KernelKind::avx2_fused;
  TransitionConfig transition{};
  TimingScope timing_scope = TimingScope::full_repetition;
  std::uint32_t slot_tile = 4;
  std::uint32_t warmup_repetitions = 2;
  std::uint32_t timed_repetitions = 10;
  std::uint32_t sequences_per_repetition = 1;
  std::uint32_t seed = 0xC001CAFEU;
  bool phase_profile = false;
  bool require_affinity = true;
  bool allow_smt_siblings = false;
  std::vector<ShardSpec> shards;
};

struct PerWorkerMetrics {
  std::uint32_t logical_cpu = 0;
  std::uint32_t physical_core_index = 0;
  bool affinity_succeeded = false;
  std::uint32_t affinity_error = 0;
  double compute_seconds = 0.0;
  double transition_seconds = 0.0;
  double synchronization_seconds = 0.0;
  double cold_prepare_seconds = 0.0;
  std::uint64_t local_sink = 0;
};

struct RunResult {
  bool valid = false;
  std::string error;
  double elapsed_seconds = 0.0;
  std::uint64_t mac_total = 0;
  double mac_per_second = 0.0;
  std::uint64_t base_weight_bytes = 0;
  std::uint64_t allocated_weight_bytes = 0;
  std::uint64_t logical_weight_load_bytes = 0;
  std::uint64_t one_pass_weight_bytes = 0;
  std::uint64_t distinct_weight_storage_bytes = 0;
  double logical_weight_load_gb_per_second = 0.0;
  double one_pass_weight_gb_per_second = 0.0;
  std::uint64_t output_checksum = 0;
  std::uint64_t state_checksum = 0;
  std::uint64_t round_sink = 0;
  std::uint64_t clipped_cells = 0;
  std::uint64_t transition_cells = 0;
  std::uint64_t weight_hash_signature = 0;
  bool clone_hashes_equal = false;
  bool clone_addresses_distinct = false;
  bool all_affinity_succeeded = false;
  std::vector<PerWorkerMetrics> workers;
};

[[nodiscard]] const char* to_string(WeightVariant value) noexcept;
[[nodiscard]] const char* to_string(KernelKind value) noexcept;
[[nodiscard]] const char* to_string(TransitionKind value) noexcept;
[[nodiscard]] const char* to_string(GateKind value) noexcept;
[[nodiscard]] const char* to_string(TimingScope value) noexcept;
[[nodiscard]] WeightVariant parse_weight_variant(const std::string& text);
[[nodiscard]] KernelKind parse_kernel_kind(const std::string& text);
[[nodiscard]] TransitionKind parse_transition_kind(const std::string& text);
[[nodiscard]] GateKind parse_gate_kind(const std::string& text);
[[nodiscard]] TimingScope parse_timing_scope(const std::string& text);

}  // namespace cnrl

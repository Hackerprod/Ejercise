#ifndef _WIN32
#error "t0m_recurrence_probe requires Windows APIs"
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <immintrin.h>
#include <intrin.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kWeightTile = 16;
constexpr std::uint32_t kDefaultDimension = 512;
constexpr std::uint32_t kDefaultDepth = 16;
constexpr std::uint32_t kDefaultWorkers = 4;
constexpr std::array<std::uint32_t, 4> kDefaultCpus{0, 2, 4, 6};
constexpr long double kRmsEpsilon = 1.0e-6L;
constexpr long double kRequantizationScale = 32.0L;
constexpr std::int32_t kQuantizedMinimum = -127;
constexpr std::int32_t kQuantizedMaximum = 127;

enum class Mode { fused, repeat };
enum class Component { full, gemv_only, transition_only };
enum class Variant { a, b };

struct Options {
  std::uint32_t dimension = kDefaultDimension;
  std::uint32_t slots = 4;
  std::uint32_t recurrent_depth = kDefaultDepth;
  std::uint32_t timed_repetitions = 5;
  std::uint32_t warmup = 1;
  std::uint32_t workers = kDefaultWorkers;
  Mode mode = Mode::fused;
  Component component = Component::full;
  Variant variant = Variant::a;
  std::vector<std::uint32_t> cpus;
  std::vector<std::uint32_t> rows_per_worker{128, 128, 128, 128};
  bool cpus_set = false;
  bool self_test = false;
};

struct CpuTarget {
  std::uint32_t logical_index = 0;
  WORD group = 0;
  BYTE group_index = 0;
};

struct QpcClock {
  LARGE_INTEGER frequency{};

  QpcClock() {
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0) {
      throw std::runtime_error("QueryPerformanceFrequency failed");
    }
  }

  [[nodiscard]] LARGE_INTEGER now() const {
    LARGE_INTEGER value{};
    if (!QueryPerformanceCounter(&value)) {
      throw std::runtime_error("QueryPerformanceCounter failed");
    }
    return value;
  }

  [[nodiscard]] double elapsed(LARGE_INTEGER begin, LARGE_INTEGER end) const {
    return static_cast<double>(end.QuadPart - begin.QuadPart) /
           static_cast<double>(frequency.QuadPart);
  }
};

class AffinityGuard {
 public:
  AffinityGuard(WORD group, BYTE processor) {
    if (!GetThreadGroupAffinity(GetCurrentThread(), &previous_)) {
      error_ = GetLastError();
      return;
    }
    saved_ = true;
    GROUP_AFFINITY requested{};
    requested.Mask = static_cast<KAFFINITY>(1) << processor;
    requested.Group = group;
    if (!SetThreadGroupAffinity(GetCurrentThread(), &requested, nullptr)) {
      error_ = GetLastError();
      return;
    }
    active_ = true;
  }

  ~AffinityGuard() {
    if (active_ && saved_) (void)SetThreadGroupAffinity(GetCurrentThread(), &previous_, nullptr);
  }

  [[nodiscard]] bool succeeded() const { return active_; }
  [[nodiscard]] DWORD error() const { return error_; }

 private:
  GROUP_AFFINITY previous_{};
  bool saved_ = false;
  bool active_ = false;
  DWORD error_ = ERROR_SUCCESS;
};

[[nodiscard]] std::uint32_t parse_u32(std::string_view text, std::string_view option,
                                      bool allow_zero) {
  std::uint32_t value = 0;
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
      (!allow_zero && value == 0)) {
    throw std::runtime_error("invalid value for " + std::string(option));
  }
  return value;
}

[[nodiscard]] std::vector<std::uint32_t> parse_list(std::string_view text,
                                                    std::string_view option,
                                                    bool allow_zero) {
  std::vector<std::uint32_t> values;
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    values.push_back(parse_u32(text.substr(start, end - start), option, allow_zero));
    start = end == text.size() ? text.size() : end + 1;
  }
  if (values.empty()) throw std::runtime_error(std::string(option) + " requires values");
  return values;
}

[[nodiscard]] Mode parse_mode(std::string_view text) {
  if (text == "fused") return Mode::fused;
  if (text == "repeat") return Mode::repeat;
  throw std::runtime_error("invalid --mode; expected fused or repeat");
}

[[nodiscard]] Component parse_component(std::string_view text) {
  if (text == "full") return Component::full;
  if (text == "gemv-only") return Component::gemv_only;
  if (text == "transition-only") return Component::transition_only;
  throw std::runtime_error("invalid --component; expected full, gemv-only, or transition-only");
}

[[nodiscard]] Variant parse_variant(std::string_view text) {
  if (text == "A" || text == "a") return Variant::a;
  if (text == "B" || text == "b") return Variant::b;
  throw std::runtime_error("invalid --variant; expected A or B");
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    auto require_value = [&](std::string_view option) -> std::string_view {
      if (index + 1 >= argc) throw std::runtime_error("missing value for " + std::string(option));
      return argv[++index];
    };
    if (argument == "--D") {
      options.dimension = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--S") {
      options.slots = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--R") {
      options.recurrent_depth = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--mode") {
      options.mode = parse_mode(require_value(argument));
    } else if (argument == "--component" || argument == "--phase") {
      options.component = parse_component(require_value(argument));
    } else if (argument == "--variant") {
      options.variant = parse_variant(require_value(argument));
    } else if (argument == "--workers") {
      options.workers = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--cpus") {
      options.cpus = parse_list(require_value(argument), argument, true);
      options.cpus_set = true;
    } else if (argument == "--rows-per-worker") {
      options.rows_per_worker = parse_list(require_value(argument), argument, false);
    } else if (argument == "--timed-repetitions") {
      options.timed_repetitions = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--warmup") {
      options.warmup = parse_u32(require_value(argument), argument, true);
    } else if (argument == "--self-test") {
      options.self_test = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: t0m_recurrence_probe [--D N] [--S 1|2|4|8|16] [--R N] "
                   "[--mode fused|repeat] [--component full|gemv-only|transition-only] "
                   "[--variant A|B] "
                   "[--workers N] [--cpus LIST] "
                   "[--rows-per-worker LIST] [--timed-repetitions N] [--warmup N] [--self-test]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + std::string(argument));
    }
  }
  if (options.slots != 1 && options.slots != 2 && options.slots != 4 &&
      options.slots != 8 && options.slots != 16) {
    throw std::runtime_error("--S must be one of 1, 2, 4, 8, or 16");
  }
  return options;
}

void validate_measurement_options(const Options& options, std::size_t target_count) {
  if (options.rows_per_worker.size() != options.workers) {
    throw std::runtime_error("--rows-per-worker count must equal --workers");
  }
  if (options.cpus_set && options.cpus.size() != options.workers) {
    throw std::runtime_error("--cpus count must equal --workers");
  }
  const std::uint64_t row_sum = std::accumulate(options.rows_per_worker.begin(),
                                                options.rows_per_worker.end(), std::uint64_t{0});
  if (row_sum != options.dimension) {
    throw std::runtime_error("recurrence requires sum(--rows-per-worker) == D");
  }
  if (options.workers > target_count) throw std::runtime_error("--workers exceed active CPUs");
}

[[nodiscard]] std::vector<CpuTarget> enumerate_cpus() {
  std::vector<CpuTarget> targets;
  std::uint32_t logical_index = 0;
  for (WORD group = 0; group < GetActiveProcessorGroupCount(); ++group) {
    const DWORD count = GetActiveProcessorCount(group);
    for (DWORD processor = 0; processor < count; ++processor) {
      targets.push_back(CpuTarget{logical_index++, group, static_cast<BYTE>(processor)});
    }
  }
  return targets;
}

[[nodiscard]] bool runtime_avx2_supported() {
  int registers[4]{};
  __cpuid(registers, 0);
  const int max_leaf = registers[0];
  if (max_leaf < 1) return false;
  __cpuidex(registers, 1, 0);
  if ((registers[2] & (1 << 27)) == 0 || (registers[2] & (1 << 28)) == 0) return false;
  if ((_xgetbv(0) & 0x6) != 0x6 || max_leaf < 7) return false;
  __cpuidex(registers, 7, 0);
  return (registers[1] & (1 << 5)) != 0;
}

[[nodiscard]] std::uint32_t next_random(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

struct Shard {
  std::uint32_t row_offset = 0;
  std::uint32_t rows = 0;
  std::vector<std::vector<std::int8_t>> weight_blocks;
};

struct Workload {
  std::uint32_t dimension = 0;
  std::uint32_t slots = 0;
  std::uint32_t recurrent_depth = 0;
  Variant variant = Variant::a;
  std::vector<Shard> shards;
  std::vector<std::int8_t> initial_state;
};

[[nodiscard]] Workload make_workload(std::uint32_t dimension, std::uint32_t slots,
                                      std::uint32_t recurrent_depth,
                                      const std::vector<std::uint32_t>& rows,
                                      Variant variant, std::uint32_t seed) {
  Workload workload{dimension, slots, recurrent_depth, variant, {}, {}};
  workload.initial_state.resize(static_cast<std::size_t>(slots) * dimension);
  std::uint32_t state_seed = seed ^ 0xA5A5F00DU;
  for (std::int8_t& value : workload.initial_state) {
    value = static_cast<std::int8_t>(static_cast<int>(next_random(state_seed) % 255U) - 127);
  }
  std::uint32_t row_offset = 0;
  for (std::size_t shard_index = 0; shard_index < rows.size(); ++shard_index) {
    Shard shard{row_offset, rows[shard_index], {}};
    const std::uint32_t block_count = variant == Variant::b ? recurrent_depth : 1U;
    shard.weight_blocks.reserve(block_count);
    for (std::uint32_t round = 0; round < block_count; ++round) {
      shard.weight_blocks.emplace_back(static_cast<std::size_t>(shard.rows) * dimension);
      // A keeps original seed. B adds static-probe-style round term.
      std::uint32_t weight_seed = seed ^
          (0x9E3779B9U * (static_cast<std::uint32_t>(shard_index) + 1U));
      if (variant == Variant::b) weight_seed ^= 0x85EBCA6BU * (round + 1U);
      for (std::int8_t& value : shard.weight_blocks.back()) {
        value = static_cast<std::int8_t>(static_cast<int>(next_random(weight_seed) % 255U) - 127);
      }
    }
    workload.shards.push_back(std::move(shard));
    row_offset += rows[shard_index];
  }
  return workload;
}

void add_boundary_data(Workload& workload) {
  if (workload.shards.empty() || workload.shards.front().rows == 0) return;
  for (std::vector<std::int8_t>& weights : workload.shards.front().weight_blocks) {
    std::fill(weights.begin(), weights.begin() + workload.dimension, static_cast<std::int8_t>(127));
  }
  for (std::uint32_t dimension = 0; dimension < workload.dimension; ++dimension) {
    workload.initial_state[dimension] = 127;
  }
}

[[nodiscard]] const std::vector<std::int8_t>& weights_for_round(const Shard& shard,
                                                                  Variant variant,
                                                                  std::uint32_t round) {
  return shard.weight_blocks[variant == Variant::b ? round : 0U];
}

[[nodiscard]] bool checked_add_i64(std::int64_t left, std::int64_t right, std::int64_t& result) {
  if ((right > 0 && left > std::numeric_limits<std::int64_t>::max() - right) ||
      (right < 0 && left < std::numeric_limits<std::int64_t>::min() - right)) {
    return false;
  }
  result = left + right;
  return true;
}

[[nodiscard]] bool checked_mul_i64(std::int64_t left, std::int64_t right, std::int64_t& result) {
  if (left == 0 || right == 0) {
    result = 0;
    return true;
  }
  if (left == -1 && right == std::numeric_limits<std::int64_t>::min()) return false;
  if (right == -1 && left == std::numeric_limits<std::int64_t>::min()) return false;
  if (left > 0) {
    if (right > 0) {
      if (left > std::numeric_limits<std::int64_t>::max() / right) return false;
    } else if (right < std::numeric_limits<std::int64_t>::min() / left) {
      return false;
    }
    result = left * right;
    return true;
  }
  if (right > 0) {
    if (left < std::numeric_limits<std::int64_t>::min() / right) return false;
  } else if (left < std::numeric_limits<std::int64_t>::max() / right) {
    return false;
  }
  result = left * right;
  return true;
}

[[nodiscard]] std::int64_t checked_dot_scalar(const std::int8_t* weights,
                                              const std::int8_t* state,
                                              std::uint32_t dimension,
                                              bool& overflow) {
  std::int64_t sum = 0;
  for (std::uint32_t index = 0; index < dimension; ++index) {
    std::int64_t product = 0;
    if (!checked_mul_i64(static_cast<std::int64_t>(weights[index]),
                         static_cast<std::int64_t>(state[index]), product) ||
        !checked_add_i64(sum, product, sum)) {
      overflow = true;
      return 0;
    }
  }
  return sum;
}

[[nodiscard]] std::int32_t horizontal_sum_i32(__m256i value) {
  __m128i low = _mm256_castsi256_si128(value);
  __m128i high = _mm256_extracti128_si256(value, 1);
  __m128i sum = _mm_add_epi32(low, high);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

[[nodiscard]] std::int64_t checked_dot_avx2(const std::int8_t* weights,
                                            const std::int8_t* state,
                                            std::uint32_t dimension,
                                            bool& overflow) {
  std::int64_t sum = 0;
  std::uint32_t index = 0;
  for (; index + kWeightTile <= dimension; index += kWeightTile) {
    const __m256i weight16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
        reinterpret_cast<const __m128i*>(weights + index)));
    const __m256i state16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
        reinterpret_cast<const __m128i*>(state + index)));
    const std::int64_t chunk = horizontal_sum_i32(_mm256_madd_epi16(weight16, state16));
    if (!checked_add_i64(sum, chunk, sum)) {
      overflow = true;
      return 0;
    }
  }
  for (; index < dimension; ++index) {
    std::int64_t product = 0;
    if (!checked_mul_i64(static_cast<std::int64_t>(weights[index]),
                         static_cast<std::int64_t>(state[index]), product) ||
        !checked_add_i64(sum, product, sum)) {
      overflow = true;
      return 0;
    }
  }
  return sum;
}

void run_reference_gemv(const Workload& workload, const std::vector<std::int8_t>& state,
                        std::vector<std::int64_t>& output, bool& overflow, std::uint32_t round) {
  for (const Shard& shard : workload.shards) {
    const std::vector<std::int8_t>& weights = weights_for_round(shard, workload.variant, round);
    for (std::uint32_t slot = 0; slot < workload.slots; ++slot) {
      const std::int8_t* state_slot = state.data() + static_cast<std::size_t>(slot) * workload.dimension;
      for (std::uint32_t row = 0; row < shard.rows; ++row) {
        output[static_cast<std::size_t>(slot) * workload.dimension + shard.row_offset + row] =
            checked_dot_scalar(weights.data() + static_cast<std::size_t>(row) * workload.dimension,
                               state_slot, workload.dimension, overflow);
      }
    }
  }
}

template <std::size_t S>
void run_fused_impl(const Shard& shard, const std::vector<std::int8_t>& weights,
                   std::uint32_t dimension,
                   const std::vector<std::int8_t>& state,
                   std::vector<std::int64_t>& output, bool& overflow) {
  static_assert(S == 1 || S == 2 || S == 4 || S == 8 || S == 16);
  for (std::uint32_t row = 0; row < shard.rows; ++row) {
    std::array<std::int64_t, S> sums{};
    const std::size_t row_start = static_cast<std::size_t>(row) * dimension;
    std::uint32_t dimension_base = 0;
    for (; dimension_base + kWeightTile <= dimension; dimension_base += kWeightTile) {
      const __m256i weight16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
          reinterpret_cast<const __m128i*>(weights.data() + row_start + dimension_base)));
      for (std::size_t slot = 0; slot < S; ++slot) {
        const std::size_t state_start = static_cast<std::size_t>(slot) * dimension + dimension_base;
        const __m256i state16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(state.data() + state_start)));
        const std::int64_t chunk = horizontal_sum_i32(_mm256_madd_epi16(weight16, state16));
        if (!checked_add_i64(sums[slot], chunk, sums[slot])) overflow = true;
      }
    }
    for (; dimension_base < dimension; ++dimension_base) {
      const std::int64_t weight = weights[row_start + dimension_base];
      for (std::size_t slot = 0; slot < S; ++slot) {
        std::int64_t product = 0;
        if (!checked_mul_i64(weight,
                             static_cast<std::int64_t>(state[static_cast<std::size_t>(slot) * dimension + dimension_base]),
                             product) ||
            !checked_add_i64(sums[slot], product, sums[slot])) {
          overflow = true;
        }
      }
    }
    for (std::size_t slot = 0; slot < S; ++slot) {
      output[slot * dimension + shard.row_offset + row] = sums[slot];
    }
  }
}

void run_fused_gemv(const Workload& workload, const std::vector<std::int8_t>& state,
                    std::vector<std::int64_t>& output, bool& overflow, std::uint32_t round) {
  switch (workload.slots) {
    case 1: for (const Shard& shard : workload.shards) run_fused_impl<1>(shard, weights_for_round(shard, workload.variant, round), workload.dimension, state, output, overflow); break;
    case 2: for (const Shard& shard : workload.shards) run_fused_impl<2>(shard, weights_for_round(shard, workload.variant, round), workload.dimension, state, output, overflow); break;
    case 4: for (const Shard& shard : workload.shards) run_fused_impl<4>(shard, weights_for_round(shard, workload.variant, round), workload.dimension, state, output, overflow); break;
    case 8: for (const Shard& shard : workload.shards) run_fused_impl<8>(shard, weights_for_round(shard, workload.variant, round), workload.dimension, state, output, overflow); break;
    case 16: for (const Shard& shard : workload.shards) run_fused_impl<16>(shard, weights_for_round(shard, workload.variant, round), workload.dimension, state, output, overflow); break;
    default: throw std::runtime_error("unsupported slot specialization");
  }
}

void run_repeat_gemv(const Workload& workload, const std::vector<std::int8_t>& state,
                     std::vector<std::int64_t>& output, bool& overflow, std::uint32_t round) {
  for (const Shard& shard : workload.shards) {
    const std::vector<std::int8_t>& weights = weights_for_round(shard, workload.variant, round);
    for (std::uint32_t slot = 0; slot < workload.slots; ++slot) {
      const std::int8_t* state_slot = state.data() + static_cast<std::size_t>(slot) * workload.dimension;
      for (std::uint32_t row = 0; row < shard.rows; ++row) {
        output[static_cast<std::size_t>(slot) * workload.dimension + shard.row_offset + row] =
            checked_dot_avx2(weights.data() + static_cast<std::size_t>(row) * workload.dimension,
                             state_slot, workload.dimension, overflow);
      }
    }
  }
}

struct TransitionStats {
  bool finite = true;
  bool overflow = false;
  std::uint64_t clipped_cells = 0;
  std::uint64_t cells = 0;
};

struct TransitionWorkspace {
  std::vector<std::int64_t> residuals;
  std::vector<double> floating_residuals;
};

#if defined(_MSC_VER)
#define T0M_NOINLINE __declspec(noinline)
#else
#define T0M_NOINLINE __attribute__((noinline))
#endif

[[nodiscard]] long double round_half_away_from_zero(long double value) {
  return value >= 0.0L ? std::floor(value + 0.5L) : std::ceil(value - 0.5L);
}

T0M_NOINLINE TransitionStats apply_transition_reference(const std::vector<std::int64_t>& output,
                                                        std::vector<std::int8_t>& state,
                                                        std::uint32_t slots,
                                                        std::uint32_t dimension) {
  TransitionStats stats;
  stats.cells = static_cast<std::uint64_t>(slots) * dimension;
  for (std::uint32_t slot = 0; slot < slots; ++slot) {
    const std::size_t base = static_cast<std::size_t>(slot) * dimension;
    std::int64_t sum_squares = 0;
    std::vector<std::int64_t> residuals(dimension);
    for (std::uint32_t index = 0; index < dimension; ++index) {
      if (!checked_add_i64(output[base + index], static_cast<std::int64_t>(state[base + index]),
                           residuals[index])) {
        stats.overflow = true;
        residuals[index] = 0;
      }
      std::int64_t square = 0;
      if (!checked_mul_i64(residuals[index], residuals[index], square) ||
          !checked_add_i64(sum_squares, square, sum_squares)) {
        stats.overflow = true;
      }
    }
    const long double rms = std::sqrt(static_cast<long double>(sum_squares) /
                                      static_cast<long double>(dimension) + kRmsEpsilon);
    if (!std::isfinite(rms) || rms <= 0.0L) {
      stats.finite = false;
      continue;
    }
    for (std::uint32_t index = 0; index < dimension; ++index) {
      const long double normalized = static_cast<long double>(residuals[index]) / rms;
      const long double scaled = kRequantizationScale * normalized;
      const long double rounded = round_half_away_from_zero(scaled);
      if (!std::isfinite(normalized) || !std::isfinite(scaled) || !std::isfinite(rounded) ||
          rounded < static_cast<long double>(std::numeric_limits<std::int32_t>::min()) ||
          rounded > static_cast<long double>(std::numeric_limits<std::int32_t>::max())) {
        stats.finite = false;
        continue;
      }
      const std::int32_t quantized = static_cast<std::int32_t>(rounded);
      if (quantized < kQuantizedMinimum || quantized > kQuantizedMaximum) {
        ++stats.clipped_cells;
      }
      const std::int32_t clamped = std::clamp(quantized, kQuantizedMinimum, kQuantizedMaximum);
      state[base + index] = static_cast<std::int8_t>(clamped);
    }
  }
  return stats;
}

T0M_NOINLINE TransitionStats apply_transition_fast(const std::vector<std::int64_t>& output,
                                                   std::vector<std::int8_t>& state,
                                                   std::uint32_t slots,
                                                   std::uint32_t dimension,
                                                   TransitionWorkspace& workspace) {
  TransitionStats stats;
  stats.cells = static_cast<std::uint64_t>(slots) * dimension;
  const std::size_t cell_count = static_cast<std::size_t>(slots) * dimension;
  workspace.residuals.resize(cell_count);
  workspace.floating_residuals.resize(cell_count);
  constexpr std::int64_t kExactDoubleIntegerLimit = 9007199254740992LL;
  const __m256d sign_mask = _mm256_set1_pd(-0.0);
  const __m256d half = _mm256_set1_pd(0.5);
  const __m256d scale = _mm256_set1_pd(static_cast<double>(kRequantizationScale));
  const __m256d min_i32 = _mm256_set1_pd(static_cast<double>(std::numeric_limits<std::int32_t>::min()));
  const __m256d max_i32 = _mm256_set1_pd(static_cast<double>(std::numeric_limits<std::int32_t>::max()));
  const __m256d one = _mm256_set1_pd(1.0);
  const double epsilon = std::numeric_limits<double>::epsilon();

  for (std::uint32_t slot = 0; slot < slots; ++slot) {
    const std::size_t base = static_cast<std::size_t>(slot) * dimension;
    std::int64_t sum_squares = 0;
    for (std::uint32_t index = 0; index < dimension; ++index) {
      std::int64_t residual = 0;
      if (!checked_add_i64(output[base + index], static_cast<std::int64_t>(state[base + index]), residual)) {
        stats.overflow = true;
      }
      workspace.residuals[base + index] = residual;
      workspace.floating_residuals[base + index] = static_cast<double>(residual);
      std::int64_t square = 0;
      if (!checked_mul_i64(residual, residual, square) ||
          !checked_add_i64(sum_squares, square, sum_squares)) {
        stats.overflow = true;
      }
    }

    const long double scalar_rms = std::sqrt(static_cast<long double>(sum_squares) /
                                             static_cast<long double>(dimension) + kRmsEpsilon);
    if (!std::isfinite(scalar_rms) || scalar_rms <= 0.0L) {
      stats.finite = false;
      continue;
    }

    // Double reduction is exact only while every integer square and partial sum fit exactly.
    double rms = static_cast<double>(scalar_rms);
    if (!stats.overflow && sum_squares <= kExactDoubleIntegerLimit && dimension >= 4 &&
        sizeof(long double) == sizeof(double)) {
      __m256d vector_sum = _mm256_setzero_pd();
      std::uint32_t index = 0;
      for (; index + 4 <= dimension; index += 4) {
        const __m256d values = _mm256_loadu_pd(workspace.floating_residuals.data() + base + index);
        vector_sum = _mm256_add_pd(vector_sum, _mm256_mul_pd(values, values));
      }
      alignas(32) double lanes[4];
      _mm256_store_pd(lanes, vector_sum);
      double vector_sum_scalar = lanes[0] + lanes[1] + lanes[2] + lanes[3];
      for (; index < dimension; ++index) {
        const double value = workspace.floating_residuals[base + index];
        vector_sum_scalar += value * value;
      }
      if (vector_sum_scalar == static_cast<double>(sum_squares)) {
        const __m256d sum = _mm256_set1_pd(vector_sum_scalar);
        const __m256d divisor = _mm256_set1_pd(static_cast<double>(dimension));
        const __m256d mean = _mm256_add_pd(_mm256_div_pd(sum, divisor), _mm256_set1_pd(static_cast<double>(kRmsEpsilon)));
        alignas(32) double vector_rms[4];
        _mm256_store_pd(vector_rms, _mm256_sqrt_pd(mean));
        if (vector_rms[0] == rms) rms = vector_rms[0];
      }
    }
    const long double selected_rms = static_cast<long double>(rms);

    auto apply_scalar = [&](std::uint32_t index) {
      const long double normalized = static_cast<long double>(workspace.residuals[base + index]) / selected_rms;
      const long double scaled = kRequantizationScale * normalized;
      const long double rounded = round_half_away_from_zero(scaled);
      if (!std::isfinite(normalized) || !std::isfinite(scaled) || !std::isfinite(rounded) ||
          rounded < static_cast<long double>(std::numeric_limits<std::int32_t>::min()) ||
          rounded > static_cast<long double>(std::numeric_limits<std::int32_t>::max())) {
        stats.finite = false;
        return;
      }
      const std::int32_t quantized = static_cast<std::int32_t>(rounded);
      if (quantized < kQuantizedMinimum || quantized > kQuantizedMaximum) ++stats.clipped_cells;
      state[base + index] = static_cast<std::int8_t>(std::clamp(quantized, kQuantizedMinimum, kQuantizedMaximum));
    };

    const __m256d rms_vector = _mm256_set1_pd(rms);
    for (std::uint32_t index = 0; index + 4 <= dimension; index += 4) {
      const __m256d values = _mm256_loadu_pd(workspace.floating_residuals.data() + base + index);
      const __m256d normalized = _mm256_div_pd(values, rms_vector);
      const __m256d scaled = _mm256_mul_pd(normalized, scale);
      const __m256d absolute_scaled = _mm256_andnot_pd(sign_mask, scaled);
      const __m256d absolute_rounded = _mm256_round_pd(
          _mm256_add_pd(absolute_scaled, half), _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
      const __m256d rounded = _mm256_or_pd(absolute_rounded, _mm256_and_pd(scaled, sign_mask));
      const __m256d finite_mask = _mm256_and_pd(_mm256_cmp_pd(normalized, normalized, _CMP_ORD_Q),
                                                _mm256_cmp_pd(scaled, scaled, _CMP_ORD_Q));
      const __m256d range_mask = _mm256_and_pd(_mm256_cmp_pd(rounded, min_i32, _CMP_GE_OQ),
                                                _mm256_cmp_pd(rounded, max_i32, _CMP_LE_OQ));
      const __m256d magnitude = _mm256_max_pd(absolute_scaled, one);
      const __m256d safety = _mm256_mul_pd(_mm256_set1_pd(8.0 * epsilon), magnitude);
      const __m256d distance = _mm256_sub_pd(absolute_scaled,
                                              _mm256_sub_pd(absolute_rounded, half));
      const __m256d away_from_half = _mm256_cmp_pd(_mm256_andnot_pd(sign_mask, distance), safety, _CMP_GT_OQ);
      const int safe_mask = _mm256_movemask_pd(_mm256_and_pd(_mm256_and_pd(finite_mask, range_mask), away_from_half));
      if (safe_mask == 0xF) {
        const __m128i quantized_values = _mm256_cvttpd_epi32(rounded);
        alignas(16) std::int32_t quantized[4];
        _mm_store_si128(reinterpret_cast<__m128i*>(quantized), quantized_values);
        for (std::uint32_t lane = 0; lane < 4; ++lane) {
          if (quantized[lane] < kQuantizedMinimum || quantized[lane] > kQuantizedMaximum) ++stats.clipped_cells;
          state[base + index + lane] = static_cast<std::int8_t>(
              std::clamp(quantized[lane], kQuantizedMinimum, kQuantizedMaximum));
        }
      } else {
        for (std::uint32_t lane = 0; lane < 4; ++lane) apply_scalar(index + lane);
      }
    }
    for (std::uint32_t index = dimension & ~3U; index < dimension; ++index) apply_scalar(index);
  }
  return stats;
}

[[nodiscard]] std::uint64_t checksum_state(const std::vector<std::int8_t>& state) {
  std::uint64_t checksum = 0;
  std::uint64_t cell_index = 1;
  for (const std::int8_t value : state) {
    checksum = checksum * 1099511628211ULL ^
               (static_cast<std::uint64_t>(static_cast<std::int64_t>(value)) + cell_index++);
  }
  return checksum;
}

[[nodiscard]] std::uint64_t checksum_output(const std::vector<std::int64_t>& output) {
  std::uint64_t checksum = 1469598103934665603ULL;
  std::uint64_t cell_index = 1;
  for (const std::int64_t value : output) {
    checksum = checksum * 1099511628211ULL ^
               (static_cast<std::uint64_t>(value) + cell_index++);
  }
  return checksum;
}

[[nodiscard]] std::vector<std::int64_t> make_synthetic_output(std::uint32_t slots,
                                                               std::uint32_t dimension) {
  std::vector<std::int64_t> output(static_cast<std::size_t>(slots) * dimension);
  for (std::uint32_t slot = 0; slot < slots; ++slot) {
    for (std::uint32_t index = 0; index < dimension; ++index) {
      const std::int64_t pattern = static_cast<std::int64_t>((index * 17U + slot * 31U) % 127U);
      output[static_cast<std::size_t>(slot) * dimension + index] = pattern - 63;
    }
  }
  return output;
}

[[nodiscard]] std::uint64_t count_equal_cells(const std::vector<std::int8_t>& left,
                                              const std::vector<std::int8_t>& right) {
  if (left.size() != right.size()) throw std::runtime_error("state cell counts differ");
  std::uint64_t equal_cells = 0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (left[index] == right[index]) ++equal_cells;
  }
  return equal_cells;
}

[[nodiscard]] std::string format_rate(std::uint64_t clipped_cells, std::uint64_t total_cells) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(9)
         << (total_cells == 0
                 ? 0.0L
                 : static_cast<long double>(clipped_cells) / static_cast<long double>(total_cells));
  return stream.str();
}

[[nodiscard]] std::string join_u64(const std::vector<std::uint64_t>& values) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) stream << ';';
    stream << values[index];
  }
  return stream.str();
}

[[nodiscard]] std::string join_bool(const std::vector<std::uint8_t>& values);

struct SequenceResult {
  std::vector<std::int8_t> state;
  std::vector<std::vector<std::int8_t>> round_states;
  std::vector<std::uint64_t> round_checksums;
  std::vector<std::uint8_t> round_finite;
  std::vector<std::uint8_t> round_overflow;
  std::vector<std::uint64_t> round_clipped_cells;
  std::uint64_t clipped_cells = 0;
  std::uint64_t total_cells = 0;
  bool all_rounds_valid = true;
};

enum class Kernel { reference, fused, repeat };

SequenceResult run_sequence(const Workload& workload, Kernel kernel) {
  SequenceResult result{workload.initial_state, {}, {}, {}, {}, {}, 0,
                        static_cast<std::uint64_t>(workload.slots) * workload.dimension * workload.recurrent_depth,
                        true};
  result.round_states.reserve(workload.recurrent_depth);
  std::vector<std::int64_t> output(static_cast<std::size_t>(workload.slots) * workload.dimension);
  TransitionWorkspace transition_workspace;
  if (kernel != Kernel::reference) {
    transition_workspace.residuals.resize(output.size());
    transition_workspace.floating_residuals.resize(output.size());
  }
  result.round_checksums.reserve(workload.recurrent_depth);
  for (std::uint32_t round = 0; round < workload.recurrent_depth; ++round) {
    bool overflow = false;
    if (kernel == Kernel::reference) {
      run_reference_gemv(workload, result.state, output, overflow, round);
    } else if (kernel == Kernel::fused) {
      run_fused_gemv(workload, result.state, output, overflow, round);
    } else {
      run_repeat_gemv(workload, result.state, output, overflow, round);
    }
    TransitionStats stats = kernel == Kernel::reference
                                ? apply_transition_reference(output, result.state, workload.slots, workload.dimension)
                                : apply_transition_fast(output, result.state, workload.slots, workload.dimension,
                                                        transition_workspace);
    stats.overflow = stats.overflow || overflow;
    result.all_rounds_valid = result.all_rounds_valid && stats.finite && !stats.overflow;
    result.clipped_cells += stats.clipped_cells;
    result.round_checksums.push_back(checksum_state(result.state));
    result.round_states.push_back(result.state);
    result.round_finite.push_back(stats.finite ? 1U : 0U);
    result.round_overflow.push_back(stats.overflow ? 1U : 0U);
    result.round_clipped_cells.push_back(stats.clipped_cells);
  }
  return result;
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("T0-M recurrence correction failed: " + message);
}

void compare_sequences(const SequenceResult& expected, const SequenceResult& actual,
                       const std::string& label) {
  require(expected.round_checksums.size() == actual.round_checksums.size(), label + " round count differs");
  for (std::size_t round = 0; round < expected.round_checksums.size(); ++round) {
    require(expected.round_states[round] == actual.round_states[round],
            label + " state cell differs at round " + std::to_string(round));
    require(expected.round_checksums[round] == actual.round_checksums[round],
            label + " checksum differs at round " + std::to_string(round));
    require(expected.round_finite[round] == actual.round_finite[round],
            label + " finite status differs at round " + std::to_string(round));
    require(expected.round_overflow[round] == actual.round_overflow[round],
            label + " overflow status differs at round " + std::to_string(round));
    require(expected.round_clipped_cells[round] == actual.round_clipped_cells[round],
            label + " clipping differs at round " + std::to_string(round));
  }
  require(expected.state == actual.state, label + " final state differs");
  require(expected.clipped_cells == actual.clipped_cells, label + " clipping count differs");
  require(actual.all_rounds_valid, label + " has nonfinite or overflow round");
}

void run_self_test(const QpcClock&) {
  require(round_half_away_from_zero(0.5L) == 1.0L, "positive half rounding incorrect");
  require(round_half_away_from_zero(-0.5L) == -1.0L, "negative half rounding incorrect");
  require(round_half_away_from_zero(1.49L) == 1.0L, "positive rounding incorrect");
  require(round_half_away_from_zero(-1.51L) == -2.0L, "negative rounding incorrect");

  for (const Variant variant : {Variant::a, Variant::b}) {
    for (const std::uint32_t dimension : {37U, 53U, 512U}) {
      for (const std::uint32_t slots : {1U, 4U}) {
      const std::vector<std::uint32_t> rows{dimension / 4U, dimension / 4U + 1U,
                                            dimension / 4U + 2U,
                                            dimension - (dimension / 4U * 3U + 3U)};
      Workload workload = make_workload(dimension, slots, 16, rows, variant,
                                        0x13579BDFU ^ dimension ^ slots);
      add_boundary_data(workload);
      const SequenceResult reference = run_sequence(workload, Kernel::reference);
      const SequenceResult fused = run_sequence(workload, Kernel::fused);
      const SequenceResult repeat = run_sequence(workload, Kernel::repeat);
      require(reference.all_rounds_valid, "reference has nonfinite or overflow round");
      compare_sequences(reference, fused, "fused");
      compare_sequences(reference, repeat, "repeat");
      require(reference.clipped_cells > 0, "boundary data produced no clipping");
      require(reference.clipped_cells < reference.total_cells,
              "boundary data caused unexplained all-cell saturation");
      const std::uint64_t round_cells = static_cast<std::uint64_t>(reference.round_states.front().size());
      std::uint64_t fused_equal_total = 0;
      std::uint64_t repeat_equal_total = 0;
      std::uint64_t fused_comparison_total = 0;
      std::uint64_t repeat_comparison_total = 0;
      for (std::size_t round = 0; round < reference.round_states.size(); ++round) {
        const std::uint64_t fused_equal = count_equal_cells(reference.round_states[round],
                                                             fused.round_states[round]);
        const std::uint64_t repeat_equal = count_equal_cells(reference.round_states[round],
                                                             repeat.round_states[round]);
        fused_equal_total += fused_equal;
        repeat_equal_total += repeat_equal;
        fused_comparison_total += static_cast<std::uint64_t>(fused.round_states[round].size());
        repeat_comparison_total += static_cast<std::uint64_t>(repeat.round_states[round].size());
       std::cerr << "self_test_round,variant=" << (variant == Variant::a ? "A" : "B")
                 << ",D=" << dimension << ",S=" << slots
                  << ",R=16,round=" << (round + 1)
                  << ",reference_checksum=" << reference.round_checksums[round]
                  << ",fused_checksum=" << fused.round_checksums[round]
                  << ",repeat_checksum=" << repeat.round_checksums[round]
                  << ",reference_finite=" << (reference.round_finite[round] != 0 ? "true" : "false")
                  << ",fused_finite=" << (fused.round_finite[round] != 0 ? "true" : "false")
                  << ",repeat_finite=" << (repeat.round_finite[round] != 0 ? "true" : "false")
                  << ",reference_overflow=" << (reference.round_overflow[round] != 0 ? "true" : "false")
                  << ",fused_overflow=" << (fused.round_overflow[round] != 0 ? "true" : "false")
                  << ",repeat_overflow=" << (repeat.round_overflow[round] != 0 ? "true" : "false")
                  << ",reference_clipped_cells=" << reference.round_clipped_cells[round]
                  << ",fused_clipped_cells=" << fused.round_clipped_cells[round]
                  << ",repeat_clipped_cells=" << repeat.round_clipped_cells[round]
                  << ",total_cells=" << round_cells
                  << ",fused_cells_equal_to_reference=" << fused_equal
                  << ",repeat_cells_equal_to_reference=" << repeat_equal
                  << ",reference_clipping_rate="
                  << format_rate(reference.round_clipped_cells[round], round_cells)
                  << ",fused_clipping_rate="
                  << format_rate(fused.round_clipped_cells[round], round_cells)
                  << ",repeat_clipping_rate="
                  << format_rate(repeat.round_clipped_cells[round], round_cells) << '\n';
      }
       std::cerr << "self_test_case,variant=" << (variant == Variant::a ? "A" : "B")
                 << ",D=" << dimension << ",S=" << slots
                 << ",R=16,clipped_cells=" << reference.clipped_cells
                 << ",clipping_rate=" << std::setprecision(8)
                 << static_cast<double>(reference.clipped_cells) /
                        static_cast<double>(reference.total_cells) << '\n';
      const std::uint64_t comparison_cells_expected =
          static_cast<std::uint64_t>(reference.round_states.size()) * round_cells;
       std::cerr << "self_test_case_vector,variant=" << (variant == Variant::a ? "A" : "B")
                 << ",D=" << dimension << ",S=" << slots
                << ",R=16,reference_checksums=" << join_u64(reference.round_checksums)
                << ",fused_checksums=" << join_u64(fused.round_checksums)
                << ",repeat_checksums=" << join_u64(repeat.round_checksums)
                << ",reference_finite=" << join_bool(reference.round_finite)
                << ",fused_finite=" << join_bool(fused.round_finite)
                << ",repeat_finite=" << join_bool(repeat.round_finite)
                << ",reference_overflow=" << join_bool(reference.round_overflow)
                << ",fused_overflow=" << join_bool(fused.round_overflow)
                << ",repeat_overflow=" << join_bool(repeat.round_overflow)
                << ",reference_clipped_cells=" << join_u64(reference.round_clipped_cells)
                << ",fused_clipped_cells=" << join_u64(fused.round_clipped_cells)
                << ",repeat_clipped_cells=" << join_u64(repeat.round_clipped_cells)
                << ",total_cells=" << reference.total_cells
                << ",reference_total_clipped_cells=" << reference.clipped_cells
                << ",fused_total_clipped_cells=" << fused.clipped_cells
                << ",repeat_total_clipped_cells=" << repeat.clipped_cells
                << ",reference_total_clipping_rate="
                << format_rate(reference.clipped_cells, reference.total_cells)
                << ",fused_total_clipping_rate=" << format_rate(fused.clipped_cells, fused.total_cells)
                << ",repeat_total_clipping_rate=" << format_rate(repeat.clipped_cells, repeat.total_cells)
                << ",comparison_cells_expected=" << comparison_cells_expected
                << ",fused_comparison_cells_expected=" << comparison_cells_expected
                << ",repeat_comparison_cells_expected=" << comparison_cells_expected
                << ",fused_comparison_cells=" << fused_comparison_total
                << ",repeat_comparison_cells=" << repeat_comparison_total
                << ",fused_cells_equal_to_reference_total=" << fused_equal_total
                << ",repeat_cells_equal_to_reference_total=" << repeat_equal_total << '\n';
      }
    }
  }
  std::cerr << "T0-M recurrence correction passed: reference/fused/repeat match every cell and round "
               "for variants A/B; S=1,4; D=37,53,512; R=16\n";
}

struct TimedResult {
  double elapsed_seconds = 0.0;
  double elapsed_per_timed_step = 0.0;
  double qpc_ticks_per_timed_step = 0.0;
  double tsc_cycles_per_timed_step = 0.0;
  bool tsc_supported = false;
  std::uint64_t mac_total = 0;
  std::uint64_t final_checksum = 0;
  std::vector<std::uint64_t> round_checksums;
  std::vector<std::uint8_t> round_finite;
  std::vector<std::uint8_t> round_overflow;
  std::vector<std::uint64_t> round_clipped_cells;
  std::uint64_t clipped_cells = 0;
  std::uint64_t total_cells = 0;
  bool all_rounds_valid = false;
  bool all_affinity_succeeded = false;
  bool all_timed_repetitions_exact = false;
  std::vector<std::uint8_t> affinity;
  std::vector<DWORD> affinity_errors;
};

TimedResult run_timed(const Workload& workload, Mode mode, Component component,
                      const std::vector<CpuTarget>& targets,
                      std::uint32_t timed_repetitions, std::uint32_t warmup,
                      const QpcClock& clock) {
  const std::uint32_t worker_count = static_cast<std::uint32_t>(workload.shards.size());
  std::vector<std::int8_t> state = workload.initial_state;
  std::vector<std::int64_t> output(static_cast<std::size_t>(workload.slots) * workload.dimension);
  TransitionWorkspace transition_workspace;
  transition_workspace.residuals.resize(output.size());
  transition_workspace.floating_residuals.resize(output.size());
  if (component == Component::transition_only) {
    output = make_synthetic_output(workload.slots, workload.dimension);
  }
  std::vector<std::uint8_t> affinity(worker_count, 0);
  std::vector<DWORD> affinity_errors(worker_count, ERROR_SUCCESS);
  std::vector<std::uint8_t> kernel_overflow(worker_count, 0);
  std::vector<std::uint32_t> completed_repetitions(worker_count, 0);
  std::barrier ready(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier depth_start(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier depth_done(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier repetition_done(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::uint32_t worker_index = 0; worker_index < worker_count; ++worker_index) {
    workers.emplace_back([&, worker_index] {
      AffinityGuard affinity_guard(targets[worker_index].group, targets[worker_index].group_index);
      affinity[worker_index] = affinity_guard.succeeded() ? 1U : 0U;
      affinity_errors[worker_index] = affinity_guard.error();
      ready.arrive_and_wait();
      const Shard& shard = workload.shards[worker_index];
      for (std::uint32_t repetition = 0; repetition < warmup + timed_repetitions; ++repetition) {
        for (std::uint32_t round = 0; round < workload.recurrent_depth; ++round) {
          depth_start.arrive_and_wait();
          bool overflow = false;
          const std::vector<std::int8_t>& round_weights =
              weights_for_round(shard, workload.variant, round);
          if (component != Component::transition_only && mode == Mode::fused) {
            switch (workload.slots) {
              case 1: run_fused_impl<1>(shard, round_weights, workload.dimension, state, output, overflow); break;
              case 2: run_fused_impl<2>(shard, round_weights, workload.dimension, state, output, overflow); break;
              case 4: run_fused_impl<4>(shard, round_weights, workload.dimension, state, output, overflow); break;
              case 8: run_fused_impl<8>(shard, round_weights, workload.dimension, state, output, overflow); break;
              case 16: run_fused_impl<16>(shard, round_weights, workload.dimension, state, output, overflow); break;
              default: overflow = true; break;
            }
          } else if (component != Component::transition_only) {
            for (std::uint32_t slot = 0; slot < workload.slots; ++slot) {
              const std::int8_t* state_slot = state.data() + static_cast<std::size_t>(slot) * workload.dimension;
              for (std::uint32_t row = 0; row < shard.rows; ++row) {
                output[static_cast<std::size_t>(slot) * workload.dimension + shard.row_offset + row] =
                    checked_dot_avx2(round_weights.data() + static_cast<std::size_t>(row) * workload.dimension,
                                     state_slot, workload.dimension, overflow);
              }
            }
          }
          kernel_overflow[worker_index] = overflow ? 1U : 0U;
          depth_done.arrive_and_wait();
        }
        repetition_done.arrive_and_wait();
        ++completed_repetitions[worker_index];
      }
    });
  }
  ready.arrive_and_wait();
  TimedResult result;
  result.affinity = affinity;
  result.affinity_errors = affinity_errors;
  result.total_cells = static_cast<std::uint64_t>(workload.slots) * workload.dimension * timed_repetitions * workload.recurrent_depth;
  result.round_checksums.reserve(workload.recurrent_depth);
  result.round_finite.reserve(workload.recurrent_depth);
  result.round_overflow.reserve(workload.recurrent_depth);
  result.round_clipped_cells.reserve(workload.recurrent_depth);
  bool all_valid = true;
  std::uint64_t clipped_cells = 0;
  double elapsed_seconds = 0.0;
  std::uint64_t qpc_ticks = 0;
  std::uint64_t tsc_cycles = 0;
  for (std::uint32_t repetition = 0; repetition < warmup + timed_repetitions; ++repetition) {
    state = workload.initial_state;
    const bool timed = repetition >= warmup;
    const LARGE_INTEGER begin = timed ? clock.now() : LARGE_INTEGER{};
    const unsigned __int64 tsc_begin = timed ? __rdtsc() : 0;
    std::vector<std::uint64_t> this_round_checksums;
    std::vector<std::uint8_t> this_round_finite;
    std::vector<std::uint8_t> this_round_overflow;
    std::vector<std::uint64_t> this_round_clipped_cells;
    if (timed) this_round_checksums.reserve(workload.recurrent_depth);
    if (timed) {
      this_round_finite.reserve(workload.recurrent_depth);
      this_round_overflow.reserve(workload.recurrent_depth);
      this_round_clipped_cells.reserve(workload.recurrent_depth);
    }
    for (std::uint32_t round = 0; round < workload.recurrent_depth; ++round) {
      depth_start.arrive_and_wait();
      depth_done.arrive_and_wait();
      TransitionStats stats;
      if (component != Component::gemv_only) {
        stats = apply_transition_fast(output, state, workload.slots, workload.dimension,
                                      transition_workspace);
      }
      const bool kernel_valid = std::all_of(kernel_overflow.begin(), kernel_overflow.end(),
                                            [](std::uint8_t value) { return value == 0; });
      const bool valid = component == Component::gemv_only
                             ? kernel_valid
                             : stats.finite && !stats.overflow && kernel_valid;
      all_valid = all_valid && valid;
      if (timed) {
        clipped_cells += stats.clipped_cells;
        this_round_checksums.push_back(component == Component::gemv_only
                                           ? checksum_output(output) : checksum_state(state));
        this_round_finite.push_back((component == Component::gemv_only || stats.finite) ? 1U : 0U);
        this_round_overflow.push_back(valid ? 0U : 1U);
        this_round_clipped_cells.push_back(stats.clipped_cells);
      }
    }
    repetition_done.arrive_and_wait();
    if (timed) {
      const LARGE_INTEGER end = clock.now();
      elapsed_seconds += clock.elapsed(begin, end);
      qpc_ticks += static_cast<std::uint64_t>(end.QuadPart - begin.QuadPart);
      tsc_cycles += static_cast<std::uint64_t>(__rdtsc() - tsc_begin);
      result.round_checksums = std::move(this_round_checksums);
      result.round_finite = std::move(this_round_finite);
      result.round_overflow = std::move(this_round_overflow);
      result.round_clipped_cells = std::move(this_round_clipped_cells);
    }
  }
  for (auto& worker : workers) worker.join();
  result.elapsed_seconds = elapsed_seconds;
  const double timed_steps = static_cast<double>(timed_repetitions) * workload.recurrent_depth;
  result.elapsed_per_timed_step = timed_steps > 0.0 ? elapsed_seconds / timed_steps : 0.0;
  result.qpc_ticks_per_timed_step = timed_steps > 0.0
                                        ? static_cast<double>(qpc_ticks) / timed_steps : 0.0;
  result.tsc_cycles_per_timed_step = timed_steps > 0.0
                                         ? static_cast<double>(tsc_cycles) / timed_steps : 0.0;
  result.tsc_supported = true;
  result.mac_total = static_cast<std::uint64_t>(workload.dimension) * workload.dimension *
                     workload.slots * workload.recurrent_depth * timed_repetitions;
  result.final_checksum = component == Component::gemv_only ? checksum_output(output)
                                                             : checksum_state(state);
  result.clipped_cells = clipped_cells;
  result.all_rounds_valid = all_valid;
  result.all_affinity_succeeded = std::all_of(affinity.begin(), affinity.end(),
                                              [](std::uint8_t value) { return value != 0; });
  result.all_timed_repetitions_exact = true;
  for (const std::uint32_t completed : completed_repetitions) {
    result.all_timed_repetitions_exact = result.all_timed_repetitions_exact &&
                                          completed == warmup + timed_repetitions;
  }
  return result;
}

[[nodiscard]] std::string mode_name(Mode mode) { return mode == Mode::fused ? "fused" : "repeat"; }

[[nodiscard]] std::string variant_name(Variant variant) { return variant == Variant::a ? "A" : "B"; }

[[nodiscard]] std::string component_name(Component component) {
  switch (component) {
    case Component::full: return "full";
    case Component::gemv_only: return "gemv-only";
    case Component::transition_only: return "transition-only";
  }
  return "unknown";
}

[[nodiscard]] std::string join_u32(const std::vector<std::uint32_t>& values) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) stream << ',';
    stream << values[index];
  }
  return stream.str();
}

[[nodiscard]] std::string join_bool(const std::vector<std::uint8_t>& values) {
  std::ostringstream stream;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) stream << ';';
    stream << (values[index] != 0 ? "true" : "false");
  }
  return stream.str();
}

void print_csv(const Options& options, const TimedResult& result,
               const std::vector<std::uint32_t>& selected_cpus) {
  std::cout << "D,S,R,variant,rows_per_worker,component,mode,kernel,elapsed_seconds,elapsed_per_timed_step,"
                "qpc_ticks_per_timed_step,tsc_cycles_per_timed_step,tsc_supported,mac_total,mac_per_second,"
                "checksum_kind,validation_invariant,final_checksum,"
               "per_round_checksums,per_round_finite,per_round_overflow,per_round_clipped_cells,"
               "per_round_clipping_rates,clipped_cells,clipping_rate,all_rounds_valid,worker_count,cpus,"
               "affinity,affinity_errors,affinity_succeeded,timed_repetitions,warmup,timed_repetitions_exact\n";
  const std::uint64_t mac_total = options.component == Component::transition_only ? 0 : result.mac_total;
  const double mac_per_second = result.elapsed_seconds > 0.0
                                     ? static_cast<double>(mac_total) / result.elapsed_seconds : 0.0;
  const double clipping_rate = result.total_cells > 0
                                   ? static_cast<double>(result.clipped_cells) /
                                         static_cast<double>(result.total_cells) : 0.0;
  std::vector<std::uint32_t> worker_indices;
  for (std::uint32_t index = 0; index < options.workers; ++index) worker_indices.push_back(index);
  std::vector<std::uint32_t> affinity_values;
  for (const std::uint8_t value : result.affinity) affinity_values.push_back(value);
  std::vector<std::uint32_t> error_values;
  for (const DWORD value : result.affinity_errors) error_values.push_back(value);
  std::ostringstream round_rates;
  const double round_cell_count = static_cast<double>(options.slots) * options.dimension;
  for (std::size_t index = 0; index < result.round_clipped_cells.size(); ++index) {
    if (index != 0) round_rates << ';';
    round_rates << std::setprecision(17) <<
        (round_cell_count > 0.0 ? static_cast<double>(result.round_clipped_cells[index]) / round_cell_count : 0.0);
  }
  std::cout << options.dimension << ',' << options.slots << ',' << options.recurrent_depth << ','
            << variant_name(options.variant) << ",\""
            << join_u32(options.rows_per_worker) << "\"," << component_name(options.component) << ','
            << mode_name(options.mode)
            << ',' << (options.component == Component::transition_only ? "none" :
                       "avx2_" + mode_name(options.mode)) << ',' << std::setprecision(17)
            << result.elapsed_seconds << ',' << result.elapsed_per_timed_step << ','
            << result.qpc_ticks_per_timed_step << ',' << result.tsc_cycles_per_timed_step << ','
            << (result.tsc_supported ? "true" : "false") << ',' << mac_total << ','
            << mac_per_second << ',' << (options.component == Component::gemv_only ? "output" : "state") << ','
            << (options.component == Component::gemv_only
                    ? "output checksum; GEMV output is deterministic and state remains frozen"
                    : options.component == Component::transition_only
                        ? "state checksum; apply_transition uses deterministic synthetic output"
                        : "state checksum; actual GEMV followed by apply_transition") << ','
            << result.final_checksum << ",\"" << join_u64(result.round_checksums) << "\","
            << '"' << join_bool(result.round_finite) << "\",\""
            << join_bool(result.round_overflow) << "\",\""
            << join_u64(result.round_clipped_cells) << "\",\""
            << round_rates.str() << "\"," << result.clipped_cells << ',' << clipping_rate << ','
            << (result.all_rounds_valid ? "true" : "false") << ',' << options.workers << ",\""
            << join_u32(selected_cpus) << "\",\"" << join_u32(affinity_values) << "\",\""
            << join_u32(error_values) << "\"," << (result.all_affinity_succeeded ? "true" : "false")
            << ',' << options.timed_repetitions << ',' << options.warmup << ",true\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::vector<CpuTarget> available_targets = enumerate_cpus();
    const QpcClock clock;
    if (!runtime_avx2_supported()) throw std::runtime_error("AVX2 unavailable; recurrence probe requires AVX2");
    if (options.self_test) {
      run_self_test(clock);
      return 0;
    }
    validate_measurement_options(options, available_targets.size());
    std::vector<std::uint32_t> selected_cpu_indices;
    if (options.cpus_set) {
      selected_cpu_indices = options.cpus;
    } else if (options.workers == kDefaultWorkers && available_targets.size() > kDefaultCpus.back()) {
      selected_cpu_indices.assign(kDefaultCpus.begin(), kDefaultCpus.end());
    } else {
      for (std::uint32_t index = 0; index < options.workers; ++index) selected_cpu_indices.push_back(index);
    }
    std::vector<CpuTarget> selected_targets;
    for (const std::uint32_t cpu : selected_cpu_indices) {
      if (cpu >= available_targets.size()) throw std::runtime_error("--cpus contains out-of-range CPU");
      selected_targets.push_back(available_targets[cpu]);
    }
    Workload workload = make_workload(options.dimension, options.slots, options.recurrent_depth,
                                       options.rows_per_worker, options.variant, 0xC001CAFEU);
    const TimedResult result = run_timed(workload, options.mode, options.component, selected_targets,
                                         options.timed_repetitions, options.warmup, clock);
    if (!result.all_affinity_succeeded) throw std::runtime_error("affinity gate failed; speed result rejected");
    print_csv(options, result, selected_cpu_indices);
    return result.all_rounds_valid ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}

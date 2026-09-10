#ifndef _WIN32
#error "t0m_int8_probe requires Windows APIs"
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
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kWeightTile = 16;
constexpr std::uint32_t kCorrectionShardCount = 4;
constexpr std::size_t kEvictionBytes = 64ULL * 1024ULL * 1024ULL;

enum class Mode { fused, repeat };
enum class Variant { a, b, c };

struct Options {
  std::uint32_t dimension = 64;
  std::uint32_t slots = 4;
  std::uint32_t recurrent_depth = 1;
  std::uint32_t iterations = 1;
  std::uint32_t timed_repetitions = 5;
  std::uint32_t warmup = 1;
  std::uint32_t slot_tile = 4;
  std::uint32_t workers = 1;
  Mode mode = Mode::fused;
  Variant variant = Variant::a;
  std::vector<std::uint32_t> cpus;
  std::vector<std::uint32_t> rows_per_worker{128};
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
    if (active_ && saved_) {
      (void)SetThreadGroupAffinity(GetCurrentThread(), &previous_, nullptr);
    }
  }

  [[nodiscard]] bool succeeded() const { return active_; }
  [[nodiscard]] DWORD error() const { return error_; }

 private:
  GROUP_AFFINITY previous_{};
  bool saved_ = false;
  bool active_ = false;
  DWORD error_ = ERROR_SUCCESS;
};

[[nodiscard]] std::uint32_t parse_u32(std::string_view text,
                                       std::string_view option,
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

[[nodiscard]] Variant parse_variant(std::string_view text) {
  if (text == "A") return Variant::a;
  if (text == "B") return Variant::b;
  if (text == "C") return Variant::c;
  throw std::runtime_error("invalid --variant; expected A, B, or C");
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
    } else if (argument == "--iterations") {
      options.iterations = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--timed-repetitions") {
      options.timed_repetitions = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--warmup") {
      options.warmup = parse_u32(require_value(argument), argument, true);
    } else if (argument == "--mode") {
      options.mode = parse_mode(require_value(argument));
    } else if (argument == "--variant") {
      options.variant = parse_variant(require_value(argument));
    } else if (argument == "--S-tile") {
      options.slot_tile = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--workers") {
      options.workers = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--cpus") {
      options.cpus = parse_list(require_value(argument), argument, true);
    } else if (argument == "--rows-per-worker") {
      options.rows_per_worker = parse_list(require_value(argument), argument, false);
    } else if (argument == "--self-test") {
      options.self_test = true;
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: t0m_int8_probe [--D N] [--S 1|2|4|8|16] [--R N] "
                   "[--mode fused|repeat] [--variant A|B|C] [--S-tile 2|4|8] "
                   "[--workers N] [--cpus LIST] [--rows-per-worker LIST] "
                   "[--iterations N] [--timed-repetitions N] [--warmup N]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + std::string(argument));
    }
  }
  if (options.slots != 1 && options.slots != 2 && options.slots != 4 &&
      options.slots != 8 && options.slots != 16) {
    throw std::runtime_error("--S must be one of 1, 2, 4, 8, or 16");
  }
  if (options.slot_tile != 2 && options.slot_tile != 4 && options.slot_tile != 8) {
    throw std::runtime_error("--S-tile must be one of 2, 4, or 8");
  }
  if (!options.cpus.empty() && options.cpus.size() != options.workers) {
    throw std::runtime_error("--cpus count must equal --workers");
  }
  if (options.rows_per_worker.size() != options.workers) {
    throw std::runtime_error("--rows-per-worker count must equal --workers");
  }
  return options;
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

[[nodiscard]] std::uint32_t next_random(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

struct WeightBlock {
  std::vector<std::int8_t> values;
};

struct ShardData {
  std::uint32_t rows = 0;
  std::vector<WeightBlock> blocks;
  std::vector<std::int8_t> activations;
};

struct Workload {
  std::uint32_t dimension = 0;
  std::uint32_t slots = 0;
  std::uint32_t recurrent_depth = 0;
  Variant variant = Variant::a;
  std::vector<ShardData> shards;
};

[[nodiscard]] Workload make_workload(std::uint32_t dimension,
                                     std::uint32_t slots,
                                     std::uint32_t recurrent_depth,
                                     Variant variant,
                                     const std::vector<std::uint32_t>& rows) {
  Workload workload{dimension, slots, recurrent_depth, variant, {}};
  workload.shards.reserve(rows.size());
  for (std::size_t shard_index = 0; shard_index < rows.size(); ++shard_index) {
    ShardData shard;
    shard.rows = rows[shard_index];
    shard.activations.resize(static_cast<std::size_t>(slots) * dimension);
    std::uint32_t activation_state = 0xA5A5F00DU ^
                                      (0x9E3779B9U * (static_cast<std::uint32_t>(shard_index) + 1U));
    for (std::int8_t& value : shard.activations) {
      value = static_cast<std::int8_t>(static_cast<int>(next_random(activation_state) % 31U) - 15);
    }
    const std::uint32_t block_count = variant == Variant::b ? recurrent_depth : 1;
    shard.blocks.reserve(block_count);
    for (std::uint32_t depth_index = 0; depth_index < block_count; ++depth_index) {
      WeightBlock block;
      block.values.resize(static_cast<std::size_t>(shard.rows) * dimension);
      std::uint32_t weight_state = 0xC001CAFEU ^
          (0x9E3779B9U * (static_cast<std::uint32_t>(shard_index) + 1U)) ^
          (0x85EBCA6BU * (depth_index + 1U));
      for (std::int8_t& value : block.values) {
        value = static_cast<std::int8_t>(static_cast<int>(next_random(weight_state) % 31U) - 15);
      }
      shard.blocks.push_back(std::move(block));
    }
    workload.shards.push_back(std::move(shard));
  }
  return workload;
}

[[nodiscard]] std::uint64_t calculate_mac_total(const std::vector<std::uint32_t>& rows,
                                                std::uint32_t dimension,
                                                std::uint32_t slots,
                                                std::uint32_t recurrent_depth,
                                                std::uint32_t iterations,
                                                std::uint32_t timed_repetitions) {
  std::uint64_t total = 0;
  for (const std::uint32_t shard_rows : rows) {
    total += static_cast<std::uint64_t>(shard_rows) * dimension * slots * recurrent_depth *
             iterations * timed_repetitions;
  }
  return total;
}

#if defined(_MSC_VER)
#define T0M_NOINLINE __declspec(noinline)
#define T0M_FORCEINLINE __forceinline
#else
#define T0M_NOINLINE
#define T0M_FORCEINLINE inline
#endif

[[nodiscard]] bool runtime_avx2_supported() {
  int registers[4]{};
  __cpuid(registers, 0);
  const int max_leaf = registers[0];
  if (max_leaf < 1) return false;
  __cpuidex(registers, 1, 0);
  if ((registers[2] & (1 << 27)) == 0 || (registers[2] & (1 << 28)) == 0) {
    return false;
  }
  if ((_xgetbv(0) & 0x6) != 0x6 || max_leaf < 7) return false;
  __cpuidex(registers, 7, 0);
  return (registers[1] & (1 << 5)) != 0;
}

[[nodiscard]] T0M_FORCEINLINE std::int32_t horizontal_sum_i32(__m256i value) {
  __m128i low = _mm256_castsi256_si128(value);
  __m128i high = _mm256_extracti128_si256(value, 1);
  __m128i sum = _mm_add_epi32(low, high);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

T0M_FORCEINLINE void accumulate_dot_int8_avx2(
    __m256i& accumulator, std::int64_t& scalar_tail,
    const std::int8_t* weights, const std::int8_t* activations,
    std::uint32_t count) {
  std::uint32_t dimension_offset = 0;
  for (; dimension_offset + 16 <= count; dimension_offset += 16) {
    const __m256i weight16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
        reinterpret_cast<const __m128i*>(weights + dimension_offset)));
    const __m256i activation16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
        reinterpret_cast<const __m128i*>(activations + dimension_offset)));
    accumulator = _mm256_add_epi32(
        accumulator, _mm256_madd_epi16(weight16, activation16));
  }
  for (; dimension_offset < count; ++dimension_offset) {
    scalar_tail += static_cast<std::int64_t>(weights[dimension_offset]) *
                   static_cast<std::int64_t>(activations[dimension_offset]);
  }
}

T0M_NOINLINE void run_fused(const ShardData& shard,
                            std::uint32_t dimension,
                            std::uint32_t slots,
                            const WeightBlock& block,
                            std::uint32_t slot_tile,
                            std::vector<std::int64_t>& output) {
  for (std::uint32_t row = 0; row < shard.rows; ++row) {
    const std::size_t row_start = static_cast<std::size_t>(row) * dimension;
    for (std::uint32_t slot_base = 0; slot_base < slots; slot_base += slot_tile) {
      const std::uint32_t slot_count = std::min(slot_tile, slots - slot_base);
      std::array<__m256i, 8> accum{};
      std::array<std::int64_t, 8> scalar_tails{};
      for (std::uint32_t slot_offset = 0; slot_offset < slot_count; ++slot_offset) {
        accum[slot_offset] = _mm256_setzero_si256();
      }
      for (std::uint32_t dimension_base = 0; dimension_base < dimension;
           dimension_base += kWeightTile) {
        const std::uint32_t dimension_count =
            std::min(kWeightTile, dimension - dimension_base);
        if (dimension_count == kWeightTile) {
          const __m256i weight16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
              reinterpret_cast<const __m128i*>(block.values.data() + row_start + dimension_base)));
          for (std::uint32_t slot_offset = 0; slot_offset < slot_count; ++slot_offset) {
            const std::size_t activation_start =
                static_cast<std::size_t>(slot_base + slot_offset) * dimension + dimension_base;
            const __m256i activation16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(
                reinterpret_cast<const __m128i*>(shard.activations.data() + activation_start)));
            accum[slot_offset] = _mm256_add_epi32(
                accum[slot_offset], _mm256_madd_epi16(weight16, activation16));
          }
        } else {
          for (std::uint32_t slot_offset = 0; slot_offset < slot_count; ++slot_offset) {
            const std::size_t activation_start =
                static_cast<std::size_t>(slot_base + slot_offset) * dimension + dimension_base;
            for (std::uint32_t dimension_offset = 0; dimension_offset < dimension_count;
                 ++dimension_offset) {
              scalar_tails[slot_offset] +=
                  static_cast<std::int64_t>(block.values[row_start + dimension_base + dimension_offset]) *
                  static_cast<std::int64_t>(shard.activations[activation_start + dimension_offset]);
            }
          }
        }
      }
      for (std::uint32_t slot_offset = 0; slot_offset < slot_count; ++slot_offset) {
        output[static_cast<std::size_t>(slot_base + slot_offset) * shard.rows + row] =
            static_cast<std::int64_t>(horizontal_sum_i32(accum[slot_offset])) +
            scalar_tails[slot_offset];
      }
    }
  }
}

T0M_NOINLINE void run_repeat_gemv(const ShardData& shard,
                                   std::uint32_t dimension,
                                   std::uint32_t slots,
                                   const WeightBlock& block,
                                   std::vector<std::int64_t>& output) {
  for (std::uint32_t slot = 0; slot < slots; ++slot) {
    for (std::uint32_t row = 0; row < shard.rows; ++row) {
      __m256i accumulator = _mm256_setzero_si256();
      std::int64_t scalar_tail = 0;
      const std::size_t row_start = static_cast<std::size_t>(row) * dimension;
      const std::size_t activation_start = static_cast<std::size_t>(slot) * dimension;
      for (std::uint32_t dimension_base = 0; dimension_base < dimension;
           dimension_base += kWeightTile) {
        const std::uint32_t dimension_count =
            std::min(kWeightTile, dimension - dimension_base);
        accumulate_dot_int8_avx2(
            accumulator, scalar_tail,
            block.values.data() + row_start + dimension_base,
            shard.activations.data() + activation_start + dimension_base,
            dimension_count);
      }
      output[static_cast<std::size_t>(slot) * shard.rows + row] =
          static_cast<std::int64_t>(horizontal_sum_i32(accumulator)) + scalar_tail;
    }
  }
}

void run_reference_gemv(const ShardData& shard,
                        std::uint32_t dimension,
                        std::uint32_t slots,
                        const WeightBlock& block,
                        std::vector<std::int64_t>& output) {
  for (std::uint32_t slot = 0; slot < slots; ++slot) {
    for (std::uint32_t row = 0; row < shard.rows; ++row) {
      std::int64_t sum = 0;
      const std::size_t row_start = static_cast<std::size_t>(row) * dimension;
      const std::size_t activation_start = static_cast<std::size_t>(slot) * dimension;
      for (std::uint32_t dimension_index = 0; dimension_index < dimension;
           ++dimension_index) {
        sum += static_cast<std::int64_t>(block.values[row_start + dimension_index]) *
               static_cast<std::int64_t>(shard.activations[activation_start + dimension_index]);
      }
      output[static_cast<std::size_t>(slot) * shard.rows + row] = sum;
    }
  }
}

void run_reference(const ShardData& shard,
                   std::uint32_t dimension,
                   std::uint32_t slots,
                   std::uint32_t recurrent_depth,
                   Variant variant,
                   std::vector<std::int64_t>& output) {
  for (std::uint32_t depth_index = 0; depth_index < recurrent_depth; ++depth_index) {
    const WeightBlock& block = shard.blocks[variant == Variant::b ? depth_index : 0];
    run_reference_gemv(shard, dimension, slots, block, output);
  }
}

void run_shard_depth(const ShardData& shard,
                     std::uint32_t dimension,
                     std::uint32_t slots,
                     Variant variant,
                     Mode mode,
                     std::uint32_t slot_tile,
                     std::uint32_t depth_index,
                     std::vector<std::int64_t>& output) {
  const WeightBlock& block = shard.blocks[variant == Variant::b ? depth_index : 0];
  if (mode == Mode::fused) {
    run_fused(shard, dimension, slots, block, slot_tile, output);
  } else {
    run_repeat_gemv(shard, dimension, slots, block, output);
  }
}

void run_shard(const ShardData& shard,
               std::uint32_t dimension,
               std::uint32_t slots,
               std::uint32_t recurrent_depth,
               Variant variant,
               Mode mode,
               std::uint32_t slot_tile,
               std::vector<std::int64_t>& output) {
  for (std::uint32_t depth_index = 0; depth_index < recurrent_depth; ++depth_index) {
    run_shard_depth(shard, dimension, slots, variant, mode, slot_tile, depth_index, output);
  }
}

void touch_eviction(const std::vector<std::uint8_t>& buffer, std::uint64_t& checksum) {
  volatile const std::uint8_t* bytes = buffer.data();
  for (std::size_t offset = 0; offset < buffer.size(); offset += 64) {
    checksum = (checksum * 1315423911U) ^ bytes[offset];
  }
}

void flush_cache_lines(const ShardData& shard, const WeightBlock& block) {
  for (std::size_t offset = 0; offset < block.values.size(); offset += 64) {
    _mm_clflush(block.values.data() + offset);
  }
  for (std::size_t offset = 0; offset < shard.activations.size(); offset += 64) {
    _mm_clflush(shard.activations.data() + offset);
  }
  _mm_mfence();
}

void prepare_cold_pass(const ShardData& shard,
                       const WeightBlock& block,
                       const std::vector<std::uint8_t>& eviction,
                       std::uint64_t& eviction_checksum) {
  touch_eviction(eviction, eviction_checksum);
  flush_cache_lines(shard, block);
}

[[nodiscard]] std::uint64_t checksum_outputs(
    const std::vector<std::vector<std::int64_t>>& outputs,
    const std::vector<std::uint32_t>& rows,
    std::uint32_t slots) {
  std::uint64_t checksum = 0;
  std::uint64_t cell_index = 1;
  for (std::size_t shard_index = 0; shard_index < outputs.size(); ++shard_index) {
    for (std::uint32_t slot = 0; slot < slots; ++slot) {
      for (std::uint32_t row = 0; row < rows[shard_index]; ++row) {
        checksum += static_cast<std::uint64_t>(outputs[shard_index][static_cast<std::size_t>(slot) * rows[shard_index] + row]) *
                    cell_index++;
      }
    }
  }
  return checksum;
}

struct TimedResult {
  std::uint32_t worker_count = 0;
  std::string worker_list;
  std::string affinity;
  std::string affinity_error;
  std::string rows_per_worker;
  std::string bytes_per_worker;
  bool all_affinity_succeeded = false;
  bool all_timed_repetitions_exact = false;
  double elapsed_seconds = 0.0;
  std::uint64_t mac_total = 0;
  double mac_per_second = 0.0;
  std::uint64_t checksum = 0;
  std::uint64_t eviction_checksum = 0;
  std::size_t eviction_bytes = 0;
  std::vector<std::vector<std::int64_t>> outputs;
};

[[nodiscard]] std::uint64_t bytes_for_worker(std::uint32_t rows,
                                             std::uint32_t dimension,
                                             std::uint32_t recurrent_depth,
                                             Variant variant) {
  return static_cast<std::uint64_t>(rows) * dimension *
         (variant == Variant::b ? recurrent_depth : 1U);
}

[[nodiscard]] TimedResult run_timed(const Workload& workload,
                                    const std::vector<CpuTarget>& targets,
                                    const std::vector<std::uint32_t>& rows,
                                    std::uint32_t worker_count,
                                    Mode mode,
                                    std::uint32_t slot_tile,
                                    std::uint32_t iterations,
                                    std::uint32_t timed_repetitions,
                                    std::uint32_t warmup,
                                    const QpcClock& clock) {
  if (worker_count != 1 && worker_count != workload.shards.size()) {
    throw std::runtime_error("worker count must be one or one worker per shard");
  }
  if (targets.size() != worker_count) throw std::runtime_error("CPU/worker count mismatch");
  std::vector<std::vector<std::int64_t>> outputs(workload.shards.size());
  for (std::size_t shard_index = 0; shard_index < workload.shards.size(); ++shard_index) {
    outputs[shard_index].assign(static_cast<std::size_t>(workload.slots) * workload.shards[shard_index].rows, 0);
  }
  std::vector<std::uint8_t> affinity_succeeded(worker_count, 0);
  std::vector<DWORD> affinity_errors(worker_count, ERROR_SUCCESS);
  std::vector<std::uint32_t> completed_repetitions(worker_count, 0);
  std::vector<std::uint64_t> eviction_checksums(worker_count, 0);
  std::barrier ready(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier start(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier phase_ready(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier phase_done(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier cold_ready(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier cold_done(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::uint32_t worker_index = 0; worker_index < worker_count; ++worker_index) {
    workers.emplace_back([&, worker_index] {
      AffinityGuard affinity(targets[worker_index].group, targets[worker_index].group_index);
      affinity_succeeded[worker_index] = affinity.succeeded() ? 1U : 0U;
      affinity_errors[worker_index] = affinity.error();
      const std::size_t first_shard = worker_count == 1 ? 0 : worker_index;
      const std::size_t shard_end = worker_count == 1 ? workload.shards.size() : worker_index + 1;
      const bool cold_variant = workload.variant == Variant::c;
      std::vector<std::uint8_t> eviction;
      if (cold_variant) eviction.resize(kEvictionBytes, 0xA5U);
      std::uint64_t eviction_checksum = 0;
      auto execute_repetition = [&] {
        for (std::size_t shard_index = first_shard; shard_index < shard_end; ++shard_index) {
          for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            run_shard(workload.shards[shard_index], workload.dimension, workload.slots,
                      workload.recurrent_depth, workload.variant, mode, slot_tile,
                      outputs[shard_index]);
          }
        }
      };
      auto execute_cold_pass = [&](std::uint32_t depth_index) {
        for (std::size_t shard_index = first_shard; shard_index < shard_end; ++shard_index) {
          const WeightBlock& block = workload.shards[shard_index].blocks[0];
          prepare_cold_pass(workload.shards[shard_index], block, eviction, eviction_checksum);
          run_shard_depth(workload.shards[shard_index], workload.dimension, workload.slots,
                          workload.variant, mode, slot_tile, depth_index,
                          outputs[shard_index]);
        }
      };
      for (std::uint32_t warmup_index = 0; warmup_index < warmup; ++warmup_index) {
        if (!cold_variant) {
          execute_repetition();
        } else {
          for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            for (std::uint32_t depth_index = 0; depth_index < workload.recurrent_depth; ++depth_index) {
              execute_cold_pass(depth_index);
            }
          }
        }
      }
      ready.arrive_and_wait();
      if (!cold_variant) {
        start.arrive_and_wait();
        for (std::uint32_t repetition = 0; repetition < timed_repetitions; ++repetition) {
          for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            for (std::uint32_t depth_index = 0; depth_index < workload.recurrent_depth; ++depth_index) {
              phase_ready.arrive_and_wait();
              for (std::size_t shard_index = first_shard; shard_index < shard_end; ++shard_index) {
                run_shard_depth(workload.shards[shard_index], workload.dimension, workload.slots,
                                workload.variant, mode, slot_tile, depth_index,
                                outputs[shard_index]);
              }
              phase_done.arrive_and_wait();
            }
          }
          ++completed_repetitions[worker_index];
        }
      } else {
        for (std::uint32_t repetition = 0; repetition < timed_repetitions; ++repetition) {
          for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            for (std::uint32_t depth_index = 0; depth_index < workload.recurrent_depth; ++depth_index) {
              for (std::size_t shard_index = first_shard; shard_index < shard_end; ++shard_index) {
                const WeightBlock& block = workload.shards[shard_index].blocks[0];
                touch_eviction(eviction, eviction_checksum);
                flush_cache_lines(workload.shards[shard_index], block);
              }
              cold_ready.arrive_and_wait();
              start.arrive_and_wait();
              for (std::size_t shard_index = first_shard; shard_index < shard_end; ++shard_index) {
                run_shard_depth(workload.shards[shard_index], workload.dimension, workload.slots,
                                workload.variant, mode, slot_tile, depth_index,
                                outputs[shard_index]);
              }
              cold_done.arrive_and_wait();
            }
          }
          ++completed_repetitions[worker_index];
        }
      }
      eviction_checksums[worker_index] = eviction_checksum;
    });
  }
  ready.arrive_and_wait();
  double elapsed_seconds = 0.0;
  if (workload.variant != Variant::c) {
    start.arrive_and_wait();
    const std::uint32_t timed_units = timed_repetitions * iterations * workload.recurrent_depth;
    for (std::uint32_t unit = 0; unit < timed_units; ++unit) {
      phase_ready.arrive_and_wait();
      const LARGE_INTEGER phase_begin = clock.now();
      phase_done.arrive_and_wait();
      const LARGE_INTEGER phase_end = clock.now();
      elapsed_seconds += clock.elapsed(phase_begin, phase_end);
    }
    for (auto& worker : workers) worker.join();
  } else {
    const std::uint32_t timed_units = timed_repetitions * iterations * workload.recurrent_depth;
    for (std::uint32_t unit = 0; unit < timed_units; ++unit) {
      cold_ready.arrive_and_wait();
      const LARGE_INTEGER begin = clock.now();
      start.arrive_and_wait();
      cold_done.arrive_and_wait();
      const LARGE_INTEGER end = clock.now();
      elapsed_seconds += clock.elapsed(begin, end);
    }
    for (auto& worker : workers) worker.join();
  }

  TimedResult result;
  result.worker_count = worker_count;
  result.outputs = std::move(outputs);
  result.elapsed_seconds = elapsed_seconds;
  result.mac_total = calculate_mac_total(rows, workload.dimension, workload.slots,
                                         workload.recurrent_depth, iterations,
                                         timed_repetitions);
  result.mac_per_second = result.elapsed_seconds > 0.0
                              ? static_cast<double>(result.mac_total) / result.elapsed_seconds
                              : 0.0;
  result.checksum = checksum_outputs(result.outputs, rows, workload.slots);
  result.eviction_checksum = std::accumulate(eviction_checksums.begin(), eviction_checksums.end(), std::uint64_t{0});
  result.eviction_bytes = workload.variant == Variant::c ? kEvictionBytes : 0;
  result.all_affinity_succeeded = true;
  result.all_timed_repetitions_exact = true;
  for (std::uint32_t worker_index = 0; worker_index < worker_count; ++worker_index) {
    if (worker_index != 0) {
      result.worker_list += ',';
      result.affinity += ',';
      result.affinity_error += ',';
      result.rows_per_worker += ',';
      result.bytes_per_worker += ',';
    }
    result.worker_list += std::to_string(targets[worker_index].logical_index);
    result.affinity += affinity_succeeded[worker_index] ? "true" : "false";
    result.affinity_error += std::to_string(affinity_errors[worker_index]);
    const std::size_t row_index = worker_count == 1 ? 0 : worker_index;
    const std::uint32_t row_value = worker_count == 1
                                        ? static_cast<std::uint32_t>(
                                              std::accumulate(rows.begin(), rows.end(), std::uint64_t{0}))
                                        : rows[row_index];
    result.rows_per_worker += std::to_string(row_value);
    result.bytes_per_worker += std::to_string(bytes_for_worker(row_value, workload.dimension,
                                                                workload.recurrent_depth,
                                                                workload.variant));
    result.all_affinity_succeeded = result.all_affinity_succeeded && affinity_succeeded[worker_index] != 0;
    result.all_timed_repetitions_exact =
        result.all_timed_repetitions_exact && completed_repetitions[worker_index] == timed_repetitions;
  }
  return result;
}

[[nodiscard]] std::string mode_name(Mode mode) { return mode == Mode::fused ? "fused" : "repeat"; }
[[nodiscard]] char variant_name(Variant variant) {
  return variant == Variant::a ? 'A' : (variant == Variant::b ? 'B' : 'C');
}

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("T0-M correction failed: " + message);
}

void verify_checksum_depends_on_every_cell(const std::vector<std::vector<std::int64_t>>& outputs,
                                           const std::vector<std::uint32_t>& rows,
                                           std::uint32_t slots,
                                           std::uint64_t original_checksum) {
  for (std::size_t shard_index = 0; shard_index < outputs.size(); ++shard_index) {
    for (std::uint32_t slot = 0; slot < slots; ++slot) {
      for (std::uint32_t row = 0; row < rows[shard_index]; ++row) {
        auto changed = outputs;
        ++changed[shard_index][static_cast<std::size_t>(slot) * rows[shard_index] + row];
        require(checksum_outputs(changed, rows, slots) != original_checksum,
                "checksum omitted an output cell");
      }
    }
  }
}

void run_correction_self_test(const std::vector<CpuTarget>& available_targets,
                             const QpcClock& clock) {
  constexpr std::array<std::uint32_t, kCorrectionShardCount> kPhysicalCpuIndices{0, 2, 4, 6};
  require(available_targets.size() > kPhysicalCpuIndices.back(),
          "topology-confirmed physical CPU targets 0,2,4,6 are required");
  const std::uint32_t dimension = 13;
  const std::uint32_t slots = 4;
  const std::uint32_t recurrent_depth = 3;
  const std::uint32_t iterations = 2;
  const std::uint32_t timed_repetitions = 3;
  const std::vector<std::uint32_t> rows{3, 5, 7, 9};
  const std::uint64_t expected_mac_total = calculate_mac_total(
      rows, dimension, slots, recurrent_depth, iterations, timed_repetitions);
  for (const Variant variant : {Variant::a, Variant::b, Variant::c}) {
    const Workload workload = make_workload(dimension, slots, recurrent_depth, variant, rows);
    bool has_positive = false;
    bool has_negative = false;
    for (const ShardData& shard : workload.shards) {
      for (const std::int8_t value : shard.activations) {
        has_positive = has_positive || value > 0;
        has_negative = has_negative || value < 0;
      }
      for (const WeightBlock& block : shard.blocks) {
        for (const std::int8_t value : block.values) {
          has_positive = has_positive || value > 0;
          has_negative = has_negative || value < 0;
        }
      }
    }
    require(has_positive && has_negative, "correction data lacks positive or negative int8 values");
    for (const std::uint32_t slot_tile : {2U, 4U, 8U}) {
      std::vector<std::vector<std::int64_t>> reference(workload.shards.size());
      std::vector<std::vector<std::int64_t>> fused(workload.shards.size());
      std::vector<std::vector<std::int64_t>> repeat(workload.shards.size());
      for (std::size_t shard_index = 0; shard_index < workload.shards.size(); ++shard_index) {
        reference[shard_index].assign(static_cast<std::size_t>(slots) * rows[shard_index], 0);
        fused[shard_index].assign(reference[shard_index].size(), 0);
        repeat[shard_index].assign(reference[shard_index].size(), 0);
        run_reference(workload.shards[shard_index], dimension, slots, recurrent_depth,
                      variant, reference[shard_index]);
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
          run_shard(workload.shards[shard_index], dimension, slots, recurrent_depth,
                    variant, Mode::fused, slot_tile, fused[shard_index]);
          run_shard(workload.shards[shard_index], dimension, slots, recurrent_depth,
                    variant, Mode::repeat, slot_tile, repeat[shard_index]);
        }
      }
      const std::uint64_t reference_checksum = checksum_outputs(reference, rows, slots);
      require(reference_checksum != 0, "reference checksum unexpectedly zero");
      verify_checksum_depends_on_every_cell(reference, rows, slots, reference_checksum);
      for (std::size_t shard_index = 0; shard_index < workload.shards.size(); ++shard_index) {
        require(reference[shard_index] == fused[shard_index], "fused output differs from reference");
        require(reference[shard_index] == repeat[shard_index], "repeat output differs from reference");
        require(fused[shard_index] == repeat[shard_index], "fused output differs from repeat");
      }
      const std::uint64_t fused_checksum = checksum_outputs(fused, rows, slots);
      const std::uint64_t repeat_checksum = checksum_outputs(repeat, rows, slots);
      require(fused_checksum == repeat_checksum && fused_checksum == reference_checksum,
              "checksums differ");

      std::vector<CpuTarget> four_targets;
      four_targets.reserve(kCorrectionShardCount);
      for (const std::uint32_t cpu_index : kPhysicalCpuIndices) {
        four_targets.push_back(available_targets[cpu_index]);
      }
      for (const Mode mode : {Mode::fused, Mode::repeat}) {
        const TimedResult one_worker = run_timed(workload, {available_targets[0]}, rows, 1,
                                                 mode, slot_tile, iterations,
                                                 timed_repetitions, 0, clock);
        const TimedResult four_workers = run_timed(workload, four_targets, rows,
                                                   kCorrectionShardCount, mode, slot_tile,
                                                   iterations, timed_repetitions, 0, clock);
        require(one_worker.outputs == four_workers.outputs, "one/four worker outputs differ");
        require(one_worker.checksum == four_workers.checksum, "one/four worker checksums differ");
        require(one_worker.mac_total == expected_mac_total &&
                    four_workers.mac_total == expected_mac_total,
                "one/four worker mac_total mismatch");
        require(one_worker.all_timed_repetitions_exact && four_workers.all_timed_repetitions_exact,
                "timed repetition count mismatch");
      }
    }
  }
  std::cerr << "T0-M correction passed: every Y[S x O_i] cell, fused/repeat/reference, "
               "A/B/C, S_tile=2/4/8, four shards, and one/four-worker accounting gate\n";
}

void print_csv(const Options& options, const TimedResult& result) {
  std::cout << "D,S,R,O_i,B_i,rows_per_worker,bytes_per_worker,mode,variant,S_tile,iterations,"
                "timed_repetitions,warmup,worker_count,worker_list,affinity,affinity_error,"
                "affinity_succeeded,timed_repetitions_exact,avx2_supported,kernel_used,"
                "eviction_bytes,eviction_checksum,elapsed_seconds,mac_total,"
                "mac_per_second,checksum\n"
             << options.dimension << ',' << options.slots << ',' << options.recurrent_depth << ",\""
             << result.rows_per_worker << "\",\"" << result.bytes_per_worker << "\",\""
             << result.rows_per_worker << "\",\"" << result.bytes_per_worker << "\","
             << mode_name(options.mode) << ',' << variant_name(options.variant) << ','
            << options.slot_tile << ',' << options.iterations << ',' << options.timed_repetitions << ','
            << options.warmup << ',' << result.worker_count << ",\"" << result.worker_list
            << "\",\"" << result.affinity << "\",\"" << result.affinity_error << "\","
             << (result.all_affinity_succeeded ? "true" : "false") << ','
             << (result.all_timed_repetitions_exact ? "true" : "false") << ','
             << "true,avx2," << result.eviction_bytes << ',' << result.eviction_checksum << ','
             << result.elapsed_seconds << ',' << result.mac_total << ',' << result.mac_per_second
             << ',' << result.checksum << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const std::vector<CpuTarget> targets = enumerate_cpus();
    const QpcClock clock;
    if (!runtime_avx2_supported()) {
      throw std::runtime_error("AVX2 unavailable; t0m_int8_probe requires AVX2");
    }
    run_correction_self_test(targets, clock);
    if (options.self_test) return 0;
    if (options.cpus.empty()) {
      if (options.workers > targets.size()) throw std::runtime_error("--workers exceed active CPUs");
    }
    std::vector<CpuTarget> selected_targets;
    if (options.cpus.empty()) {
      selected_targets.assign(targets.begin(), targets.begin() + options.workers);
    } else {
      for (const std::uint32_t cpu : options.cpus) {
        if (cpu >= targets.size()) throw std::runtime_error("--cpus contains out-of-range CPU");
        selected_targets.push_back(targets[cpu]);
      }
    }
    const Workload workload = make_workload(options.dimension, options.slots,
                                            options.recurrent_depth, options.variant,
                                            options.rows_per_worker);
    const TimedResult result = run_timed(workload, selected_targets, options.rows_per_worker,
                                         options.workers, options.mode, options.slot_tile,
                                         options.iterations, options.timed_repetitions,
                                         options.warmup, clock);
    if (!result.all_timed_repetitions_exact || !result.all_affinity_succeeded) {
      throw std::runtime_error("accounting or affinity gate failed; speed result rejected");
    }
    print_csv(options, result);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}

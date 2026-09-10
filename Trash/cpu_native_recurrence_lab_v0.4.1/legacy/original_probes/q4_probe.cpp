#ifndef _WIN32
#error "cpu_native_q4_probe requires Windows APIs"
#endif

#include <windows.h>
#include <immintrin.h>
#include <intrin.h>

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <barrier>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#define CPU_NATIVE_NOINLINE __declspec(noinline)
#else
#define CPU_NATIVE_NOINLINE __attribute__((noinline))
#endif

namespace {

struct Options {
  std::uint32_t m = 1024;
  std::uint32_t k = 4096;
  std::uint32_t depth = 4;
  std::uint32_t iterations = 2;
  std::uint32_t repetitions = 5;
  std::uint32_t warmup = 2;
  bool selected_cpu = false;
  std::uint32_t cpu = 0;
  std::uint32_t parallel_workers = 0;
  bool parallel_workers_explicit = false;
  std::vector<std::uint32_t> parallel_cpu_indices;
  bool self_test = false;
  bool m_explicit = false;
  bool target_kib_specified = false;
  std::uint32_t target_kib = 0;
  enum class Variant : char { a = 'A', b = 'B', c = 'C' };
  Variant variant = Variant::a;
  enum class Kernel : char { scalar, avx2, automatic };
  Kernel kernel = Kernel::scalar;
};

using Variant = Options::Variant;
using Kernel = Options::Kernel;

constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;

struct CpuTarget {
  std::uint32_t logical_index;
  WORD group;
  BYTE group_index;
};

struct WeightBlock {
  std::vector<std::uint8_t> packed_weights;
  std::vector<float> scales;
};

struct ProbeData {
  std::uint32_t m;
  std::uint32_t k;
  std::vector<WeightBlock> weight_blocks;
  std::vector<float> input;
};

struct CpuResult {
  CpuTarget target{};
  bool affinity_succeeded = false;
  DWORD affinity_error = ERROR_SUCCESS;
  double elapsed_seconds = 0.0;
  std::uint64_t mac_count = 0;
  double mac_per_second = 0.0;
  float checksum = 0.0F;
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

  AffinityGuard(const AffinityGuard&) = delete;
  AffinityGuard& operator=(const AffinityGuard&) = delete;

  ~AffinityGuard() {
    if (active_ && saved_) {
      // Restore only after successful pinning; failure cannot destroy saved state.
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
  const auto* first = text.data();
  const auto* last = first + text.size();
  const auto parsed = std::from_chars(first, last, value);
  if (parsed.ec != std::errc{} || parsed.ptr != last ||
      (!allow_zero && value == 0)) {
    throw std::runtime_error("invalid value for " + std::string(option));
  }
  return value;
}

[[nodiscard]] std::vector<std::uint32_t> parse_cpu_list(std::string_view text) {
  std::vector<std::uint32_t> indices;
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t separator = text.find(',', start);
    const std::size_t end = separator == std::string_view::npos
                                ? text.size()
                                : separator;
    indices.push_back(parse_u32(text.substr(start, end - start),
                                 "--parallel-cpus", true));
    start = end == text.size() ? text.size() : end + 1;
  }
  if (indices.empty()) {
    throw std::runtime_error("--parallel-cpus requires at least one index");
  }
  return indices;
}

[[nodiscard]] Variant parse_variant(std::string_view text) {
  if (text == "A") {
    return Variant::a;
  }
  if (text == "B") {
    return Variant::b;
  }
  if (text == "C") {
    return Variant::c;
  }
  throw std::runtime_error("invalid value for --variant (expected A, B, or C)");
}

[[nodiscard]] Kernel parse_kernel(std::string_view text) {
  if (text == "scalar") {
    return Kernel::scalar;
  }
  if (text == "avx2") {
    return Kernel::avx2;
  }
  if (text == "auto") {
    return Kernel::automatic;
  }
  throw std::runtime_error(
      "invalid value for --kernel (expected scalar, avx2, or auto)");
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    if (argument == "--help" || argument == "-h") {
      std::cout
          << "Usage: cpu_native_q4_probe [options]\n"
          << "  --m N             output rows (default 1024)\n"
          << "  --K N             input columns (default 4096)\n"
          << "  --target-kib N    derive m for packed-Q4 plus scale bytes (K=512)\n"
          << "  --depth N         repeated matrix-vector passes (default 4)\n"
          << "  --variant A|B|C   weight reuse, distinct blocks, or cache eviction\n"
          << "  --kernel NAME     scalar, avx2, or auto dispatch (default scalar)\n"
          << "  --iterations N    kernel invocations per repetition (default 2)\n"
          << "  --repetitions N   timed repetitions per CPU (default 5)\n"
          << "  --warmup N        untimed warmup repetitions (default 2)\n"
          << "  --cpu N           probe one global logical CPU (default: all)\n"
          << "  --parallel-workers N  run N logical CPUs simultaneously\n"
          << "  --parallel-cpus LIST  explicit comma-separated logical CPUs\n"
          << "  --self-test       run deterministic scalar smoke test\n";
      std::exit(0);
    }
    if (argument == "--self-test") {
      options.self_test = true;
      continue;
    }

    auto require_value = [&](std::string_view option) -> std::string_view {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + std::string(option));
      }
      return argv[++i];
    };

    if (argument == "--m") {
      options.m = parse_u32(require_value(argument), argument, false);
      options.m_explicit = true;
    } else if (argument == "--K") {
      options.k = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--target-kib") {
      options.target_kib = parse_u32(require_value(argument), argument, false);
      options.target_kib_specified = true;
    } else if (argument == "--depth") {
      options.depth = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--variant") {
      options.variant = parse_variant(require_value(argument));
    } else if (argument == "--kernel") {
      options.kernel = parse_kernel(require_value(argument));
    } else if (argument == "--iterations") {
      options.iterations = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--repetitions") {
      options.repetitions = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--warmup") {
      options.warmup = parse_u32(require_value(argument), argument, true);
    } else if (argument == "--cpu") {
      options.cpu = parse_u32(require_value(argument), argument, true);
      options.selected_cpu = true;
    } else if (argument == "--parallel-workers") {
      options.parallel_workers =
          parse_u32(require_value(argument), argument, false);
      options.parallel_workers_explicit = true;
    } else if (argument == "--parallel-cpus") {
      options.parallel_cpu_indices = parse_cpu_list(require_value(argument));
    } else {
      throw std::runtime_error("unknown option: " + std::string(argument));
    }
  }
  if (options.m_explicit && options.target_kib_specified) {
    throw std::runtime_error("--m and --target-kib are mutually exclusive");
  }
  if (!options.parallel_cpu_indices.empty()) {
    if (options.parallel_workers_explicit &&
        options.parallel_workers != options.parallel_cpu_indices.size()) {
      throw std::runtime_error(
          "--parallel-workers must match --parallel-cpus count");
    }
    options.parallel_workers =
        static_cast<std::uint32_t>(options.parallel_cpu_indices.size());
  }
  return options;
}

[[nodiscard]] std::vector<CpuTarget> enumerate_cpus() {
  const WORD group_count = GetActiveProcessorGroupCount();
  if (group_count == 0) {
    throw std::runtime_error("GetActiveProcessorGroupCount returned zero");
  }

  std::vector<CpuTarget> targets;
  std::uint32_t logical_index = 0;
  for (WORD group = 0; group < group_count; ++group) {
    const DWORD count = GetActiveProcessorCount(group);
    if (count == 0 || count > 64) {
      throw std::runtime_error("unsupported processor-group size");
    }
    for (DWORD processor = 0; processor < count; ++processor) {
      targets.push_back(
          CpuTarget{logical_index++, group, static_cast<BYTE>(processor)});
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

[[nodiscard]] std::uint64_t checked_multiply(std::uint64_t left,
                                             std::uint64_t right) {
  if (right != 0 && left > (std::numeric_limits<std::uint64_t>::max)() / right) {
    throw std::runtime_error("value exceeds uint64_t");
  }
  return left * right;
}

[[nodiscard]] std::uint64_t weight_bytes_for_shape(std::uint32_t m,
                                                   std::uint32_t k) {
  const std::uint64_t weight_count = static_cast<std::uint64_t>(m) * k;
  const std::uint64_t packed_count = (weight_count + 1) / 2;
  const std::uint64_t scale_count = (weight_count + 31) / 32;
  if (packed_count > (std::numeric_limits<std::uint64_t>::max)() -
                         scale_count * sizeof(float)) {
    throw std::runtime_error("probe dimensions exceed addressable memory");
  }
  return packed_count + scale_count * sizeof(float);
}

[[nodiscard]] std::uint32_t derive_m(std::uint32_t target_kib,
                                     std::uint32_t k) {
  if (k != 512) {
    throw std::runtime_error("--target-kib requires --K 512");
  }
  const std::uint64_t target_bytes = checked_multiply(target_kib, 1024U);
  const std::uint64_t bytes_per_row = weight_bytes_for_shape(1, k);
  std::uint64_t m = target_bytes / bytes_per_row;
  if (m == 0 || m > (std::numeric_limits<std::uint32_t>::max)()) {
    throw std::runtime_error("--target-kib is too small or too large");
  }
  while (m > 0 && weight_bytes_for_shape(static_cast<std::uint32_t>(m), k) >
                         target_bytes) {
    --m;
  }
  return static_cast<std::uint32_t>(m);
}

[[nodiscard]] WeightBlock make_weight_block(std::uint32_t m,
                                             std::uint32_t k,
                                             std::uint32_t& state) {
  const std::uint64_t weight_count = checked_multiply(m, k);
  const std::uint64_t packed_count = (weight_count + 1) / 2;
  const std::uint64_t scale_count = (weight_count + 31) / 32;
  if (packed_count > (std::numeric_limits<std::size_t>::max)() ||
      scale_count > (std::numeric_limits<std::size_t>::max)()) {
    throw std::runtime_error("probe dimensions exceed addressable memory");
  }

  WeightBlock block{std::vector<std::uint8_t>(static_cast<std::size_t>(packed_count)),
                    std::vector<float>(static_cast<std::size_t>(scale_count))};
  for (auto& byte : block.packed_weights) {
    const auto low = static_cast<std::uint8_t>(next_random(state) & 0x0FU);
    const auto high = static_cast<std::uint8_t>(next_random(state) & 0x0FU);
    byte = static_cast<std::uint8_t>(low | (high << 4));
  }
  for (auto& scale : block.scales) {
    scale = 0.015625F +
            static_cast<float>(next_random(state) & 0xFFU) / 4096.0F;
  }
  return block;
}

[[nodiscard]] ProbeData make_probe_data(std::uint32_t m,
                                        std::uint32_t k,
                                        std::uint32_t depth,
                                        Variant variant) {
  ProbeData data{m, k, {}, std::vector<float>(k)};
  const std::uint32_t block_count = variant == Variant::b ? depth : 1;
  data.weight_blocks.reserve(block_count);
  for (std::uint32_t block = 0; block < block_count; ++block) {
    std::uint32_t state = 0xC001CAFEU ^
                          (0x9E3779B9U * (block + 1U));
    data.weight_blocks.push_back(make_weight_block(m, k, state));
  }
  std::uint32_t input_state = 0xA5A5F00DU;
  for (auto& value : data.input) {
    value = (static_cast<float>(next_random(input_state) & 0xFFU) - 128.0F) /
            128.0F;
  }
  return data;
}

CPU_NATIVE_NOINLINE void touch_eviction(const std::vector<std::uint8_t>& buffer,
                                         std::uint64_t& checksum) {
  volatile const std::uint8_t* bytes = buffer.data();
  std::uint64_t local = checksum;
  for (std::size_t offset = 0; offset < buffer.size(); offset += 64) {
    local = (local * 1315423911U) ^ bytes[offset];
  }
  checksum = local;
}

CPU_NATIVE_NOINLINE float run_kernel(const ProbeData& data,
                                      std::uint32_t depth,
                                      std::uint32_t iterations,
                                      Variant variant,
                                      std::vector<float>& output,
                                      const std::vector<std::uint8_t>* eviction,
                                      std::uint64_t& eviction_checksum) {
  float checksum = 0.0F;
  for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
    for (std::uint32_t pass = 0; pass < depth; ++pass) {
      const WeightBlock& weights = data.weight_blocks[pass % data.weight_blocks.size()];
      for (std::uint32_t row = 0; row < data.m; ++row) {
        float sum = 0.0F;
        const std::uint64_t row_start = static_cast<std::uint64_t>(row) * data.k;
        for (std::uint32_t column = 0; column < data.k; ++column) {
          const std::uint64_t weight_index = row_start + column;
          const std::uint8_t packed =
              weights.packed_weights[static_cast<std::size_t>(weight_index >> 1)];
          const std::uint8_t nibble =
              (weight_index & 1U) == 0 ? packed & 0x0FU : packed >> 4;
          const float weight =
              (static_cast<int>(nibble) - 8) *
              weights.scales[static_cast<std::size_t>(weight_index >> 5)];
          sum += data.input[column] * weight;
        }
        output[row] = sum;
      }
      for (const float value : output) {
        checksum += value;
      }
      if (variant == Variant::c && pass + 1U < depth) {
        if (eviction == nullptr || eviction->size() != kEvictionBytes) {
          throw std::runtime_error("variant C requires 64 MiB eviction buffer");
        }
        touch_eviction(*eviction, eviction_checksum);
      }
    }
  }
  if (variant == Variant::c) {
    checksum += static_cast<float>(eviction_checksum & 0xFFU) * 0.000001F;
  }
  return checksum;
}

struct CpuFeatures {
  bool avx2 = false;
  bool fma = false;
};

[[nodiscard]] CpuFeatures detect_cpu_features() {
  CpuFeatures features;
  int registers[4]{};
  __cpuid(registers, 0);
  const int maximum_leaf = registers[0];
  if (maximum_leaf < 1) {
    return features;
  }

  __cpuidex(registers, 1, 0);
  constexpr int kOsxsaveBit = 1 << 27;
  constexpr int kAvxBit = 1 << 28;
  if ((registers[2] & (kOsxsaveBit | kAvxBit)) !=
      (kOsxsaveBit | kAvxBit)) {
    return features;
  }
  if ((_xgetbv(0) & 0x6U) != 0x6U) {
    return features;
  }
  features.fma = (registers[2] & (1 << 12)) != 0;

  if (maximum_leaf < 7) {
    return features;
  }
  __cpuidex(registers, 7, 0);
  constexpr int kAvx2Bit = 1 << 5;
  features.avx2 = (registers[1] & kAvx2Bit) != 0;
  return features;
}

[[nodiscard]] __m256 q4_to_float8(__m128i values, float scale) {
  const __m128i q16 = _mm_cvtepi8_epi16(values);
  const __m128i q32_low = _mm_cvtepi16_epi32(q16);
  const __m128i q32_high =
      _mm_cvtepi16_epi32(_mm_srli_si128(q16, sizeof(std::int32_t) * 2));
  const __m128 low = _mm_mul_ps(_mm_cvtepi32_ps(q32_low),
                                _mm_set1_ps(scale));
  const __m128 high = _mm_mul_ps(_mm_cvtepi32_ps(q32_high),
                                 _mm_set1_ps(scale));
  return _mm256_set_m128(high, low);
}

[[nodiscard]] float horizontal_sum(__m256 value) {
  __m128 sum = _mm_add_ps(_mm256_castps256_ps128(value),
                          _mm256_extractf128_ps(value, 1));
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

CPU_NATIVE_NOINLINE float run_kernel_avx2(
    const ProbeData& data, std::uint32_t depth, std::uint32_t iterations,
    Variant variant, std::vector<float>& output,
    const std::vector<std::uint8_t>* eviction,
    std::uint64_t& eviction_checksum) {
  float checksum = 0.0F;
  const __m128i nibble_mask = _mm_set1_epi8(0x0F);
  const __m128i nibble_bias = _mm_set1_epi8(8);
  for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
    for (std::uint32_t pass = 0; pass < depth; ++pass) {
      const WeightBlock& weights = data.weight_blocks[pass % data.weight_blocks.size()];
      for (std::uint32_t row = 0; row < data.m; ++row) {
        const std::uint64_t row_start = static_cast<std::uint64_t>(row) * data.k;
        __m256 accumulator = _mm256_setzero_ps();
        std::uint32_t column = 0;
        for (; column + 32 <= data.k; column += 32) {
          const std::uint64_t weight_index = row_start + column;
          const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(
              weights.packed_weights.data() + (weight_index >> 1)));
          const __m128i low_nibbles = _mm_and_si128(packed, nibble_mask);
          const __m128i high_nibbles = _mm_and_si128(
              _mm_srli_epi16(packed, 4), nibble_mask);
          const __m128i interleaved_low =
              _mm_sub_epi8(_mm_unpacklo_epi8(low_nibbles, high_nibbles),
                           nibble_bias);
          const __m128i interleaved_high =
              _mm_sub_epi8(_mm_unpackhi_epi8(low_nibbles, high_nibbles),
                           nibble_bias);
          const float scale = weights.scales[weight_index >> 5];
          accumulator = _mm256_fmadd_ps(
              q4_to_float8(interleaved_low, scale),
              _mm256_loadu_ps(data.input.data() + column), accumulator);
          accumulator = _mm256_fmadd_ps(
              q4_to_float8(_mm_srli_si128(interleaved_low, 8), scale),
              _mm256_loadu_ps(data.input.data() + column + 8), accumulator);
          accumulator = _mm256_fmadd_ps(
              q4_to_float8(interleaved_high, scale),
              _mm256_loadu_ps(data.input.data() + column + 16), accumulator);
          accumulator = _mm256_fmadd_ps(
              q4_to_float8(_mm_srli_si128(interleaved_high, 8), scale),
              _mm256_loadu_ps(data.input.data() + column + 24), accumulator);
        }

        float sum = horizontal_sum(accumulator);
        for (; column < data.k; ++column) {
          const std::uint64_t weight_index = row_start + column;
          const std::uint8_t packed =
              weights.packed_weights[static_cast<std::size_t>(weight_index >> 1)];
          const std::uint8_t nibble =
              (weight_index & 1U) == 0 ? packed & 0x0FU : packed >> 4;
          const float weight =
              (static_cast<int>(nibble) - 8) *
              weights.scales[static_cast<std::size_t>(weight_index >> 5)];
          sum += data.input[column] * weight;
        }
        output[row] = sum;
      }
      for (const float value : output) {
        checksum += value;
      }
      if (variant == Variant::c && pass + 1U < depth) {
        if (eviction == nullptr || eviction->size() != kEvictionBytes) {
          throw std::runtime_error("variant C requires 64 MiB eviction buffer");
        }
        touch_eviction(*eviction, eviction_checksum);
      }
    }
  }
  if (variant == Variant::c) {
    checksum += static_cast<float>(eviction_checksum & 0xFFU) * 0.000001F;
  }
  return checksum;
}

using KernelFunction = float (*)(const ProbeData&, std::uint32_t,
                                 std::uint32_t, Variant, std::vector<float>&,
                                 const std::vector<std::uint8_t>*,
                                 std::uint64_t&);

void run_kernel_correction_test() {
  const ProbeData data = make_probe_data(8, 64, 2, Variant::a);
  std::vector<float> output(data.m, 0.0F);
  std::uint64_t scalar_eviction_checksum = 0;
  std::uint64_t avx2_eviction_checksum = 0;
  const float scalar = run_kernel(data, 2, 3, Variant::a, output, nullptr,
                                  scalar_eviction_checksum);
  const float avx2 = run_kernel_avx2(data, 2, 3, Variant::a, output, nullptr,
                                     avx2_eviction_checksum);
  const float difference = std::fabs(scalar - avx2);
  const float magnitude = (std::fabs(scalar) > std::fabs(avx2))
                              ? std::fabs(scalar)
                              : std::fabs(avx2);
  const float tolerance = 0.001F * ((magnitude > 1.0F) ? magnitude : 1.0F);
  if (difference > tolerance) {
    throw std::runtime_error("AVX2 correction test failed: scalar=" +
                             std::to_string(scalar) + ", avx2=" +
                             std::to_string(avx2) + ", difference=" +
                             std::to_string(difference) + ", tolerance=" +
                             std::to_string(tolerance));
  }
  std::cerr << "AVX2 correction test passed; scalar_checksum=" << scalar
            << ", avx2_checksum=" << avx2 << ", abs_difference=" << difference
            << ", tolerance=" << tolerance << '\n';
}

[[nodiscard]] CpuResult run_cpu(const CpuTarget& target,
                                 const ProbeData& data,
                                 const Options& options,
                                 KernelFunction kernel,
                                 std::uint64_t mac_count,
                                 const QpcClock& clock) {
  CpuResult result;
  result.target = target;
  std::thread worker([&] {
    AffinityGuard affinity(target.group, target.group_index);
    result.affinity_succeeded = affinity.succeeded();
    result.affinity_error = affinity.error();
    std::vector<float> output(data.m, 0.0F);
    std::vector<std::uint8_t> eviction;
    if (options.variant == Variant::c) {
      eviction.resize(kEvictionBytes);
      std::uint32_t state = 0x51EDBEEFU;
      for (auto& byte : eviction) {
        byte = static_cast<std::uint8_t>(next_random(state));
      }
    }

    for (std::uint32_t warmup = 0; warmup < options.warmup; ++warmup) {
      std::uint64_t eviction_checksum = 0;
      (void)kernel(data, options.depth, options.iterations, options.variant,
                   output, eviction.empty() ? nullptr : &eviction,
                   eviction_checksum);
    }

    const LARGE_INTEGER begin = clock.now();
    float checksum = 0.0F;
    std::uint64_t eviction_checksum = 0;
    for (std::uint32_t repetition = 0; repetition < options.repetitions;
         ++repetition) {
      checksum += kernel(data, options.depth, options.iterations,
                         options.variant, output,
                         eviction.empty() ? nullptr : &eviction,
                         eviction_checksum);
    }
    const LARGE_INTEGER end = clock.now();
    result.elapsed_seconds = clock.elapsed(begin, end);
    result.mac_count = mac_count;
    result.mac_per_second =
        result.elapsed_seconds > 0.0
            ? static_cast<double>(result.mac_count) / result.elapsed_seconds
            : 0.0;
    result.checksum = checksum;
  });
  worker.join();
  return result;
}

struct ParallelResult {
  std::uint32_t worker_count = 0;
  std::string logical_cpu_indices;
  bool all_affinity_succeeded = false;
  DWORD first_affinity_error = ERROR_SUCCESS;
  double elapsed_seconds = 0.0;
  std::uint64_t mac_count = 0;
  double mac_per_second = 0.0;
  float checksum_sum = 0.0F;
};

[[nodiscard]] ParallelResult run_parallel(
    const std::vector<CpuTarget>& targets, const Options& options,
    KernelFunction kernel, std::uint64_t mac_count, const QpcClock& clock) {
  const std::uint32_t worker_count = options.parallel_workers;
  if (worker_count == 0 || worker_count > targets.size()) {
    throw std::runtime_error("parallel worker count is outside active CPU range");
  }

  std::vector<CpuResult> results(worker_count);
  std::barrier ready(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier release(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier done(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::uint32_t worker = 0; worker < worker_count; ++worker) {
    workers.emplace_back([&, worker] {
      const CpuTarget& target = targets[worker];
      results[worker].target = target;
      ProbeData data = make_probe_data(options.m, options.k, options.depth,
                                       options.variant);
      AffinityGuard affinity(target.group, target.group_index);
      results[worker].affinity_succeeded = affinity.succeeded();
      results[worker].affinity_error = affinity.error();
      std::vector<float> output(data.m, 0.0F);
      std::vector<std::uint8_t> eviction;
      if (options.variant == Variant::c) {
        eviction.resize(kEvictionBytes);
        std::uint32_t state = 0x51EDBEEFU;
        for (auto& byte : eviction) {
          byte = static_cast<std::uint8_t>(next_random(state));
        }
      }
      for (std::uint32_t warmup = 0; warmup < options.warmup; ++warmup) {
        std::uint64_t eviction_checksum = 0;
        (void)kernel(data, options.depth, options.iterations, options.variant,
                     output, eviction.empty() ? nullptr : &eviction,
                     eviction_checksum);
      }

      ready.arrive_and_wait();
      release.arrive_and_wait();
      float checksum = 0.0F;
      std::uint64_t eviction_checksum = 0;
      for (std::uint32_t repetition = 0; repetition < options.repetitions;
           ++repetition) {
        checksum += kernel(data, options.depth, options.iterations,
                           options.variant, output,
                           eviction.empty() ? nullptr : &eviction,
                           eviction_checksum);
      }
      results[worker].mac_count = mac_count;
      results[worker].checksum = checksum;
      done.arrive_and_wait();
    });
  }

  ready.arrive_and_wait();
  const LARGE_INTEGER begin = clock.now();
  release.arrive_and_wait();
  done.arrive_and_wait();
  const LARGE_INTEGER end = clock.now();
  for (auto& worker : workers) {
    worker.join();
  }

  ParallelResult result;
  result.worker_count = worker_count;
  for (std::uint32_t worker = 0; worker < worker_count; ++worker) {
    if (worker != 0) {
      result.logical_cpu_indices += ',';
    }
    result.logical_cpu_indices +=
        std::to_string(targets[worker].logical_index);
  }
  result.all_affinity_succeeded = true;
  for (const CpuResult& worker : results) {
    result.all_affinity_succeeded =
        result.all_affinity_succeeded && worker.affinity_succeeded;
    if (result.first_affinity_error == ERROR_SUCCESS &&
        worker.affinity_error != ERROR_SUCCESS) {
      result.first_affinity_error = worker.affinity_error;
    }
    if (result.mac_count >
        (std::numeric_limits<std::uint64_t>::max)() - worker.mac_count) {
      throw std::runtime_error("parallel MAC count exceeds uint64_t");
    }
    result.mac_count += worker.mac_count;
    result.checksum_sum += worker.checksum;
  }
  result.elapsed_seconds = clock.elapsed(begin, end);
  result.mac_per_second =
      result.elapsed_seconds > 0.0
          ? static_cast<double>(result.mac_count) / result.elapsed_seconds
          : 0.0;
  return result;
}

void print_csv_header() {
  std::cout
      << "probe,kernel_requested,kernel_used,avx2_supported,fma_supported,target_kib,"
         "actual_weight_bytes_per_block,allocated_weight_bytes,"
         "m,K,depth,variant,iterations,repetitions,warmup,logical_cpu_index,"
         "processor_group,group_processor_index,affinity_succeeded,"
         "affinity_error,eviction_bytes,elapsed_seconds,mac_count,"
         "mac_per_second,checksum\n";
}

[[nodiscard]] std::uint64_t allocated_weight_bytes(const ProbeData& data) {
  std::uint64_t total = 0;
  for (const auto& block : data.weight_blocks) {
    const std::uint64_t block_bytes = block.packed_weights.size() +
                                      block.scales.size() * sizeof(float);
    if (total > (std::numeric_limits<std::uint64_t>::max)() - block_bytes) {
      throw std::runtime_error("allocated weight bytes exceed uint64_t");
    }
    total += block_bytes;
  }
  return total;
}

[[nodiscard]] char variant_name(Variant variant) {
  return static_cast<char>(variant);
}

[[nodiscard]] const char* kernel_name(Kernel kernel) {
  switch (kernel) {
    case Kernel::scalar:
      return "scalar";
    case Kernel::avx2:
      return "avx2";
    case Kernel::automatic:
      return "auto";
  }
  return "unknown";
}

void print_csv_row(const Options& options,
                   const ProbeData& data,
                   const CpuResult& result, const char* kernel_used,
                   const CpuFeatures& features) {
  const auto target_kib = options.target_kib_specified ? options.target_kib : 0;
  const auto actual_bytes = weight_bytes_for_shape(data.m, data.k);
  const auto allocated_bytes = allocated_weight_bytes(data);
  const auto eviction_bytes = options.variant == Variant::c ? kEvictionBytes : 0;
  std::cout << "q4_gemv," << kernel_name(options.kernel) << ',' << kernel_used
            << ',' << (features.avx2 ? "true" : "false") << ','
            << (features.fma ? "true" : "false") << ',' << target_kib
            << ',' << actual_bytes << ','
            << allocated_bytes << ',' << options.m << ',' << options.k << ','
            << options.depth << ',' << variant_name(options.variant) << ','
            << options.iterations << ','
            << options.repetitions << ',' << options.warmup << ','
            << result.target.logical_index << ',' << result.target.group << ','
            << static_cast<unsigned>(result.target.group_index) << ','
            << (result.affinity_succeeded ? "true" : "false") << ','
            << result.affinity_error << ',' << eviction_bytes << ','
            << result.elapsed_seconds << ',' << result.mac_count << ','
            << result.mac_per_second << ',' << result.checksum << '\n';
}

void print_parallel_csv_header() {
  std::cout
      << "probe,kernel_requested,kernel_used,avx2_supported,fma_supported,"
         "target_kib,actual_weight_bytes_per_block,allocated_weight_bytes_per_worker,"
         "m,K,depth,variant,iterations,repetitions,warmup,worker_count,"
         "logical_cpu_indices,all_affinity_succeeded,affinity_error,"
         "batch_elapsed_seconds,batch_mac_count,batch_mac_per_second,checksum_sum\n";
}

void print_parallel_csv_row(const Options& options, const ProbeData& data,
                            const ParallelResult& result,
                            const char* kernel_used,
                            const CpuFeatures& features) {
  const auto target_kib = options.target_kib_specified ? options.target_kib : 0;
  const auto actual_bytes = weight_bytes_for_shape(data.m, data.k);
  const auto allocated_bytes = allocated_weight_bytes(data);
  std::cout << "q4_gemv_batch," << kernel_name(options.kernel) << ','
            << kernel_used << ',' << (features.avx2 ? "true" : "false") << ','
            << (features.fma ? "true" : "false") << ',' << target_kib << ','
            << actual_bytes << ',' << allocated_bytes << ',' << options.m << ','
            << options.k << ',' << options.depth << ','
            << variant_name(options.variant) << ',' << options.iterations << ','
            << options.repetitions << ',' << options.warmup << ','
            << result.worker_count << ",\"" << result.logical_cpu_indices
            << "\","
            << (result.all_affinity_succeeded ? "true" : "false") << ','
            << result.first_affinity_error << ',' << result.elapsed_seconds << ','
            << result.mac_count << ',' << result.mac_per_second << ','
            << result.checksum_sum << '\n';
}

void run_self_test() {
  const std::uint64_t expected_macs = 8U * 16U * 2U * 3U;
  const Variant variants[] = {Variant::a, Variant::b, Variant::c};
  for (const Variant variant : variants) {
    const ProbeData data = make_probe_data(8, 16, 2, variant);
    std::vector<float> output(data.m, 0.0F);
    std::vector<std::uint8_t> eviction;
    if (variant == Variant::c) {
      eviction.resize(kEvictionBytes, 0xA5U);
    }
    std::uint64_t first_eviction_checksum = 0;
    std::uint64_t second_eviction_checksum = 0;
    const float first = run_kernel(data, 2, 3, variant, output,
                                   eviction.empty() ? nullptr : &eviction,
                                   first_eviction_checksum);
    const float second = run_kernel(data, 2, 3, variant, output,
                                    eviction.empty() ? nullptr : &eviction,
                                    second_eviction_checksum);
    const std::size_t expected_blocks = variant == Variant::b ? 2U : 1U;
    if (data.weight_blocks.size() != expected_blocks || first != second ||
        !std::isfinite(first) || first == 0.0F) {
      throw std::runtime_error("self-test failed");
    }
  }
  std::cout << "self_test,status=pass,mac_count=" << expected_macs
            << ",variants=A,B,C\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_options(argc, argv);
    std::cout << std::setprecision(17);
    const CpuFeatures features = detect_cpu_features();
    if (options.self_test) {
      run_self_test();
      if (features.avx2 && features.fma) {
        run_kernel_correction_test();
      } else {
        std::cerr << "AVX2/FMA correction test skipped; required features unavailable\n";
      }
      return 0;
    }

    KernelFunction kernel = run_kernel;
    const char* kernel_used = "scalar";
    if (options.kernel != Kernel::scalar) {
      if (features.avx2 && features.fma) {
        run_kernel_correction_test();
        kernel = run_kernel_avx2;
        kernel_used = "avx2";
      } else {
        std::cerr << "AVX2 unavailable; falling back to scalar\n";
      }
    }

    if (options.target_kib_specified) {
      options.m = derive_m(options.target_kib, options.k);
    }

    const auto targets = enumerate_cpus();
    if (options.selected_cpu && options.cpu >= targets.size()) {
      throw std::runtime_error("--cpu is outside active logical CPU range");
    }
    const auto mac_count = checked_multiply(
        checked_multiply(
            checked_multiply(static_cast<std::uint64_t>(options.m), options.k),
            options.depth),
        checked_multiply(options.iterations, options.repetitions));
    const QpcClock clock;
    if (options.parallel_workers > 0) {
      if (options.selected_cpu) {
        throw std::runtime_error("--cpu cannot be combined with --parallel-workers");
      }
      std::vector<CpuTarget> parallel_targets;
      if (options.parallel_cpu_indices.empty()) {
        if (options.parallel_workers > targets.size()) {
          throw std::runtime_error(
              "--parallel-workers exceeds active logical CPU count");
        }
        parallel_targets.assign(targets.begin(),
                                targets.begin() + options.parallel_workers);
      } else {
        parallel_targets.reserve(options.parallel_cpu_indices.size());
        for (const std::uint32_t logical_index : options.parallel_cpu_indices) {
          if (logical_index >= targets.size()) {
            throw std::runtime_error(
                "--parallel-cpus contains an index outside active CPU range");
          }
          parallel_targets.push_back(targets[logical_index]);
        }
      }
      const ProbeData metadata = make_probe_data(options.m, options.k,
                                                 options.depth, options.variant);
      std::cerr << "T0 parallel Q4 throughput probe; wall-clock batch timing; "
                   "not an exact model-kernel equivalence claim\n"
                << "shape m=" << options.m << " K=" << options.k
                << ", target_kib="
                << (options.target_kib_specified ? options.target_kib : 0)
                << ", variant=" << variant_name(options.variant)
                << ", kernel_requested=" << kernel_name(options.kernel)
                << ", kernel_used=" << kernel_used
                << ", workers=" << options.parallel_workers
                << ", logical_cpu_indices=";
      for (std::size_t index = 0; index < parallel_targets.size(); ++index) {
        if (index != 0) {
          std::cerr << ',';
        }
        std::cerr << parallel_targets[index].logical_index;
      }
      std::cerr << '\n';
      const ParallelResult result =
          run_parallel(parallel_targets, options, kernel, mac_count, clock);
      print_parallel_csv_header();
      print_parallel_csv_row(options, metadata, result, kernel_used, features);
      return 0;
    }
    const ProbeData data = make_probe_data(options.m, options.k, options.depth,
                                           options.variant);

    std::cerr << "T0 scalar Q4 throughput probe; not an exact model-kernel "
                 "equivalence claim\n"
              << "shape m=" << options.m << " K=" << options.k
              << ", target_kib="
              << (options.target_kib_specified ? options.target_kib : 0)
              << ", actual_weight_bytes_per_block="
              << weight_bytes_for_shape(options.m, options.k)
              << ", allocated_weight_bytes=" << allocated_weight_bytes(data)
              << ", variant=" << variant_name(options.variant)
              << ", kernel_requested=" << kernel_name(options.kernel)
              << ", kernel_used=" << kernel_used
              << ", avx2_supported=" << (features.avx2 ? "true" : "false")
              << ", fma_supported=" << (features.fma ? "true" : "false")
              << ", eviction_bytes="
              << (options.variant == Variant::c ? kEvictionBytes : 0)
              << ", active_logical_cpus=" << targets.size() << '\n';
    print_csv_header();

    for (const auto& target : targets) {
      if (options.selected_cpu && target.logical_index != options.cpu) {
        continue;
      }
      const CpuResult result =
          run_cpu(target, data, options, kernel, mac_count, clock);
      print_csv_row(options, data, result, kernel_used, features);
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}

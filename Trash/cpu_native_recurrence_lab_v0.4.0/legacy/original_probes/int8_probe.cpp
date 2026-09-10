#ifndef _WIN32
#error "int8_probe requires Windows APIs"
#endif

#include <windows.h>
#include <immintrin.h>
#include <intrin.h>

#include <barrier>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kEvictionBytes = 64U * 1024U * 1024U;

struct Options {
  std::uint32_t m = 64;
  std::uint32_t k = 64;
  std::uint32_t depth = 1;
  std::uint32_t iterations = 1;
  std::uint32_t repetitions = 20;
  std::uint32_t warmup = 5;
  std::uint32_t cpu = 0;
  bool selected_cpu = false;
  std::uint32_t target_kib = 0;
  bool target_kib_specified = false;
  std::uint32_t parallel_workers = 0;
  std::vector<std::uint32_t> parallel_cpus;
  std::vector<std::uint32_t> parallel_rows;
  enum class Variant : char { a = 'A', b = 'B', c = 'C' };
  enum class Kernel : char { scalar, avx2, automatic };
  Variant variant = Variant::a;
  Kernel kernel = Kernel::scalar;
};

using Variant = Options::Variant;
using Kernel = Options::Kernel;

struct CpuFeatures {
  bool avx2 = false;
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
  const auto parsed = std::from_chars(text.data(), text.data() + text.size(),
                                      value);
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
  if (values.empty()) {
    throw std::runtime_error(std::string(option) + " requires values");
  }
  return values;
}

[[nodiscard]] Variant parse_variant(std::string_view text) {
  if (text == "A") return Variant::a;
  if (text == "B") return Variant::b;
  if (text == "C") return Variant::c;
  throw std::runtime_error("invalid --variant; expected A, B, or C");
}

[[nodiscard]] Kernel parse_kernel(std::string_view text) {
  if (text == "scalar") return Kernel::scalar;
  if (text == "avx2") return Kernel::avx2;
  if (text == "auto") return Kernel::automatic;
  throw std::runtime_error("invalid --kernel; expected scalar, avx2, or auto");
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    auto require_value = [&](std::string_view option) -> std::string_view {
      if (index + 1 >= argc) {
        throw std::runtime_error("missing value for " + std::string(option));
      }
      return argv[++index];
    };
    if (argument == "--m") {
      options.m = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--K") {
      options.k = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--target-kib") {
      options.target_kib = parse_u32(require_value(argument), argument, false);
      options.target_kib_specified = true;
    } else if (argument == "--depth") {
      options.depth = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--iterations") {
      options.iterations = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--repetitions") {
      options.repetitions = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--warmup") {
      options.warmup = parse_u32(require_value(argument), argument, true);
    } else if (argument == "--variant") {
      options.variant = parse_variant(require_value(argument));
    } else if (argument == "--kernel") {
      options.kernel = parse_kernel(require_value(argument));
    } else if (argument == "--cpu") {
      options.cpu = parse_u32(require_value(argument), argument, true);
      options.selected_cpu = true;
    } else if (argument == "--parallel-workers") {
      options.parallel_workers = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--parallel-cpus") {
      options.parallel_cpus = parse_list(require_value(argument), argument, true);
    } else if (argument == "--parallel-rows") {
      options.parallel_rows = parse_list(require_value(argument), argument, false);
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: int8_probe [--m N|--target-kib N] [--K N] [--depth N] "
                   "[--variant A|B|C] [--kernel scalar|avx2|auto] [--cpu N] "
                   "[--parallel-workers N|--parallel-cpus LIST --parallel-rows LIST]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + std::string(argument));
    }
  }
  if (options.target_kib_specified && options.k != 512) {
    throw std::runtime_error("--target-kib requires --K 512");
  }
  if (!options.parallel_cpus.empty()) {
    if (options.parallel_workers != 0 &&
        options.parallel_workers != options.parallel_cpus.size()) {
      throw std::runtime_error("--parallel-workers must match --parallel-cpus");
    }
    options.parallel_workers =
        static_cast<std::uint32_t>(options.parallel_cpus.size());
  }
  if (!options.parallel_rows.empty()) {
    if (options.parallel_workers != 0 &&
        options.parallel_workers != options.parallel_rows.size()) {
      throw std::runtime_error("--parallel-workers must match --parallel-rows");
    }
    options.parallel_workers =
        static_cast<std::uint32_t>(options.parallel_rows.size());
  }
  return options;
}

[[nodiscard]] CpuFeatures detect_cpu_features() {
  CpuFeatures features;
  int registers[4]{};
  __cpuid(registers, 0);
  if (registers[0] < 7) return features;
  __cpuidex(registers, 7, 0);
  features.avx2 = (registers[1] & (1 << 5)) != 0;
  return features;
}

[[nodiscard]] std::vector<CpuTarget> enumerate_cpus() {
  const WORD group_count = GetActiveProcessorGroupCount();
  std::vector<CpuTarget> targets;
  std::uint32_t logical_index = 0;
  for (WORD group = 0; group < group_count; ++group) {
    const DWORD count = GetActiveProcessorCount(group);
    for (DWORD processor = 0; processor < count; ++processor) {
      targets.push_back(
          CpuTarget{logical_index++, group, static_cast<BYTE>(processor)});
    }
  }
  return targets;
}

struct WeightBlock {
  std::vector<std::int8_t> weights;
};

struct Int8Data {
  std::uint32_t m = 0;
  std::uint32_t k = 0;
  std::vector<WeightBlock> blocks;
  std::vector<std::int8_t> input;
};

[[nodiscard]] std::uint32_t next_random(std::uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}

[[nodiscard]] Int8Data make_data(std::uint32_t m, std::uint32_t k,
                                 std::uint32_t depth, Variant variant) {
  Int8Data data{m, k, {}, std::vector<std::int8_t>(k)};
  const std::uint32_t block_count = variant == Variant::b ? depth : 1;
  data.blocks.reserve(block_count);
  for (std::uint32_t block_index = 0; block_index < block_count; ++block_index) {
    std::uint32_t state = 0xC001CAFEU ^ (0x9E3779B9U * (block_index + 1U));
    WeightBlock block{std::vector<std::int8_t>(static_cast<std::size_t>(m) * k)};
    for (std::int8_t& value : block.weights) {
      value = static_cast<std::int8_t>(static_cast<int>(next_random(state) & 0xFFU) - 128);
    }
    data.blocks.push_back(std::move(block));
  }
  std::uint32_t input_state = 0xA5A5F00DU;
  for (std::int8_t& value : data.input) {
    value = static_cast<std::int8_t>(static_cast<int>(next_random(input_state) & 0xFFU) - 128);
  }
  return data;
}

using PassFunction = std::int64_t (*)(const Int8Data&, std::uint32_t,
                                      Variant, std::vector<std::int32_t>&);

[[nodiscard]] std::int64_t run_pass_int8_scalar(const Int8Data& data,
                                                std::uint32_t pass,
                                                Variant variant,
                                                std::vector<std::int32_t>& output) {
  (void)variant;
  const WeightBlock& block = data.blocks[pass % data.blocks.size()];
  std::int64_t checksum = 0;
  for (std::uint32_t row = 0; row < data.m; ++row) {
    std::int32_t sum = 0;
    const std::size_t start = static_cast<std::size_t>(row) * data.k;
    for (std::uint32_t column = 0; column < data.k; ++column) {
      sum += static_cast<std::int32_t>(block.weights[start + column]) *
             static_cast<std::int32_t>(data.input[column]);
    }
    output[row] = sum;
    checksum += sum;
  }
  return checksum;
}

[[nodiscard]] std::int32_t horizontal_sum_i32(__m256i value) {
  __m128i low = _mm256_castsi256_si128(value);
  __m128i high = _mm256_extracti128_si256(value, 1);
  __m128i sum = _mm_add_epi32(low, high);
  sum = _mm_hadd_epi32(sum, sum);
  sum = _mm_hadd_epi32(sum, sum);
  return _mm_cvtsi128_si32(sum);
}

[[nodiscard]] std::int64_t run_pass_int8_avx2(const Int8Data& data,
                                              std::uint32_t pass,
                                              Variant variant,
                                              std::vector<std::int32_t>& output) {
  (void)variant;
  const WeightBlock& block = data.blocks[pass % data.blocks.size()];
  std::int64_t checksum = 0;
  for (std::uint32_t row = 0; row < data.m; ++row) {
    const std::size_t start = static_cast<std::size_t>(row) * data.k;
    __m256i accumulator = _mm256_setzero_si256();
    std::uint32_t column = 0;
    for (; column + 16 <= data.k; column += 16) {
      const auto* weights = reinterpret_cast<const __m128i*>(block.weights.data() + start + column);
      const auto* input = reinterpret_cast<const __m128i*>(data.input.data() + column);
      const __m256i weight16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(weights));
      const __m256i input16 = _mm256_cvtepi8_epi16(_mm_loadu_si128(input));
      accumulator = _mm256_add_epi32(accumulator, _mm256_madd_epi16(weight16, input16));
    }
    std::int32_t sum = horizontal_sum_i32(accumulator);
    for (; column < data.k; ++column) {
      sum += static_cast<std::int32_t>(block.weights[start + column]) *
             static_cast<std::int32_t>(data.input[column]);
    }
    output[row] = sum;
    checksum += sum;
  }
  return checksum;
}

void run_kernel_correction_test() {
  const Int8Data data = make_data(8, 64, 2, Variant::a);
  std::vector<std::int32_t> output(data.m, 0);
  std::int64_t scalar = 0;
  std::int64_t avx2 = 0;
  for (std::uint32_t pass = 0; pass < 2; ++pass) {
    scalar += run_pass_int8_scalar(data, pass, Variant::a, output);
    avx2 += run_pass_int8_avx2(data, pass, Variant::a, output);
  }
  if (scalar != avx2) {
    throw std::runtime_error("int8 AVX2 correction test failed");
  }
  std::cerr << "int8 AVX2 correction test passed; scalar_checksum=" << scalar
            << ", avx2_checksum=" << avx2 << ", difference=0\n";
}

void touch_eviction(const std::vector<std::uint8_t>& buffer,
                    std::uint64_t& checksum) {
  volatile const std::uint8_t* bytes = buffer.data();
  for (std::size_t offset = 0; offset < buffer.size(); offset += 64) {
    checksum = (checksum * 1315423911U) ^ bytes[offset];
  }
}

void flush_cache_lines(const Int8Data& data, std::uint32_t pass) {
  const WeightBlock& block = data.blocks[pass % data.blocks.size()];
  for (std::size_t offset = 0; offset < block.weights.size(); offset += 64) {
    _mm_clflush(block.weights.data() + offset);
  }
  for (std::size_t offset = 0; offset < data.input.size(); offset += 64) {
    _mm_clflush(data.input.data() + offset);
  }
  _mm_mfence();
}

void prepare_cold_pass(const Int8Data& data,
                       const std::vector<std::uint8_t>& eviction,
                       std::uint64_t& eviction_checksum,
                       std::uint32_t pass) {
  touch_eviction(eviction, eviction_checksum);
  flush_cache_lines(data, pass);
}

struct SingleResult {
  bool affinity_succeeded = false;
  DWORD affinity_error = ERROR_SUCCESS;
  double elapsed_seconds = 0.0;
  std::uint64_t mac_count = 0;
  double mac_per_second = 0.0;
  std::int64_t checksum = 0;
};

[[nodiscard]] SingleResult run_single(const CpuTarget& target,
                                      const Options& options,
                                      PassFunction pass_function,
                                      const QpcClock& clock) {
  Int8Data data = make_data(options.m, options.k, options.depth, options.variant);
  std::vector<std::int32_t> output(data.m, 0);
  std::vector<std::uint8_t> eviction;
  if (options.variant == Variant::c) eviction.resize(kEvictionBytes, 0xA5U);
  SingleResult result;
  AffinityGuard affinity(target.group, target.group_index);
  result.affinity_succeeded = affinity.succeeded();
  result.affinity_error = affinity.error();
  double kernel_elapsed = 0.0;
  auto run_all = [&] {
    std::uint64_t eviction_checksum = 0;
    std::int64_t checksum = 0;
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
      for (std::uint32_t pass = 0; pass < options.depth; ++pass) {
        if (options.variant == Variant::c) {
          prepare_cold_pass(data, eviction, eviction_checksum, pass);
        }
        const LARGE_INTEGER begin = clock.now();
        checksum += pass_function(data, pass, options.variant, output);
        const LARGE_INTEGER end = clock.now();
        kernel_elapsed += clock.elapsed(begin, end);
      }
    }
    return checksum;
  };
  for (std::uint32_t warmup = 0; warmup < options.warmup; ++warmup) {
    (void)run_all();
    kernel_elapsed = 0.0;
  }
  for (std::uint32_t repetition = 0; repetition < options.repetitions; ++repetition) {
    result.checksum += run_all();
  }
  result.elapsed_seconds = kernel_elapsed;
  result.mac_count = static_cast<std::uint64_t>(options.m) * options.k * options.depth *
                     options.iterations * options.repetitions;
  result.mac_per_second = result.elapsed_seconds > 0.0
                              ? static_cast<double>(result.mac_count) / result.elapsed_seconds
                              : 0.0;
  return result;
}

struct ParallelResult {
  std::uint32_t worker_count = 0;
  std::string logical_cpu_indices;
  std::string rows_per_worker;
  bool all_affinity_succeeded = false;
  DWORD first_affinity_error = ERROR_SUCCESS;
  double kernel_elapsed_seconds = 0.0;
  double wall_elapsed_seconds = 0.0;
  std::uint64_t mac_count = 0;
  double mac_per_second = 0.0;
  std::int64_t checksum = 0;
};

[[nodiscard]] ParallelResult run_parallel(const std::vector<CpuTarget>& targets,
                                          const std::vector<std::uint32_t>& rows,
                                          const Options& options,
                                          PassFunction pass_function,
                                          const QpcClock& clock) {
  const std::size_t worker_count = targets.size();
  const std::size_t pass_count = static_cast<std::size_t>(options.repetitions) *
                                 options.iterations * options.depth;
  std::vector<SingleResult> results(worker_count);
  std::barrier ready(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier start(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier phase_ready(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier phase_done(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::size_t index = 0; index < worker_count; ++index) {
    workers.emplace_back([&, index] {
      Int8Data data = make_data(rows[index], options.k, options.depth, options.variant);
      std::vector<std::int32_t> output(data.m, 0);
      std::vector<std::uint8_t> eviction;
      if (options.variant == Variant::c) eviction.resize(kEvictionBytes, 0xA5U);
      AffinityGuard affinity(targets[index].group, targets[index].group_index);
      results[index].affinity_succeeded = affinity.succeeded();
      results[index].affinity_error = affinity.error();
      std::uint64_t eviction_checksum = 0;
      for (std::uint32_t warmup = 0; warmup < options.warmup; ++warmup) {
        for (std::uint32_t pass = 0; pass < options.depth; ++pass) {
          if (options.variant == Variant::c) {
            prepare_cold_pass(data, eviction, eviction_checksum, pass);
          }
          (void)pass_function(data, pass, options.variant, output);
        }
      }
      ready.arrive_and_wait();
      start.arrive_and_wait();
      for (std::size_t slot = 0; slot < pass_count; ++slot) {
        if (options.variant == Variant::c) {
          prepare_cold_pass(data, eviction, eviction_checksum,
                            static_cast<std::uint32_t>(slot % options.depth));
        }
         phase_ready.arrive_and_wait();
        const std::uint32_t pass = static_cast<std::uint32_t>(slot % options.depth);
        results[index].checksum += pass_function(data, pass, options.variant, output);
        phase_done.arrive_and_wait();
      }
    });
  }
  ready.arrive_and_wait();
  const LARGE_INTEGER wall_begin = clock.now();
  start.arrive_and_wait();
  double kernel_elapsed = 0.0;
  for (std::size_t slot = 0; slot < pass_count; ++slot) {
    phase_ready.arrive_and_wait();
    const LARGE_INTEGER phase_begin = clock.now();
    phase_done.arrive_and_wait();
    const LARGE_INTEGER phase_end = clock.now();
    kernel_elapsed += clock.elapsed(phase_begin, phase_end);
  }
  const LARGE_INTEGER wall_end = clock.now();
  for (auto& worker : workers) worker.join();

  ParallelResult result;
  result.worker_count = static_cast<std::uint32_t>(worker_count);
  for (std::size_t index = 0; index < worker_count; ++index) {
    if (index != 0) {
      result.logical_cpu_indices += ',';
      result.rows_per_worker += ',';
    }
    result.logical_cpu_indices += std::to_string(targets[index].logical_index);
    result.rows_per_worker += std::to_string(rows[index]);
    result.all_affinity_succeeded =
        index == 0 ? results[index].affinity_succeeded
                   : result.all_affinity_succeeded && results[index].affinity_succeeded;
    if (result.first_affinity_error == ERROR_SUCCESS &&
        results[index].affinity_error != ERROR_SUCCESS) {
      result.first_affinity_error = results[index].affinity_error;
    }
    result.checksum += results[index].checksum;
    result.mac_count += static_cast<std::uint64_t>(rows[index]) * options.k *
                        options.depth * options.iterations;
  }
  result.mac_count *= options.repetitions;
  result.kernel_elapsed_seconds = kernel_elapsed;
  result.wall_elapsed_seconds = clock.elapsed(wall_begin, wall_end);
  result.mac_per_second = kernel_elapsed > 0.0
                              ? static_cast<double>(result.mac_count) / kernel_elapsed
                              : 0.0;
  return result;
}

[[nodiscard]] const char* kernel_name(Kernel kernel) {
  if (kernel == Kernel::scalar) return "scalar";
  if (kernel == Kernel::avx2) return "avx2";
  return "auto";
}

[[nodiscard]] char variant_name(Variant variant) { return static_cast<char>(variant); }

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_options(argc, argv);
    const CpuFeatures features = detect_cpu_features();
    PassFunction pass_function = run_pass_int8_scalar;
    const char* kernel_used = "scalar";
    if (options.kernel != Kernel::scalar) {
      if (features.avx2) {
        run_kernel_correction_test();
        pass_function = run_pass_int8_avx2;
        kernel_used = "avx2";
      } else {
        std::cerr << "AVX2 unavailable; falling back to scalar\n";
      }
    }
    if (options.target_kib_specified) {
      options.m = static_cast<std::uint32_t>(
          (static_cast<std::uint64_t>(options.target_kib) * 1024U) / options.k);
      if (options.m == 0) throw std::runtime_error("target size too small");
    }
    const auto targets = enumerate_cpus();
    if (options.selected_cpu && options.parallel_workers > 0) {
      throw std::runtime_error("--cpu cannot combine with parallel mode");
    }
    const QpcClock clock;
    if (options.parallel_workers > 0) {
      std::vector<CpuTarget> selected;
      if (options.parallel_cpus.empty()) {
        if (options.parallel_workers > targets.size())
          throw std::runtime_error("parallel workers exceed active CPUs");
        selected.assign(targets.begin(), targets.begin() + options.parallel_workers);
      } else {
        for (const std::uint32_t logical : options.parallel_cpus) {
          if (logical >= targets.size()) throw std::runtime_error("parallel CPU out of range");
          selected.push_back(targets[logical]);
        }
      }
      std::vector<std::uint32_t> rows;
      if (options.parallel_rows.empty()) {
        rows.assign(selected.size(), options.m);
      } else {
        rows = options.parallel_rows;
      }
      if (rows.size() != selected.size()) throw std::runtime_error("row/CPU count mismatch");
      const ParallelResult result =
          run_parallel(selected, rows, options, pass_function, clock);
      std::cout << "probe,kernel_requested,kernel_used,avx2_supported,target_kib,m,K,depth,variant,"
                   "iterations,repetitions,warmup,worker_count,logical_cpu_indices,rows_per_worker,"
                   "all_affinity_succeeded,affinity_error,kernel_elapsed_seconds,wall_elapsed_seconds,"
                   "mac_count,mac_per_second,checksum\n"
                << "int8_batch," << kernel_name(options.kernel) << ',' << kernel_used << ','
                << (features.avx2 ? "true" : "false") << ','
                << (options.target_kib_specified ? options.target_kib : 0) << ',' << options.m << ','
                << options.k << ',' << options.depth << ',' << variant_name(options.variant) << ','
                << options.iterations << ',' << options.repetitions << ',' << options.warmup << ','
                << result.worker_count << ",\"" << result.logical_cpu_indices << "\",\""
                << result.rows_per_worker << "\"," << (result.all_affinity_succeeded ? "true" : "false")
                << ',' << result.first_affinity_error << ',' << result.kernel_elapsed_seconds << ','
                << result.wall_elapsed_seconds << ',' << result.mac_count << ','
                << result.mac_per_second << ',' << result.checksum << '\n';
      return 0;
    }

    if (options.cpu >= targets.size()) throw std::runtime_error("--cpu out of range");
    const SingleResult result = run_single(targets[options.cpu], options, pass_function, clock);
    std::cout << "probe,kernel_requested,kernel_used,avx2_supported,target_kib,m,K,depth,variant,"
                 "iterations,repetitions,warmup,logical_cpu_index,affinity_succeeded,affinity_error,"
                 "elapsed_seconds,mac_count,mac_per_second,checksum\n"
              << "int8," << kernel_name(options.kernel) << ',' << kernel_used << ','
              << (features.avx2 ? "true" : "false") << ','
              << (options.target_kib_specified ? options.target_kib : 0) << ',' << options.m << ','
              << options.k << ',' << options.depth << ',' << variant_name(options.variant) << ','
              << options.iterations << ',' << options.repetitions << ',' << options.warmup << ','
              << options.cpu << ',' << (result.affinity_succeeded ? "true" : "false") << ','
              << result.affinity_error << ',' << result.elapsed_seconds << ',' << result.mac_count << ','
              << result.mac_per_second << ',' << result.checksum << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}

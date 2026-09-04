#ifndef _WIN32
#error "stream_bandwidth requires Windows APIs"
#endif

#include <windows.h>

#include <barrier>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

struct Options {
  std::uint32_t mebibytes = 256;
  std::uint32_t repetitions = 4;
  std::vector<std::uint32_t> logical_cpus = {0, 2, 4, 6};
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
  explicit AffinityGuard(std::uint32_t logical_cpu) {
    if (!GetThreadGroupAffinity(GetCurrentThread(), &previous_)) {
      error_ = GetLastError();
      return;
    }
    saved_ = true;
    GROUP_AFFINITY requested{};
    requested.Mask = static_cast<KAFFINITY>(1) << logical_cpu;
    requested.Group = 0;
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

  AffinityGuard(const AffinityGuard&) = delete;
  AffinityGuard& operator=(const AffinityGuard&) = delete;

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

[[nodiscard]] std::vector<std::uint32_t> parse_cpu_list(std::string_view text) {
  std::vector<std::uint32_t> cpus;
  std::size_t start = 0;
  while (start < text.size()) {
    const std::size_t comma = text.find(',', start);
    const std::size_t end = comma == std::string_view::npos ? text.size() : comma;
    cpus.push_back(parse_u32(text.substr(start, end - start), "--cpus", true));
    start = end == text.size() ? text.size() : end + 1;
  }
  if (cpus.empty()) {
    throw std::runtime_error("--cpus requires at least one logical CPU");
  }
  return cpus;
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
    if (argument == "--mebibytes") {
      options.mebibytes = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--repetitions") {
      options.repetitions = parse_u32(require_value(argument), argument, false);
    } else if (argument == "--cpus") {
      options.logical_cpus = parse_cpu_list(require_value(argument));
    } else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: stream_bandwidth [--mebibytes N] [--repetitions N] "
                   "[--cpus 0,2,4,6]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + std::string(argument));
    }
  }
  return options;
}

struct WorkerResult {
  bool affinity_succeeded = false;
  DWORD affinity_error = ERROR_SUCCESS;
  std::uint64_t checksum = 0;
};

struct BatchResult {
  bool all_affinity_succeeded = false;
  DWORD first_affinity_error = ERROR_SUCCESS;
  double elapsed_seconds = 0.0;
  std::uint64_t bytes_copied = 0;
  double gigabytes_per_second = 0.0;
  std::uint64_t checksum = 0;
};

[[nodiscard]] BatchResult run_batch(const Options& options,
                                    const QpcClock& clock) {
  const std::uint64_t bytes_per_buffer =
      static_cast<std::uint64_t>(options.mebibytes) * 1024U * 1024U;
  if (bytes_per_buffer > (std::numeric_limits<std::size_t>::max)()) {
    throw std::runtime_error("buffer size exceeds addressable memory");
  }
  const std::size_t buffer_size = static_cast<std::size_t>(bytes_per_buffer);
  const std::size_t worker_count = options.logical_cpus.size();
  std::vector<WorkerResult> results(worker_count);
  std::barrier ready(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier release(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::barrier done(static_cast<std::ptrdiff_t>(worker_count + 1));
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::size_t index = 0; index < worker_count; ++index) {
    workers.emplace_back([&, index] {
      std::vector<std::uint8_t> source(buffer_size);
      std::vector<std::uint8_t> destination(buffer_size, 0);
      for (std::size_t offset = 0; offset < source.size(); offset += 4096) {
        source[offset] = static_cast<std::uint8_t>(offset >> 12);
      }
      AffinityGuard affinity(options.logical_cpus[index]);
      results[index].affinity_succeeded = affinity.succeeded();
      results[index].affinity_error = affinity.error();
      ready.arrive_and_wait();
      release.arrive_and_wait();
      for (std::uint32_t repetition = 0; repetition < options.repetitions;
           ++repetition) {
        std::memcpy(destination.data(), source.data(), source.size());
      }
      std::uint64_t checksum = 0;
      for (std::size_t offset = 0; offset < destination.size(); offset += 4096) {
        checksum += destination[offset];
      }
      results[index].checksum = checksum;
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

  BatchResult result;
  result.all_affinity_succeeded = true;
  for (const WorkerResult& worker : results) {
    result.all_affinity_succeeded =
        result.all_affinity_succeeded && worker.affinity_succeeded;
    if (result.first_affinity_error == ERROR_SUCCESS &&
        worker.affinity_error != ERROR_SUCCESS) {
      result.first_affinity_error = worker.affinity_error;
    }
    result.checksum += worker.checksum;
  }
  result.elapsed_seconds = clock.elapsed(begin, end);
  result.bytes_copied = bytes_per_buffer * 2U * options.repetitions * worker_count;
  result.gigabytes_per_second =
      static_cast<double>(result.bytes_copied) / result.elapsed_seconds /
      1'000'000'000.0;
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const QpcClock clock;
    const BatchResult result = run_batch(options, clock);
    std::cout << "probe,mebibytes_per_buffer,repetitions,worker_count,"
                 "logical_cpu_indices,all_affinity_succeeded,affinity_error,"
                 "elapsed_seconds,bytes_copied,gigabytes_per_second,checksum\n"
              << "stream_copy," << options.mebibytes << ','
              << options.repetitions << ',' << options.logical_cpus.size() << ",\"";
    for (std::size_t index = 0; index < options.logical_cpus.size(); ++index) {
      if (index != 0) {
        std::cout << ',';
      }
      std::cout << options.logical_cpus[index];
    }
    std::cout << "\"," << (result.all_affinity_succeeded ? "true" : "false")
              << ',' << result.first_affinity_error << ','
              << result.elapsed_seconds << ',' << result.bytes_copied << ','
              << result.gigabytes_per_second << ',' << result.checksum << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}

#include <algorithm>
#include <atomic>
#include <barrier>
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

#include <immintrin.h>

#include "cnrl/aligned_buffer.hpp"
#include "cnrl/argparse.hpp"
#include "cnrl/checked_math.hpp"
#include "cnrl/platform.hpp"
#include "cnrl/random.hpp"

namespace {

struct Options {
  std::string mode = "read";
  std::uint32_t mib_per_worker = 256;
  std::uint32_t repetitions = 4;
  std::vector<std::uint32_t> cpus;
  bool allow_smt_siblings = false;
  bool require_affinity = true;
};

struct WorkerBuffers {
  cnrl::AlignedBuffer<std::uint8_t> source;
  cnrl::AlignedBuffer<std::uint8_t> destination;
};

struct WorkerResult {
  bool affinity_succeeded = false;
  std::uint32_t affinity_error = 0;
  std::uint64_t checksum = 0;
};

std::uint64_t read_stream(const std::uint8_t* data, std::size_t bytes) noexcept {
  __m256i accumulator0 = _mm256_setzero_si256();
  __m256i accumulator1 = _mm256_setzero_si256();
  __m256i accumulator2 = _mm256_setzero_si256();
  __m256i accumulator3 = _mm256_setzero_si256();
  std::size_t offset = 0;
  for (; offset + 128U <= bytes; offset += 128U) {
    accumulator0 = _mm256_add_epi64(
        accumulator0,
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset)));
    accumulator1 = _mm256_xor_si256(
        accumulator1,
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 32U)));
    accumulator2 = _mm256_add_epi64(
        accumulator2,
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 64U)));
    accumulator3 = _mm256_xor_si256(
        accumulator3,
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(data + offset + 96U)));
  }
  const __m256i total = _mm256_xor_si256(
      _mm256_add_epi64(accumulator0, accumulator2),
      _mm256_xor_si256(accumulator1, accumulator3));
  alignas(32) std::uint64_t lanes[4];
  _mm256_store_si256(reinterpret_cast<__m256i*>(lanes), total);
  std::uint64_t checksum = lanes[0] ^ lanes[1] ^ lanes[2] ^ lanes[3];
  for (; offset < bytes; ++offset) {
    checksum = (checksum * 1315423911ULL) ^ data[offset];
  }
  return checksum;
}

std::string join_u32(const std::vector<std::uint32_t>& values) {
  std::string output;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output.push_back(';');
    output += std::to_string(values[index]);
  }
  return output;
}

Options parse_options(int argc, char** argv) {
  Options options;
  cnrl::ArgParser parser(argc, argv);
  while (!parser.done()) {
    const auto argument = parser.next();
    if (argument == "--mode") {
      options.mode = std::string(parser.value(argument));
    } else if (argument == "--mib") {
      options.mib_per_worker = cnrl::parse_u32(parser.value(argument), argument);
    } else if (argument == "--repetitions") {
      options.repetitions = cnrl::parse_u32(parser.value(argument), argument);
    } else if (argument == "--cpus") {
      options.cpus = cnrl::parse_u32_list(parser.value(argument), argument, true);
    } else if (argument == "--allow-smt-siblings") {
      options.allow_smt_siblings = true;
    } else if (argument == "--allow-affinity-failure") {
      options.require_affinity = false;
    } else if (argument == "--help" || argument == "-h") {
      std::cout
          << "cnrl_bandwidth --mode read|copy [--mib 256] [--repetitions 4] "
             "[--cpus LIST] [--allow-smt-siblings] [--allow-affinity-failure]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown option: " + std::string(argument));
    }
  }
  if (options.mode != "read" && options.mode != "copy") {
    throw std::runtime_error("mode must be read or copy");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse_options(argc, argv);
    if (!cnrl::detect_cpu_features().avx2) {
      throw std::runtime_error("AVX2 is required by the bandwidth kernel");
    }

    const auto topology = cnrl::discover_cpu_topology();
    if (options.cpus.empty()) {
      options.cpus = cnrl::choose_one_logical_per_physical_core(topology);
      if (options.cpus.size() > 4U) options.cpus.resize(4U);
    }
    if (options.cpus.empty()) throw std::runtime_error("no CPU was selected");

    std::vector<cnrl::LogicalProcessor> processors;
    std::vector<std::uint32_t> physical_cores;
    processors.reserve(options.cpus.size());
    physical_cores.reserve(options.cpus.size());
    for (const std::uint32_t cpu : options.cpus) {
      processors.push_back(cnrl::find_logical_processor(topology, cpu));
      physical_cores.push_back(cnrl::find_physical_core_index(topology, cpu));
    }
    if (!options.allow_smt_siblings) {
      auto sorted = physical_cores;
      std::sort(sorted.begin(), sorted.end());
      if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::runtime_error(
            "selected CPUs contain SMT siblings; use --allow-smt-siblings only deliberately");
      }
    }

    const std::uint64_t bytes_wide = cnrl::checked_mul_u64(
        cnrl::checked_mul_u64(options.mib_per_worker, 1024U), 1024U);
    if (bytes_wide > static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)())) {
      throw std::runtime_error("requested buffer exceeds size_t");
    }
    const std::size_t bytes = static_cast<std::size_t>(bytes_wide);

    std::vector<WorkerBuffers> buffers(options.cpus.size());
    for (auto& buffer : buffers) {
      buffer.source.resize(bytes);
      if (options.mode == "copy") buffer.destination.resize(bytes);
    }
    std::vector<WorkerResult> results(options.cpus.size());

    std::barrier initialized(static_cast<std::ptrdiff_t>(options.cpus.size() + 1U));
    std::barrier release(static_cast<std::ptrdiff_t>(options.cpus.size() + 1U));
    std::barrier done(static_cast<std::ptrdiff_t>(options.cpus.size() + 1U));
    std::vector<std::thread> threads;
    threads.reserve(options.cpus.size());

    for (std::size_t worker = 0; worker < options.cpus.size(); ++worker) {
      threads.emplace_back([&, worker] {
        cnrl::AffinityGuard affinity(processors[worker]);
        results[worker].affinity_succeeded = affinity.succeeded();
        results[worker].affinity_error = affinity.error();

        cnrl::XorShift32 random(0x51EDBEEFU ^ static_cast<std::uint32_t>(worker));
        for (std::size_t index = 0; index < bytes; ++index) {
          buffers[worker].source[index] = static_cast<std::uint8_t>(random.next());
        }
        if (options.mode == "copy") {
          buffers[worker].destination.fill_zero();
          std::memcpy(buffers[worker].destination.data(),
                      buffers[worker].source.data(), bytes);
          results[worker].checksum ^= buffers[worker].destination[bytes / 2U];
        } else {
          results[worker].checksum ^= read_stream(buffers[worker].source.data(), bytes);
        }

        initialized.arrive_and_wait();
        release.arrive_and_wait();
        std::uint64_t checksum = 0;
        for (std::uint32_t repetition = 0; repetition < options.repetitions; ++repetition) {
          if (options.mode == "read") {
            checksum ^= read_stream(buffers[worker].source.data(), bytes);
          } else {
            std::memcpy(buffers[worker].destination.data(),
                        buffers[worker].source.data(), bytes);
            checksum ^= buffers[worker].destination[
                (static_cast<std::size_t>(repetition) * 4096U) % bytes];
          }
        }
        results[worker].checksum ^= checksum;
        done.arrive_and_wait();
      });
    }

    initialized.arrive_and_wait();
    cnrl::MonotonicClock clock;
    const std::uint64_t begin = clock.now();
    release.arrive_and_wait();
    done.arrive_and_wait();
    const std::uint64_t end = clock.now();
    for (auto& thread : threads) thread.join();

    const double seconds = clock.seconds_between(begin, end);
    const std::uint64_t payload = cnrl::checked_mul_u64(
        cnrl::checked_mul_u64(bytes_wide, options.repetitions),
        options.cpus.size());
    const std::uint64_t estimated_traffic = options.mode == "copy"
        ? cnrl::checked_mul_u64(payload, 2U) : payload;
    std::uint64_t checksum = 0;
    bool all_affinity = true;
    std::uint32_t first_affinity_error = 0;
    for (const auto& result : results) {
      checksum ^= result.checksum;
      all_affinity = all_affinity && result.affinity_succeeded;
      if (first_affinity_error == 0U && result.affinity_error != 0U) {
        first_affinity_error = result.affinity_error;
      }
    }
    const bool valid = seconds > 0.0 &&
                       (!options.require_affinity || all_affinity);

    std::cout
        << "mode,mib_per_worker,repetitions,worker_count,allow_smt_siblings,cpus,"
           "physical_cores,elapsed_seconds,payload_bytes,estimated_traffic_bytes,"
           "payload_gb_per_second,estimated_traffic_gb_per_second,"
           "all_affinity_succeeded,affinity_error,checksum,valid,error\n";
    std::cout << options.mode << ',' << options.mib_per_worker << ','
              << options.repetitions << ',' << options.cpus.size() << ','
              << (options.allow_smt_siblings ? "true" : "false") << ",\""
              << join_u32(options.cpus) << "\",\"" << join_u32(physical_cores)
              << "\"," << seconds << ',' << payload << ',' << estimated_traffic
              << ',' << static_cast<double>(payload) / seconds / 1.0e9 << ','
              << static_cast<double>(estimated_traffic) / seconds / 1.0e9 << ','
              << (all_affinity ? "true" : "false") << ','
              << first_affinity_error << ',' << checksum << ','
              << (valid ? "true" : "false") << ",\""
              << (valid ? "" : "timing or affinity failure") << "\"\n";
    return valid ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}

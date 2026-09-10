#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "cnrl/argparse.hpp"
#include "cnrl/benchmark.hpp"
#include "cnrl/checked_math.hpp"
#include "cnrl/platform.hpp"
#include "cnrl/sharding.hpp"

namespace {

std::string csv_escape(const std::string& value) {
  std::string escaped = "\"";
  for (const char character : value) {
    if (character == '"') escaped.push_back('"');
    escaped.push_back(character);
  }
  escaped.push_back('"');
  return escaped;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  if (values.size() % 2U != 0U) return values[middle];
  return (values[middle - 1U] + values[middle]) * 0.5;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::uint32_t dimension = 512;
    std::uint32_t weight_kib = 256;
    std::uint32_t depth = 16;
    std::uint32_t repetitions = 7;
    std::uint32_t warmup = 2;
    std::uint32_t passes = 3;
    std::vector<std::uint32_t> cpus;
    bool allow_affinity_failure = false;
    bool allow_smt_siblings = false;

    cnrl::ArgParser parser(argc, argv);
    while (!parser.done()) {
      const auto argument = parser.next();
      if (argument == "--D") {
        dimension = cnrl::parse_u32(parser.value(argument), argument);
      } else if (argument == "--weight-kib") {
        weight_kib = cnrl::parse_u32(parser.value(argument), argument);
      } else if (argument == "--R") {
        depth = cnrl::parse_u32(parser.value(argument), argument);
      } else if (argument == "--repetitions") {
        repetitions = cnrl::parse_u32(parser.value(argument), argument);
      } else if (argument == "--warmup") {
        warmup = cnrl::parse_u32(parser.value(argument), argument, true);
      } else if (argument == "--passes") {
        passes = cnrl::parse_u32(parser.value(argument), argument);
      } else if (argument == "--cpus") {
        cpus = cnrl::parse_u32_list(parser.value(argument), argument, true);
      } else if (argument == "--allow-affinity-failure") {
        allow_affinity_failure = true;
      } else if (argument == "--allow-smt-siblings") {
        allow_smt_siblings = true;
      } else if (argument == "--help" || argument == "-h") {
        std::cout
            << "cnrl_calibrate [--cpus LIST] [--D 512] [--weight-kib 256] "
               "[--R 16] [--passes 3] [--allow-smt-siblings] "
               "[--allow-affinity-failure]\n";
        return 0;
      } else {
        throw std::runtime_error("unknown option: " + std::string(argument));
      }
    }

    const auto topology = cnrl::discover_cpu_topology();
    if (cpus.empty()) {
      cpus = cnrl::choose_one_logical_per_physical_core(topology);
      if (cpus.size() > 4U) cpus.resize(4U);
    }
    if (cpus.empty()) throw std::runtime_error("no physical CPU was selected");

    std::vector<std::uint32_t> physical_cores;
    std::vector<std::uint32_t> efficiency_classes;
    physical_cores.reserve(cpus.size());
    efficiency_classes.reserve(cpus.size());
    for (const std::uint32_t cpu : cpus) {
      const std::uint32_t physical = cnrl::find_physical_core_index(topology, cpu);
      physical_cores.push_back(physical);
      efficiency_classes.push_back(topology.physical_cores.at(physical).efficiency_class);
    }
    if (!allow_smt_siblings) {
      auto sorted = physical_cores;
      std::sort(sorted.begin(), sorted.end());
      if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::runtime_error(
            "selected CPUs contain SMT siblings; use --allow-smt-siblings only deliberately");
      }
    }

    const std::uint64_t requested_bytes = cnrl::checked_mul_u64(weight_kib, 1024U);
    const std::uint64_t rows_wide = std::max<std::uint64_t>(
        1U, requested_bytes / dimension);
    if (rows_wide > UINT32_MAX) {
      throw std::runtime_error("calibration row count exceeds uint32");
    }
    const std::uint32_t rows = static_cast<std::uint32_t>(rows_wide);

    std::vector<std::vector<double>> measurements(cpus.size());
    std::vector<bool> affinity(cpus.size(), true);
    std::vector<std::string> errors(cpus.size());

    for (std::uint32_t pass = 0; pass < passes; ++pass) {
      std::vector<std::size_t> order(cpus.size());
      std::iota(order.begin(), order.end(), std::size_t{0});
      if (pass % 3U == 1U) {
        std::reverse(order.begin(), order.end());
      } else if (pass % 3U == 2U && order.size() > 1U) {
        std::rotate(order.begin(), order.begin() + 1, order.end());
      }
      for (const std::size_t index : order) {
        cnrl::RunConfig config;
        config.gate = cnrl::GateKind::calibrate;
        config.shape = {dimension, 1, depth};
        config.variant = cnrl::WeightVariant::shared;
        config.kernel = cnrl::KernelKind::avx2_repeat;
        config.transition.kind = cnrl::TransitionKind::frozen;
        config.warmup_repetitions = warmup;
        config.timed_repetitions = repetitions;
        config.require_affinity = !allow_affinity_failure;
        config.shards = cnrl::make_shards({cpus[index]}, {rows});
        const auto result = cnrl::run_benchmark(config);
        affinity[index] = affinity[index] && result.all_affinity_succeeded;
        if (result.valid) {
          measurements[index].push_back(result.mac_per_second);
        } else if (errors[index].empty()) {
          errors[index] = result.error;
        }
      }
    }

    std::cout
        << "logical_cpu,physical_core_index,efficiency_class,D,rows,weight_bytes,R,"
           "internal_repetitions,passes,mac_per_second,min_mac_per_second,"
           "max_mac_per_second,mean_mac_per_second,relative_stddev,"
           "allow_smt_siblings,affinity_succeeded,valid,error\n";
    bool every_row_valid = true;
    for (std::size_t index = 0; index < cpus.size(); ++index) {
      const bool valid = measurements[index].size() == passes &&
                         (allow_affinity_failure || affinity[index]);
      every_row_valid = every_row_valid && valid;
      double minimum = 0.0;
      double maximum = 0.0;
      double mean = 0.0;
      double med = 0.0;
      double relative_stddev = 0.0;
      if (!measurements[index].empty()) {
        minimum = *std::min_element(measurements[index].begin(),
                                    measurements[index].end());
        maximum = *std::max_element(measurements[index].begin(),
                                    measurements[index].end());
        mean = std::accumulate(measurements[index].begin(),
                               measurements[index].end(), 0.0) /
               static_cast<double>(measurements[index].size());
        med = median(measurements[index]);
        double variance = 0.0;
        for (const double value : measurements[index]) {
          const double delta = value - mean;
          variance += delta * delta;
        }
        variance /= static_cast<double>(measurements[index].size());
        relative_stddev = mean > 0.0 ? std::sqrt(variance) / mean : 0.0;
      }
      std::string error = errors[index];
      if (!valid && error.empty()) error = "one or more calibration passes failed";
      std::cout << cpus[index] << ',' << physical_cores[index] << ','
                << efficiency_classes[index] << ',' << dimension << ',' << rows
                << ',' << cnrl::checked_mul_u64(rows, dimension) << ',' << depth
                << ',' << repetitions << ',' << passes << ',' << med << ','
                << minimum << ',' << maximum << ',' << mean << ','
                << relative_stddev << ','
                << (allow_smt_siblings ? "true" : "false") << ','
                << (affinity[index] ? "true" : "false") << ','
                << (valid ? "true" : "false") << ',' << csv_escape(error) << '\n';
    }
    return every_row_valid ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}

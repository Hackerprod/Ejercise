#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "cnrl/argparse.hpp"
#include "cnrl/benchmark.hpp"
#include "cnrl/checked_math.hpp"
#include "cnrl/csv.hpp"
#include "cnrl/platform.hpp"
#include "cnrl/sharding.hpp"
#include "cnrl/types.hpp"

namespace {
struct Options {
  cnrl::RunConfig run;
  std::vector<std::uint32_t> cpus;
  std::vector<std::uint32_t> rows;
  std::vector<double> rates;
  std::uint32_t average_weight_kib_per_core = 512;
  std::uint32_t row_alignment = 1;
  bool all_physical = false;
  bool header = true;
  bool D_explicit = false;
  bool S_explicit = false;
  bool transition_explicit = false;
  bool timing_explicit = false;
  bool square_output = false;
};

void help() {
  std::cout <<
    "cnrl_gate --gate t0r|t0m|t0rm [options]\n\n"
    "Core shape:\n"
    "  --D N --S N --R N --slot-tile 1|2|4|8 (default 4; tile 8 is a measured stress path)\n"
    "  --kernel scalar|repeat|fused\n"
    "  --variant shared|clone|untied|cold\n"
    "  --transition frozen|fixed|group-rms|global-rms\n\n"
    "Placement:\n"
    "  --cpus 0,2,4,6 --rows 435,408,246,383\n"
    "  --rates 19.3,18.1,10.9,17.0 (derive rows proportionally)\n"
    "  --average-weight-kib-per-core N (static T0-R/T0-M average, default 512)\n"
    "  --weight-kib-per-core N (compatibility alias; also an average target)\n"
    "  --row-alignment N (default 1; use >1 only when deliberately testing coarse shards)\n"
    "  --all-physical (otherwise at most four physical cores are selected)\n"
    "  --square-output (force sum(rows)=D in an isolated bridge)\n\n"
    "Timing:\n"
    "  --warmup N --repetitions N --sequences N\n"
    "  --timing full|round --profile --allow-affinity-failure --allow-smt-siblings --no-header\n\n"
    "Transition scaling:\n"
    "  --projection-shift N --state-mult N --output-mult N --final-shift N\n"
    "  --target-rms X --epsilon X\n";
}

Options parse(int argc, char** argv) {
  Options options;
  cnrl::ArgParser parser(argc, argv);
  while (!parser.done()) {
    const auto arg = parser.next();
    if (arg == "--help" || arg == "-h") { help(); std::exit(0); }
    else if (arg == "--gate") options.run.gate = cnrl::parse_gate_kind(std::string(parser.value(arg)));
    else if (arg == "--D") { options.run.shape.dimension = cnrl::parse_u32(parser.value(arg), arg); options.D_explicit = true; }
    else if (arg == "--S") { options.run.shape.slots = cnrl::parse_u32(parser.value(arg), arg); options.S_explicit = true; }
    else if (arg == "--R") options.run.shape.depth = cnrl::parse_u32(parser.value(arg), arg);
    else if (arg == "--slot-tile") options.run.slot_tile = cnrl::parse_u32(parser.value(arg), arg);
    else if (arg == "--kernel") options.run.kernel = cnrl::parse_kernel_kind(std::string(parser.value(arg)));
    else if (arg == "--variant") options.run.variant = cnrl::parse_weight_variant(std::string(parser.value(arg)));
    else if (arg == "--transition") { options.run.transition.kind = cnrl::parse_transition_kind(std::string(parser.value(arg))); options.transition_explicit = true; }
    else if (arg == "--cpus") options.cpus = cnrl::parse_u32_list(parser.value(arg), arg, true);
    else if (arg == "--rows") options.rows = cnrl::parse_u32_list(parser.value(arg), arg);
    else if (arg == "--rates") options.rates = cnrl::parse_double_list(parser.value(arg), arg);
    else if (arg == "--average-weight-kib-per-core" || arg == "--weight-kib-per-core") {
      options.average_weight_kib_per_core = cnrl::parse_u32(parser.value(arg), arg);
    }
    else if (arg == "--row-alignment") options.row_alignment = cnrl::parse_u32(parser.value(arg), arg);
    else if (arg == "--warmup") options.run.warmup_repetitions = cnrl::parse_u32(parser.value(arg), arg, true);
    else if (arg == "--repetitions") options.run.timed_repetitions = cnrl::parse_u32(parser.value(arg), arg);
    else if (arg == "--sequences") options.run.sequences_per_repetition = cnrl::parse_u32(parser.value(arg), arg);
    else if (arg == "--seed") options.run.seed = cnrl::parse_u32(parser.value(arg), arg, true);
    else if (arg == "--timing") { options.run.timing_scope = cnrl::parse_timing_scope(std::string(parser.value(arg))); options.timing_explicit = true; }
    else if (arg == "--profile") options.run.phase_profile = true;
    else if (arg == "--allow-affinity-failure") options.run.require_affinity = false;
    else if (arg == "--allow-smt-siblings") options.run.allow_smt_siblings = true;
    else if (arg == "--all-physical") options.all_physical = true;
    else if (arg == "--square-output") options.square_output = true;
    else if (arg == "--no-header") options.header = false;
    else if (arg == "--projection-shift") options.run.transition.projection_shift = cnrl::parse_u32(parser.value(arg), arg, true);
    else if (arg == "--state-mult") options.run.transition.state_multiplier = cnrl::parse_i32(parser.value(arg), arg);
    else if (arg == "--output-mult") options.run.transition.output_multiplier = cnrl::parse_i32(parser.value(arg), arg);
    else if (arg == "--final-shift") options.run.transition.final_shift = cnrl::parse_u32(parser.value(arg), arg, true);
    else if (arg == "--target-rms") options.run.transition.target_rms = cnrl::parse_double(parser.value(arg), arg);
    else if (arg == "--epsilon") options.run.transition.epsilon = cnrl::parse_double(parser.value(arg), arg);
    else throw std::runtime_error("unknown option: " + std::string(arg));
  }

  if (!options.D_explicit) options.run.shape.dimension = options.run.gate == cnrl::GateKind::t0rm ? 1472U : 512U;
  if (!options.S_explicit) options.run.shape.slots = options.run.gate == cnrl::GateKind::t0r ? 1U : 8U;
  if (!options.transition_explicit) {
    options.run.transition.kind = options.run.gate == cnrl::GateKind::t0rm
        ? cnrl::TransitionKind::fixed_point : cnrl::TransitionKind::frozen;
  }
  if (options.run.gate != cnrl::GateKind::t0rm && options.run.transition.kind != cnrl::TransitionKind::frozen) {
    throw std::runtime_error("T0-R/T0-M are isolated gates and require --transition frozen");
  }
  if (options.run.gate == cnrl::GateKind::t0rm && options.run.transition.kind == cnrl::TransitionKind::frozen) {
    throw std::runtime_error("T0-RM requires a real transition");
  }
  if (options.run.variant == cnrl::WeightVariant::cold && !options.timing_explicit) {
    options.run.timing_scope = cnrl::TimingScope::round_window;
  }
  return options;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse(argc, argv);
    const auto topology = cnrl::discover_cpu_topology();
    if (options.cpus.empty()) {
      options.cpus = cnrl::choose_one_logical_per_physical_core(topology);
      if (!options.all_physical && options.cpus.size() > 4) options.cpus.resize(4);
    }
    if (options.cpus.empty()) throw std::runtime_error("no physical cores were discovered");
    if (!options.rows.empty() && options.rows.size() != options.cpus.size()) {
      throw std::runtime_error("--rows count must equal --cpus count");
    }
    if (!options.rates.empty() && options.rates.size() != options.cpus.size()) {
      throw std::runtime_error("--rates count must equal --cpus count");
    }
    if (options.rows.empty()) {
      const std::vector<double> rates = options.rates.empty()
          ? std::vector<double>(options.cpus.size(), 1.0) : options.rates;
      if (options.run.transition.kind != cnrl::TransitionKind::frozen || options.square_output) {
        options.rows = cnrl::proportional_rows(
            options.run.shape.dimension, rates, options.row_alignment);
      } else {
        const std::uint64_t requested_bytes = cnrl::checked_mul_u64(
            options.average_weight_kib_per_core, 1024U);
        const std::uint64_t rows_per_core_wide = std::max<std::uint64_t>(
            1U, requested_bytes / options.run.shape.dimension);
        const std::uint64_t total_rows_wide = cnrl::checked_mul_u64(
            rows_per_core_wide, options.cpus.size());
        if (total_rows_wide > UINT32_MAX) {
          throw std::runtime_error("requested static output rows exceed uint32");
        }
        options.rows = cnrl::proportional_rows(
            static_cast<std::uint32_t>(total_rows_wide), rates,
            options.row_alignment);
      }
    }
    options.run.shards = cnrl::make_shards(options.cpus, options.rows);
    const auto result = cnrl::run_benchmark(options.run);
    if (options.header) cnrl::write_run_csv_header(std::cout);
    cnrl::write_run_csv_row(std::cout, options.run, result);
    if (!result.valid) {
      std::cerr << "gate rejected: " << result.error << '\n';
      return 2;
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}

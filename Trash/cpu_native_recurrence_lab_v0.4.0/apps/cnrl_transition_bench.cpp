#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "cnrl/aligned_buffer.hpp"
#include "cnrl/argparse.hpp"
#include "cnrl/platform.hpp"
#include "cnrl/random.hpp"
#include "cnrl/sharding.hpp"
#include "cnrl/spin_barrier.hpp"
#include "cnrl/transitions.hpp"
#include "cnrl/types.hpp"

namespace {
struct Options {
  cnrl::Shape shape{1472, 8, 1};
  cnrl::TransitionConfig transition{};
  std::vector<std::uint32_t> cpus;
  std::vector<std::uint32_t> rows;
  std::vector<double> rates;
  std::uint32_t warmup = 100;
  std::uint32_t repetitions = 1000;
  std::uint32_t seed = 0x7A4D13C9U;
  bool require_affinity = true;
  bool allow_smt_siblings = false;
};

void print_help() {
  std::cout
      << "cnrl_transition_bench --transition fixed|group-rms|global-rms [options]\n"
         "  --D 1472 --S 8 --cpus 0,2,4,6 --rates 19.3,18.1,10.9,17.0\n"
         "  --rows 435,408,246,383 --warmup 100 --repetitions 1000\n"
         "  --projection-shift 12 --state-mult 1 --output-mult 1 --final-shift 0\n"
         "  --target-rms 32 --epsilon 1e-6\n";
}

Options parse(int argc, char** argv) {
  Options options;
  options.transition.kind = cnrl::TransitionKind::fixed_point;
  cnrl::ArgParser parser(argc, argv);
  while (!parser.done()) {
    const auto argument = parser.next();
    if (argument == "--help" || argument == "-h") {
      print_help();
      std::exit(0);
    } else if (argument == "--transition") {
      options.transition.kind = cnrl::parse_transition_kind(
          std::string(parser.value(argument)));
    } else if (argument == "--D") {
      options.shape.dimension = cnrl::parse_u32(parser.value(argument), argument);
    } else if (argument == "--S") {
      options.shape.slots = cnrl::parse_u32(parser.value(argument), argument);
    } else if (argument == "--cpus") {
      options.cpus = cnrl::parse_u32_list(parser.value(argument), argument, true);
    } else if (argument == "--rows") {
      options.rows = cnrl::parse_u32_list(parser.value(argument), argument);
    } else if (argument == "--rates") {
      options.rates = cnrl::parse_double_list(parser.value(argument), argument);
    } else if (argument == "--warmup") {
      options.warmup = cnrl::parse_u32(parser.value(argument), argument, true);
    } else if (argument == "--repetitions") {
      options.repetitions = cnrl::parse_u32(parser.value(argument), argument);
    } else if (argument == "--seed") {
      options.seed = cnrl::parse_u32(parser.value(argument), argument, true);
    } else if (argument == "--projection-shift") {
      options.transition.projection_shift =
          cnrl::parse_u32(parser.value(argument), argument, true);
    } else if (argument == "--state-mult") {
      options.transition.state_multiplier =
          cnrl::parse_i32(parser.value(argument), argument);
    } else if (argument == "--output-mult") {
      options.transition.output_multiplier =
          cnrl::parse_i32(parser.value(argument), argument);
    } else if (argument == "--final-shift") {
      options.transition.final_shift =
          cnrl::parse_u32(parser.value(argument), argument, true);
    } else if (argument == "--target-rms") {
      options.transition.target_rms =
          cnrl::parse_double(parser.value(argument), argument);
    } else if (argument == "--epsilon") {
      options.transition.epsilon =
          cnrl::parse_double(parser.value(argument), argument);
    } else if (argument == "--allow-affinity-failure") {
      options.require_affinity = false;
    } else if (argument == "--allow-smt-siblings") {
      options.allow_smt_siblings = true;
    } else {
      throw std::runtime_error("unknown option: " + std::string(argument));
    }
  }
  if (options.transition.kind == cnrl::TransitionKind::frozen) {
    throw std::runtime_error("transition microbenchmark requires a real transition");
  }
  return options;
}

std::string join_u32(const std::vector<std::uint32_t>& values) {
  std::string result;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) result.push_back(';');
    result += std::to_string(values[index]);
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    Options options = parse(argc, argv);
    if (!cnrl::detect_cpu_features().avx2) {
      throw std::runtime_error("AVX2 is required by the transition benchmark");
    }
    cnrl::validate_transition_config(options.shape, options.transition);
    const auto topology = cnrl::discover_cpu_topology();
    if (options.cpus.empty()) {
      options.cpus = cnrl::choose_one_logical_per_physical_core(topology);
      if (options.cpus.size() > 4U) options.cpus.resize(4U);
    }
    if (options.cpus.empty()) throw std::runtime_error("no CPU was selected");
    if (!options.rows.empty() && options.rows.size() != options.cpus.size()) {
      throw std::runtime_error("--rows count must equal --cpus count");
    }
    if (!options.rates.empty() && options.rates.size() != options.cpus.size()) {
      throw std::runtime_error("--rates count must equal --cpus count");
    }
    if (options.rows.empty()) {
      const auto rates = options.rates.empty()
          ? std::vector<double>(options.cpus.size(), 1.0) : options.rates;
      options.rows = cnrl::proportional_rows(options.shape.dimension, rates, 1);
    }
    const auto shards = cnrl::make_shards(options.cpus, options.rows);
    cnrl::validate_shards(options.shape, shards, true);

    std::vector<cnrl::LogicalProcessor> processors;
    std::vector<std::uint32_t> physical_cores;
    processors.reserve(shards.size());
    physical_cores.reserve(shards.size());
    for (const auto& shard : shards) {
      processors.push_back(cnrl::find_logical_processor(topology, shard.logical_cpu));
      physical_cores.push_back(
          cnrl::find_physical_core_index(topology, shard.logical_cpu));
    }
    if (!options.allow_smt_siblings) {
      auto sorted = physical_cores;
      std::sort(sorted.begin(), sorted.end());
      if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
        throw std::runtime_error("selected CPUs contain SMT siblings");
      }
    }

    const std::size_t cells = static_cast<std::size_t>(options.shape.slots) *
                              options.shape.dimension;
    cnrl::AlignedBuffer<std::int8_t> initial(cells), state_a(cells), state_b(cells);
    cnrl::AlignedBuffer<std::int32_t> output(cells);
    cnrl::XorShift32 random(options.seed);
    for (std::size_t index = 0; index < cells; ++index) {
      initial[index] = random.symmetric_i8(31);
      output[index] = static_cast<std::int32_t>(random.symmetric_i8(100)) * 512;
    }
    cnrl::TransitionWorkspace workspace;
    cnrl::prepare_transition_workspace(
        workspace, static_cast<std::uint32_t>(shards.size()), options.shape);
    cnrl::SpinBarrier barrier(static_cast<std::uint32_t>(shards.size()));
    cnrl::MonotonicClock clock;
    std::atomic<bool> go{false};
    std::atomic<std::uint32_t> ready{0};
    std::atomic<std::uint64_t> begin{0};
    std::atomic<std::uint64_t> end{0};
    std::vector<cnrl::PerWorkerMetrics> metrics(shards.size());
    std::vector<cnrl::TransitionStats> stats(shards.size());
    std::vector<std::thread> threads;
    threads.reserve(shards.size());

    for (std::size_t worker = 0; worker < shards.size(); ++worker) {
      threads.emplace_back([&, worker] {
        cnrl::AffinityGuard affinity(processors[worker]);
        cnrl::PerWorkerMetrics local_metrics;
        local_metrics.logical_cpu = shards[worker].logical_cpu;
        local_metrics.physical_core_index = physical_cores[worker];
        local_metrics.affinity_succeeded = affinity.succeeded();
        local_metrics.affinity_error = affinity.error();
        cnrl::TransitionStats local_stats;
        ready.fetch_add(1, std::memory_order_release);
        while (!go.load(std::memory_order_acquire)) {
          _mm_pause();
        }

        auto* current = state_a.data();
        auto* next = state_b.data();
        auto reset_local_state = [&] {
          for (std::uint32_t slot = 0; slot < options.shape.slots; ++slot) {
            const std::size_t base = static_cast<std::size_t>(slot) *
                                     options.shape.dimension +
                                     shards[worker].row_offset;
            std::copy_n(initial.data() + base, shards[worker].rows,
                        state_a.data() + base);
            std::fill_n(state_b.data() + base, shards[worker].rows,
                        std::int8_t{0});
          }
        };
        auto one_transition = [&] {
          switch (options.transition.kind) {
            case cnrl::TransitionKind::fixed_point:
              cnrl::transition_fixed_point_local(
                  current, output.data(), next, options.shape, shards[worker],
                  options.transition, local_stats);
              barrier.arrive_and_wait();
              break;
            case cnrl::TransitionKind::group_rms:
              cnrl::transition_group_rms_local(
                  current, output.data(), next, options.shape, shards[worker],
                  options.transition, workspace, local_stats);
              barrier.arrive_and_wait();
              break;
            case cnrl::TransitionKind::global_rms:
              cnrl::transition_global_rms_prepare(
                  current, output.data(), options.shape, shards[worker],
                  options.transition, workspace);
              barrier.arrive_and_wait();
              if (worker == 0U) {
                cnrl::transition_global_rms_reduce(
                    static_cast<std::uint32_t>(shards.size()), options.shape,
                    options.transition, workspace);
              }
              barrier.arrive_and_wait();
              cnrl::transition_global_rms_apply(
                  next, options.shape, shards[worker], options.transition,
                  workspace, local_stats);
              barrier.arrive_and_wait();
              break;
            case cnrl::TransitionKind::frozen:
              break;
          }
          std::swap(current, next);
        };

        reset_local_state();
        barrier.arrive_and_wait();
        for (std::uint32_t iteration = 0; iteration < options.warmup; ++iteration) {
          one_transition();
        }
        reset_local_state();
        current = state_a.data();
        next = state_b.data();
        local_stats = {};
        barrier.arrive_and_wait();
        if (worker == 0U) begin.store(clock.now(), std::memory_order_relaxed);
        barrier.arrive_and_wait();
        for (std::uint32_t iteration = 0; iteration < options.repetitions; ++iteration) {
          one_transition();
        }
        barrier.arrive_and_wait();
        if (worker == 0U) end.store(clock.now(), std::memory_order_relaxed);
        barrier.arrive_and_wait();
        metrics[worker] = local_metrics;
        stats[worker] = local_stats;
      });
    }

    while (ready.load(std::memory_order_acquire) !=
           static_cast<std::uint32_t>(shards.size())) {
      std::this_thread::yield();
    }
    go.store(true, std::memory_order_release);
    for (auto& thread : threads) thread.join();

    bool all_affinity = true;
    std::uint32_t first_affinity_error = 0;
    std::uint64_t clipped = 0;
    std::uint64_t updated_cells = 0;
    for (std::size_t worker = 0; worker < metrics.size(); ++worker) {
      all_affinity = all_affinity && metrics[worker].affinity_succeeded;
      if (first_affinity_error == 0U && metrics[worker].affinity_error != 0U) {
        first_affinity_error = metrics[worker].affinity_error;
      }
      clipped += stats[worker].clipped_cells;
      updated_cells += stats[worker].cells;
    }
    const double seconds = clock.seconds_between(
        begin.load(std::memory_order_relaxed), end.load(std::memory_order_relaxed));
    const std::uint64_t cell_updates = static_cast<std::uint64_t>(cells) *
                                       options.repetitions;
    const bool final_in_b = options.repetitions % 2U != 0U;
    const auto checksum = cnrl::checksum_i8(
        final_in_b ? state_b.data() : state_a.data(), cells);
    const bool valid = seconds > 0.0 && updated_cells == cell_updates &&
                       (!options.require_affinity || all_affinity);

    std::cout
        << "D,S,transition,warmup,repetitions,worker_count,cpus,physical_cores,rows,"
           "projection_shift,state_multiplier,output_multiplier,final_shift,target_rms,"
           "elapsed_seconds,cell_updates,updated_cells,cell_updates_per_second,ns_per_cell,clipped_cells,"
           "all_affinity_succeeded,affinity_error,state_checksum,valid,error\n";
    std::cout << options.shape.dimension << ',' << options.shape.slots << ','
              << cnrl::to_string(options.transition.kind) << ',' << options.warmup << ','
              << options.repetitions << ',' << shards.size() << ",\""
              << join_u32(options.cpus) << "\",\"" << join_u32(physical_cores)
              << "\",\"" << join_u32(options.rows) << "\","
              << options.transition.projection_shift << ','
              << options.transition.state_multiplier << ','
              << options.transition.output_multiplier << ','
              << options.transition.final_shift << ','
              << options.transition.target_rms << ',' << seconds << ','
              << cell_updates << ',' << updated_cells << ','
              << static_cast<double>(cell_updates) / seconds << ','
              << seconds * 1.0e9 / static_cast<double>(cell_updates) << ','
              << clipped << ',' << (all_affinity ? "true" : "false") << ','
              << first_affinity_error << ',' << checksum << ','
              << (valid ? "true" : "false") << ",\""
              << (valid ? "" : "timing, accounting, or affinity failure") << "\"\n";
    return valid ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 2;
  }
}

#include "cnrl/benchmark.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>
#include <immintrin.h>

#include "cnrl/aligned_buffer.hpp"
#include "cnrl/checked_math.hpp"
#include "cnrl/hash.hpp"
#include "cnrl/kernels.hpp"
#include "cnrl/platform.hpp"
#include "cnrl/random.hpp"
#include "cnrl/sharding.hpp"
#include "cnrl/state.hpp"
#include "cnrl/spin_barrier.hpp"
#include "cnrl/transitions.hpp"
#include "cnrl/weights.hpp"

namespace cnrl {
namespace {

std::uint64_t calculate_mac_total(const RunConfig& config) {
  std::uint64_t rows = total_output_rows(config.shards);
  std::uint64_t total = checked_mul_u64(rows, config.shape.dimension);
  total = checked_mul_u64(total, config.shape.slots);
  total = checked_mul_u64(total, config.shape.depth);
  total = checked_mul_u64(total, config.sequences_per_repetition);
  total = checked_mul_u64(total, config.timed_repetitions);
  return total;
}


}  // namespace

void validate_run_config(const RunConfig& config) {
  if (config.shape.dimension == 0 || config.shape.slots == 0 || config.shape.depth == 0) {
    throw std::invalid_argument("D, S, and R must be positive");
  }
  if (config.shape.slots > 32) throw std::invalid_argument("S must be <=32");
  if (config.slot_tile != 1 && config.slot_tile != 2 &&
      config.slot_tile != 4 && config.slot_tile != 8) {
    throw std::invalid_argument("slot tile must be 1,2,4,8");
  }
  if (config.timed_repetitions == 0 || config.sequences_per_repetition == 0) {
    throw std::invalid_argument("timed repetitions and sequences must be positive");
  }
  switch (config.gate) {
    case GateKind::t0r:
      if (config.shape.slots != 1U || config.transition.kind != TransitionKind::frozen) {
        throw std::invalid_argument("T0-R requires S=1 and a frozen state");
      }
      break;
    case GateKind::t0m:
      if (config.transition.kind != TransitionKind::frozen) {
        throw std::invalid_argument("T0-M requires a frozen state");
      }
      break;
    case GateKind::t0rm:
      if (config.transition.kind == TransitionKind::frozen) {
        throw std::invalid_argument("T0-RM requires a real transition");
      }
      break;
    case GateKind::calibrate:
      if (config.shape.slots != 1U || config.transition.kind != TransitionKind::frozen) {
        throw std::invalid_argument("calibration requires S=1 and a frozen state");
      }
      break;
  }
  const bool recurrent = config.transition.kind != TransitionKind::frozen;
  validate_shards(config.shape, config.shards, recurrent);
  if (recurrent) validate_transition_config(config.shape, config.transition);
  const CpuFeatures features = detect_cpu_features();
  if (config.kernel != KernelKind::scalar && !features.avx2) {
    throw std::invalid_argument("AVX2 kernel requested on a CPU without AVX2");
  }
  if (config.variant == WeightVariant::cold && config.timing_scope != TimingScope::round_window) {
    throw std::invalid_argument("cold control must use round-window timing so clflush is outside the timer");
  }
}

RunResult run_benchmark(const RunConfig& config) {
  RunResult result;
  try {
    validate_run_config(config);
    const bool recurrent = config.transition.kind != TransitionKind::frozen;
    const std::uint32_t output_rows = total_output_rows(config.shards);
    const std::uint32_t worker_count = static_cast<std::uint32_t>(config.shards.size());
    const CpuTopology topology = discover_cpu_topology();
    std::vector<LogicalProcessor> worker_processors;
    std::vector<std::uint32_t> worker_physical_cores;
    worker_processors.reserve(worker_count);
    worker_physical_cores.reserve(worker_count);
    for (const auto& shard : config.shards) {
      worker_processors.push_back(find_logical_processor(topology, shard.logical_cpu));
      worker_physical_cores.push_back(find_physical_core_index(topology, shard.logical_cpu));
    }
    if (!config.allow_smt_siblings) {
      auto sorted_cores = worker_physical_cores;
      std::sort(sorted_cores.begin(), sorted_cores.end());
      if (std::adjacent_find(sorted_cores.begin(), sorted_cores.end()) != sorted_cores.end()) {
        throw std::invalid_argument(
            "selected logical CPUs include SMT siblings; pass --allow-smt-siblings only for an explicit SMT experiment");
      }
    }
    const WeightBank bank = make_weight_bank(config);

    AlignedBuffer<std::int8_t> initial_state;
    AlignedBuffer<std::int8_t> state_a;
    AlignedBuffer<std::int8_t> state_b;
    initialize_state(initial_state, config.shape, config.seed);
    state_a.resize(initial_state.size());
    state_b.resize(initial_state.size());
    std::memcpy(state_a.data(), initial_state.data(), initial_state.bytes());
    state_b.fill_zero();
    AlignedBuffer<std::int32_t> output(static_cast<std::size_t>(config.shape.slots) * output_rows);
    output.fill_zero();
    TransitionWorkspace transition_workspace;
    prepare_transition_workspace(transition_workspace, worker_count, config.shape);
    for (const auto& shard : config.shards) {
      KernelCall validation_call;
      validation_call.weights = bank.shards[shard.worker_index].block(0);
      validation_call.state = state_a.data();
      validation_call.output = output.data();
      validation_call.rows = shard.rows;
      validation_call.dimension = config.shape.dimension;
      validation_call.slots = config.shape.slots;
      validation_call.row_offset = shard.row_offset;
      validation_call.output_stride = output_rows;
      validation_call.slot_tile = config.slot_tile;
      validate_kernel_call(validation_call);
    }

    SpinBarrier barrier(worker_count);
    MonotonicClock clock;
    std::atomic<std::uint32_t> ready{0};
    std::atomic<bool> go{false};
    std::vector<PerWorkerMetrics> worker_metrics(worker_count);
    std::vector<TransitionStats> transition_stats(worker_count);
    std::vector<std::uint64_t> sinks(worker_count, 0);
    std::atomic<std::uint64_t> measured_ticks{0};

    auto wait_barrier = [&](PerWorkerMetrics& metrics, bool record) {
      if (!record) {
        barrier.arrive_and_wait();
        return;
      }
      const auto begin = clock.now();
      barrier.arrive_and_wait();
      metrics.synchronization_seconds += clock.seconds_between(begin, clock.now());
    };

    std::vector<std::thread> threads;
    threads.reserve(worker_count);
    for (std::uint32_t worker = 0; worker < worker_count; ++worker) {
      threads.emplace_back([&, worker] {
        const ShardSpec& shard = config.shards[worker];
        const auto& logical = worker_processors[worker];
        AffinityGuard affinity(logical);
        PerWorkerMetrics metrics{};
        TransitionStats local_transition_stats{};
        std::uint64_t local_sink = 0;
        metrics.logical_cpu = shard.logical_cpu;
        metrics.physical_core_index = worker_physical_cores[worker];
        metrics.affinity_succeeded = affinity.succeeded();
        metrics.affinity_error = affinity.error();
        ready.fetch_add(1, std::memory_order_release);
        while (!go.load(std::memory_order_acquire)) _mm_pause();

        std::uint64_t local_measured_ticks = 0;
        const std::uint32_t total_reps = config.warmup_repetitions + config.timed_repetitions;
        for (std::uint32_t repetition = 0; repetition < total_reps; ++repetition) {
          const bool timed = repetition >= config.warmup_repetitions;
          if (repetition == config.warmup_repetitions) {
            local_transition_stats = TransitionStats{};
            local_sink = 0;
            metrics.compute_seconds = 0.0;
            metrics.transition_seconds = 0.0;
            metrics.synchronization_seconds = 0.0;
            metrics.cold_prepare_seconds = 0.0;
            wait_barrier(metrics, false);
          }

          for (std::uint32_t sequence = 0; sequence < config.sequences_per_repetition; ++sequence) {
            std::int8_t* current = state_a.data();
            std::int8_t* next = state_b.data();
            if (recurrent) {
              copy_state_shard(initial_state.data(), state_a.data(), config.shape, shard);
              clear_state_shard(state_b.data(), config.shape, shard);
            }
            wait_barrier(metrics, false);
            std::uint64_t sequence_begin = 0;
            if (timed && config.timing_scope == TimingScope::full_repetition) {
              wait_barrier(metrics, false);
              if (worker == 0) sequence_begin = clock.now();
              wait_barrier(metrics, false);
            }

            for (std::uint32_t round = 0; round < config.shape.depth; ++round) {
              const ShardWeights& weights = bank.shards[worker];
              if (config.variant == WeightVariant::cold) {
                const auto cold_begin = timed && config.phase_profile ? clock.now() : 0;
                flush_cache_range(weights.block(round), static_cast<std::size_t>(weights.block_bytes));
                if (timed && config.phase_profile) {
                  metrics.cold_prepare_seconds += clock.seconds_between(cold_begin, clock.now());
                }
                wait_barrier(metrics, false);
              }

              std::uint64_t round_begin = 0;
              if (timed && config.timing_scope == TimingScope::round_window) {
                wait_barrier(metrics, false);
                if (worker == 0) round_begin = clock.now();
                wait_barrier(metrics, false);
              }

              KernelCall call;
              call.weights = weights.block(round);
              call.state = current;
              call.output = output.data();
              call.rows = shard.rows;
              call.dimension = config.shape.dimension;
              call.slots = config.shape.slots;
              call.row_offset = shard.row_offset;
              call.output_stride = output_rows;
              call.slot_tile = config.slot_tile;
              const auto compute_begin = timed && config.phase_profile ? clock.now() : 0;
              run_kernel_unchecked(config.kernel, call);
              if (timed && config.phase_profile) {
                metrics.compute_seconds += clock.seconds_between(compute_begin, clock.now());
              }

              if (!recurrent) {
                const std::uint32_t slot = worker % config.shape.slots;
                const std::size_t cell = static_cast<std::size_t>(slot) * output_rows + shard.row_offset;
                local_sink = hash_combine(local_sink,
                    static_cast<std::uint64_t>(static_cast<std::uint32_t>(output[cell])) ^ round);
                wait_barrier(metrics, timed && config.phase_profile);
              } else {
                switch (config.transition.kind) {
                  case TransitionKind::fixed_point: {
                    const auto begin = timed && config.phase_profile ? clock.now() : 0;
                    transition_fixed_point_local(current, output.data(), next, config.shape,
                                                 shard, config.transition, local_transition_stats);
                    if (timed && config.phase_profile) metrics.transition_seconds += clock.seconds_between(begin, clock.now());
                    wait_barrier(metrics, timed && config.phase_profile);
                    break;
                  }
                  case TransitionKind::group_rms: {
                    const auto begin = timed && config.phase_profile ? clock.now() : 0;
                    transition_group_rms_local(current, output.data(), next, config.shape,
                                               shard, config.transition, transition_workspace,
                                               local_transition_stats);
                    if (timed && config.phase_profile) metrics.transition_seconds += clock.seconds_between(begin, clock.now());
                    wait_barrier(metrics, timed && config.phase_profile);
                    break;
                  }
                  case TransitionKind::global_rms: {
                    auto begin = timed && config.phase_profile ? clock.now() : 0;
                    transition_global_rms_prepare(current, output.data(), config.shape, shard,
                                                  config.transition, transition_workspace);
                    if (timed && config.phase_profile) metrics.transition_seconds += clock.seconds_between(begin, clock.now());
                    wait_barrier(metrics, timed && config.phase_profile);
                    if (worker == 0) {
                      begin = timed && config.phase_profile ? clock.now() : 0;
                      transition_global_rms_reduce(worker_count, config.shape, config.transition,
                                                   transition_workspace);
                      if (timed && config.phase_profile) metrics.transition_seconds += clock.seconds_between(begin, clock.now());
                    }
                    wait_barrier(metrics, timed && config.phase_profile);
                    begin = timed && config.phase_profile ? clock.now() : 0;
                    transition_global_rms_apply(next, config.shape, shard, config.transition,
                                                transition_workspace, local_transition_stats);
                    if (timed && config.phase_profile) metrics.transition_seconds += clock.seconds_between(begin, clock.now());
                    wait_barrier(metrics, timed && config.phase_profile);
                    break;
                  }
                  case TransitionKind::frozen:
                    break;
                }
                std::swap(current, next);
                const std::size_t cell = static_cast<std::size_t>(worker % config.shape.slots) *
                                         config.shape.dimension + shard.row_offset;
                local_sink = hash_combine(local_sink,
                    static_cast<std::uint64_t>(static_cast<std::uint8_t>(current[cell])) ^ round);
              }

              if (timed && config.timing_scope == TimingScope::round_window) {
                if (worker == 0) local_measured_ticks += clock.now() - round_begin;
                wait_barrier(metrics, false);
              }
            }
            if (timed && config.timing_scope == TimingScope::full_repetition) {
              wait_barrier(metrics, false);
              if (worker == 0) local_measured_ticks += clock.now() - sequence_begin;
              wait_barrier(metrics, false);
            }
          }
        }
        metrics.local_sink = local_sink;
        worker_metrics[worker] = metrics;
        transition_stats[worker] = local_transition_stats;
        sinks[worker] = local_sink;
        if (worker == 0) measured_ticks.store(local_measured_ticks, std::memory_order_release);
      });
    }

    while (ready.load(std::memory_order_acquire) != worker_count) _mm_pause();
    go.store(true, std::memory_order_release);
    for (auto& thread : threads) thread.join();

    result.elapsed_seconds = clock.seconds_between(0, measured_ticks.load(std::memory_order_acquire));
    result.mac_total = calculate_mac_total(config);
    result.mac_per_second = result.elapsed_seconds > 0.0
        ? static_cast<double>(result.mac_total) / result.elapsed_seconds : 0.0;
    result.base_weight_bytes = bank.base_weight_bytes;
    result.allocated_weight_bytes = bank.allocated_weight_bytes;
    result.distinct_weight_storage_bytes = bank.allocated_weight_bytes;
    result.logical_weight_load_bytes = calculate_logical_weight_load_bytes(config);
    result.one_pass_weight_bytes = checked_mul_u64(
        checked_mul_u64(checked_mul_u64(bank.base_weight_bytes, config.shape.depth),
                        config.sequences_per_repetition),
        config.timed_repetitions);
    result.logical_weight_load_gb_per_second = result.elapsed_seconds > 0.0
        ? static_cast<double>(result.logical_weight_load_bytes) / result.elapsed_seconds / 1.0e9 : 0.0;
    result.one_pass_weight_gb_per_second = result.elapsed_seconds > 0.0
        ? static_cast<double>(result.one_pass_weight_bytes) / result.elapsed_seconds / 1.0e9 : 0.0;
    result.output_checksum = checksum_i32(output.data(), output.size());
    const std::int8_t* final_state = recurrent && (config.shape.depth % 2U != 0U)
        ? state_b.data() : state_a.data();
    result.state_checksum = checksum_i8(final_state, initial_state.size());
    result.weight_hash_signature = bank.hash_signature;
    result.clone_hashes_equal = bank.clone_hashes_equal;
    result.clone_addresses_distinct = bank.clone_addresses_distinct;
    result.workers = worker_metrics;
    result.all_affinity_succeeded = true;
    for (std::uint32_t worker = 0; worker < worker_count; ++worker) {
      result.all_affinity_succeeded = result.all_affinity_succeeded && worker_metrics[worker].affinity_succeeded;
      result.round_sink = hash_combine(result.round_sink, sinks[worker]);
      result.clipped_cells += transition_stats[worker].clipped_cells;
      result.transition_cells += transition_stats[worker].cells;
    }
    if (config.require_affinity && !result.all_affinity_succeeded) {
      throw std::runtime_error("one or more worker affinity operations failed");
    }
    result.valid = true;
  } catch (const std::exception& error) {
    result.valid = false;
    result.error = error.what();
  }
  return result;
}

}  // namespace cnrl

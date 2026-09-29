#include "v2_1.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace omega_v2_1 {
namespace {

constexpr int kWarmups = 10;
constexpr int kBlocks = 5;
constexpr int kSamplesPerBlock = 21;
constexpr std::uint32_t kScheduleSeed = 20260929;

std::size_t target_b_pool_bytes(const HardwareInfo& hardware) {
    const double by_llc = std::ceil(2.5 * static_cast<double>(hardware.llc_bytes));
    return (std::max)(static_cast<std::size_t>(by_llc), static_cast<std::size_t>(64ull * 1024ull * 1024ull));
}

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

struct UntiedPool {
    int d = 0;
    int K = 0;
    std::size_t groups = 0;
    std::size_t bytes_per_group = 0;
    std::size_t total_bytes = 0;
    bool values_equal = false;
    bool storages_disjoint = false;
    std::size_t unique_storage_count = 0;
    std::vector<std::vector<CoreWeights>> blocks;
};

bool same_q4_values(const CoreWeights& left, const CoreWeights& right) {
    const Q4Matrix* a[] = {&left.W_Q, &left.W_K, &left.W_V, &left.W_O, &left.W_gate, &left.W_up, &left.W_down};
    const Q4Matrix* b[] = {&right.W_Q, &right.W_K, &right.W_V, &right.W_O, &right.W_gate, &right.W_up, &right.W_down};
    for (std::size_t i = 0; i < 7; ++i) {
        if (a[i]->rows != b[i]->rows || a[i]->cols != b[i]->cols) return false;
        if (std::memcmp(a[i]->packed.data, b[i]->packed.data, a[i]->packed.logical_bytes) != 0) return false;
        if (std::memcmp(a[i]->scales_fp16.data, b[i]->scales_fp16.data, a[i]->scales_fp16.logical_bytes) != 0) return false;
    }
    return true;
}

std::vector<const void*> q4_buffer_addresses(const CoreWeights& weights) {
    const Q4Matrix* matrices[] = {&weights.W_Q, &weights.W_K, &weights.W_V, &weights.W_O, &weights.W_gate, &weights.W_up, &weights.W_down};
    std::vector<const void*> result;
    result.reserve(14);
    for (const Q4Matrix* matrix : matrices) {
        result.push_back(matrix->packed.data);
        result.push_back(matrix->scales_fp16.data);
    }
    return result;
}

std::array<const Q4Matrix*, 7> matrices_of(const CoreWeights& weights) {
    return {&weights.W_Q, &weights.W_K, &weights.W_V, &weights.W_O, &weights.W_gate, &weights.W_up, &weights.W_down};
}

std::string ids_string(const std::vector<CoreRecord>& workers, bool cpu_set_ids) {
    std::ostringstream out;
    for (std::size_t i = 0; i < workers.size(); ++i) {
        if (i) out << ';';
        if (workers[i].cpu_sets.empty()) continue;
        if (cpu_set_ids) out << workers[i].cpu_sets.front().id;
        else out << static_cast<unsigned>(workers[i].group) << ':' << static_cast<unsigned>(workers[i].core_index);
    }
    return out.str();
}

std::uint64_t macs_per_round(int d, int m) {
    return 16ull * static_cast<std::uint64_t>(m) * d * d + 2ull * static_cast<std::uint64_t>(m) * m * d;
}

void initialize_state(Scratch& scratch, int d, int m, int cell_index) {
    scratch.resize_for(d, m);
    for (std::size_t i = 0; i < scratch.state.size(); ++i) {
        const auto value = static_cast<int>((i * 37 + static_cast<std::size_t>(cell_index * 11)) % 127) - 63;
        scratch.state[i] = static_cast<float>(value) * (1.0f / 256.0f);
    }
}

bool finite_state(const Scratch& scratch) {
    for (float value : scratch.state) if (!std::isfinite(value)) return false;
    return true;
}

void write_sample(std::ofstream& csv,
                  const HardwareInfo& hardware,
                  const std::vector<CoreRecord>& workers,
                  const std::string& run_id,
                  int block_id,
                  int sample_id,
                  bool warmup,
                  int d,
                  int m,
                  int K,
                  char variant,
                  int b_group_id,
                  const std::string& eviction_method,
                  int round_index,
                  std::int64_t ticks,
                  double seconds,
                   std::uint64_t macs,
                   std::uint64_t checksum,
                   std::uint64_t timed_heap_allocation_count,
                   bool valid,
                  const std::string& invalid_reason) {
    csv << csv_escape(run_id) << ',' << block_id << ',' << sample_id << ',' << (warmup ? "true" : "false") << ','
        << csv_escape(iso8601_now()) << ',' << csv_escape(hardware.cpu_model) << ','
        << csv_escape(ids_string(workers, true)) << ',' << csv_escape(ids_string(workers, false)) << ','
        << d << ',' << m << ',' << K << ',' << variant << ',' << b_group_id << ',' << csv_escape(eviction_method) << ','
        << round_index << ',' << ticks << ',' << std::setprecision(17) << seconds << ',' << ','
        << macs << ',' << (2ull * macs) << ',' << checksum << ',' << timed_heap_allocation_count << ',' << ',' << ','
        << (valid ? "true" : "false") << ',' << csv_escape(invalid_reason) << '\n';
}

} // namespace

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') escaped += "\"\"";
        else escaped += ch;
    }
    escaped += '"';
    return escaped;
}

std::string iso8601_now() {
    SYSTEMTIME time{};
    GetSystemTime(&time);
    char buffer[40]{};
    sprintf_s(buffer, "%04u-%02u-%02uT%02u:%02u:%02u.%03uZ",
              time.wYear, time.wMonth, time.wDay, time.wHour, time.wMinute,
              time.wSecond, time.wMilliseconds);
    return buffer;
}

std::uint64_t checksum_floats(const float* data, std::size_t count) {
    std::uint64_t hash = 1469598103934665603ull;
    for (std::size_t i = 0; i < count; ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, data + i, sizeof(bits));
        for (int byte = 0; byte < 4; ++byte) {
            hash ^= static_cast<std::uint8_t>(bits >> (8 * byte));
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

bool run_full_sweep(const HardwareInfo& hardware,
                    const std::vector<CoreRecord>& selected_workers,
                    const std::string& eviction_method,
                    const std::filesystem::path& raw_csv,
                    const std::filesystem::path& q4_ledger_path,
                    std::string& error) {
    if (selected_workers.size() != 4) {
        error = "primary sweep requires exactly four H0-selected physical P-cores";
        return false;
    }
    if (eviction_method != "CLFLUSH" && eviction_method != "SWEEP_BUFFER") {
        error = "unknown C eviction method";
        return false;
    }
    if ((eviction_method == "CLFLUSH") != hardware.clflush_supported) {
        error = "preflight eviction method and measured CPUID CLFLUSH support differ";
        return false;
    }
    std::vector<CpuSetRecord> cpu_sets;
    cpu_sets.reserve(selected_workers.size());
    for (const CoreRecord& core : selected_workers) {
        if (core.cpu_sets.size() != 1 || core.cpu_sets.front().shard_weight <= 0.0) {
            error = "selected P-core worker lacks exactly one fixed CPU set or positive H0 v_i shard weight";
            return false;
        }
        cpu_sets.push_back(core.cpu_sets.front());
    }
    std::unique_ptr<WorkerPool> pool;
    try {
        pool = std::make_unique<WorkerPool>(cpu_sets);
    } catch (const std::exception& exception) {
        error = std::string("persistent worker-pool startup failed: ") + exception.what();
        return false;
    }
    if (!pool->ready() || !pool->affinity_intact()) {
        error = "persistent worker pool did not preserve fixed physical-core affinity";
        return false;
    }

    std::vector<CoreWeights> base_weights;
    base_weights.reserve(2);
    for (int d : {kD512, kD640}) base_weights.push_back(make_seeded_weights(d, kSeed));
    std::string q4_json = q4_ledger_json(base_weights);
    const std::size_t target_pool = target_b_pool_bytes(hardware);
    const std::size_t target_buffer = target_pool;

    std::vector<UntiedPool> b_pools;
    std::unordered_set<const void*> seen_storage;
    for (const CoreWeights& weights : base_weights) {
        for (const void* address : q4_buffer_addresses(weights)) seen_storage.insert(address);
        for (int K : {1, 4, 8}) {
            UntiedPool pool_spec;
            pool_spec.d = weights.d;
            pool_spec.K = K;
            pool_spec.bytes_per_group = static_cast<std::size_t>(K) * weights.physical_buffer_bytes;
            pool_spec.groups = ceil_div(target_pool, pool_spec.bytes_per_group);
            pool_spec.total_bytes = pool_spec.groups * pool_spec.bytes_per_group;
            pool_spec.values_equal = true;
            pool_spec.storages_disjoint = true;
            pool_spec.blocks.reserve(pool_spec.groups);
            try {
                for (std::size_t group = 0; group < pool_spec.groups; ++group) {
                    std::vector<CoreWeights> round_blocks;
                    round_blocks.reserve(K);
                    for (int round = 0; round < K; ++round) {
                        CoreWeights clone = clone_weights(weights);
                        pool_spec.values_equal = pool_spec.values_equal && same_q4_values(weights, clone);
                        for (const void* address : q4_buffer_addresses(clone)) {
                            if (seen_storage.find(address) != seen_storage.end()) pool_spec.storages_disjoint = false;
                            seen_storage.insert(address);
                            ++pool_spec.unique_storage_count;
                        }
                        round_blocks.push_back(std::move(clone));
                    }
                    pool_spec.blocks.push_back(std::move(round_blocks));
                }
            } catch (const std::bad_alloc&) {
                error = "MEASUREMENT_INVALID_B_POOL_ALLOCATION_FAILED";
                return false;
            }
            if (!pool_spec.values_equal || !pool_spec.storages_disjoint || pool_spec.total_bytes < target_pool
                || static_cast<double>(pool_spec.total_bytes) / hardware.llc_bytes < 2.5) {
                error = "MEASUREMENT_INVALID_B_POOL_INTEGRITY_OR_SIZE";
                return false;
            }
            b_pools.push_back(std::move(pool_spec));
        }
    }

    std::ofstream ledger_out(q4_ledger_path, std::ios::binary | std::ios::trunc);
    if (!ledger_out) {
        error = "cannot create q4_physical_ledger.json";
        return false;
    }
    ledger_out << "{\"schema\":\"omega-v2-1-q4-physical-ledger-v1\",\"group_size\":32,\"scale_dtype\":\"FP16\",\"zero_point\":false,\"alignment_bytes\":64,\"hardware_llc_bytes\":" << hardware.llc_bytes
               << ",\"required_b_pool_bytes\":" << target_pool << ",\"core_families\":";
    ledger_out << q4_json.substr(q4_json.find('['), q4_json.rfind(']') - q4_json.find('[') + 1);
    ledger_out << ",\"b_pool_by_d_k\":[";
    for (std::size_t i = 0; i < b_pools.size(); ++i) {
        if (i) ledger_out << ',';
        const UntiedPool& pool_spec = b_pools[i];
        ledger_out << "{\"d\":" << pool_spec.d << ",\"K\":" << pool_spec.K << ",\"groups\":" << pool_spec.groups
                   << ",\"bytes_per_group\":" << pool_spec.bytes_per_group << ",\"pool_total_bytes\":" << pool_spec.total_bytes
                   << ",\"pool_to_llc_ratio\":" << (static_cast<double>(pool_spec.total_bytes) / hardware.llc_bytes)
                   << ",\"q4_values_bitwise_equal_to_A\":" << (pool_spec.values_equal ? "true" : "false")
                   << ",\"distinct_buffer_storage_count\":" << pool_spec.unique_storage_count
                   << ",\"all_B_storages_disjoint\":" << (pool_spec.storages_disjoint ? "true" : "false")
                   << ",\"passes_2p5_llc_and_64mib\":" << (pool_spec.total_bytes >= target_pool ? "true" : "false") << '}';
    }
    ledger_out << "]}";
    ledger_out.close();

    std::ofstream csv(raw_csv, std::ios::binary | std::ios::trunc);
    if (!csv) {
        error = "cannot create raw_measurements.csv";
        return false;
    }
    csv << "run_id,block_id,sample_id,is_warmup,timestamp,cpu_model,worker_cpu_sets,worker_core_ids,d,m,K,variant,B_group_id,eviction_method,round_index,qpc_ticks,qpc_seconds,rdtscp_delta_if_available,effective_macs,effective_flops,output_checksum,timed_heap_allocation_count,frequency_if_available,temperature_if_available,valid,invalid_reason\n";

    std::mt19937 schedule(kScheduleSeed);
    std::uint64_t b_visit_counter = 0;
    const std::string run_id = "V2-1-" + iso8601_now();
    const std::size_t sweep_buffer_bytes = target_buffer;
    for (const CoreWeights& weights : base_weights) {
        const int d = weights.d;
        for (int K : {1, 4, 8}) {
            const auto pool_it = std::find_if(b_pools.begin(), b_pools.end(), [&](const UntiedPool& item) { return item.d == d && item.K == K; });
            if (pool_it == b_pools.end()) {
                error = "MEASUREMENT_INVALID_B_POOL_MISSING";
                return false;
            }
            const UntiedPool& untied_pool = *pool_it;
            for (int m : {1, 4, 8, 16}) {
                Scratch scratch;
                scratch.resize_for(d, m);
                std::vector<float> initial(static_cast<std::size_t>(m) * d);
                for (std::size_t i = 0; i < initial.size(); ++i) {
                    const int value = static_cast<int>((i * 37 + static_cast<std::size_t>(d + m)) % 127) - 63;
                    initial[i] = static_cast<float>(value) * (1.0f / 256.0f);
                }
                bool measurement_process_failed = false;
                auto run_sample = [&](char variant, int block_id, int sample_id, bool warmup) {
                    std::copy(initial.begin(), initial.end(), scratch.state.begin());
                    int b_group = -1;
                    if (variant == 'A') touch_weights(weights, hardware.cache_line_bytes);
                    if (variant == 'B') {
                        b_group = static_cast<int>(b_visit_counter++ % untied_pool.blocks.size());
                    }
                    for (int round = 0; round < K; ++round) {
                        const CoreWeights* selected = &weights;
                        if (variant == 'B') selected = &untied_pool.blocks[static_cast<std::size_t>(b_group)][static_cast<std::size_t>(round)];
                        std::string invalid_reason;
                        if (variant == 'C' && !evict_weights(weights, hardware.cache_line_bytes, eviction_method, sweep_buffer_bytes, invalid_reason)) {
                            error = "MEASUREMENT_PROCESS_FAILURE: C eviction failed during sweep: " + invalid_reason;
                            write_sample(csv, hardware, selected_workers, run_id, block_id, sample_id, warmup, d, m, K, variant,
                                         b_group, eviction_method, round, 0, 0.0, macs_per_round(d, m), 0, 0, false, error);
                            measurement_process_failed = true;
                            return;
                        }
                        begin_timed_allocation_count();
                        const std::int64_t start = qpc_ticks();
                        v2_full_block_round(*selected, scratch, *pool);
                        const std::int64_t stop = qpc_ticks();
                        const std::uint64_t timed_allocations = end_timed_allocation_count();
                        const bool timer_ok = stop > start;
                        const bool finite_ok = finite_state(scratch);
                        const bool affinity_ok = pool->affinity_intact();
                        // Allocation is a contractual test failure, but it is not one of the
                        // predeclared criteria for discarding an otherwise valid timed sample.
                        const bool valid = timer_ok && finite_ok && affinity_ok;
                        if (!timer_ok) invalid_reason = "QPC_TIMER_ERROR";
                        else if (!finite_ok) invalid_reason = "NONFINITE_OUTPUT";
                        else if (!affinity_ok) invalid_reason = "AFFINITY_LOST";
                        const std::int64_t ticks = timer_ok ? stop - start : 0;
                        const double seconds = timer_ok ? static_cast<double>(ticks) / qpc_frequency() : 0.0;
                        write_sample(csv, hardware, selected_workers, run_id, block_id, sample_id, warmup, d, m, K, variant,
                                      b_group, eviction_method, round, ticks, seconds, macs_per_round(d, m),
                                      checksum_floats(scratch.state.data(), scratch.state.size()), timed_allocations, valid, invalid_reason);
                        if (!affinity_ok) {
                            error = "MEASUREMENT_PROCESS_FAILURE: worker affinity was lost during a timed round";
                            measurement_process_failed = true;
                            return;
                        }
                    }
                };
                std::vector<char> variants{'A', 'B', 'C'};
                for (char variant : variants) {
                    for (int warmup = 0; warmup < kWarmups; ++warmup) {
                        run_sample(variant, -1, -kWarmups + warmup, true);
                        if (measurement_process_failed) { csv.close(); return false; }
                    }
                }
                for (int block = 0; block < kBlocks; ++block) {
                    std::shuffle(variants.begin(), variants.end(), schedule);
                    for (int sample = 0; sample < kSamplesPerBlock; ++sample) {
                        const int sample_id = block * kSamplesPerBlock + sample;
                        for (char variant : variants) {
                            run_sample(variant, block, sample_id, false);
                            if (measurement_process_failed) { csv.close(); return false; }
                        }
                    }
                }
            }
        }
    }
    csv.close();
    return true;
}

} // namespace omega_v2_1

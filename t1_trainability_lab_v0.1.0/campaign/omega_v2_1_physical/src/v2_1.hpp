#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#include <winternl.h>
#include <intrin.h>
#include <immintrin.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace omega_v2_1 {

constexpr int kD512 = 512;
constexpr int kD640 = 640;
constexpr int kGroup = 32;
constexpr std::size_t kAlignment = 64;
constexpr std::uint32_t kSeed = 20260929;

struct CpuSetRecord {
    DWORD id = 0;
    WORD group = 0;
    BYTE logical_index = 0;
    BYTE core_index = 0;
    BYTE last_level_cache_index = 0;
    BYTE numa_node_index = 0;
    BYTE efficiency_class = 0;
    bool parked = false;
    bool allocated = false;
    bool realtime = false;
    double shard_weight = 0.0;
};

struct CoreRecord {
    WORD group = 0;
    BYTE core_index = 0;
    BYTE efficiency_class = 0;
    BYTE intel_core_type = 0;
    std::vector<CpuSetRecord> cpu_sets;
    double v_i = 0.0;
    bool classified_p_core = false;
};

struct CacheRecord {
    BYTE level = 0;
    BYTE type = 0;
    DWORD size_bytes = 0;
    WORD line_bytes = 0;
    BYTE associativity = 0;
    std::vector<std::pair<WORD, KAFFINITY>> group_masks;
};

struct HardwareInfo {
    std::string cpu_model;
    DWORD family = 0;
    DWORD model = 0;
    DWORD stepping = 0;
    std::string windows_version;
    DWORD windows_build = 0;
    DWORD physical_core_count = 0;
    DWORD logical_processor_count = 0;
    DWORD processor_group_count = 0;
    DWORD p_core_count = 0;
    DWORD e_core_count = 0;
    DWORD coordinator_cpu_set_id = MAXDWORD;
    WORD coordinator_group = 0;
    BYTE coordinator_logical_index = 0;
    DWORD cache_line_bytes = 0;
    std::uint64_t l1d_bytes_total = 0;
    std::uint64_t l1i_bytes_total = 0;
    std::uint64_t l2_bytes_total = 0;
    std::uint64_t llc_bytes = 0;
    std::uint64_t ram_bytes = 0;
    double qpc_overhead_ns = 0.0;
    bool avx2 = false;
    bool fma = false;
    bool avx_vnni = false;
    bool avx512f_os_enabled = false;
    bool clflush_supported = false;
    bool qpc_monotonic = false;
    std::int64_t qpc_frequency = 0;
    std::vector<CoreRecord> cores;
    std::vector<CacheRecord> caches;
};

struct AlignedBuffer {
    std::uint8_t* data = nullptr;
    std::size_t logical_bytes = 0;
    std::size_t allocated_bytes = 0;
    AlignedBuffer() = default;
    explicit AlignedBuffer(std::size_t logical);
    ~AlignedBuffer();
    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    void reset(std::size_t logical);
};

struct Q4Matrix {
    int rows = 0;
    int cols = 0;
    AlignedBuffer packed;
    AlignedBuffer scales_fp16;
    std::size_t scale_count = 0;
    std::size_t alignment_padding_bytes = 0;
    void pack(const std::vector<float>& values, int row_count, int column_count);
    std::int8_t value_at(int row, int column) const;
    float scale_at(int row, int column) const;
    float dequant_at(int row, int column) const;
};

struct CoreWeights {
    int d = 0;
    Q4Matrix W_Q, W_K, W_V, W_O, W_gate, W_up, W_down;
    std::size_t logical_weight_bytes = 0;
    std::size_t logical_scale_bytes = 0;
    std::size_t alignment_padding_bytes = 0;
    std::size_t physical_buffer_bytes = 0;
};

CoreWeights make_seeded_weights(int d, std::uint32_t seed);
CoreWeights clone_weights(const CoreWeights& source);

struct Scratch {
    int d = 0;
    int m = 0;
    std::vector<float> state, normalized, query, key, value, scores, attention, projected, hidden;
    std::vector<float> mlp_normalized, gate, up, gated, down, output;
    void resize_for(int dimension, int slots);
};

class WorkerPool {
public:
    WorkerPool();
    explicit WorkerPool(const std::vector<CpuSetRecord>& workers);
    ~WorkerPool();
    WorkerPool(const WorkerPool&) = delete;
    WorkerPool& operator=(const WorkerPool&) = delete;
    bool start(const std::vector<CpuSetRecord>& workers, std::string& error);
    void stop();
    void parallel_for(std::size_t count, void* context, void (*function)(void*, std::size_t));
    const std::vector<DWORD>& selected_cpu_set_ids() const { return cpu_set_ids_; }
    const std::vector<std::string>& affinity_results() const { return affinity_results_; }
    const std::vector<double>& worker_weights() const { return worker_weights_; }
    std::vector<std::pair<std::size_t, std::size_t>> row_shards(std::size_t row_count) const;
    bool ready() const { return ready_.load(); }
    bool affinity_intact() const { return affinity_intact_.load(); }

private:
    struct ThreadState;
    std::vector<std::unique_ptr<ThreadState>> threads_;
    std::vector<DWORD> cpu_set_ids_;
    std::vector<double> worker_weights_;
    std::vector<std::string> affinity_results_;
    std::mutex mutex_;
    std::condition_variable cv_work_;
    std::condition_variable cv_done_;
    std::condition_variable cv_ready_;
    std::vector<std::pair<std::size_t, std::size_t>> row_ranges_;
    std::size_t work_count_ = 0;
    std::size_t completed_ = 0;
    std::size_t startup_completed_ = 0;
    std::uint64_t generation_ = 0;
    bool shutdown_ = false;
    void* job_context_ = nullptr;
    void (*job_function_)(void*, std::size_t) = nullptr;
    std::atomic<bool> ready_{false};
    std::atomic<bool> affinity_intact_{true};
    std::string startup_error_;
    void worker_main(std::size_t worker_index, CpuSetRecord cpu_set);
};

bool query_hardware(HardwareInfo& info, std::string& error);
bool classify_intel_hybrid_cores(HardwareInfo& info, std::string& error);
std::string hardware_json(const HardwareInfo& info);
std::string q4_ledger_json(const std::vector<CoreWeights>& weight_families);
std::string q4_scalar_json_test(WorkerPool& pool);
std::string q4_round_trip_test();
std::string full_block_scalar_reference_test(WorkerPool& pool);
std::string full_block_abc_correctness(WorkerPool& pool, const CoreWeights& weights);
void q4_linear(const Q4Matrix& weights, const float* input, float* output, int token_rows, WorkerPool& pool);
void q4_linear_single_thread(const Q4Matrix& weights, const float* input, float* output, int token_rows);
void v2_full_block_round(const CoreWeights& weights, Scratch& scratch, WorkerPool& pool);
void v2_full_block_round_single_thread(const CoreWeights& weights, Scratch& scratch);

double qpc_seconds();
std::int64_t qpc_ticks();
std::int64_t qpc_frequency();
bool monotonic_qpc_test(std::string& error);
bool set_current_cpu_set(DWORD id, WORD group, BYTE logical_index, std::string& error);
bool probe_eviction(const CoreWeights& weights, const HardwareInfo& hardware, std::string& method, double& effectiveness, std::string& error);
bool evict_weights(const CoreWeights& weights, DWORD cache_line_bytes, const std::string& method, std::size_t sweep_buffer_bytes, std::string& error);
void touch_weights(const CoreWeights& weights, DWORD cache_line_bytes);

struct H0Row {
    WORD group = 0;
    BYTE core_index = 0;
    BYTE efficiency_class = 0;
    BYTE intel_core_type = 0;
    DWORD logical_index = 0;
    DWORD cpu_set_id = 0;
    int m = 0;
    double median_macs_per_second = 0.0;
    std::vector<double> sample_macs_per_second;
    int warmup_count = 0;
    int measured_repetitions = 0;
    bool affinity_ok = false;
};

std::vector<H0Row> measure_h0_cores(const HardwareInfo& hardware, const CoreWeights& d512_weights, std::string& error);
std::vector<CoreRecord> select_p_cores_by_h0(HardwareInfo& hardware, const std::vector<H0Row>& h0, std::string& p_class_method, std::string& error);

struct RawSample {
    std::string run_id;
    int block_id = 0;
    int sample_id = 0;
    int d = 0;
    int m = 0;
    int K = 0;
    char variant = 'A';
    int round_index = 0;
    int b_group_id = -1;
    std::int64_t qpc_ticks = 0;
    double qpc_seconds = 0.0;
    std::uint64_t effective_macs = 0;
    std::uint64_t effective_flops = 0;
    std::uint64_t output_checksum = 0;
    bool valid = true;
    std::string invalid_reason;
};

bool run_full_sweep(const HardwareInfo& hardware,
                    const std::vector<CoreRecord>& selected_workers,
                    const std::string& eviction_method,
                    const std::filesystem::path& raw_csv,
                    const std::filesystem::path& q4_ledger_path,
                    std::string& error);

std::string csv_escape(const std::string& value);
std::string iso8601_now();
std::uint64_t checksum_floats(const float* data, std::size_t count);
void begin_timed_allocation_count();
std::uint64_t end_timed_allocation_count();

} // namespace omega_v2_1

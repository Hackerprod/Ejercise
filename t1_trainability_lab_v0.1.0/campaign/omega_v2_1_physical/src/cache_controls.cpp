#include "v2_1.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

namespace omega_v2_1 {
namespace {

std::vector<std::pair<const std::uint8_t*, std::size_t>> buffers(const CoreWeights& weights) {
    std::vector<std::pair<const std::uint8_t*, std::size_t>> result;
    const Q4Matrix* matrices[] = {&weights.W_Q, &weights.W_K, &weights.W_V, &weights.W_O, &weights.W_gate, &weights.W_up, &weights.W_down};
    result.reserve(14);
    for (const Q4Matrix* matrix : matrices) {
        result.emplace_back(matrix->packed.data, matrix->packed.allocated_bytes);
        result.emplace_back(matrix->scales_fp16.data, matrix->scales_fp16.allocated_bytes);
    }
    return result;
}

double median(std::vector<double> values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    const double upper = values[mid];
    if ((values.size() & 1u) != 0) return upper;
    std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
    return (values[mid - 1] + upper) * 0.5;
}

std::vector<const volatile std::uint8_t*> probe_lines(const CoreWeights& weights, DWORD line_bytes) {
    const auto all_buffers = buffers(weights);
    std::size_t total_lines = 0;
    for (const auto& buffer : all_buffers) total_lines += (buffer.second + line_bytes - 1) / line_bytes;
    const std::size_t desired = (std::min)(total_lines, static_cast<std::size_t>(1024));
    std::vector<const volatile std::uint8_t*> result;
    result.reserve(desired);
    if (desired == 0) return result;
    for (std::size_t target = 0; target < desired; ++target) {
        const std::size_t wanted_line = (target * total_lines) / desired;
        std::size_t prefix = 0;
        for (const auto& buffer : all_buffers) {
            const std::size_t lines = (buffer.second + line_bytes - 1) / line_bytes;
            if (wanted_line < prefix + lines) {
                const std::size_t local = wanted_line - prefix;
                result.push_back(reinterpret_cast<const volatile std::uint8_t*>(buffer.first + local * line_bytes));
                break;
            }
            prefix += lines;
        }
    }
    return result;
}

double time_probe_lines(const std::vector<const volatile std::uint8_t*>& lines) {
    volatile std::uint64_t checksum = 0;
    LARGE_INTEGER start{}, stop{};
    QueryPerformanceCounter(&start);
    for (const volatile std::uint8_t* line : lines) checksum += *line;
    QueryPerformanceCounter(&stop);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    return static_cast<double>(stop.QuadPart - start.QuadPart) / static_cast<double>(qpc_frequency()) / lines.size();
}

} // namespace

std::int64_t qpc_ticks() {
    LARGE_INTEGER value{};
    QueryPerformanceCounter(&value);
    return value.QuadPart;
}

std::int64_t qpc_frequency() {
    LARGE_INTEGER value{};
    QueryPerformanceFrequency(&value);
    return value.QuadPart;
}

double qpc_seconds() { return static_cast<double>(qpc_ticks()) / static_cast<double>(qpc_frequency()); }

bool monotonic_qpc_test(std::string& error) {
    const std::int64_t frequency = qpc_frequency();
    if (frequency <= 0) {
        error = "QueryPerformanceFrequency returned a non-positive frequency";
        return false;
    }
    std::int64_t previous = qpc_ticks();
    for (int i = 0; i < 100000; ++i) {
        const std::int64_t current = qpc_ticks();
        if (current < previous) {
            error = "QueryPerformanceCounter moved backwards";
            return false;
        }
        previous = current;
    }
    return true;
}

void touch_weights(const CoreWeights& weights, DWORD cache_line_bytes) {
    const auto all_buffers = buffers(weights);
    volatile std::uint64_t checksum = 0;
    for (const auto& buffer : all_buffers) {
        for (std::size_t offset = 0; offset < buffer.second; offset += cache_line_bytes) checksum += buffer.first[offset];
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    (void)checksum;
}

bool evict_weights(const CoreWeights& weights, DWORD cache_line_bytes, const std::string& method, std::size_t sweep_buffer_bytes, std::string& error) {
    int registers[4]{};
    __cpuidex(registers, 1, 0);
    const bool clflush = (static_cast<unsigned>(registers[3]) & (1u << 19)) != 0;
    if (method == "CLFLUSH") {
        if (!clflush) {
            error = "selected CLFLUSH method is not supported by CPUID";
            return false;
        }
        const auto all_buffers = buffers(weights);
        for (const auto& buffer : all_buffers) {
            for (std::size_t offset = 0; offset < buffer.second; offset += cache_line_bytes) {
                _mm_clflush(const_cast<std::uint8_t*>(buffer.first + offset));
            }
        }
        _mm_mfence();
        return true;
    }
    if (method != "SWEEP_BUFFER") {
        error = "unknown eviction method";
        return false;
    }
    if (sweep_buffer_bytes < 64ull * 1024ull * 1024ull) {
        error = "SWEEP_BUFFER request below 64 MiB minimum";
        return false;
    }
    static AlignedBuffer sweep;
    if (sweep.allocated_bytes < sweep_buffer_bytes) sweep.reset(sweep_buffer_bytes);
    volatile std::uint64_t checksum = 0;
    for (std::size_t offset = 0; offset < sweep_buffer_bytes; offset += kAlignment) checksum += sweep.data[offset];
    std::atomic_signal_fence(std::memory_order_seq_cst);
    (void)checksum;
    return true;
}

bool probe_eviction(const CoreWeights& weights, const HardwareInfo& hardware, std::string& method, double& effectiveness, std::string& error) {
    const std::size_t target = (std::max)(static_cast<std::size_t>(64ull * 1024ull * 1024ull), static_cast<std::size_t>(2.5 * hardware.llc_bytes));
    method = hardware.clflush_supported ? "CLFLUSH" : "SWEEP_BUFFER";
    const auto lines = probe_lines(weights, hardware.cache_line_bytes);
    if (lines.size() < 128) {
        error = "eviction probe could not select at least 128 cache lines";
        return false;
    }
    std::vector<double> cached, evicted;
    cached.reserve(21);
    evicted.reserve(21);
    for (int sample = 0; sample < 21; ++sample) {
        touch_weights(weights, hardware.cache_line_bytes);
        const double cached_time = time_probe_lines(lines);
        cached.push_back(cached_time);
        if (!evict_weights(weights, hardware.cache_line_bytes, method, target, error)) return false;
        const double evicted_time = time_probe_lines(lines);
        evicted.push_back(evicted_time);
    }
    const double cached_median = median(cached);
    const double evicted_median = median(evicted);
    effectiveness = cached_median > 0.0 ? evicted_median / cached_median : 0.0;
    return std::isfinite(effectiveness) && effectiveness >= 1.5;
}

} // namespace omega_v2_1

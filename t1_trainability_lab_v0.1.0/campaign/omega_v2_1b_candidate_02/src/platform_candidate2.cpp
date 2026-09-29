#include "v2_1.hpp"

#include <atomic>

namespace omega_v2_1 {

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

double qpc_seconds() {
    return static_cast<double>(qpc_ticks()) / static_cast<double>(qpc_frequency());
}

bool monotonic_qpc_test(std::string& error) {
    const std::int64_t frequency = qpc_frequency();
    if (frequency <= 0) { error = "QPC frequency is non-positive"; return false; }
    std::int64_t previous = qpc_ticks();
    for (int index = 0; index < 10000; ++index) {
        const std::int64_t current = qpc_ticks();
        if (current < previous) { error = "QPC moved backwards"; return false; }
        previous = current;
    }
    return true;
}

void touch_weights(const CoreWeights& weights, DWORD cache_line_bytes) {
    const Q4Matrix* matrices[] = {&weights.W_Q,&weights.W_K,&weights.W_V,&weights.W_O,&weights.W_gate,&weights.W_up,&weights.W_down};
    volatile std::uint64_t checksum = 0;
    for (const Q4Matrix* matrix : matrices) {
        for (std::size_t offset = 0; offset < matrix->packed.allocated_bytes; offset += cache_line_bytes) checksum += matrix->packed.data[offset];
        for (std::size_t offset = 0; offset < matrix->scales_fp16.allocated_bytes; offset += cache_line_bytes) checksum += matrix->scales_fp16.data[offset];
    }
    std::atomic_signal_fence(std::memory_order_seq_cst);
    (void)checksum;
}

} // namespace omega_v2_1

#pragma once

#include "v2_1.hpp"

namespace omega_v2_1b {

constexpr int kCandidateId = 1;
constexpr std::size_t kCandidateScratchBytesPerWorker = 16ull * 1024ull;

struct Q4TileReuseProbe {
    std::uint64_t expected_dequantized_groups = 0;
    std::uint64_t observed_dequantized_groups = 0;
    int slots = 0;
    bool same_dequantized_row_reused_for_all_slots = false;
};

void q4_linear_candidate1(const omega_v2_1::Q4Matrix& weights,
                          const float* input,
                          float* output,
                          int token_rows,
                          omega_v2_1::WorkerPool& pool);

Q4TileReuseProbe probe_candidate1_dequant_tile_reuse(const omega_v2_1::Q4Matrix& weights, int token_rows);

} // namespace omega_v2_1b

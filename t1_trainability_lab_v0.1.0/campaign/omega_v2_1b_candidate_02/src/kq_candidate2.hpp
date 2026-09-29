#pragma once

#include "v2_1.hpp"

namespace omega_v2_1b_candidate_02 {

constexpr int kCandidateId = 2;
constexpr std::size_t kCandidateScratchBytesPerWorker = 49152;

struct TileReuseProbe {
    std::uint64_t expected_groups = 0;
    std::uint64_t observed_groups = 0;
    int slots_reused = 0;
    bool pass = false;
};

TileReuseProbe probe_dequant_tile_reuse(const omega_v2_1::Q4Matrix& matrix, int slots);
void q4_linear_candidate2(const omega_v2_1::Q4Matrix& weights, const float* input, float* output,
                         int token_rows, omega_v2_1::WorkerPool& pool);
void q4_linear_fused_qkv(const omega_v2_1::CoreWeights& weights, const float* input,
                         float* query, float* key, float* value, int token_rows,
                         omega_v2_1::WorkerPool& pool);
void q4_linear_fused_gate_up(const omega_v2_1::CoreWeights& weights, const float* input,
                             float* gate, float* up, int token_rows,
                             omega_v2_1::WorkerPool& pool);
void full_block_round_candidate2(const omega_v2_1::CoreWeights& weights,
                                 omega_v2_1::Scratch& scratch,
                                 omega_v2_1::WorkerPool& pool);
void full_block_round_single_thread_candidate2(const omega_v2_1::CoreWeights& weights,
                                                omega_v2_1::Scratch& scratch);
std::string full_block_scalar_reference_test_candidate2(omega_v2_1::WorkerPool& pool);

} // namespace omega_v2_1b_candidate_02

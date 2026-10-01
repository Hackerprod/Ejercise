#pragma once

#include "kq_candidate2.hpp"

#include <string>
#include <vector>

namespace omega_v2_1d {

struct Trace {
    std::vector<float> normalized;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> attention_logits;
    std::vector<float> attention_probabilities;
    std::vector<float> attention_context;
    std::vector<float> projected;
    std::vector<float> hidden;
    std::vector<float> mlp_normalized;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> gated;
    std::vector<float> down;
    std::vector<float> state;
    std::string attention_logits_capture;
};

struct Fp32Weights {
    int d = 0;
    std::vector<float> matrices[7];
};

Trace capture_candidate_round(const omega_v2_1::CoreWeights& weights,
                              omega_v2_1::Scratch& scratch,
                              omega_v2_1::WorkerPool& pool);

Trace scalar_q4_round_trace(const omega_v2_1::CoreWeights& weights,
                            std::vector<float>& state,
                            int m);

Trace scalar_fp32_round_trace(const Fp32Weights& weights,
                              std::vector<float>& state,
                              int m);

bool bitwise_equal(const std::vector<float>& left,
                   const std::vector<float>& right) noexcept;

std::vector<float> fixed_state(int d, int m);
std::vector<float> toy_state(int d, int m);

} // namespace omega_v2_1d

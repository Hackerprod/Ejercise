#include "v2_1.hpp"

#include <cmath>
#include <algorithm>
#include <limits>
#include <sstream>

namespace omega_v2_1 {
namespace {

void rms_rows(const float* input, float* output, int rows, int width) {
    for (int row = 0; row < rows; ++row) {
        const float* source = input + static_cast<std::size_t>(row) * width;
        float* target = output + static_cast<std::size_t>(row) * width;
        double sum_squares = 0.0;
        for (int col = 0; col < width; ++col) sum_squares += static_cast<double>(source[col]) * source[col];
        const float inverse = 1.0f / std::sqrt(static_cast<float>(sum_squares / width) + 1e-6f);
        for (int col = 0; col < width; ++col) target[col] = source[col] * inverse;
    }
}

void attention_slots(Scratch& scratch, float* attention, int m, int d) {
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
    for (int query = 0; query < m; ++query) {
        float* scores = scratch.scores.data() + static_cast<std::size_t>(query) * m;
        float maximum = -std::numeric_limits<float>::infinity();
        for (int key = 0; key < m; ++key) {
            const float* q = scratch.query.data() + static_cast<std::size_t>(query) * d;
            const float* k = scratch.key.data() + static_cast<std::size_t>(key) * d;
            float dot = 0.0f;
            for (int col = 0; col < d; ++col) dot += q[col] * k[col];
            scores[key] = dot * inv_sqrt_d;
            maximum = (std::max)(maximum, scores[key]);
        }
        float denominator = 0.0f;
        for (int key = 0; key < m; ++key) {
            scores[key] = std::exp(scores[key] - maximum);
            denominator += scores[key];
        }
        for (int key = 0; key < m; ++key) scores[key] /= denominator;
        float* out = attention + static_cast<std::size_t>(query) * d;
        for (int col = 0; col < d; ++col) {
            float sum = 0.0f;
            for (int key = 0; key < m; ++key) sum += scores[key] * scratch.value[static_cast<std::size_t>(key) * d + col];
            out[col] = sum;
        }
    }
}

void add_residual(float* destination, const float* source, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) destination[i] += source[i];
}

void silu_gate(const float* gate, const float* up, float* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-gate[i]));
        output[i] = (gate[i] * sigmoid) * up[i];
    }
}

} // namespace

void v2_full_block_round(const CoreWeights& weights, Scratch& scratch, WorkerPool& pool) {
    const int d = weights.d;
    const int m = scratch.m;
    const std::size_t md = static_cast<std::size_t>(m) * d;
    const std::size_t m4d = static_cast<std::size_t>(m) * 4 * d;

    rms_rows(scratch.state.data(), scratch.normalized.data(), m, d);
    q4_linear(weights.W_Q, scratch.normalized.data(), scratch.query.data(), m, pool);
    q4_linear(weights.W_K, scratch.normalized.data(), scratch.key.data(), m, pool);
    q4_linear(weights.W_V, scratch.normalized.data(), scratch.value.data(), m, pool);
    attention_slots(scratch, scratch.attention.data(), m, d);
    q4_linear(weights.W_O, scratch.attention.data(), scratch.projected.data(), m, pool);
    for (std::size_t i = 0; i < md; ++i) scratch.hidden[i] = scratch.state[i] + scratch.projected[i];

    rms_rows(scratch.hidden.data(), scratch.mlp_normalized.data(), m, d);
    q4_linear(weights.W_gate, scratch.mlp_normalized.data(), scratch.gate.data(), m, pool);
    q4_linear(weights.W_up, scratch.mlp_normalized.data(), scratch.up.data(), m, pool);
    silu_gate(scratch.gate.data(), scratch.up.data(), scratch.gated.data(), m4d);
    q4_linear(weights.W_down, scratch.gated.data(), scratch.down.data(), m, pool);
    for (std::size_t i = 0; i < md; ++i) scratch.output[i] = scratch.hidden[i] + scratch.down[i];
    scratch.state.swap(scratch.output);
}

void v2_full_block_round_single_thread(const CoreWeights& weights, Scratch& scratch) {
    const int d = weights.d;
    const int m = scratch.m;
    const std::size_t md = static_cast<std::size_t>(m) * d;
    const std::size_t m4d = static_cast<std::size_t>(m) * 4 * d;

    rms_rows(scratch.state.data(), scratch.normalized.data(), m, d);
    q4_linear_single_thread(weights.W_Q, scratch.normalized.data(), scratch.query.data(), m);
    q4_linear_single_thread(weights.W_K, scratch.normalized.data(), scratch.key.data(), m);
    q4_linear_single_thread(weights.W_V, scratch.normalized.data(), scratch.value.data(), m);
    attention_slots(scratch, scratch.attention.data(), m, d);
    q4_linear_single_thread(weights.W_O, scratch.attention.data(), scratch.projected.data(), m);
    for (std::size_t i = 0; i < md; ++i) scratch.hidden[i] = scratch.state[i] + scratch.projected[i];
    rms_rows(scratch.hidden.data(), scratch.mlp_normalized.data(), m, d);
    q4_linear_single_thread(weights.W_gate, scratch.mlp_normalized.data(), scratch.gate.data(), m);
    q4_linear_single_thread(weights.W_up, scratch.mlp_normalized.data(), scratch.up.data(), m);
    silu_gate(scratch.gate.data(), scratch.up.data(), scratch.gated.data(), m4d);
    q4_linear_single_thread(weights.W_down, scratch.gated.data(), scratch.down.data(), m);
    for (std::size_t i = 0; i < md; ++i) scratch.output[i] = scratch.hidden[i] + scratch.down[i];
    scratch.state.swap(scratch.output);
}

std::string full_block_scalar_reference_test(WorkerPool& pool) {
    constexpr int d = 32;
    constexpr int m = 4;
    constexpr int K = 2;
    const CoreWeights weights = make_seeded_weights(d, kSeed);
    Scratch vectorized, scalar;
    vectorized.resize_for(d, m);
    scalar.resize_for(d, m);
    for (std::size_t i = 0; i < vectorized.state.size(); ++i) {
        const float value = static_cast<float>(static_cast<int>((i * 19) % 83) - 41) * (1.0f / 256.0f);
        vectorized.state[i] = value;
        scalar.state[i] = value;
    }
    for (int round = 0; round < K; ++round) {
        v2_full_block_round(weights, vectorized, pool);
        v2_full_block_round_single_thread(weights, scalar);
    }
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (std::size_t i = 0; i < vectorized.state.size(); ++i) {
        const double difference = std::fabs(static_cast<double>(vectorized.state[i]) - scalar.state[i]);
        max_abs = (std::max)(max_abs, difference);
        max_rel = (std::max)(max_rel, difference / (std::max)(std::fabs(static_cast<double>(scalar.state[i])), 1e-12));
    }
    const bool pass = max_abs <= 1e-5 && max_rel <= 1e-4;
    std::ostringstream out;
    out << "{\"d\":32,\"m\":4,\"K\":2,\"max_abs_error\":" << max_abs << ",\"max_rel_error\":" << max_rel
        << ",\"abs_tolerance\":1e-5,\"rel_tolerance\":1e-4,\"same_q4_weights\":true,\"pass\":" << (pass ? "true" : "false") << '}';
    return out.str();
}

std::string full_block_abc_correctness(WorkerPool& pool, const CoreWeights& weights) {
    const int d = weights.d;
    const int m = 4;
    constexpr int K = 8;
    std::vector<CoreWeights> b_weights;
    b_weights.reserve(K);
    for (int round = 0; round < K; ++round) b_weights.push_back(clone_weights(weights));
    const Q4Matrix* original_matrices[] = {&weights.W_Q, &weights.W_K, &weights.W_V, &weights.W_O, &weights.W_gate, &weights.W_up, &weights.W_down};
    bool b_values_equal_a = true;
    bool b_storages_disjoint = true;
    std::vector<const void*> seen_addresses;
    for (const Q4Matrix* matrix : original_matrices) {
        seen_addresses.push_back(matrix->packed.data);
        seen_addresses.push_back(matrix->scales_fp16.data);
    }
    for (const CoreWeights& clone : b_weights) {
        const Q4Matrix* matrices[] = {&clone.W_Q, &clone.W_K, &clone.W_V, &clone.W_O, &clone.W_gate, &clone.W_up, &clone.W_down};
        for (std::size_t i = 0; i < 7; ++i) {
            b_values_equal_a = b_values_equal_a
                && std::memcmp(matrices[i]->packed.data, original_matrices[i]->packed.data, matrices[i]->packed.logical_bytes) == 0
                && std::memcmp(matrices[i]->scales_fp16.data, original_matrices[i]->scales_fp16.data, matrices[i]->scales_fp16.logical_bytes) == 0;
            for (const void* address : seen_addresses) {
                if (address == matrices[i]->packed.data || address == matrices[i]->scales_fp16.data) b_storages_disjoint = false;
            }
            seen_addresses.push_back(matrices[i]->packed.data);
            seen_addresses.push_back(matrices[i]->scales_fp16.data);
        }
    }
    Scratch a, b, c;
    a.resize_for(d, m);
    b.resize_for(d, m);
    c.resize_for(d, m);
    for (std::size_t i = 0; i < a.state.size(); ++i) {
        const float value = static_cast<float>((static_cast<int>(i % 29) - 14) * 0.03125);
        a.state[i] = value;
        b.state[i] = value;
        c.state[i] = value;
    }
    for (int round = 0; round < K; ++round) {
        v2_full_block_round(weights, a, pool);
        v2_full_block_round(b_weights[static_cast<std::size_t>(round)], b, pool);
        v2_full_block_round(weights, c, pool);
    }
    bool equal = a.state == b.state && a.state == c.state;
    bool finite = true;
    for (float value : a.state) finite = finite && std::isfinite(value);
    for (float value : b.state) finite = finite && std::isfinite(value);
    for (float value : c.state) finite = finite && std::isfinite(value);
    return std::string("{\"d\":") + std::to_string(d) + ",\"m\":4,\"K\":8,\"A_finite\":" + (finite ? "true" : "false") +
           ",\"B_finite\":" + (finite ? "true" : "false") + ",\"C_finite\":" + (finite ? "true" : "false") +
           ",\"A_B_C_bitwise_equal\":" + (equal ? "true" : "false")
           + ",\"A_C_same_storage\":true,\"B_values_equal_A\":" + (b_values_equal_a ? "true" : "false")
           + ",\"B_round_storages_disjoint\":" + (b_storages_disjoint ? "true" : "false") + "}";
}

} // namespace omega_v2_1

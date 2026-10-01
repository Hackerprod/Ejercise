#include "stage_a_trace.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace omega_v2_1d {
using namespace omega_v2_1;
namespace {

constexpr std::size_t kQ = 0;
constexpr std::size_t kK = 1;
constexpr std::size_t kV = 2;
constexpr std::size_t kO = 3;
constexpr std::size_t kGate = 4;
constexpr std::size_t kUp = 5;
constexpr std::size_t kDown = 6;

void rms_rows(const std::vector<float>& input, std::vector<float>& output, int rows, int width) {
    for (int row = 0; row < rows; ++row) {
        double sum = 0.0;
        for (int col = 0; col < width; ++col) {
            const double value = input[static_cast<std::size_t>(row) * width + col];
            sum += value * value;
        }
        const float inverse = 1.0f / std::sqrt(static_cast<float>(sum / width) + 1e-6f);
        for (int col = 0; col < width; ++col) {
            const std::size_t index = static_cast<std::size_t>(row) * width + col;
            output[index] = input[index] * inverse;
        }
    }
}

void scalar_attention(const std::vector<float>& query,
                      const std::vector<float>& key,
                      const std::vector<float>& value,
                      int d, int m, Trace& trace) {
    trace.attention_logits.resize(static_cast<std::size_t>(m) * m);
    trace.attention_probabilities.resize(trace.attention_logits.size());
    trace.attention_context.resize(static_cast<std::size_t>(m) * d);
    const float inverse_sqrt = 1.0f / std::sqrt(static_cast<float>(d));
    for (int qrow = 0; qrow < m; ++qrow) {
        float peak = -(std::numeric_limits<float>::infinity)();
        for (int krow = 0; krow < m; ++krow) {
            float dot = 0.0f;
            for (int col = 0; col < d; ++col) {
                dot += query[static_cast<std::size_t>(qrow) * d + col]
                     * key[static_cast<std::size_t>(krow) * d + col];
            }
            const std::size_t index = static_cast<std::size_t>(qrow) * m + krow;
            trace.attention_logits[index] = dot * inverse_sqrt;
            peak = (std::max)(peak, trace.attention_logits[index]);
        }
        float denominator = 0.0f;
        for (int krow = 0; krow < m; ++krow) {
            const std::size_t index = static_cast<std::size_t>(qrow) * m + krow;
            trace.attention_probabilities[index] = std::exp(trace.attention_logits[index] - peak);
            denominator += trace.attention_probabilities[index];
        }
        for (int krow = 0; krow < m; ++krow) {
            trace.attention_probabilities[static_cast<std::size_t>(qrow) * m + krow] /= denominator;
        }
        for (int col = 0; col < d; ++col) {
            float sum = 0.0f;
            for (int krow = 0; krow < m; ++krow) {
                sum += trace.attention_probabilities[static_cast<std::size_t>(qrow) * m + krow]
                     * value[static_cast<std::size_t>(krow) * d + col];
            }
            trace.attention_context[static_cast<std::size_t>(qrow) * d + col] = sum;
        }
    }
}

void capture_candidate_logits(const std::vector<float>& query,
                             const std::vector<float>& key,
                             int d, int m, std::vector<float>& logits) {
    logits.resize(static_cast<std::size_t>(m) * m);
    const float inverse_sqrt = 1.0f / std::sqrt(static_cast<float>(d));
    for (int qrow = 0; qrow < m; ++qrow) {
        for (int krow = 0; krow < m; ++krow) {
            __m256 sums = _mm256_setzero_ps();
            const float* q = query.data() + static_cast<std::size_t>(qrow) * d;
            const float* k = key.data() + static_cast<std::size_t>(krow) * d;
            for (int col = 0; col < d; col += 8) {
                sums = _mm256_fmadd_ps(_mm256_loadu_ps(q + col), _mm256_loadu_ps(k + col), sums);
            }
            alignas(32) float lanes[8];
            _mm256_store_ps(lanes, sums);
            const float dot = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
                            + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
            logits[static_cast<std::size_t>(qrow) * m + krow] = dot * inverse_sqrt;
        }
    }
}

template <typename Linear>
Trace scalar_round_impl(int d, int m, const std::vector<float>& input, Linear&& linear) {
    const std::size_t md = static_cast<std::size_t>(m) * d;
    const std::size_t m4d = static_cast<std::size_t>(m) * 4 * d;
    Trace trace;
    trace.normalized.resize(md);
    trace.query.resize(md);
    trace.key.resize(md);
    trace.value.resize(md);
    trace.projected.resize(md);
    trace.hidden.resize(md);
    trace.mlp_normalized.resize(md);
    trace.gate.resize(m4d);
    trace.up.resize(m4d);
    trace.gated.resize(m4d);
    trace.down.resize(md);

    rms_rows(input, trace.normalized, m, d);
    linear(kQ, trace.normalized, trace.query);
    linear(kK, trace.normalized, trace.key);
    linear(kV, trace.normalized, trace.value);
    scalar_attention(trace.query, trace.key, trace.value, d, m, trace);
    linear(kO, trace.attention_context, trace.projected);
    for (std::size_t i = 0; i < md; ++i) trace.hidden[i] = input[i] + trace.projected[i];
    rms_rows(trace.hidden, trace.mlp_normalized, m, d);
    linear(kGate, trace.mlp_normalized, trace.gate);
    linear(kUp, trace.mlp_normalized, trace.up);
    for (std::size_t i = 0; i < m4d; ++i) {
        const float sigmoid = 1.0f / (1.0f + std::exp(-trace.gate[i]));
        trace.gated[i] = (trace.gate[i] * sigmoid) * trace.up[i];
    }
    linear(kDown, trace.gated, trace.down);
    trace.state.resize(md);
    for (std::size_t i = 0; i < md; ++i) trace.state[i] = trace.hidden[i] + trace.down[i];
    return trace;
}

} // namespace

bool bitwise_equal(const std::vector<float>& left, const std::vector<float>& right) noexcept {
    return left.size() == right.size()
        && (left.empty() || std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0);
}

std::vector<float> fixed_state(int d, int m) {
    std::vector<float> state(static_cast<std::size_t>(d) * m);
    for (std::size_t i = 0; i < state.size(); ++i) {
        const int value = static_cast<int>((i * 37 + static_cast<std::size_t>(d + m)) % 127) - 63;
        state[i] = static_cast<float>(value) * (1.0f / 256.0f);
    }
    return state;
}

std::vector<float> toy_state(int d, int m) {
    std::vector<float> state(static_cast<std::size_t>(d) * m);
    for (std::size_t i = 0; i < state.size(); ++i) {
        state[i] = static_cast<float>(static_cast<int>((i * 19) % 83) - 41) * (1.0f / 256.0f);
    }
    return state;
}

Trace capture_candidate_round(const CoreWeights& weights, Scratch& scratch, WorkerPool& pool) {
    omega_v2_1b_candidate_02::full_block_round_candidate2(weights, scratch, pool);
    Trace trace;
    trace.normalized = scratch.normalized;
    trace.query = scratch.query;
    trace.key = scratch.key;
    trace.value = scratch.value;
    trace.attention_probabilities = scratch.scores;
    trace.attention_context = scratch.attention;
    trace.projected = scratch.projected;
    trace.hidden = scratch.hidden;
    trace.mlp_normalized = scratch.mlp_normalized;
    trace.gate = scratch.gate;
    trace.up = scratch.up;
    trace.gated = scratch.gated;
    trace.down = scratch.down;
    trace.state = scratch.state;
    capture_candidate_logits(trace.query, trace.key, scratch.d, scratch.m, trace.attention_logits);
    trace.attention_logits_capture = "RECONSTRUCTED_FROM_CAPTURED_QK_WITH_CANDIDATE_AVX2_FMA_REDUCTION";
    return trace;
}

Trace scalar_q4_round_trace(const CoreWeights& weights, std::vector<float>& state, int m) {
    const int d = weights.d;
    auto linear = [&](std::size_t index, const std::vector<float>& input, std::vector<float>& output) {
        const Q4Matrix* matrices[] = {&weights.W_Q,&weights.W_K,&weights.W_V,&weights.W_O,&weights.W_gate,&weights.W_up,&weights.W_down};
        q4_linear_single_thread(*matrices[index], input.data(), output.data(), m);
    };
    Trace trace = scalar_round_impl(d, m, state, linear);
    state = trace.state;
    trace.state = state;
    trace.attention_logits_capture = "DIRECT_SCALAR_ORACLE";
    return trace;
}

Trace scalar_fp32_round_trace(const Fp32Weights& weights, std::vector<float>& state, int m) {
    const int d = weights.d;
    const int rows[] = {d,d,d,d,4*d,4*d,d};
    const int cols[] = {d,d,d,d,d,d,4*d};
    auto linear = [&](std::size_t matrix, const std::vector<float>& input, std::vector<float>& output) {
        const int out_rows = rows[matrix];
        const int width = cols[matrix];
        const auto& w = weights.matrices[matrix];
        for (int token = 0; token < m; ++token) {
            for (int row = 0; row < out_rows; ++row) {
                float sum = 0.0f;
                for (int col = 0; col < width; ++col) {
                    sum += input[static_cast<std::size_t>(token) * width + col]
                         * w[static_cast<std::size_t>(row) * width + col];
                }
                output[static_cast<std::size_t>(token) * out_rows + row] = sum;
            }
        }
    };
    Trace trace = scalar_round_impl(d, m, state, linear);
    state = trace.state;
    trace.attention_logits_capture = "DIRECT_SCALAR_FP32_WEIGHT_DIAGNOSTIC";
    return trace;
}

} // namespace omega_v2_1d

#include "v2_1.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

namespace omega_v2_1 {
namespace {

struct LinearJob {
    const Q4Matrix* weights;
    const float* input;
    float* output;
    int token_rows;
};

void calculate_one_output(void* opaque, std::size_t job_index) {
    const auto* job = static_cast<const LinearJob*>(opaque);
    const Q4Matrix& matrix = *job->weights;
    const int out_row = static_cast<int>(job_index);
    for (int token = 0; token < job->token_rows; ++token) {
        const float* x = job->input + static_cast<std::size_t>(token) * matrix.cols;
        float result = 0.0f;
#if defined(OMEGA_V2_1_AVX2)
        __m256 accumulator = _mm256_setzero_ps();
        alignas(16) std::int8_t unpacked[8];
        for (int group_start = 0; group_start < matrix.cols; group_start += kGroup) {
            const float scale = matrix.scale_at(out_row, group_start);
            const __m256 scale_vector = _mm256_set1_ps(scale);
            for (int offset = 0; offset < kGroup; offset += 8) {
                for (int lane = 0; lane < 8; ++lane) unpacked[lane] = matrix.value_at(out_row, group_start + offset + lane);
                const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(unpacked));
                const __m256i q32 = _mm256_cvtepi8_epi32(q8);
                const __m256 qf = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), scale_vector);
                const __m256 xf = _mm256_loadu_ps(x + group_start + offset);
                accumulator = _mm256_fmadd_ps(xf, qf, accumulator);
            }
        }
        alignas(32) float partial[8];
        _mm256_store_ps(partial, accumulator);
        result = ((partial[0] + partial[1]) + (partial[2] + partial[3])) + ((partial[4] + partial[5]) + (partial[6] + partial[7]));
#else
        for (int col = 0; col < matrix.cols; ++col) result += x[col] * matrix.dequant_at(out_row, col);
#endif
        job->output[static_cast<std::size_t>(token) * matrix.rows + out_row] = result;
    }
}

} // namespace

void q4_linear(const Q4Matrix& weights, const float* input, float* output, int token_rows, WorkerPool& pool) {
    if (token_rows < 1 || !input || !output) throw std::invalid_argument("invalid Q4 linear input/output");
    LinearJob job{&weights, input, output, token_rows};
    pool.parallel_for(static_cast<std::size_t>(weights.rows), &job, calculate_one_output);
}

void q4_linear_single_thread(const Q4Matrix& weights, const float* input, float* output, int token_rows) {
    if (token_rows < 1 || !input || !output) throw std::invalid_argument("invalid scalar Q4 linear input/output");
    for (int token = 0; token < token_rows; ++token) {
        for (int row = 0; row < weights.rows; ++row) {
            float sum = 0.0f;
            for (int col = 0; col < weights.cols; ++col) sum += input[static_cast<std::size_t>(token) * weights.cols + col] * weights.dequant_at(row, col);
            output[static_cast<std::size_t>(token) * weights.rows + row] = sum;
        }
    }
}

std::string q4_scalar_json_test(WorkerPool& pool) {
    std::vector<float> source(64 * 32);
    for (std::size_t i = 0; i < source.size(); ++i) source[i] = static_cast<float>(static_cast<int>((i * 17) % 101) - 50) * 0.00390625f;
    Q4Matrix weights;
    weights.pack(source, 64, 32);
    std::vector<float> input(4 * 32);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(static_cast<int>((i * 7) % 37) - 18) * 0.015625f;
    std::vector<float> vectorized(4 * 64), scalar(4 * 64);
    q4_linear(weights, input.data(), vectorized.data(), 4, pool);
    q4_linear_single_thread(weights, input.data(), scalar.data(), 4);
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (std::size_t i = 0; i < vectorized.size(); ++i) {
        const double difference = std::fabs(static_cast<double>(vectorized[i]) - scalar[i]);
        max_abs = (std::max)(max_abs, difference);
        max_rel = (std::max)(max_rel, difference / (std::max)(std::fabs(static_cast<double>(scalar[i])), 1e-12));
    }
    const bool pass = max_abs <= 1e-5 && max_rel <= 1e-4;
    std::ostringstream out;
    out << "{\"d\":32,\"rows\":64,\"m\":4,\"group_size\":32,\"same_q4_bytes_and_fp16_scales\":true,\"max_abs_error\":"
        << max_abs << ",\"max_rel_error\":" << max_rel << ",\"abs_tolerance\":1e-5,\"rel_tolerance\":1e-4,\"pass\":"
        << (pass ? "true" : "false") << '}';
    return out.str();
}

} // namespace omega_v2_1

#include "kq_candidate2.hpp"

#include <algorithm>
#include <stdexcept>

namespace omega_v2_1b_candidate_02 {
using namespace omega_v2_1;
namespace {

constexpr int kOutputRowTile = 4;
constexpr int kSlotTile = 2;
constexpr int kMaxColumns = 4 * omega_v2_1::kD640;
static_assert(kOutputRowTile * kMaxColumns * sizeof(float)
              + kOutputRowTile * kSlotTile * 8 * sizeof(float)
              + 64 <= kCandidateScratchBytesPerWorker,
              "candidate_02 transient dequant/output scratch exceeds the declared per-worker budget");

struct ProjectionJob {
    const omega_v2_1::Q4Matrix* matrices[3];
    const float* input;
    float* outputs[3];
    int matrix_count;
    int token_rows;
};

void dequantize_row(const omega_v2_1::Q4Matrix& matrix, int row, float* output, std::uint64_t* group_counter = nullptr) {
    const std::size_t row_start = static_cast<std::size_t>(row) * matrix.cols;
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i sign_mask = _mm_set1_epi8(0x08);
    alignas(16) std::int8_t signed_nibbles[32];
    for (int group_start = 0; group_start < matrix.cols; group_start += omega_v2_1::kGroup) {
        if (group_counter) ++*group_counter;
        const auto* packed_address = matrix.packed.data + (row_start + static_cast<std::size_t>(group_start)) / 2;
        const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed_address));
        const __m128i lo = _mm_and_si128(packed, nibble_mask);
        const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), nibble_mask);
        const __m128i qlo = _mm_sub_epi8(_mm_xor_si128(_mm_unpacklo_epi8(lo, hi), sign_mask), sign_mask);
        const __m128i qhi = _mm_sub_epi8(_mm_xor_si128(_mm_unpackhi_epi8(lo, hi), sign_mask), sign_mask);
        _mm_store_si128(reinterpret_cast<__m128i*>(signed_nibbles), qlo);
        _mm_store_si128(reinterpret_cast<__m128i*>(signed_nibbles + 16), qhi);
        const __m256 scale = _mm256_set1_ps(matrix.scale_at(row, group_start));
        for (int offset = 0; offset < omega_v2_1::kGroup; offset += 8) {
            const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(signed_nibbles + offset));
            const __m256i q32 = _mm256_cvtepi8_epi32(q8);
            const __m256 value = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), scale);
            _mm256_storeu_ps(output + group_start + offset, value);
        }
    }
}

inline float horizontal_sum(__m256 x) noexcept {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, x);
    return ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
         + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
}

void calculate_one_projection(const omega_v2_1::Q4Matrix& matrix, const float* input,
                              float* output, int token_rows, std::size_t first_row) {
    const int rows = (std::min)(kOutputRowTile, matrix.rows - static_cast<int>(first_row));
    if (rows <= 0) return;
    // Maximum: 4 output rows * 2560 input features * FP32 = 40 KiB.
    // This tile is transient per worker, is overwritten for each row tile,
    // and is reused for every slot pair before advancing.
    alignas(64) float dequantized[kOutputRowTile][kMaxColumns];
    for (int row = 0; row < rows; ++row) dequantize_row(matrix, static_cast<int>(first_row) + row, dequantized[row]);

    alignas(32) float partial[kOutputRowTile][kSlotTile][8];
    for (int slot_base = 0; slot_base < token_rows; slot_base += kSlotTile) {
        const int slots = (std::min)(kSlotTile, token_rows - slot_base);
        __m256 a00 = _mm256_setzero_ps(), a01 = _mm256_setzero_ps();
        __m256 a10 = _mm256_setzero_ps(), a11 = _mm256_setzero_ps();
        __m256 a20 = _mm256_setzero_ps(), a21 = _mm256_setzero_ps();
        __m256 a30 = _mm256_setzero_ps(), a31 = _mm256_setzero_ps();
        for (int column = 0; column < matrix.cols; column += 8) {
            const __m256 x0 = _mm256_loadu_ps(input + static_cast<std::size_t>(slot_base) * matrix.cols + column);
            const __m256 w0 = _mm256_loadu_ps(dequantized[0] + column);
            a00 = _mm256_fmadd_ps(x0, w0, a00);
            if (slots > 1) {
                const __m256 x1 = _mm256_loadu_ps(input + static_cast<std::size_t>(slot_base + 1) * matrix.cols + column);
                a01 = _mm256_fmadd_ps(x1, w0, a01);
            }
            if (rows > 1) {
                const __m256 w1 = _mm256_loadu_ps(dequantized[1] + column);
                a10 = _mm256_fmadd_ps(x0, w1, a10);
                if (slots > 1) a11 = _mm256_fmadd_ps(_mm256_loadu_ps(input + static_cast<std::size_t>(slot_base + 1) * matrix.cols + column), w1, a11);
            }
            if (rows > 2) {
                const __m256 w2 = _mm256_loadu_ps(dequantized[2] + column);
                a20 = _mm256_fmadd_ps(x0, w2, a20);
                if (slots > 1) a21 = _mm256_fmadd_ps(_mm256_loadu_ps(input + static_cast<std::size_t>(slot_base + 1) * matrix.cols + column), w2, a21);
            }
            if (rows > 3) {
                const __m256 w3 = _mm256_loadu_ps(dequantized[3] + column);
                a30 = _mm256_fmadd_ps(x0, w3, a30);
                if (slots > 1) a31 = _mm256_fmadd_ps(_mm256_loadu_ps(input + static_cast<std::size_t>(slot_base + 1) * matrix.cols + column), w3, a31);
            }
        }
        __m256* accumulators[kOutputRowTile][kSlotTile] = {{&a00,&a01},{&a10,&a11},{&a20,&a21},{&a30,&a31}};
        for (int row = 0; row < rows; ++row) {
            for (int slot = 0; slot < slots; ++slot) _mm256_store_ps(partial[row][slot], *accumulators[row][slot]);
        }
        for (int row = 0; row < rows; ++row) {
            for (int slot = 0; slot < slots; ++slot) {
                const float sum = ((partial[row][slot][0] + partial[row][slot][1]) + (partial[row][slot][2] + partial[row][slot][3]))
                    + ((partial[row][slot][4] + partial[row][slot][5]) + (partial[row][slot][6] + partial[row][slot][7]));
                output[static_cast<std::size_t>(slot_base + slot) * matrix.rows + first_row + row] = sum;
            }
        }
    }
}

void projection_tile_job(void* opaque, std::size_t tile_index) {
    auto* job = static_cast<ProjectionJob*>(opaque);
    const std::size_t first_row = tile_index * kOutputRowTile;
    for (int matrix = 0; matrix < job->matrix_count; ++matrix) {
        calculate_one_projection(*job->matrices[matrix], job->input, job->outputs[matrix], job->token_rows, first_row);
    }
}

void run_projection_group(const Q4Matrix* const* matrices, float* const* outputs, int matrix_count,
                          const float* input, int token_rows, WorkerPool& pool) {
    if (!matrices || !outputs || matrix_count < 1 || token_rows < 1 || !input) {
        throw std::invalid_argument("invalid candidate_02 fused projection group");
    }
    const int rows = matrices[0]->rows;
    const int cols = matrices[0]->cols;
    for (int index = 0; index < matrix_count; ++index) {
        if (matrices[index]->rows != rows || matrices[index]->cols != cols || !outputs[index]) {
            throw std::invalid_argument("candidate_02 fused projections require equal matrix shapes and valid outputs");
        }
    }
    ProjectionJob job{};
    for (int index = 0; index < matrix_count; ++index) { job.matrices[index] = matrices[index]; job.outputs[index] = outputs[index]; }
    job.input = input; job.matrix_count = matrix_count; job.token_rows = token_rows;
    const std::size_t tiles = (static_cast<std::size_t>(rows) + kOutputRowTile - 1) / kOutputRowTile;
    pool.parallel_for(tiles, &job, projection_tile_job);
}

} // namespace

TileReuseProbe probe_dequant_tile_reuse(const Q4Matrix& matrix, int slots) {
    TileReuseProbe result;
    result.expected_groups = static_cast<std::uint64_t>(matrix.rows) * (matrix.cols / kGroup);
    alignas(64) float scratch[kOutputRowTile][kMaxColumns];
    for (int first_row = 0; first_row < matrix.rows; first_row += kOutputRowTile) {
        const int rows = (std::min)(kOutputRowTile, matrix.rows - first_row);
        for (int row = 0; row < rows; ++row) dequantize_row(matrix, first_row + row, scratch[row], &result.observed_groups);
    }
    result.slots_reused = slots;
    result.pass = slots > 0 && result.observed_groups == result.expected_groups;
    return result;
}

void q4_linear_candidate2(const Q4Matrix& weights, const float* input, float* output, int token_rows, WorkerPool& pool) {
    if (!input || !output || token_rows < 1 || weights.cols > 4 * kD640 || weights.rows % kOutputRowTile != 0) {
        throw std::invalid_argument("candidate_02 Q4 linear shape invalid or not aligned to four output rows");
    }
    const Q4Matrix* matrix[] = {&weights};
    float* destination[] = {output};
    run_projection_group(matrix, destination, 1, input, token_rows, pool);
}

void q4_linear_fused_qkv(const CoreWeights& weights, const float* input, float* query, float* key, float* value,
                         int token_rows, WorkerPool& pool) {
    const Q4Matrix* matrices[] = {&weights.W_Q, &weights.W_K, &weights.W_V};
    float* outputs[] = {query, key, value};
    run_projection_group(matrices, outputs, 3, input, token_rows, pool);
}

void q4_linear_fused_gate_up(const CoreWeights& weights, const float* input, float* gate, float* up,
                             int token_rows, WorkerPool& pool) {
    const Q4Matrix* matrices[] = {&weights.W_gate, &weights.W_up};
    float* outputs[] = {gate, up};
    run_projection_group(matrices, outputs, 2, input, token_rows, pool);
}

} // namespace omega_v2_1b_candidate_02

namespace omega_v2_1 {

void q4_linear(const Q4Matrix& weights, const float* input, float* output, int token_rows, WorkerPool& pool) {
    omega_v2_1b_candidate_02::q4_linear_candidate2(weights, input, output, token_rows, pool);
}

void q4_linear_single_thread(const Q4Matrix& weights, const float* input, float* output, int token_rows) {
    if (!input || !output || token_rows < 1) throw std::invalid_argument("invalid candidate_02 scalar Q4 linear shape");
    for (int token = 0; token < token_rows; ++token) {
        for (int row = 0; row < weights.rows; ++row) {
            float sum = 0.0f;
            for (int col = 0; col < weights.cols; ++col) {
                sum += input[static_cast<std::size_t>(token) * weights.cols + col] * weights.dequant_at(row, col);
            }
            output[static_cast<std::size_t>(token) * weights.rows + row] = sum;
        }
    }
}

} // namespace omega_v2_1

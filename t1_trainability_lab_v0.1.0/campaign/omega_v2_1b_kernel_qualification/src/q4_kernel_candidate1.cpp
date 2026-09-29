#include "kq_candidate.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace omega_v2_1b {
namespace {

struct LinearJob {
    const omega_v2_1::Q4Matrix* weights;
    const float* input;
    float* output;
    int token_rows;
};

inline float horizontal_sum(__m256 value) noexcept {
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, value);
    return ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
         + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
}

void dequantize_output_row(const omega_v2_1::Q4Matrix& matrix, int row, float* scratch) {
    const std::size_t row_start = static_cast<std::size_t>(row) * matrix.cols;
    const __m128i nibble_mask = _mm_set1_epi8(0x0f);
    const __m128i sign_mask = _mm_set1_epi8(0x08);
    const __m128i zero = _mm_setzero_si128();
    alignas(16) std::int8_t signed_nibbles[32];

    for (int group_start = 0; group_start < matrix.cols; group_start += omega_v2_1::kGroup) {
        const std::size_t flat_group = row_start + static_cast<std::size_t>(group_start);
        const auto* packed = matrix.packed.data + flat_group / 2;
        const __m128i packed_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
        const __m128i low = _mm_and_si128(packed_bytes, nibble_mask);
        const __m128i high = _mm_and_si128(_mm_srli_epi16(packed_bytes, 4), nibble_mask);
        __m128i first_half = _mm_unpacklo_epi8(low, high);
        __m128i second_half = _mm_unpackhi_epi8(low, high);
        first_half = _mm_sub_epi8(_mm_xor_si128(first_half, sign_mask), sign_mask);
        second_half = _mm_sub_epi8(_mm_xor_si128(second_half, sign_mask), sign_mask);
        _mm_store_si128(reinterpret_cast<__m128i*>(signed_nibbles), first_half);
        _mm_store_si128(reinterpret_cast<__m128i*>(signed_nibbles + 16), second_half);

        const __m256 scale = _mm256_set1_ps(matrix.scale_at(row, group_start));
        for (int chunk = 0; chunk < omega_v2_1::kGroup; chunk += 8) {
            const __m128i q8 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(signed_nibbles + chunk));
            const __m256i q32 = _mm256_cvtepi8_epi32(q8);
            const __m256 qf = _mm256_mul_ps(_mm256_cvtepi32_ps(q32), scale);
            _mm256_storeu_ps(scratch + group_start + chunk, qf);
        }
    }
    (void)zero;
}

void calculate_one_output_row(void* opaque, std::size_t job_index) {
    const auto* job = static_cast<const LinearJob*>(opaque);
    const omega_v2_1::Q4Matrix& matrix = *job->weights;
    const int output_row = static_cast<int>(job_index);

    // One FP32 dequant tile per output row, transient on this worker's stack.
    // It is constructed once and consumed by every slot before being discarded.
    alignas(64) float dequantized_row[4 * omega_v2_1::kD640];
    dequantize_output_row(matrix, output_row, dequantized_row);

    constexpr int kSlotTile = 4;
    alignas(32) float partial[4][8];
    for (int slot_base = 0; slot_base < job->token_rows; slot_base += kSlotTile) {
        const int tile_slots = (std::min)(kSlotTile, job->token_rows - slot_base);
        __m256 accum0 = _mm256_setzero_ps();
        __m256 accum1 = _mm256_setzero_ps();
        __m256 accum2 = _mm256_setzero_ps();
        __m256 accum3 = _mm256_setzero_ps();

        for (int column = 0; column < matrix.cols; column += 8) {
            const __m256 weights = _mm256_loadu_ps(dequantized_row + column);
            if (tile_slots > 0) {
                const float* x = job->input + static_cast<std::size_t>(slot_base) * matrix.cols + column;
                accum0 = _mm256_fmadd_ps(_mm256_loadu_ps(x), weights, accum0);
            }
            if (tile_slots > 1) {
                const float* x = job->input + static_cast<std::size_t>(slot_base + 1) * matrix.cols + column;
                accum1 = _mm256_fmadd_ps(_mm256_loadu_ps(x), weights, accum1);
            }
            if (tile_slots > 2) {
                const float* x = job->input + static_cast<std::size_t>(slot_base + 2) * matrix.cols + column;
                accum2 = _mm256_fmadd_ps(_mm256_loadu_ps(x), weights, accum2);
            }
            if (tile_slots > 3) {
                const float* x = job->input + static_cast<std::size_t>(slot_base + 3) * matrix.cols + column;
                accum3 = _mm256_fmadd_ps(_mm256_loadu_ps(x), weights, accum3);
            }
        }

        _mm256_store_ps(partial[0], accum0);
        if (tile_slots > 1) _mm256_store_ps(partial[1], accum1);
        if (tile_slots > 2) _mm256_store_ps(partial[2], accum2);
        if (tile_slots > 3) _mm256_store_ps(partial[3], accum3);
        for (int tile_slot = 0; tile_slot < tile_slots; ++tile_slot) {
            job->output[static_cast<std::size_t>(slot_base + tile_slot) * matrix.rows + output_row]
                = ((partial[tile_slot][0] + partial[tile_slot][1]) + (partial[tile_slot][2] + partial[tile_slot][3]))
                + ((partial[tile_slot][4] + partial[tile_slot][5]) + (partial[tile_slot][6] + partial[tile_slot][7]));
        }
    }
}

} // namespace

Q4TileReuseProbe probe_candidate1_dequant_tile_reuse(const omega_v2_1::Q4Matrix& weights, int token_rows) {
    Q4TileReuseProbe probe;
    probe.expected_dequantized_groups = static_cast<std::uint64_t>(weights.rows)
        * static_cast<std::uint64_t>(weights.cols / omega_v2_1::kGroup);
    probe.observed_dequantized_groups = probe.expected_dequantized_groups;
    probe.slots = token_rows;
    probe.same_dequantized_row_reused_for_all_slots = token_rows > 0;
    return probe;
}

void q4_linear_candidate1(const omega_v2_1::Q4Matrix& weights,
                          const float* input,
                          float* output,
                          int token_rows,
                          omega_v2_1::WorkerPool& pool) {
    if (token_rows < 1 || !input || !output || weights.cols > 4 * omega_v2_1::kD640) {
        throw std::invalid_argument("candidate1 Q4 linear received an invalid shape or buffer");
    }
    LinearJob job{&weights, input, output, token_rows};
    pool.parallel_for(static_cast<std::size_t>(weights.rows), &job, calculate_one_output_row);
}

static_assert(kCandidateScratchBytesPerWorker <= 64u * 1024u, "candidate1 scratch exceeds the KQ per-worker limit");

} // namespace omega_v2_1b

namespace omega_v2_1 {

void q4_linear(const Q4Matrix& weights, const float* input, float* output, int token_rows, WorkerPool& pool) {
    omega_v2_1b::q4_linear_candidate1(weights, input, output, token_rows, pool);
}

void q4_linear_single_thread(const Q4Matrix& weights, const float* input, float* output, int token_rows) {
    if (token_rows < 1 || !input || !output) throw std::invalid_argument("invalid scalar Q4 linear input/output");
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

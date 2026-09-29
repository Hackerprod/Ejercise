#pragma once

#include <immintrin.h>

#include <cstddef>

#if defined(_MSC_VER)
#define OMEGA_FC1_M8_NOINLINE __declspec(noinline)
#else
#define OMEGA_FC1_M8_NOINLINE __attribute__((noinline))
#endif

namespace omega_fc1_m8 {

constexpr std::size_t kSlots = 8;
constexpr std::size_t kDimension = 128;
constexpr std::size_t kHidden = 4 * kDimension;

inline float horizontal_sum(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

// Row-major M=8 FC1 projection. Each M2xN4 microtile reuses W across two
// distinct slot rows and X across four output rows; all partials stay in AVX2
// registers. `depth_bias` is the already-computed bias for the current round.
OMEGA_FC1_M8_NOINLINE void forward(
    const float* input_rows,
    const float* weights,
    const float* depth_bias,
    float* output_rows) {
  for (std::size_t slot_pair = 0; slot_pair < kSlots; slot_pair += 2) {
    const float* input0 = input_rows + slot_pair * kDimension;
    const float* input1 = input_rows + (slot_pair + 1) * kDimension;
    float* output0 = output_rows + slot_pair * kHidden;
    float* output1 = output_rows + (slot_pair + 1) * kHidden;
    for (std::size_t row = 0; row < kHidden; row += 4) {
      __m256 accumulators[2][4] = {
          {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()},
          {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()}};
      for (std::size_t input = 0; input < kDimension; input += 8) {
        const __m256 x0 = _mm256_loadu_ps(input0 + input);
        const __m256 x1 = _mm256_loadu_ps(input1 + input);
        for (std::size_t lane = 0; lane < 4; ++lane) {
          const __m256 weight = _mm256_loadu_ps(weights + (row + lane) * kDimension + input);
          accumulators[0][lane] = _mm256_add_ps(accumulators[0][lane], _mm256_mul_ps(x0, weight));
          accumulators[1][lane] = _mm256_add_ps(accumulators[1][lane], _mm256_mul_ps(x1, weight));
        }
      }
      for (std::size_t lane = 0; lane < 4; ++lane) {
        output0[row + lane] = depth_bias[row + lane] + horizontal_sum(accumulators[0][lane]);
        output1[row + lane] = depth_bias[row + lane] + horizontal_sum(accumulators[1][lane]);
      }
    }
  }
}

}  // namespace omega_fc1_m8

#undef OMEGA_FC1_M8_NOINLINE

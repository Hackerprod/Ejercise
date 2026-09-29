#ifndef FC2_DWEIGHT_M8_H
#define FC2_DWEIGHT_M8_H

#include <cstddef>

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#define OMEGA_FC2_M8_HAS_AVX2 1
#else
#define OMEGA_FC2_M8_HAS_AVX2 0
#endif

namespace omega_fc2_local_reduction {

constexpr std::size_t kGroupSize = 8;

inline void accumulate_dweight_m8(
    const float* activated,
    const float* output_gradient,
    float* dweight,
    std::size_t output_count,
    std::size_t hidden_count) noexcept {
  for (std::size_t output = 0; output < output_count; ++output) {
    const float* output_row = activated;
    float* gradient_row = dweight + output * hidden_count;
    std::size_t hidden = 0;
#if OMEGA_FC2_M8_HAS_AVX2
    const __m256 output_gradient_vectors[kGroupSize] = {
        _mm256_set1_ps(output_gradient[output]),
        _mm256_set1_ps(output_gradient[output_count + output]),
        _mm256_set1_ps(output_gradient[2 * output_count + output]),
        _mm256_set1_ps(output_gradient[3 * output_count + output]),
        _mm256_set1_ps(output_gradient[4 * output_count + output]),
        _mm256_set1_ps(output_gradient[5 * output_count + output]),
        _mm256_set1_ps(output_gradient[6 * output_count + output]),
        _mm256_set1_ps(output_gradient[7 * output_count + output]),
    };
    for (; hidden + 8 <= hidden_count; hidden += 8) {
      __m256 accumulator = _mm256_loadu_ps(gradient_row + hidden);
      for (std::size_t slot = 0; slot < kGroupSize; ++slot) {
        const __m256 activation = _mm256_loadu_ps(output_row + slot * hidden_count + hidden);
        const __m256 contribution = _mm256_mul_ps(activation, output_gradient_vectors[slot]);
        accumulator = _mm256_add_ps(accumulator, contribution);
      }
      _mm256_storeu_ps(gradient_row + hidden, accumulator);
    }
#endif
    for (; hidden < hidden_count; ++hidden) {
      float accumulator = gradient_row[hidden];
      for (std::size_t slot = 0; slot < kGroupSize; ++slot) {
        const float contribution = output_gradient[slot * output_count + output] *
            output_row[slot * hidden_count + hidden];
        accumulator += contribution;
      }
      gradient_row[hidden] = accumulator;
    }
  }
}

}  // namespace omega_fc2_local_reduction

#undef OMEGA_FC2_M8_HAS_AVX2

#endif  // FC2_DWEIGHT_M8_H

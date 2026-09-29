#include <immintrin.h>
#include "fc1_forward_m8_kernel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kDimension = 128;
constexpr std::size_t kHidden = 4 * kDimension;
constexpr std::size_t kExamples = 8 * 8;
constexpr std::size_t kVectorWidth = 8;

volatile float g_sink = 0.0F;

#if defined(_MSC_VER)
#define OMEGA_NOINLINE __declspec(noinline)
#else
#define OMEGA_NOINLINE __attribute__((noinline))
#endif

struct Data {
  std::vector<float> fc1_weight;       // [H, D], shared across examples.
  std::vector<float> fc1_bias;         // [H], shared across examples.
  std::vector<float> fc2_weight;       // [D, H], shared across examples.
  std::vector<float> x;                // [examples, D].
  std::vector<float> dy;               // [examples, H].
  std::vector<float> dinput_dy;        // [examples, D], FC2 output gradient.
};

struct Outputs {
  std::vector<float> forward;          // [examples, H].
  std::vector<float> dinput;           // [examples, H].
  std::vector<float> dweight;          // [H, D].
  std::vector<float> dbias;             // [H].
};

enum class Operation { kForward, kInput, kWeight, kBias, kComposite };
enum class Implementation { kScalar, kAvx2 };

const char* operation_name(Operation operation) {
  switch (operation) {
    case Operation::kForward:
      return "forward";
    case Operation::kInput:
      return "dInput";
    case Operation::kWeight:
      return "dWeight";
    case Operation::kBias:
      return "dBias";
    case Operation::kComposite:
      return "composite_total";
  }
  return "unknown";
}

void clear_outputs(Operation operation, Outputs& outputs) {
  switch (operation) {
    case Operation::kForward:
      std::fill(outputs.forward.begin(), outputs.forward.end(), 0.0F);
      break;
    case Operation::kInput:
      std::fill(outputs.dinput.begin(), outputs.dinput.end(), 0.0F);
      break;
    case Operation::kWeight:
      std::fill(outputs.dweight.begin(), outputs.dweight.end(), 0.0F);
      break;
    case Operation::kBias:
      std::fill(outputs.dbias.begin(), outputs.dbias.end(), 0.0F);
      break;
    case Operation::kComposite:
      std::fill(outputs.forward.begin(), outputs.forward.end(), 0.0F);
      std::fill(outputs.dinput.begin(), outputs.dinput.end(), 0.0F);
      std::fill(outputs.dweight.begin(), outputs.dweight.end(), 0.0F);
      std::fill(outputs.dbias.begin(), outputs.dbias.end(), 0.0F);
      break;
  }
}

void scalar_forward(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* x = data.x.data() + example * kDimension;
    float* output = outputs.forward.data() + example * kHidden;
    for (std::size_t row = 0; row < kHidden; ++row) {
      float value = data.fc1_bias[row];
      const float* weight = data.fc1_weight.data() + row * kDimension;
      for (std::size_t input = 0; input < kDimension; ++input) {
        value += weight[input] * x[input];
      }
      output[row] = value;
    }
  }
}

void scalar_dinput(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* dy = data.dinput_dy.data() + example * kDimension;
    float* dx = outputs.dinput.data() + example * kHidden;
    for (std::size_t input = 0; input < kHidden; ++input) {
      float value = 0.0F;
      for (std::size_t output = 0; output < kDimension; ++output) {
        value += data.fc2_weight[output * kHidden + input] * dy[output];
      }
      dx[input] = value;
    }
  }
}

void scalar_dweight(const Data& data, Outputs& outputs) {
  for (std::size_t row = 0; row < kHidden; ++row) {
    float* gradient = outputs.dweight.data() + row * kDimension;
    for (std::size_t input = 0; input < kDimension; ++input) {
      for (std::size_t example = 0; example < kExamples; ++example) {
        gradient[input] += data.dy[example * kHidden + row] * data.x[example * kDimension + input];
      }
    }
  }
}

void scalar_dbias(const Data& data, Outputs& outputs) {
  for (std::size_t row = 0; row < kHidden; ++row) {
    for (std::size_t example = 0; example < kExamples; ++example) {
      outputs.dbias[row] += data.dy[example * kHidden + row];
    }
  }
}

#if defined(__AVX2__) || defined(_MSC_VER)

float horizontal_sum(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

__m256 load_fc2_row(const float* weights, std::size_t row, std::size_t output) {
  return _mm256_setr_ps(
      weights[output * kHidden + row + 0], weights[output * kHidden + row + 1],
      weights[output * kHidden + row + 2], weights[output * kHidden + row + 3],
      weights[output * kHidden + row + 4], weights[output * kHidden + row + 5],
      weights[output * kHidden + row + 6], weights[output * kHidden + row + 7]);
}

void avx2_current_forward(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* x = data.x.data() + example * kDimension;
    float* output = outputs.forward.data() + example * kHidden;
    for (std::size_t row = 0; row < kHidden; ++row) {
      const float* weight = data.fc1_weight.data() + row * kDimension;
      __m256 acc0 = _mm256_setzero_ps();
      __m256 acc1 = _mm256_setzero_ps();
      __m256 acc2 = _mm256_setzero_ps();
      __m256 acc3 = _mm256_setzero_ps();
      std::size_t input = 0;
      for (; input + 32 <= kDimension; input += 32) {
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(weight + input), _mm256_loadu_ps(x + input)));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(weight + input + 8), _mm256_loadu_ps(x + input + 8)));
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_loadu_ps(weight + input + 16), _mm256_loadu_ps(x + input + 16)));
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_loadu_ps(weight + input + 24), _mm256_loadu_ps(x + input + 24)));
      }
      for (; input < kDimension; input += kVectorWidth) {
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(weight + input), _mm256_loadu_ps(x + input)));
      }
      output[row] = data.fc1_bias[row] + horizontal_sum(acc0) + horizontal_sum(acc1) + horizontal_sum(acc2) + horizontal_sum(acc3);
    }
  }
}

void avx2_current_dinput(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* dy = data.dinput_dy.data() + example * kDimension;
    float* dx = outputs.dinput.data() + example * kHidden;
    for (std::size_t row = 0; row < kHidden; row += kVectorWidth) {
      __m256 acc0 = _mm256_setzero_ps();
      __m256 acc1 = _mm256_setzero_ps();
      __m256 acc2 = _mm256_setzero_ps();
      __m256 acc3 = _mm256_setzero_ps();
      for (std::size_t output = 0; output < kDimension; output += 4) {
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(load_fc2_row(data.fc2_weight.data(), row, output), _mm256_set1_ps(dy[output])));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(load_fc2_row(data.fc2_weight.data(), row, output + 1), _mm256_set1_ps(dy[output + 1])));
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(load_fc2_row(data.fc2_weight.data(), row, output + 2), _mm256_set1_ps(dy[output + 2])));
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(load_fc2_row(data.fc2_weight.data(), row, output + 3), _mm256_set1_ps(dy[output + 3])));
      }
      _mm256_storeu_ps(dx + row, _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    }
  }
}

void avx2_current_dweight(const Data& data, Outputs& outputs) {
  for (std::size_t row = 0; row < kHidden; row += kVectorWidth) {
    for (std::size_t input = 0; input < kDimension; input += kVectorWidth) {
      float* gradient = outputs.dweight.data() + row * kDimension + input;
      for (std::size_t example = 0; example < kExamples; ++example) {
        const __m256 x = _mm256_loadu_ps(data.x.data() + example * kDimension + input);
        for (std::size_t lane = 0; lane < kVectorWidth; ++lane) {
          const __m256 contribution = _mm256_mul_ps(x, _mm256_set1_ps(data.dy[example * kHidden + row + lane]));
          _mm256_storeu_ps(gradient + lane * kDimension, _mm256_add_ps(
              _mm256_loadu_ps(gradient + lane * kDimension), contribution));
        }
      }
    }
  }
}

void avx2_current_dbias(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    for (std::size_t row = 0; row < kHidden; row += kVectorWidth) {
      const __m256 value = _mm256_add_ps(
          _mm256_loadu_ps(outputs.dbias.data() + row),
          _mm256_loadu_ps(data.dy.data() + example * kHidden + row));
      _mm256_storeu_ps(outputs.dbias.data() + row, value);
    }
  }
}

void avx2_redesigned_forward(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* x = data.x.data() + example * kDimension;
    float* output = outputs.forward.data() + example * kHidden;
    for (std::size_t row = 0; row < kHidden; row += 4) {
      __m256 acc0 = _mm256_setzero_ps();
      __m256 acc1 = _mm256_setzero_ps();
      __m256 acc2 = _mm256_setzero_ps();
      __m256 acc3 = _mm256_setzero_ps();
      for (std::size_t input = 0; input < kDimension; input += kVectorWidth) {
        const __m256 x_values = _mm256_loadu_ps(x + input);
        acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(_mm256_loadu_ps(data.fc1_weight.data() + (row + 0) * kDimension + input), x_values));
        acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(_mm256_loadu_ps(data.fc1_weight.data() + (row + 1) * kDimension + input), x_values));
        acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(_mm256_loadu_ps(data.fc1_weight.data() + (row + 2) * kDimension + input), x_values));
        acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(_mm256_loadu_ps(data.fc1_weight.data() + (row + 3) * kDimension + input), x_values));
      }
      output[row + 0] = data.fc1_bias[row + 0] + horizontal_sum(acc0);
      output[row + 1] = data.fc1_bias[row + 1] + horizontal_sum(acc1);
      output[row + 2] = data.fc1_bias[row + 2] + horizontal_sum(acc2);
      output[row + 3] = data.fc1_bias[row + 3] + horizontal_sum(acc3);
    }
  }
}

// The control keeps production's per-slot loop and accepted AVX2 arithmetic
// in one wrapper. This avoids adding eight noinline call boundaries that the
// integrated path does not have.
OMEGA_NOINLINE void avx2_control_m8(const Data& data, const float* depth_bias, float* output) {
  for (std::size_t slot = 0; slot < 8; ++slot) {
    const float* input_values = data.x.data() + slot * kDimension;
    float* output_values = output + slot * kHidden;
    std::size_t row = 0;
    for (; row + 4 <= kHidden; row += 4) {
      __m256 accumulators[4][4] = {
          {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()},
          {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()},
          {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()},
          {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()}};
      std::size_t input_dimension = 0;
      for (; input_dimension + 32 <= kDimension; input_dimension += 32) {
        const __m256 input0 = _mm256_loadu_ps(input_values + input_dimension);
        const __m256 input1 = _mm256_loadu_ps(input_values + input_dimension + 8);
        const __m256 input2 = _mm256_loadu_ps(input_values + input_dimension + 16);
        const __m256 input3 = _mm256_loadu_ps(input_values + input_dimension + 24);
        for (std::size_t lane = 0; lane < 4; ++lane) {
          const float* weight_row = data.fc1_weight.data() + (row + lane) * kDimension + input_dimension;
          accumulators[lane][0] = _mm256_add_ps(accumulators[lane][0], _mm256_mul_ps(
              _mm256_loadu_ps(weight_row), input0));
          accumulators[lane][1] = _mm256_add_ps(accumulators[lane][1], _mm256_mul_ps(
              _mm256_loadu_ps(weight_row + 8), input1));
          accumulators[lane][2] = _mm256_add_ps(accumulators[lane][2], _mm256_mul_ps(
              _mm256_loadu_ps(weight_row + 16), input2));
          accumulators[lane][3] = _mm256_add_ps(accumulators[lane][3], _mm256_mul_ps(
              _mm256_loadu_ps(weight_row + 24), input3));
        }
      }
      for (; input_dimension + 8 <= kDimension; input_dimension += 8) {
        const __m256 input_vector = _mm256_loadu_ps(input_values + input_dimension);
        for (std::size_t lane = 0; lane < 4; ++lane) {
          accumulators[lane][0] = _mm256_add_ps(accumulators[lane][0], _mm256_mul_ps(
              _mm256_loadu_ps(data.fc1_weight.data() + (row + lane) * kDimension + input_dimension), input_vector));
        }
      }
      float hidden_values[4] = {
          depth_bias[row], depth_bias[row + 1], depth_bias[row + 2], depth_bias[row + 3]};
      for (std::size_t lane = 0; lane < 4; ++lane) {
        hidden_values[lane] += horizontal_sum(accumulators[lane][0]) + horizontal_sum(accumulators[lane][1]) +
            horizontal_sum(accumulators[lane][2]) + horizontal_sum(accumulators[lane][3]);
      }
      for (; input_dimension < kDimension; ++input_dimension) {
        const float input_value = input_values[input_dimension];
        for (std::size_t lane = 0; lane < 4; ++lane) {
          hidden_values[lane] += data.fc1_weight[(row + lane) * kDimension + input_dimension] * input_value;
        }
      }
      for (std::size_t lane = 0; lane < 4; ++lane) output_values[row + lane] = hidden_values[lane];
    }
    for (; row < kHidden; ++row) {
      float hidden = depth_bias[row];
      for (std::size_t input_dimension = 0; input_dimension < kDimension; ++input_dimension) {
        hidden += data.fc1_weight[row * kDimension + input_dimension] * input_values[input_dimension];
      }
      output_values[row] = hidden;
    }
  }
}

void avx2_redesigned_dinput(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* dy = data.dinput_dy.data() + example * kDimension;
    float* dx = outputs.dinput.data() + example * kHidden;
    for (std::size_t output = 0; output < kDimension; ++output) {
      const __m256 dy_value = _mm256_set1_ps(dy[output]);
      const float* weight_row = data.fc2_weight.data() + output * kHidden;
      for (std::size_t input = 0; input < kHidden; input += kVectorWidth) {
        const __m256 value = _mm256_add_ps(
            _mm256_loadu_ps(dx + input),
            _mm256_mul_ps(_mm256_loadu_ps(weight_row + input), dy_value));
        _mm256_storeu_ps(dx + input, value);
      }
    }
  }
}

void avx2_redesigned_dweight(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    for (std::size_t input = 0; input < kDimension; input += kVectorWidth) {
      const __m256 x = _mm256_loadu_ps(data.x.data() + example * kDimension + input);
      for (std::size_t row = 0; row < kHidden; ++row) {
        float* gradient = outputs.dweight.data() + row * kDimension + input;
        const __m256 contribution = _mm256_mul_ps(x, _mm256_set1_ps(data.dy[example * kHidden + row]));
        _mm256_storeu_ps(gradient, _mm256_add_ps(_mm256_loadu_ps(gradient), contribution));
      }
    }
  }
}

void avx2_redesigned_dbias(const Data& data, Outputs& outputs) {
  for (std::size_t row = 0; row < kHidden; row += kVectorWidth) {
    __m256 value = _mm256_setzero_ps();
    for (std::size_t example = 0; example < kExamples; ++example) {
      value = _mm256_add_ps(value, _mm256_loadu_ps(data.dy.data() + example * kHidden + row));
    }
    _mm256_storeu_ps(outputs.dbias.data() + row, value);
  }
}

#else
#error "omega_fc1_microbenchmark requires AVX2"
#endif

void run_operation(Operation operation, Implementation implementation, bool redesigned, const Data& data, Outputs& outputs) {
  if (implementation == Implementation::kScalar) {
    switch (operation) {
      case Operation::kForward:
        scalar_forward(data, outputs);
        return;
      case Operation::kInput:
        scalar_dinput(data, outputs);
        return;
      case Operation::kWeight:
        scalar_dweight(data, outputs);
        return;
      case Operation::kBias:
        scalar_dbias(data, outputs);
        return;
      case Operation::kComposite:
        scalar_forward(data, outputs);
        scalar_dinput(data, outputs);
        scalar_dweight(data, outputs);
        scalar_dbias(data, outputs);
        return;
    }
  }

  switch (operation) {
    case Operation::kForward:
      redesigned ? avx2_redesigned_forward(data, outputs) : avx2_current_forward(data, outputs);
      return;
    case Operation::kInput:
      redesigned ? avx2_redesigned_dinput(data, outputs) : avx2_current_dinput(data, outputs);
      return;
    case Operation::kWeight:
      redesigned ? avx2_redesigned_dweight(data, outputs) : avx2_current_dweight(data, outputs);
      return;
    case Operation::kBias:
      redesigned ? avx2_redesigned_dbias(data, outputs) : avx2_current_dbias(data, outputs);
      return;
    case Operation::kComposite:
      redesigned ? avx2_redesigned_forward(data, outputs) : avx2_current_forward(data, outputs);
      redesigned ? avx2_redesigned_dinput(data, outputs) : avx2_current_dinput(data, outputs);
      redesigned ? avx2_redesigned_dweight(data, outputs) : avx2_current_dweight(data, outputs);
      redesigned ? avx2_redesigned_dbias(data, outputs) : avx2_current_dbias(data, outputs);
      return;
  }
}

OMEGA_NOINLINE void consume_outputs(const Outputs& outputs) {
  float sum = 0.0F;
  for (float value : outputs.forward) sum += value;
  for (float value : outputs.dinput) sum += value;
  for (float value : outputs.dweight) sum += value;
  for (float value : outputs.dbias) sum += value;
  g_sink += sum;
}

struct Statistics {
  double mean_ns = 0.0;
  double median_ns = 0.0;
};

Statistics summarize(std::vector<double> samples) {
  std::sort(samples.begin(), samples.end());
  double total = 0.0;
  for (double sample : samples) total += sample;
  const std::size_t middle = samples.size() / 2;
  const double median = (samples.size() % 2 == 0) ? (samples[middle - 1] + samples[middle]) * 0.5 : samples[middle];
  return {total / static_cast<double>(samples.size()), median};
}

Statistics measure(Operation operation, Implementation implementation, bool redesigned, const Data& data, Outputs& outputs,
    int warmup, int repetitions) {
  for (int iteration = 0; iteration < warmup; ++iteration) {
    clear_outputs(operation, outputs);
    run_operation(operation, implementation, redesigned, data, outputs);
    consume_outputs(outputs);
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repetitions));
  for (int iteration = 0; iteration < repetitions; ++iteration) {
    clear_outputs(operation, outputs);
    const auto start = std::chrono::steady_clock::now();
    run_operation(operation, implementation, redesigned, data, outputs);
    const auto finish = std::chrono::steady_clock::now();
    consume_outputs(outputs);
    samples.push_back(std::chrono::duration<double, std::nano>(finish - start).count());
  }
  return summarize(std::move(samples));
}

struct Comparison {
  bool finite = true;
  bool close = true;
  float max_abs = 0.0F;
  float max_rel = 0.0F;
};

Comparison compare(const std::vector<float>& expected, const std::vector<float>& actual) {
  Comparison result;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const float lhs = expected[index];
    const float rhs = actual[index];
    result.finite = result.finite && std::isfinite(lhs) && std::isfinite(rhs);
    const float absolute = std::fabs(lhs - rhs);
    const float relative = absolute / std::max(1.0F, std::fabs(lhs));
    result.max_abs = std::max(result.max_abs, absolute);
    result.max_rel = std::max(result.max_rel, relative);
  }
  result.close = result.finite && result.max_abs <= 2.0e-4F && result.max_rel <= 2.0e-5F;
  return result;
}

void fill_data(Data& data) {
  for (std::size_t index = 0; index < data.fc1_weight.size(); ++index) {
    data.fc1_weight[index] = 0.0025F * static_cast<float>(static_cast<int>(index % 23) - 11);
  }
  for (std::size_t index = 0; index < data.fc1_bias.size(); ++index) {
    data.fc1_bias[index] = 0.001F * static_cast<float>(static_cast<int>(index % 13) - 6);
  }
  for (std::size_t index = 0; index < data.fc2_weight.size(); ++index) {
    data.fc2_weight[index] = 0.002F * static_cast<float>(static_cast<int>(index % 19) - 9);
  }
  for (std::size_t index = 0; index < data.x.size(); ++index) {
    data.x[index] = 0.01F * static_cast<float>(static_cast<int>(index % 29) - 14);
  }
  for (std::size_t index = 0; index < data.dy.size(); ++index) {
    data.dy[index] = 0.01F * static_cast<float>(static_cast<int>(index % 31) - 15);
  }
  for (std::size_t index = 0; index < data.dinput_dy.size(); ++index) {
    data.dinput_dy[index] = 0.01F * static_cast<float>(static_cast<int>(index % 17) - 8);
  }
}

Outputs make_outputs() {
  return {
      std::vector<float>(kExamples * kHidden),
      std::vector<float>(kExamples * kHidden),
      std::vector<float>(kHidden * kDimension),
      std::vector<float>(kHidden),
  };
}

std::uint64_t hash_floats(const float* values, std::size_t count) {
  const auto* bytes = reinterpret_cast<const unsigned char*>(values);
  std::uint64_t hash = 14695981039346656037ULL;
  for (std::size_t index = 0; index < count * sizeof(float); ++index) {
    hash ^= bytes[index];
    hash *= 1099511628211ULL;
  }
  return hash;
}

void consume_forward_m8(const float* values) {
  float sum = 0.0F;
  for (std::size_t index = 0; index < 8 * kHidden; ++index) sum += values[index];
  g_sink += sum;
}

int run_step1_m8(const Data& data, int warmup, int repetitions, int trial) {
  Outputs control = make_outputs();
  Outputs candidate = make_outputs();
  std::vector<float> round_depth_embedding(kDimension);
  std::vector<float> round_depth_bias(kHidden);
  for (std::size_t input = 0; input < kDimension; ++input) {
    round_depth_embedding[input] = 0.002F * static_cast<float>(static_cast<int>(input % 17) - 8);
  }
  for (std::size_t row = 0; row < kHidden; ++row) {
    float value = data.fc1_bias[row];
    for (std::size_t input = 0; input < kDimension; ++input) {
      value += data.fc1_weight[row * kDimension + input] * round_depth_embedding[input];
    }
    round_depth_bias[row] = value;
  }
  bool distinct_inputs = true;
  for (std::size_t left = 0; left < 8; ++left) {
    for (std::size_t right = left + 1; right < 8; ++right) {
      bool equal = true;
      for (std::size_t input = 0; input < kDimension; ++input) {
        equal = equal && data.x[left * kDimension + input] == data.x[right * kDimension + input];
      }
      distinct_inputs = distinct_inputs && !equal;
    }
  }
  std::cout << std::setprecision(9)
            << "{\"record\":\"metadata\",\"benchmark\":\"fc1_forward_m8_step1\",\"trial\":" << trial
            << ",\"D\":" << kDimension << ",\"H\":" << kHidden << ",\"M\":8"
            << ",\"microtile\":\"M2xN4\",\"weights_reused_across_slot_rows\":true"
            << ",\"distinct_slot_inputs\":" << (distinct_inputs ? "true" : "false")
            << ",\"layout\":\"x[M,D],W[H,D],out[M,H]\",\"staging\":\"none; direct contiguous rows\""
            << ",\"bias\":\"round-0 depth_bias=b_fc1+W_fc1*depth_embedding; precomputed once outside timing, added once per output\""
            << ",\"dtype\":\"FP32\",\"fma\":false,\"threads\":1,\"warmup\":" << warmup
            << ",\"reps\":" << repetitions << ",\"timing_order\":\"alternating per pair\"}\n";
  if (!distinct_inputs) {
    std::cerr << "step1 requires eight distinct slot inputs\n";
    return 3;
  }

  avx2_control_m8(data, round_depth_bias.data(), control.forward.data());
  omega_fc1_m8::forward(data.x.data(), data.fc1_weight.data(), round_depth_bias.data(), candidate.forward.data());
  bool all_finite = true;
  bool all_close = true;
  bool all_exact = true;
  for (std::size_t slot = 0; slot < 8; ++slot) {
    float max_abs = 0.0F;
    float max_rel = 0.0F;
    bool finite = true;
    bool close = true;
    for (std::size_t output = 0; output < kHidden; ++output) {
      const float expected = control.forward[slot * kHidden + output];
      const float actual = candidate.forward[slot * kHidden + output];
      finite = finite && std::isfinite(expected) && std::isfinite(actual);
      const float absolute = std::fabs(expected - actual);
      const float relative = absolute / std::max(1.0F, std::fabs(expected));
      max_abs = std::max(max_abs, absolute);
      max_rel = std::max(max_rel, relative);
      all_exact = all_exact && expected == actual;
      close = close && absolute <= 2.0e-4F && relative <= 2.0e-5F;
    }
    all_finite = all_finite && finite;
    all_close = all_close && close;
    std::cout << std::setprecision(9)
              << "{\"record\":\"slot_verification\",\"trial\":" << trial << ",\"slot\":" << slot
              << ",\"input_fnv1a64\":" << hash_floats(data.x.data() + slot * kDimension, kDimension)
              << ",\"control_fnv1a64\":" << hash_floats(control.forward.data() + slot * kHidden, kHidden)
              << ",\"candidate_fnv1a64\":" << hash_floats(candidate.forward.data() + slot * kHidden, kHidden)
              << ",\"finite\":" << (finite ? "true" : "false")
              << ",\"close_existing_tolerance\":" << (close ? "true" : "false")
              << ",\"bitwise_equal\":" << (std::memcmp(control.forward.data() + slot * kHidden,
                  candidate.forward.data() + slot * kHidden, kHidden * sizeof(float)) == 0 ? "true" : "false")
              << ",\"max_abs\":" << max_abs << ",\"max_rel\":" << max_rel << "}\n";
  }
  if (!all_finite || !all_close) {
    std::cout << "{\"record\":\"result\",\"trial\":" << trial
              << ",\"status\":\"STOP_CORRECTNESS\",\"finite\":" << (all_finite ? "true" : "false")
              << ",\"all_slots_close\":false}\n";
    return 4;
  }

  for (int iteration = 0; iteration < warmup; ++iteration) {
    if ((iteration & 1) == 0) {
      avx2_control_m8(data, round_depth_bias.data(), control.forward.data());
      consume_forward_m8(control.forward.data());
      omega_fc1_m8::forward(data.x.data(), data.fc1_weight.data(), round_depth_bias.data(), candidate.forward.data());
      consume_forward_m8(candidate.forward.data());
    } else {
      omega_fc1_m8::forward(data.x.data(), data.fc1_weight.data(), round_depth_bias.data(), candidate.forward.data());
      consume_forward_m8(candidate.forward.data());
      avx2_control_m8(data, round_depth_bias.data(), control.forward.data());
      consume_forward_m8(control.forward.data());
    }
  }

  std::vector<double> control_samples;
  std::vector<double> candidate_samples;
  std::vector<double> reductions;
  control_samples.reserve(static_cast<std::size_t>(repetitions));
  candidate_samples.reserve(static_cast<std::size_t>(repetitions));
  reductions.reserve(static_cast<std::size_t>(repetitions));
  for (int iteration = 0; iteration < repetitions; ++iteration) {
    double control_ns = 0.0;
    double candidate_ns = 0.0;
    if ((iteration & 1) == 0) {
      auto start = std::chrono::steady_clock::now();
      avx2_control_m8(data, round_depth_bias.data(), control.forward.data());
      auto finish = std::chrono::steady_clock::now();
      control_ns = std::chrono::duration<double, std::nano>(finish - start).count();
      consume_forward_m8(control.forward.data());
      start = std::chrono::steady_clock::now();
      omega_fc1_m8::forward(data.x.data(), data.fc1_weight.data(), round_depth_bias.data(), candidate.forward.data());
      finish = std::chrono::steady_clock::now();
      candidate_ns = std::chrono::duration<double, std::nano>(finish - start).count();
      consume_forward_m8(candidate.forward.data());
    } else {
      auto start = std::chrono::steady_clock::now();
      omega_fc1_m8::forward(data.x.data(), data.fc1_weight.data(), round_depth_bias.data(), candidate.forward.data());
      auto finish = std::chrono::steady_clock::now();
      candidate_ns = std::chrono::duration<double, std::nano>(finish - start).count();
      consume_forward_m8(candidate.forward.data());
      start = std::chrono::steady_clock::now();
      avx2_control_m8(data, round_depth_bias.data(), control.forward.data());
      finish = std::chrono::steady_clock::now();
      control_ns = std::chrono::duration<double, std::nano>(finish - start).count();
      consume_forward_m8(control.forward.data());
    }
    control_samples.push_back(control_ns);
    candidate_samples.push_back(candidate_ns);
    reductions.push_back(100.0 * (control_ns - candidate_ns) / control_ns);
  }

  const Statistics control_stats = summarize(control_samples);
  const Statistics candidate_stats = summarize(candidate_samples);
  const Statistics reduction_stats = summarize(reductions);
  const std::size_t favorable_pairs = static_cast<std::size_t>(std::count_if(
      reductions.begin(), reductions.end(), [](double reduction) { return reduction > 0.0; }));
  auto print_samples = [trial](const char* label, const std::vector<double>& values) {
    std::cout << "{\"record\":\"raw_samples\",\"trial\":" << trial << ",\"series\":\"" << label << "\",\"ns\":[";
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0) std::cout << ',';
      std::cout << std::setprecision(9) << values[index];
    }
    std::cout << "]}\n";
  };
  print_samples("accepted_avx2_m1_x8", control_samples);
  print_samples("candidate_avx2_m8_reuse", candidate_samples);
  print_samples("paired_reduction_percent", reductions);
  std::cout << std::setprecision(9)
            << "{\"record\":\"summary\",\"trial\":" << trial
            << ",\"status\":\"PASS\",\"verification_exact\":" << (all_exact ? "true" : "false")
            << ",\"accepted_m1_median_ns\":" << control_stats.median_ns
            << ",\"candidate_m8_median_ns\":" << candidate_stats.median_ns
            << ",\"median_reduction_percent\":" << reduction_stats.median_ns
            << ",\"favorable_pairs\":" << favorable_pairs << ",\"pairs\":" << repetitions
            << ",\"sink\":" << g_sink << "}\n";
  return 0;
}

bool parse_positive(const char* text, int& value) {
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) return false;
  value = static_cast<int>(parsed);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string variant = "current";
  int repetitions = 30;
  int warmup = 5;
  int trial = 1;
  bool step1_m8 = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--step1-m8") {
      step1_m8 = true;
    } else if (argument == "--trial" && index + 1 < argc) {
      if (!parse_positive(argv[++index], trial)) {
        std::cerr << "invalid --trial\n";
        return 2;
      }
    } else if (argument == "--variant" && index + 1 < argc) {
      variant = argv[++index];
    } else if (argument == "--reps" && index + 1 < argc) {
      if (!parse_positive(argv[++index], repetitions)) {
        std::cerr << "invalid --reps\n";
        return 2;
      }
    } else if (argument == "--warmup" && index + 1 < argc) {
      if (!parse_positive(argv[++index], warmup)) {
        std::cerr << "invalid --warmup\n";
        return 2;
      }
    } else if (argument == "--help") {
      std::cout << "usage: omega_fc1_microbenchmark [--variant current|redesigned] [--reps N] [--warmup N] [--step1-m8 --trial N]\n";
      return 0;
    } else {
      std::cerr << "usage: omega_fc1_microbenchmark [--variant current|redesigned] [--reps N] [--warmup N] [--step1-m8 --trial N]\n";
      return 2;
    }
  }
  if (variant != "current" && variant != "redesigned") {
    std::cerr << "invalid --variant: " << variant << "\n";
    return 2;
  }

  const bool redesigned = variant == "redesigned";
  Data data{
      std::vector<float>(kHidden * kDimension),
      std::vector<float>(kHidden),
      std::vector<float>(kDimension * kHidden),
      std::vector<float>(kExamples * kDimension),
      std::vector<float>(kExamples * kHidden),
      std::vector<float>(kExamples * kDimension),
  };
  fill_data(data);

  if (step1_m8) return run_step1_m8(data, warmup, repetitions, trial);

  Outputs scalar_outputs = make_outputs();
  Outputs avx_outputs = make_outputs();
  const Operation operations[] = {Operation::kForward, Operation::kInput, Operation::kWeight, Operation::kBias, Operation::kComposite};
  bool verification_passed = true;
  for (Operation operation : operations) {
    clear_outputs(operation, scalar_outputs);
    clear_outputs(operation, avx_outputs);
    run_operation(operation, Implementation::kScalar, redesigned, data, scalar_outputs);
    run_operation(operation, Implementation::kAvx2, redesigned, data, avx_outputs);
    consume_outputs(scalar_outputs);
    consume_outputs(avx_outputs);

    const Comparison forward = compare(scalar_outputs.forward, avx_outputs.forward);
    const Comparison dinput = compare(scalar_outputs.dinput, avx_outputs.dinput);
    const Comparison dweight = compare(scalar_outputs.dweight, avx_outputs.dweight);
    const Comparison dbias = compare(scalar_outputs.dbias, avx_outputs.dbias);
    Comparison composite{forward.finite && dinput.finite && dweight.finite && dbias.finite,
        forward.close && dinput.close && dweight.close && dbias.close,
        std::max(std::max(forward.max_abs, dinput.max_abs), std::max(dweight.max_abs, dbias.max_abs)),
        std::max(std::max(forward.max_rel, dinput.max_rel), std::max(dweight.max_rel, dbias.max_rel))};
    const Comparison* selected = &composite;
    switch (operation) {
      case Operation::kForward:
        selected = &forward;
        break;
      case Operation::kInput:
        selected = &dinput;
        break;
      case Operation::kWeight:
        selected = &dweight;
        break;
      case Operation::kBias:
        selected = &dbias;
        break;
      case Operation::kComposite:
        break;
    }
    const bool passed = selected->finite && selected->close;
    verification_passed = verification_passed && passed;
    std::cout << std::setprecision(9)
              << "{\"record\":\"verification\",\"operation\":\"" << operation_name(operation)
              << "\",\"finite\":" << (selected->finite ? "true" : "false")
              << ",\"close\":" << (passed ? "true" : "false")
              << ",\"max_abs\":" << selected->max_abs << ",\"max_rel\":" << selected->max_rel << "}\n";
  }

  std::cout << "{\"record\":\"metadata\",\"benchmark\":\"fc1_microbenchmark\",\"variant\":\"" << variant
            << "\",\"D\":128,\"H\":512,\"examples\":64,\"warmup\":" << warmup << ",\"reps\":" << repetitions
            << ",\"threads\":1,\"layout\":\"fc1[H,D],fc2[D,H]\"}\n";
  for (Implementation implementation : {Implementation::kScalar, Implementation::kAvx2}) {
    for (Operation operation : operations) {
      const Statistics stats = measure(operation, implementation, redesigned, data, avx_outputs, warmup, repetitions);
      std::cout << std::setprecision(9)
                << "{\"record\":\"timing\",\"implementation\":\""
                << (implementation == Implementation::kScalar ? "scalar" : "avx2")
                << "\",\"variant\":\"" << variant << "\",\"operation\":\"" << operation_name(operation)
                << "\",\"warmup\":" << warmup << ",\"reps\":" << repetitions
                << ",\"mean_ns\":" << stats.mean_ns << ",\"median_ns\":" << stats.median_ns << "}\n";
    }
  }
  std::cout << std::setprecision(9) << "{\"record\":\"result\",\"verification_passed\":"
            << (verification_passed ? "true" : "false") << ",\"sink\":" << g_sink << "}\n";
  return verification_passed ? 0 : 1;
}

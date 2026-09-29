#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
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
constexpr std::size_t kExamples = 64;
constexpr std::size_t kVectorWidth = 8;
constexpr float kSqrtTwo = 1.41421356237309504880F;

volatile float g_sink = 0.0F;

float gelu(float value) {
  return 0.5F * value * (1.0F + std::erf(value / kSqrtTwo));
}

#if defined(_MSC_VER)
#define OMEGA_NOINLINE __declspec(noinline)
#else
#define OMEGA_NOINLINE __attribute__((noinline))
#endif

struct Data {
  std::vector<float> fc2_weight;  // [D, H].
  std::vector<float> fc1_pre;     // [examples, H], pre-GELU.
  std::vector<float> x;            // [examples, H], GELU output.
  std::vector<float> dy;           // [examples, D].
};

struct Outputs {
  std::vector<float> forward;  // [examples, D].
  std::vector<float> dinput;   // [examples, H].
  std::vector<float> dweight;  // [D, H].
  std::vector<float> dbias;    // [D].
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
    const float* x = data.x.data() + example * kHidden;
    float* output = outputs.forward.data() + example * kDimension;
    for (std::size_t d = 0; d < kDimension; ++d) {
      float value = 0.0F;
      const float* weight = data.fc2_weight.data() + d * kHidden;
      for (std::size_t row = 0; row < kHidden; ++row) {
        value += weight[row] * x[row];
      }
      output[d] = value;
    }
  }
}

void scalar_dinput(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* dy = data.dy.data() + example * kDimension;
    float* dx = outputs.dinput.data() + example * kHidden;
    for (std::size_t row = 0; row < kHidden; ++row) {
      float value = 0.0F;
      for (std::size_t d = 0; d < kDimension; ++d) {
        value += data.fc2_weight[d * kHidden + row] * dy[d];
      }
      dx[row] = value;
    }
  }
}

void scalar_dweight(const Data& data, Outputs& outputs) {
  std::vector<float> post_gelu(kHidden);
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* fc1_pre = data.fc1_pre.data() + example * kHidden;
    const float* dy = data.dy.data() + example * kDimension;
    for (std::size_t row = 0; row < kHidden; ++row) {
      post_gelu[row] = gelu(fc1_pre[row]);
    }
    for (std::size_t d = 0; d < kDimension; ++d) {
      float* gradient = outputs.dweight.data() + d * kHidden;
      for (std::size_t row = 0; row < kHidden; ++row) {
        gradient[row] += dy[d] * post_gelu[row];
      }
    }
  }
}

void scalar_dbias(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* dy = data.dy.data() + example * kDimension;
    for (std::size_t d = 0; d < kDimension; ++d) {
      outputs.dbias[d] += dy[d];
    }
  }
}

#if defined(__AVX2__) || defined(_M_AVX2)

float horizontal_sum(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

void avx2_forward(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* x = data.x.data() + example * kHidden;
    float* output = outputs.forward.data() + example * kDimension;
    std::size_t d = 0;
    for (; d + 4 <= kDimension; d += 4) {
      __m256 accumulators[4][4] = {
          {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()},
          {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()},
          {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()},
          {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()}};
      std::size_t row = 0;
      for (; row + 32 <= kHidden; row += 32) {
        const __m256 input0 = _mm256_loadu_ps(x + row);
        const __m256 input1 = _mm256_loadu_ps(x + row + 8);
        const __m256 input2 = _mm256_loadu_ps(x + row + 16);
        const __m256 input3 = _mm256_loadu_ps(x + row + 24);
        for (std::size_t output_index = 0; output_index < 4; ++output_index) {
          const float* weight = data.fc2_weight.data() + (d + output_index) * kHidden + row;
          accumulators[output_index][0] = _mm256_add_ps(accumulators[output_index][0], _mm256_mul_ps(
              _mm256_loadu_ps(weight), input0));
          accumulators[output_index][1] = _mm256_add_ps(accumulators[output_index][1], _mm256_mul_ps(
              _mm256_loadu_ps(weight + 8), input1));
          accumulators[output_index][2] = _mm256_add_ps(accumulators[output_index][2], _mm256_mul_ps(
              _mm256_loadu_ps(weight + 16), input2));
          accumulators[output_index][3] = _mm256_add_ps(accumulators[output_index][3], _mm256_mul_ps(
              _mm256_loadu_ps(weight + 24), input3));
        }
      }
      for (; row + 8 <= kHidden; row += 8) {
        const __m256 input = _mm256_loadu_ps(x + row);
        for (std::size_t output_index = 0; output_index < 4; ++output_index) {
          accumulators[output_index][0] = _mm256_add_ps(accumulators[output_index][0], _mm256_mul_ps(
              _mm256_loadu_ps(data.fc2_weight.data() + (d + output_index) * kHidden + row), input));
        }
      }
      for (std::size_t output_index = 0; output_index < 4; ++output_index) {
        output[d + output_index] = horizontal_sum(accumulators[output_index][0]) +
            horizontal_sum(accumulators[output_index][1]) + horizontal_sum(accumulators[output_index][2]) +
            horizontal_sum(accumulators[output_index][3]);
      }
    }
    for (; d < kDimension; ++d) {
      float value = 0.0F;
      const float* weight = data.fc2_weight.data() + d * kHidden;
      for (std::size_t row = 0; row < kHidden; ++row) {
        value += weight[row] * x[row];
      }
      output[d] = value;
    }
  }
}

void avx2_dinput(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* dy = data.dy.data() + example * kDimension;
    float* dx = outputs.dinput.data() + example * kHidden;
    for (std::size_t d = 0; d < kDimension; ++d) {
      const __m256 dy_value = _mm256_set1_ps(dy[d]);
      const float* weight = data.fc2_weight.data() + d * kHidden;
      std::size_t row = 0;
      for (; row + kVectorWidth <= kHidden; row += kVectorWidth) {
        _mm256_storeu_ps(dx + row, _mm256_add_ps(_mm256_loadu_ps(dx + row), _mm256_mul_ps(
            _mm256_loadu_ps(weight + row), dy_value)));
      }
      for (; row < kHidden; ++row) {
        dx[row] += weight[row] * dy[d];
      }
    }
  }
}

void avx2_dweight(const Data& data, Outputs& outputs) {
  std::vector<float> post_gelu(kHidden);
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* fc1_pre = data.fc1_pre.data() + example * kHidden;
    const float* dy = data.dy.data() + example * kDimension;
    for (std::size_t row = 0; row < kHidden; ++row) {
      post_gelu[row] = gelu(fc1_pre[row]);
    }
    for (std::size_t d = 0; d < kDimension; ++d) {
      const __m256 dy_value = _mm256_set1_ps(dy[d]);
      std::size_t row = 0;
      for (; row + kVectorWidth <= kHidden; row += kVectorWidth) {
        const __m256 x_values = _mm256_loadu_ps(post_gelu.data() + row);
        float* gradient = outputs.dweight.data() + d * kHidden + row;
        _mm256_storeu_ps(gradient, _mm256_add_ps(_mm256_loadu_ps(gradient), _mm256_mul_ps(
            x_values, dy_value)));
      }
      for (; row < kHidden; ++row) {
        outputs.dweight[d * kHidden + row] += dy[d] * post_gelu[row];
      }
    }
  }
}

void avx2_dbias(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* dy = data.dy.data() + example * kDimension;
    std::size_t d = 0;
    for (; d + kVectorWidth <= kDimension; d += kVectorWidth) {
      _mm256_storeu_ps(outputs.dbias.data() + d, _mm256_add_ps(
          _mm256_loadu_ps(outputs.dbias.data() + d), _mm256_loadu_ps(dy + d)));
    }
    for (; d < kDimension; ++d) {
      outputs.dbias[d] += dy[d];
    }
  }
}

#else
#error "omega_fc2_microbenchmark requires AVX2"
#endif

void run_operation(Operation operation, Implementation implementation, const Data& data, Outputs& outputs) {
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
      avx2_forward(data, outputs);
      return;
    case Operation::kInput:
      avx2_dinput(data, outputs);
      return;
    case Operation::kWeight:
      avx2_dweight(data, outputs);
      return;
    case Operation::kBias:
      avx2_dbias(data, outputs);
      return;
    case Operation::kComposite:
      avx2_forward(data, outputs);
      avx2_dinput(data, outputs);
      avx2_dweight(data, outputs);
      avx2_dbias(data, outputs);
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
  const double median = samples.size() % 2 == 0 ? (samples[middle - 1] + samples[middle]) * 0.5 : samples[middle];
  return {total / static_cast<double>(samples.size()), median};
}

Statistics measure(Operation operation, Implementation implementation, const Data& data, Outputs& outputs,
    int warmup, int repetitions) {
  for (int iteration = 0; iteration < warmup; ++iteration) {
    clear_outputs(operation, outputs);
    run_operation(operation, implementation, data, outputs);
    consume_outputs(outputs);
  }

  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(repetitions));
  for (int iteration = 0; iteration < repetitions; ++iteration) {
    clear_outputs(operation, outputs);
    const auto start = std::chrono::steady_clock::now();
    run_operation(operation, implementation, data, outputs);
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
  for (std::size_t index = 0; index < data.fc2_weight.size(); ++index) {
    data.fc2_weight[index] = 0.002F * static_cast<float>(static_cast<int>(index % 19) - 9);
  }
  for (std::size_t index = 0; index < data.fc1_pre.size(); ++index) {
    data.fc1_pre[index] = 0.01F * static_cast<float>(static_cast<int>(index % 29) - 14);
    data.x[index] = gelu(data.fc1_pre[index]);
  }
  for (std::size_t index = 0; index < data.dy.size(); ++index) {
    data.dy[index] = 0.01F * static_cast<float>(static_cast<int>(index % 31) - 15);
  }
}

Outputs make_outputs() {
  return {
      std::vector<float>(kExamples * kDimension),
      std::vector<float>(kExamples * kHidden),
      std::vector<float>(kDimension * kHidden),
      std::vector<float>(kDimension),
  };
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
  int repetitions = 30;
  int warmup = 5;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--reps" && index + 1 < argc) {
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
      std::cout << "usage: omega_fc2_microbenchmark [--reps N] [--warmup N]\n";
      return 0;
    } else {
      std::cerr << "usage: omega_fc2_microbenchmark [--reps N] [--warmup N]\n";
      return 2;
    }
  }

  Data data{
      std::vector<float>(kDimension * kHidden),
      std::vector<float>(kExamples * kHidden),
      std::vector<float>(kExamples * kHidden),
      std::vector<float>(kExamples * kDimension),
  };
  fill_data(data);
  Outputs scalar_outputs = make_outputs();
  Outputs avx_outputs = make_outputs();
  const Operation operations[] = {Operation::kForward, Operation::kInput, Operation::kWeight, Operation::kBias, Operation::kComposite};

  bool verification_passed = true;
  for (Operation operation : operations) {
    clear_outputs(operation, scalar_outputs);
    clear_outputs(operation, avx_outputs);
    run_operation(operation, Implementation::kScalar, data, scalar_outputs);
    run_operation(operation, Implementation::kAvx2, data, avx_outputs);
    consume_outputs(scalar_outputs);
    consume_outputs(avx_outputs);

    Comparison selected;
    switch (operation) {
      case Operation::kForward:
        selected = compare(scalar_outputs.forward, avx_outputs.forward);
        break;
      case Operation::kInput:
        selected = compare(scalar_outputs.dinput, avx_outputs.dinput);
        break;
      case Operation::kWeight:
        selected = compare(scalar_outputs.dweight, avx_outputs.dweight);
        break;
      case Operation::kBias:
        selected = compare(scalar_outputs.dbias, avx_outputs.dbias);
        break;
      case Operation::kComposite: {
        const Comparison forward = compare(scalar_outputs.forward, avx_outputs.forward);
        const Comparison dinput = compare(scalar_outputs.dinput, avx_outputs.dinput);
        const Comparison dweight = compare(scalar_outputs.dweight, avx_outputs.dweight);
        const Comparison dbias = compare(scalar_outputs.dbias, avx_outputs.dbias);
        selected.finite = forward.finite && dinput.finite && dweight.finite && dbias.finite;
        selected.close = forward.close && dinput.close && dweight.close && dbias.close;
        selected.max_abs = std::max(std::max(forward.max_abs, dinput.max_abs), std::max(dweight.max_abs, dbias.max_abs));
        selected.max_rel = std::max(std::max(forward.max_rel, dinput.max_rel), std::max(dweight.max_rel, dbias.max_rel));
        break;
      }
    }
    verification_passed = verification_passed && selected.finite && selected.close;
    std::cout << std::setprecision(9)
              << "{\"record\":\"verification\",\"operation\":\"" << operation_name(operation)
              << "\",\"finite\":" << (selected.finite ? "true" : "false")
              << ",\"close\":" << (selected.close ? "true" : "false")
              << ",\"max_abs\":" << selected.max_abs << ",\"max_rel\":" << selected.max_rel << "}\n";
  }

  std::cout << std::setprecision(9)
            << "{\"record\":\"metadata\",\"benchmark\":\"fc2_microbenchmark\",\"D\":128,\"H\":512,\"examples\":64,\"warmup\":"
            << warmup << ",\"reps\":" << repetitions << ",\"threads\":1,\"layout\":\"fc2[D,H],x[examples,H],dy[examples,D]\"}\n";
  for (Implementation implementation : {Implementation::kScalar, Implementation::kAvx2}) {
    for (Operation operation : operations) {
      const Statistics stats = measure(operation, implementation, data, avx_outputs, warmup, repetitions);
      std::cout << std::setprecision(9)
                << "{\"record\":\"timing\",\"implementation\":\""
                << (implementation == Implementation::kScalar ? "scalar" : "avx2")
                << "\",\"operation\":\"" << operation_name(operation) << "\",\"warmup\":" << warmup
                << ",\"reps\":" << repetitions << ",\"mean_ns\":" << stats.mean_ns
                << ",\"median_ns\":" << stats.median_ns << "}\n";
    }
  }
  std::cout << std::setprecision(9) << "{\"record\":\"result\",\"verification_passed\":"
            << (verification_passed ? "true" : "false") << ",\"sink\":" << g_sink << "}\n";
  return verification_passed ? 0 : 1;
}

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
constexpr std::size_t kOutputs = 3 * kDimension;
constexpr std::size_t kExamples = 64;
constexpr std::size_t kVectorWidth = 8;

volatile float g_sink = 0.0F;

#if defined(_MSC_VER)
#define OMEGA_NOINLINE __declspec(noinline)
#else
#define OMEGA_NOINLINE __attribute__((noinline))
#endif

struct Data {
  std::vector<float> weight;     // [3D, D].
  std::vector<float> bias;       // [3D].
  std::vector<float> candidate;  // [examples, D].
  std::vector<float> anchor;     // [examples, D].
  std::vector<float> input;      // [examples, D], candidate + anchor.
  std::vector<float> dy;         // [examples, 3D].
};

struct Outputs {
  std::vector<float> forward;  // [examples, 3D].
  std::vector<float> dinput;   // [examples, D].
  std::vector<float> dweight;  // [3D, D].
  std::vector<float> dbias;    // [3D].
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
    const float* input = data.input.data() + example * kDimension;
    float* output = outputs.forward.data() + example * kOutputs;
    for (std::size_t row = 0; row < kOutputs; ++row) {
      float value = data.bias[row];
      const float* weight = data.weight.data() + row * kDimension;
      for (std::size_t input_index = 0; input_index < kDimension; ++input_index) {
        value += weight[input_index] * input[input_index];
      }
      output[row] = value;
    }
  }
}

void scalar_dinput(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* dy = data.dy.data() + example * kOutputs;
    float* dinput = outputs.dinput.data() + example * kDimension;
    for (std::size_t row = 0; row < kOutputs; ++row) {
      const float output_gradient = dy[row];
      const float* weight = data.weight.data() + row * kDimension;
      for (std::size_t input = 0; input < kDimension; ++input) {
        dinput[input] += weight[input] * output_gradient;
      }
    }
  }
}

void scalar_dweight(const Data& data, Outputs& outputs) {
  for (std::size_t row = 0; row < kOutputs; ++row) {
    float* gradient = outputs.dweight.data() + row * kDimension;
    for (std::size_t input = 0; input < kDimension; ++input) {
      for (std::size_t example = 0; example < kExamples; ++example) {
        gradient[input] += data.dy[example * kOutputs + row] * data.input[example * kDimension + input];
      }
    }
  }
}

void scalar_dbias(const Data& data, Outputs& outputs) {
  for (std::size_t row = 0; row < kOutputs; ++row) {
    for (std::size_t example = 0; example < kExamples; ++example) {
      outputs.dbias[row] += data.dy[example * kOutputs + row];
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
    const float* input = data.input.data() + example * kDimension;
    float* output = outputs.forward.data() + example * kOutputs;
    std::size_t row = 0;
    for (; row + 4 <= kOutputs; row += 4) {
      __m256 accumulators[4] = {
          _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
      std::size_t input_index = 0;
      for (; input_index + kVectorWidth <= kDimension; input_index += kVectorWidth) {
        const __m256 input_values = _mm256_loadu_ps(input + input_index);
        for (std::size_t output_index = 0; output_index < 4; ++output_index) {
          accumulators[output_index] = _mm256_add_ps(accumulators[output_index], _mm256_mul_ps(
              _mm256_loadu_ps(data.weight.data() + (row + output_index) * kDimension + input_index), input_values));
        }
      }
      float values[4] = {data.bias[row], data.bias[row + 1], data.bias[row + 2], data.bias[row + 3]};
      for (std::size_t output_index = 0; output_index < 4; ++output_index) {
        values[output_index] += horizontal_sum(accumulators[output_index]);
      }
      for (; input_index < kDimension; ++input_index) {
        const float input_value = input[input_index];
        for (std::size_t output_index = 0; output_index < 4; ++output_index) {
          values[output_index] += data.weight[(row + output_index) * kDimension + input_index] * input_value;
        }
      }
      for (std::size_t output_index = 0; output_index < 4; ++output_index) {
        output[row + output_index] = values[output_index];
      }
    }
    for (; row < kOutputs; ++row) {
      float value = data.bias[row];
      const float* weight = data.weight.data() + row * kDimension;
      for (std::size_t input_index = 0; input_index < kDimension; ++input_index) {
        value += weight[input_index] * input[input_index];
      }
      output[row] = value;
    }
  }
}

void avx2_dinput(const Data& data, Outputs& outputs) {
  for (std::size_t example = 0; example < kExamples; ++example) {
    const float* dy = data.dy.data() + example * kOutputs;
    float* dinput = outputs.dinput.data() + example * kDimension;
    for (std::size_t row = 0; row < kOutputs; ++row) {
      const __m256 output_gradient = _mm256_set1_ps(dy[row]);
      const float* weight = data.weight.data() + row * kDimension;
      std::size_t input = 0;
      for (; input + kVectorWidth <= kDimension; input += kVectorWidth) {
        _mm256_storeu_ps(dinput + input, _mm256_add_ps(_mm256_loadu_ps(dinput + input), _mm256_mul_ps(
            _mm256_loadu_ps(weight + input), output_gradient)));
      }
      for (; input < kDimension; ++input) {
        dinput[input] += weight[input] * dy[row];
      }
    }
  }
}

void avx2_dweight(const Data& data, Outputs& outputs) {
  for (std::size_t row = 0; row < kOutputs; ++row) {
    float* gradient = outputs.dweight.data() + row * kDimension;
    for (std::size_t input = 0; input < kDimension; input += kVectorWidth) {
      for (std::size_t example = 0; example < kExamples; ++example) {
        const __m256 input_values = _mm256_loadu_ps(data.input.data() + example * kDimension + input);
        const __m256 output_gradient = _mm256_set1_ps(data.dy[example * kOutputs + row]);
        _mm256_storeu_ps(gradient + input, _mm256_add_ps(_mm256_loadu_ps(gradient + input),
            _mm256_mul_ps(input_values, output_gradient)));
      }
    }
  }
}

void avx2_dbias(const Data& data, Outputs& outputs) {
  for (std::size_t row = 0; row < kOutputs; row += kVectorWidth) {
    __m256 value = _mm256_loadu_ps(outputs.dbias.data() + row);
    for (std::size_t example = 0; example < kExamples; ++example) {
      value = _mm256_add_ps(value, _mm256_loadu_ps(data.dy.data() + example * kOutputs + row));
    }
    _mm256_storeu_ps(outputs.dbias.data() + row, value);
  }
}

#else
#error "omega_qkv_microbenchmark requires AVX2"
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
  for (std::size_t index = 0; index < data.weight.size(); ++index) {
    data.weight[index] = 0.002F * static_cast<float>(static_cast<int>(index % 19) - 9);
  }
  for (std::size_t index = 0; index < data.bias.size(); ++index) {
    data.bias[index] = 0.001F * static_cast<float>(static_cast<int>(index % 13) - 6);
  }
  for (std::size_t index = 0; index < data.candidate.size(); ++index) {
    data.candidate[index] = 0.01F * static_cast<float>(static_cast<int>(index % 29) - 14);
    data.anchor[index] = 0.008F * static_cast<float>(static_cast<int>(index % 23) - 11);
    data.input[index] = data.candidate[index] + data.anchor[index];
  }
  for (std::size_t index = 0; index < data.dy.size(); ++index) {
    data.dy[index] = 0.01F * static_cast<float>(static_cast<int>(index % 31) - 15);
  }
}

Outputs make_outputs() {
  return {
      std::vector<float>(kExamples * kOutputs),
      std::vector<float>(kExamples * kDimension),
      std::vector<float>(kOutputs * kDimension),
      std::vector<float>(kOutputs),
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
      std::cout << "usage: omega_qkv_microbenchmark [--reps N] [--warmup N]\n";
      return 0;
    } else {
      std::cerr << "usage: omega_qkv_microbenchmark [--reps N] [--warmup N]\n";
      return 2;
    }
  }

  Data data{
      std::vector<float>(kOutputs * kDimension),
      std::vector<float>(kOutputs),
      std::vector<float>(kExamples * kDimension),
      std::vector<float>(kExamples * kDimension),
      std::vector<float>(kExamples * kDimension),
      std::vector<float>(kExamples * kOutputs),
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
            << "{\"record\":\"metadata\",\"benchmark\":\"qkv_microbenchmark\",\"D\":128,\"outputs\":384,\"examples\":64,\"warmup\":"
            << warmup << ",\"reps\":" << repetitions << ",\"threads\":1,\"layout\":\"weight[3D,D],candidate+anchor->input[examples,D],dy[examples,3D]\"}\n";
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

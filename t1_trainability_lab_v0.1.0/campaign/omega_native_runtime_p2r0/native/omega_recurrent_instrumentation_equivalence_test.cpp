#include "omega_recurrent.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace {

constexpr size_t kBatch = 8;
constexpr size_t kSequenceLength = 256;
constexpr size_t kSlots = 8;
constexpr size_t kDimension = 128;

struct GoldenCase {
  std::vector<std::vector<float>> tensors;
};

struct ModeResult {
  size_t workspace_bytes = 0;
  std::array<std::vector<float>, 15> gradients;
};

size_t state_count() { return kBatch * kSlots * kDimension; }
size_t readout_count() { return kBatch * kSequenceLength * kSlots * kDimension; }

std::vector<size_t> tensor_counts(size_t rounds) {
  const size_t state = state_count();
  const size_t four_dimension = 4 * kDimension;
  return {
      kBatch * kSequenceLength * kSlots * kDimension, state, kSlots * kDimension * kDimension,
      kSlots * kDimension, 3 * kDimension * kDimension, 3 * kDimension, kDimension * kDimension,
      kDimension, four_dimension * kDimension, four_dimension, kDimension * four_dimension, kDimension,
      kDimension, rounds * kDimension, rounds * kDimension, state, readout_count(), readout_count(), state,
      kBatch * kSequenceLength * kSlots * kDimension, state, kSlots * kDimension * kDimension, kSlots * kDimension,
      3 * kDimension * kDimension, 3 * kDimension, kDimension * kDimension, kDimension,
      four_dimension * kDimension, four_dimension, kDimension * four_dimension, kDimension, kDimension,
      rounds * kDimension, rounds * kDimension,
  };
}

bool load_golden(const std::filesystem::path& path, size_t rounds, GoldenCase* result) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return false;
  std::uint32_t header_length = 0;
  input.read(reinterpret_cast<char*>(&header_length), sizeof(header_length));
  std::string header(header_length, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!input || header.find("\"magic\":\"OMEGA-P2R0-GOLDEN\"") == std::string::npos) return false;
  result->tensors.clear();
  for (size_t count : tensor_counts(rounds)) {
    std::vector<float> tensor(count);
    input.read(reinterpret_cast<char*>(tensor.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!input) return false;
    result->tensors.push_back(std::move(tensor));
  }
  char extra = 0;
  return !input.read(&extra, 1);
}

ModeResult run_mode(const GoldenCase& golden, size_t rounds, int instrumentation) {
  const auto& t = golden.tensors;
  const OmegaRecurrentConfig config{kSequenceLength, kBatch, kSlots, kDimension, rounds, 1, instrumentation};
  const OmegaMatrixViewF32 state_part_weight{
      t[2].data(), kSlots * kDimension, kDimension, static_cast<ptrdiff_t>(kDimension)};
  const OmegaRecurrentParams params{
      state_part_weight, t[3].data(), t[4].data(), t[5].data(), t[6].data(), t[7].data(), t[8].data(),
      t[9].data(), t[10].data(), t[11].data(), t[12].data(), t[13].data(), t[14].data(),
  };
  ModeResult result;
  result.workspace_bytes = omega_recurrent_workspace_bytes(config);
  std::vector<unsigned char> workspace(result.workspace_bytes);
  std::vector<float> next_state(state_count());
  std::vector<float> readout_states(readout_count());
  if (omega_recurrent_forward(&config, &params, t[0].data(), t[1].data(), next_state.data(), readout_states.data(), workspace.data(), workspace.size()) != 0) {
    throw std::runtime_error("forward failed");
  }

  std::vector<float> d_token_part(t[0].size());
  std::vector<float> d_previous_state(t[1].size());
  std::vector<float> d_state_part_weight(t[2].size());
  std::vector<float> d_prelude_norm_weight(t[3].size());
  std::vector<float> d_block_qkv_weight(t[4].size());
  std::vector<float> d_block_qkv_bias(t[5].size());
  std::vector<float> d_block_out_weight(t[6].size());
  std::vector<float> d_block_out_bias(t[7].size());
  std::vector<float> d_block_fc1_weight(t[8].size());
  std::vector<float> d_block_fc1_bias(t[9].size());
  std::vector<float> d_block_fc2_weight(t[10].size());
  std::vector<float> d_block_fc2_bias(t[11].size());
  std::vector<float> d_block_norm_weight(t[12].size());
  std::vector<float> d_depth_embedding(t[13].size());
  std::vector<float> d_gate_logits(t[14].size());
  std::array<std::vector<float>, 15> sums;
  std::array<std::vector<size_t>, 15> counts;
  for (size_t index = 0; index < 15; ++index) {
    if (instrumentation != 0) {
      sums[index].resize(t[index].size());
      counts[index].resize(t[index].size());
    }
  }
  std::vector<double> fp64(t[13].size());
  std::vector<size_t> levels(t[13].size());
  OmegaRecurrentGrads grads{
      d_token_part.data(), d_previous_state.data(), d_state_part_weight.data(), d_prelude_norm_weight.data(),
      d_block_qkv_weight.data(), d_block_qkv_bias.data(), d_block_out_weight.data(), d_block_out_bias.data(),
      d_block_fc1_weight.data(), d_block_fc1_bias.data(), d_block_fc2_weight.data(), d_block_fc2_bias.data(),
      d_block_norm_weight.data(), d_depth_embedding.data(), d_gate_logits.data(),
      instrumentation != 0 ? sums[0].data() : nullptr, instrumentation != 0 ? sums[1].data() : nullptr,
      instrumentation != 0 ? sums[2].data() : nullptr, instrumentation != 0 ? sums[3].data() : nullptr,
      instrumentation != 0 ? sums[4].data() : nullptr, instrumentation != 0 ? sums[5].data() : nullptr,
      instrumentation != 0 ? sums[6].data() : nullptr, instrumentation != 0 ? sums[7].data() : nullptr,
      instrumentation != 0 ? sums[8].data() : nullptr, instrumentation != 0 ? sums[9].data() : nullptr,
      instrumentation != 0 ? sums[10].data() : nullptr, instrumentation != 0 ? sums[11].data() : nullptr,
      instrumentation != 0 ? sums[12].data() : nullptr, instrumentation != 0 ? sums[13].data() : nullptr,
      instrumentation != 0 ? sums[14].data() : nullptr,
      instrumentation != 0 ? counts[0].data() : nullptr, instrumentation != 0 ? counts[1].data() : nullptr,
      instrumentation != 0 ? counts[2].data() : nullptr, instrumentation != 0 ? counts[3].data() : nullptr,
      instrumentation != 0 ? counts[4].data() : nullptr, instrumentation != 0 ? counts[5].data() : nullptr,
      instrumentation != 0 ? counts[6].data() : nullptr, instrumentation != 0 ? counts[7].data() : nullptr,
      instrumentation != 0 ? counts[8].data() : nullptr, instrumentation != 0 ? counts[9].data() : nullptr,
      instrumentation != 0 ? counts[10].data() : nullptr, instrumentation != 0 ? counts[11].data() : nullptr,
      instrumentation != 0 ? counts[12].data() : nullptr, instrumentation != 0 ? counts[13].data() : nullptr,
      instrumentation != 0 ? counts[14].data() : nullptr,
      instrumentation != 0 ? fp64.data() : nullptr, instrumentation != 0 ? levels.data() : nullptr,
  };
  if (omega_recurrent_backward(
          &config, &params, t[0].data(), t[17].data(), t[18].data(), workspace.data(), workspace.size(),
          &grads) != 0) {
    throw std::runtime_error("backward failed");
  }
  result.gradients = {std::move(d_token_part), std::move(d_previous_state), std::move(d_state_part_weight),
      std::move(d_prelude_norm_weight), std::move(d_block_qkv_weight), std::move(d_block_qkv_bias),
      std::move(d_block_out_weight), std::move(d_block_out_bias), std::move(d_block_fc1_weight),
      std::move(d_block_fc1_bias), std::move(d_block_fc2_weight), std::move(d_block_fc2_bias),
      std::move(d_block_norm_weight), std::move(d_depth_embedding), std::move(d_gate_logits)};
  return result;
}

bool compare_bytes(const ModeResult& instrumented, const ModeResult& clean) {
  for (size_t index = 0; index < instrumented.gradients.size(); ++index) {
    const auto& left = instrumented.gradients[index];
    const auto& right = clean.gradients[index];
    if (left.size() != right.size() || std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) != 0) return false;
  }
  return true;
}

#if defined(_WIN32)
constexpr size_t kDllBatch = 2;
constexpr size_t kDllSequenceLength = 3;
constexpr size_t kDllSlots = 2;
constexpr size_t kDllDimension = 8;
constexpr float kSgdLearningRate = 0.03125F;

struct DllInputs {
  OmegaRecurrentConfig config{};
  std::vector<float> token_part;
  std::vector<float> previous_state;
  std::vector<float> d_readout_states;
  std::vector<float> d_next_state;
  std::array<std::vector<float>, 13> parameters;
};

struct DllResult {
  std::vector<float> next_state;
  std::vector<float> readout_states;
  std::array<std::vector<float>, 15> gradients;
  std::array<std::vector<float>, 13> parameters_after_sgd;
  std::vector<float> sum_abs_d_block_fc1_weight;
  std::vector<size_t> count_d_block_fc1_weight;
  std::vector<float> sum_abs_d_depth_embedding;
  std::vector<size_t> count_d_depth_embedding;
  std::vector<double> fp64_d_depth_embedding;
};

std::vector<float> make_positive_values(size_t count, float base, float step, size_t seed) {
  std::vector<float> values(count);
  for (size_t index = 0; index < count; ++index) {
    const size_t residue = (index * 13 + seed * 7) % 29;
    values[index] = base + static_cast<float>(residue) * step;
  }
  return values;
}

std::array<size_t, 13> dll_parameter_counts(size_t rounds) {
  return {kDllSlots * kDllDimension * kDllDimension, kDllSlots * kDllDimension,
      3 * kDllDimension * kDllDimension, 3 * kDllDimension, kDllDimension * kDllDimension,
      kDllDimension, 4 * kDllDimension * kDllDimension, 4 * kDllDimension,
      kDllDimension * 4 * kDllDimension, kDllDimension, kDllDimension,
      rounds * kDllDimension, rounds * kDllDimension};
}

DllInputs make_dll_inputs(size_t rounds) {
  DllInputs inputs;
  inputs.config = {kDllSequenceLength, kDllBatch, kDllSlots, kDllDimension, rounds, 1, 0};
  const size_t state_count = kDllBatch * kDllSlots * kDllDimension;
  const size_t sequence_state_count = kDllBatch * kDllSequenceLength * kDllSlots * kDllDimension;
  const size_t readout_count = kDllBatch * kDllSequenceLength * kDllSlots * kDllDimension;
  inputs.token_part = make_positive_values(sequence_state_count, 0.02F, 0.0004F, 1);
  inputs.previous_state = make_positive_values(state_count, 0.03F, 0.0003F, 2);
  inputs.d_readout_states = make_positive_values(readout_count, 0.01F, 0.0002F, 3);
  inputs.d_next_state = make_positive_values(state_count, 0.015F, 0.0002F, 4);

  const auto parameter_counts = dll_parameter_counts(rounds);
  for (size_t parameter = 0; parameter < inputs.parameters.size(); ++parameter) {
    const float base = parameter == 10 ? 0.9F : 0.002F;
    const float step = parameter == 10 ? 0.001F : 0.00003F;
    inputs.parameters[parameter] = make_positive_values(parameter_counts[parameter], base, step, parameter + 5);
  }
  return inputs;
}

OmegaRecurrentParams make_dll_params(const DllInputs& inputs) {
  const auto& p = inputs.parameters;
  return {{p[0].data(), kDllSlots * kDllDimension, kDllDimension, static_cast<ptrdiff_t>(kDllDimension)},
      p[1].data(), p[2].data(), p[3].data(), p[4].data(), p[5].data(), p[6].data(), p[7].data(),
      p[8].data(), p[9].data(), p[10].data(), p[11].data(), p[12].data()};
}

struct DllApi {
  using WorkspaceBytes = size_t (*)(OmegaRecurrentConfig);
  using Forward = int (*)(const OmegaRecurrentConfig*, const OmegaRecurrentParams*, const float*, const float*,
      float*, float*, void*, size_t);
  using Backward = int (*)(const OmegaRecurrentConfig*, const OmegaRecurrentParams*, const float*, const float*,
      const float*, void*, size_t, OmegaRecurrentGrads*);

  explicit DllApi(const std::filesystem::path& path) : module(LoadLibraryW(path.c_str())) {
    if (module == nullptr) throw std::runtime_error("LoadLibraryW failed for " + path.string());
    workspace_bytes = load<WorkspaceBytes>("omega_recurrent_workspace_bytes");
    forward = load<Forward>("omega_recurrent_forward");
    backward = load<Backward>("omega_recurrent_backward");
  }

  ~DllApi() {
    if (module != nullptr) FreeLibrary(module);
  }

  DllApi(const DllApi&) = delete;
  DllApi& operator=(const DllApi&) = delete;

  WorkspaceBytes workspace_bytes = nullptr;
  Forward forward = nullptr;
  Backward backward = nullptr;

 private:
  template <typename Function>
  Function load(const char* name) {
    const FARPROC symbol = GetProcAddress(module, name);
    if (symbol == nullptr) throw std::runtime_error(std::string("GetProcAddress failed for ") + name);
    return reinterpret_cast<Function>(symbol);
  }

  HMODULE module = nullptr;
};

DllResult run_dll_case(const DllApi& api, const DllInputs& inputs, size_t diagnostic_mask) {
  DllResult result;
  const auto& config = inputs.config;
  const size_t state_count = kDllBatch * kDllSlots * kDllDimension;
  const size_t readout_count = kDllBatch * kDllSequenceLength * kDllSlots * kDllDimension;
  const size_t token_count = kDllBatch * kDllSequenceLength * kDllSlots * kDllDimension;
  const size_t depth_count = config.rounds * kDllDimension;
  const auto parameter_counts = dll_parameter_counts(config.rounds);
  const std::array<size_t, 15> gradient_counts = {token_count, state_count,
      parameter_counts[0], parameter_counts[1], parameter_counts[2], parameter_counts[3],
      parameter_counts[4], parameter_counts[5], parameter_counts[6], parameter_counts[7],
      parameter_counts[8], parameter_counts[9], parameter_counts[10], parameter_counts[11],
      parameter_counts[12]};
  for (size_t index = 0; index < result.gradients.size(); ++index) {
    result.gradients[index].resize(gradient_counts[index]);
  }

  result.next_state.resize(state_count);
  result.readout_states.resize(readout_count);
  const size_t workspace_bytes = api.workspace_bytes(config);
  if (workspace_bytes == 0) throw std::runtime_error("DLL returned zero workspace size");
  std::vector<unsigned char> workspace(workspace_bytes);
  const OmegaRecurrentParams params = make_dll_params(inputs);
  if (api.forward(&config, &params, inputs.token_part.data(), inputs.previous_state.data(),
          result.next_state.data(), result.readout_states.data(), workspace.data(), workspace.size()) != 0) {
    throw std::runtime_error("DLL forward failed");
  }

  if ((diagnostic_mask & (1U << 0)) != 0) result.sum_abs_d_block_fc1_weight.resize(parameter_counts[6]);
  if ((diagnostic_mask & (1U << 1)) != 0) result.count_d_block_fc1_weight.resize(parameter_counts[6]);
  if ((diagnostic_mask & (1U << 2)) != 0) result.sum_abs_d_depth_embedding.resize(depth_count);
  if ((diagnostic_mask & (1U << 3)) != 0) result.count_d_depth_embedding.resize(depth_count);
  if ((diagnostic_mask & (1U << 4)) != 0) result.fp64_d_depth_embedding.resize(depth_count);

  OmegaRecurrentGrads grads{};
  grads.d_token_part = result.gradients[0].data();
  grads.d_previous_state = result.gradients[1].data();
  grads.d_state_part_weight = result.gradients[2].data();
  grads.d_prelude_norm_weight = result.gradients[3].data();
  grads.d_block_qkv_weight = result.gradients[4].data();
  grads.d_block_qkv_bias = result.gradients[5].data();
  grads.d_block_out_weight = result.gradients[6].data();
  grads.d_block_out_bias = result.gradients[7].data();
  grads.d_block_fc1_weight = result.gradients[8].data();
  grads.d_block_fc1_bias = result.gradients[9].data();
  grads.d_block_fc2_weight = result.gradients[10].data();
  grads.d_block_fc2_bias = result.gradients[11].data();
  grads.d_block_norm_weight = result.gradients[12].data();
  grads.d_depth_embedding = result.gradients[13].data();
  grads.d_gate_logits = result.gradients[14].data();
  grads.sum_abs_d_block_fc1_weight = result.sum_abs_d_block_fc1_weight.empty()
      ? nullptr : result.sum_abs_d_block_fc1_weight.data();
  grads.count_d_block_fc1_weight = result.count_d_block_fc1_weight.empty()
      ? nullptr : result.count_d_block_fc1_weight.data();
  grads.sum_abs_d_depth_embedding = result.sum_abs_d_depth_embedding.empty()
      ? nullptr : result.sum_abs_d_depth_embedding.data();
  grads.count_d_depth_embedding = result.count_d_depth_embedding.empty()
      ? nullptr : result.count_d_depth_embedding.data();
  grads.fp64_d_depth_embedding = result.fp64_d_depth_embedding.empty()
      ? nullptr : result.fp64_d_depth_embedding.data();
  if (api.backward(&config, &params, inputs.token_part.data(), inputs.d_readout_states.data(),
          inputs.d_next_state.data(), workspace.data(), workspace.size(), &grads) != 0) {
    throw std::runtime_error("DLL backward failed");
  }

  result.parameters_after_sgd = inputs.parameters;
  for (size_t parameter = 0; parameter < result.parameters_after_sgd.size(); ++parameter) {
    const auto& gradient = result.gradients[parameter + 2];
    for (size_t index = 0; index < result.parameters_after_sgd[parameter].size(); ++index) {
      result.parameters_after_sgd[parameter][index] -= kSgdLearningRate * gradient[index];
    }
  }
  return result;
}

template <typename T>
bool vectors_equal_bytes(const std::vector<T>& left, const std::vector<T>& right) {
  return left.size() == right.size() &&
      std::memcmp(left.data(), right.data(), left.size() * sizeof(T)) == 0;
}

template <typename T>
bool has_nonzero(const std::vector<T>& values) {
  for (const T& value : values) {
    if (value != T{}) return true;
  }
  return false;
}

bool compare_dll_results(const DllResult& baseline, const DllResult& candidate,
    size_t diagnostic_mask, std::string* mismatch) {
  if (!vectors_equal_bytes(baseline.next_state, candidate.next_state)) {
    *mismatch = "next_state";
    return false;
  }
  if (!vectors_equal_bytes(baseline.readout_states, candidate.readout_states)) {
    *mismatch = "readout_states";
    return false;
  }
  for (size_t index = 0; index < baseline.gradients.size(); ++index) {
    if (!vectors_equal_bytes(baseline.gradients[index], candidate.gradients[index])) {
      *mismatch = "primary gradient " + std::to_string(index);
      return false;
    }
  }
  for (size_t index = 0; index < baseline.parameters_after_sgd.size(); ++index) {
    if (!vectors_equal_bytes(baseline.parameters_after_sgd[index], candidate.parameters_after_sgd[index])) {
      *mismatch = "SGD parameter " + std::to_string(index);
      return false;
    }
  }

  const std::array<const char*, 5> diagnostic_names = {"sum_abs_d_block_fc1_weight",
      "count_d_block_fc1_weight", "sum_abs_d_depth_embedding", "count_d_depth_embedding",
      "fp64_d_depth_embedding"};
  for (size_t index = 0; index < diagnostic_names.size(); ++index) {
    if ((diagnostic_mask & (size_t{1} << index)) == 0) continue;
    bool equal = false;
    bool exercised = false;
    switch (index) {
      case 0:
        equal = vectors_equal_bytes(baseline.sum_abs_d_block_fc1_weight, candidate.sum_abs_d_block_fc1_weight);
        exercised = has_nonzero(baseline.sum_abs_d_block_fc1_weight);
        break;
      case 1:
        equal = vectors_equal_bytes(baseline.count_d_block_fc1_weight, candidate.count_d_block_fc1_weight);
        exercised = has_nonzero(baseline.count_d_block_fc1_weight);
        break;
      case 2:
        equal = vectors_equal_bytes(baseline.sum_abs_d_depth_embedding, candidate.sum_abs_d_depth_embedding);
        exercised = has_nonzero(baseline.sum_abs_d_depth_embedding);
        break;
      case 3:
        equal = vectors_equal_bytes(baseline.count_d_depth_embedding, candidate.count_d_depth_embedding);
        exercised = has_nonzero(baseline.count_d_depth_embedding);
        break;
      case 4:
        equal = vectors_equal_bytes(baseline.fp64_d_depth_embedding, candidate.fp64_d_depth_embedding);
        exercised = has_nonzero(baseline.fp64_d_depth_embedding);
        break;
    }
    if (!exercised) {
      *mismatch = std::string("diagnostic not exercised: ") + diagnostic_names[index];
      return false;
    }
    if (!equal) {
      *mismatch = std::string("diagnostic mismatch: ") + diagnostic_names[index];
      return false;
    }
  }
  return true;
}
#endif

int run_dll_equivalence(const std::filesystem::path& baseline_path,
    const std::filesystem::path& candidate_path) {
#if !defined(_WIN32)
  (void)baseline_path;
  (void)candidate_path;
  std::cerr << "DLL equivalence mode requires Windows\n";
  return 2;
#else
  try {
    const DllApi baseline_api(baseline_path);
    const DllApi candidate_api(candidate_path);
    bool passed = true;
    const std::array<size_t, 2> round_cases = {1, 4};
    for (size_t rounds : round_cases) {
      const DllInputs inputs = make_dll_inputs(rounds);
      for (size_t mask = 0; mask < 32; ++mask) {
        const DllResult baseline = run_dll_case(baseline_api, inputs, mask);
        const DllResult candidate = run_dll_case(candidate_api, inputs, mask);
        std::string mismatch;
        const bool equal = compare_dll_results(baseline, candidate, mask, &mismatch);
        std::cout << "dll_equivalence K=" << rounds << " diagnostic_mask=" << mask
                  << " byte_exact=" << (equal ? "true" : "false");
        if (!equal) {
          std::cout << " mismatch=" << mismatch;
          passed = false;
        }
        std::cout << "\n";
      }
    }
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "DLL equivalence: " << error.what() << "\n";
    return 2;
  }
#endif
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--dll-equivalence") {
    if (argc != 4) {
      std::cerr << "usage: omega_recurrent_instrumentation_equivalence_test --dll-equivalence <baseline.dll> <candidate.dll>\n";
      return 2;
    }
    return run_dll_equivalence(std::filesystem::path(argv[2]), std::filesystem::path(argv[3]));
  }
  const std::filesystem::path root = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path("golden_cases");
  const std::array<std::pair<const char*, size_t>, 4> cases = {{
      {"golden_K1_window0.bin", 1}, {"golden_K1_window1.bin", 1},
      {"golden_K4_window0.bin", 4}, {"golden_K4_window1.bin", 4},
  }};
  bool passed = true;
  for (const auto& [name, rounds] : cases) {
    GoldenCase golden;
    if (!load_golden(root / name, rounds, &golden)) return 2;
    try {
      const ModeResult instrumented = run_mode(golden, rounds, 1);
      const ModeResult clean = run_mode(golden, rounds, 0);
      const bool equal = compare_bytes(instrumented, clean);
      std::cout << name << " workspace_instrumented=" << instrumented.workspace_bytes
                << " workspace_clean=" << clean.workspace_bytes
                << " gradient_bytes_identical=" << (equal ? "true" : "false") << "\n";
      passed = passed && equal;
    } catch (const std::exception& error) {
      std::cerr << name << ": " << error.what() << "\n";
      passed = false;
    }
  }
  return passed ? 0 : 1;
}

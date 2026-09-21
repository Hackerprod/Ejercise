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

}  // namespace

int main(int argc, char** argv) {
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

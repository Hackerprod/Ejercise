#include "omega_recurrent.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr size_t kBatch = 2;
constexpr size_t kSequenceLength = 3;
constexpr size_t kSlots = 2;
constexpr size_t kDimension = 3;
constexpr size_t kRounds = 2;
constexpr double kUnitRoundoff = 5.9604644775390625e-8;

size_t state_count() { return kBatch * kSlots * kDimension; }
size_t state_dimension() { return kSlots * kDimension; }
size_t readout_count() { return kBatch * kSequenceLength * state_dimension(); }

struct Inputs {
  std::vector<float> token_part = std::vector<float>(kBatch * kSequenceLength * state_dimension());
  std::vector<float> previous_state = std::vector<float>(state_count());
  std::vector<float> prelude_norm_weight = std::vector<float>(state_dimension());
  std::vector<float> block_qkv_weight = std::vector<float>(3 * kDimension * kDimension);
  std::vector<float> block_qkv_bias = std::vector<float>(3 * kDimension);
  std::vector<float> block_out_weight = std::vector<float>(kDimension * kDimension);
  std::vector<float> block_out_bias = std::vector<float>(kDimension);
  std::vector<float> block_fc1_weight = std::vector<float>(4 * kDimension * kDimension);
  std::vector<float> block_fc1_bias = std::vector<float>(4 * kDimension);
  std::vector<float> block_fc2_weight = std::vector<float>(kDimension * 4 * kDimension);
  std::vector<float> block_fc2_bias = std::vector<float>(kDimension);
  std::vector<float> block_norm_weight = std::vector<float>(kDimension);
  std::vector<float> depth_embedding = std::vector<float>(kRounds * kDimension);
  std::vector<float> gate_logits = std::vector<float>(kRounds * kDimension);
  std::vector<float> d_readout_states = std::vector<float>(readout_count());
  std::vector<float> d_next_state = std::vector<float>(state_count());
};

void fill_values(std::vector<float>* values, float scale) {
  for (size_t index = 0; index < values->size(); ++index) {
    (*values)[index] = scale * static_cast<float>((index % 11) + 1) / 11.0F;
  }
}

Inputs make_inputs() {
  Inputs inputs;
  fill_values(&inputs.token_part, 0.07F);
  fill_values(&inputs.previous_state, 0.05F);
  fill_values(&inputs.prelude_norm_weight, 0.03F);
  fill_values(&inputs.block_qkv_weight, 0.02F);
  fill_values(&inputs.block_qkv_bias, 0.01F);
  fill_values(&inputs.block_out_weight, 0.02F);
  fill_values(&inputs.block_out_bias, 0.01F);
  fill_values(&inputs.block_fc1_weight, 0.015F);
  fill_values(&inputs.block_fc1_bias, 0.01F);
  fill_values(&inputs.block_fc2_weight, 0.012F);
  fill_values(&inputs.block_fc2_bias, 0.01F);
  fill_values(&inputs.block_norm_weight, 0.03F);
  fill_values(&inputs.depth_embedding, 0.02F);
  fill_values(&inputs.gate_logits, 0.01F);
  fill_values(&inputs.d_readout_states, 0.04F);
  fill_values(&inputs.d_next_state, 0.05F);
  return inputs;
}

OmegaRecurrentParams make_params(const OmegaMatrixViewF32& state_part_weight, const Inputs& inputs) {
  return OmegaRecurrentParams{
      state_part_weight, inputs.prelude_norm_weight.data(), inputs.block_qkv_weight.data(), inputs.block_qkv_bias.data(),
      inputs.block_out_weight.data(), inputs.block_out_bias.data(), inputs.block_fc1_weight.data(), inputs.block_fc1_bias.data(),
      inputs.block_fc2_weight.data(), inputs.block_fc2_bias.data(), inputs.block_norm_weight.data(), inputs.depth_embedding.data(),
      inputs.gate_logits.data(),
  };
}

struct Result {
  std::array<std::vector<float>, 17> tensors;
};

Result run(const OmegaMatrixViewF32& state_part_weight, const Inputs& inputs) {
  const OmegaRecurrentConfig config{kSequenceLength, kBatch, kSlots, kDimension, kRounds, 1, 0};
  const OmegaRecurrentParams params = make_params(state_part_weight, inputs);
  const size_t workspace_bytes = omega_recurrent_workspace_bytes(config);
  if (workspace_bytes == 0) throw std::runtime_error("workspace size is zero");
  std::vector<unsigned char> workspace(workspace_bytes);
  std::vector<float> next_state(state_count());
  std::vector<float> readout_states(readout_count());
  if (omega_recurrent_forward(&config, &params, inputs.token_part.data(), inputs.previous_state.data(), next_state.data(),
          readout_states.data(), workspace.data(), workspace.size()) != 0) {
    throw std::runtime_error("forward failed");
  }

  std::array<std::vector<float>, 15> gradients;
  const std::array<size_t, 15> counts = {
      inputs.token_part.size(), inputs.previous_state.size(), kSlots * kDimension * kDimension, state_dimension(),
      3 * kDimension * kDimension, 3 * kDimension, kDimension * kDimension, kDimension, 4 * kDimension * kDimension,
      4 * kDimension, kDimension * 4 * kDimension, kDimension, kDimension, kRounds * kDimension, kRounds * kDimension,
  };
  for (size_t index = 0; index < gradients.size(); ++index) gradients[index].resize(counts[index]);
  OmegaRecurrentGrads grads{
      gradients[0].data(), gradients[1].data(), gradients[2].data(), gradients[3].data(), gradients[4].data(),
      gradients[5].data(), gradients[6].data(), gradients[7].data(), gradients[8].data(), gradients[9].data(),
      gradients[10].data(), gradients[11].data(), gradients[12].data(), gradients[13].data(), gradients[14].data(),
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr,
  };
  if (omega_recurrent_backward(&config, &params, inputs.token_part.data(), inputs.d_readout_states.data(), inputs.d_next_state.data(),
          workspace.data(), workspace.size(), &grads) != 0) {
    throw std::runtime_error("backward failed");
  }
  Result result;
  result.tensors[0] = std::move(next_state);
  result.tensors[1] = std::move(readout_states);
  for (size_t index = 0; index < gradients.size(); ++index) result.tensors[index + 2] = std::move(gradients[index]);
  return result;
}

bool compare_three_level(const char* name, const std::vector<float>& left, const std::vector<float>& right) {
  if (left.size() != right.size()) return false;
  size_t exact = 0;
  size_t normal = 0;
  size_t fallback = 0;
  for (size_t index = 0; index < left.size(); ++index) {
    if (std::memcmp(&left[index], &right[index], sizeof(float)) == 0) {
      ++exact;
      continue;
    }
    const double difference = std::fabs(static_cast<double>(left[index]) - static_cast<double>(right[index]));
    const double magnitude = std::fabs(static_cast<double>(right[index]));
    if (difference <= 1.0e-5 + 1.0e-4 * magnitude) {
      ++normal;
    } else if (difference <= 2.0 * kUnitRoundoff * std::max(1.0, magnitude)) {
      ++fallback;
    } else {
      std::cerr << name << " mismatch at " << index << " left=" << left[index] << " right=" << right[index] << "\n";
      return false;
    }
  }
  std::cout << name << " exact=" << exact << " normal=" << normal << " fallback=" << fallback << "\n";
  return true;
}

}  // namespace

int main() {
  try {
    const Inputs inputs = make_inputs();
    std::vector<float> packed_state_part(kSlots * kDimension * kDimension);
    fill_values(&packed_state_part, 0.025F);
    std::vector<float> strided_storage(kSlots * kDimension * 2 * kDimension, -9.0F);
    for (size_t row = 0; row < kSlots * kDimension; ++row) {
      std::memcpy(strided_storage.data() + row * 2 * kDimension + kDimension,
          packed_state_part.data() + row * kDimension, kDimension * sizeof(float));
    }
    const OmegaMatrixViewF32 packed_view{
        packed_state_part.data(), kSlots * kDimension, kDimension, static_cast<ptrdiff_t>(kDimension)};
    const OmegaMatrixViewF32 strided_view{
        strided_storage.data() + kDimension, kSlots * kDimension, kDimension, static_cast<ptrdiff_t>(2 * kDimension)};
    const Result packed = run(packed_view, inputs);
    const Result strided = run(strided_view, inputs);
    bool passed = true;
    for (size_t index = 0; index < packed.tensors.size(); ++index) {
      const char* name = index == 0 ? "forward_next_state" : index == 1 ? "forward_readout_states" : "backward_gradient";
      passed = compare_three_level(name, packed.tensors[index], strided.tensors[index]) && passed;
    }

    const OmegaRecurrentConfig config{kSequenceLength, kBatch, kSlots, kDimension, kRounds, 0, 0};
    std::vector<unsigned char> workspace(omega_recurrent_workspace_bytes(config));
    std::vector<float> next_state(state_count());
    std::vector<float> readout_states(readout_count());
    for (const ptrdiff_t invalid_stride : {static_cast<ptrdiff_t>(kDimension - 1), ptrdiff_t{-1}}) {
      const OmegaMatrixViewF32 invalid_view{
          packed_state_part.data(), kSlots * kDimension, kDimension, invalid_stride};
      const OmegaRecurrentParams invalid_params = make_params(invalid_view, inputs);
      const int invalid_status = omega_recurrent_forward(&config, &invalid_params, inputs.token_part.data(), inputs.previous_state.data(),
          next_state.data(), readout_states.data(), workspace.data(), workspace.size());
      if (invalid_status == 0) {
        std::cerr << "invalid row_stride was accepted: " << invalid_stride << "\n";
        passed = false;
      } else {
        std::cout << "invalid row_stride rejected stride=" << invalid_stride << " status=" << invalid_status << "\n";
      }
    }
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 2;
  }
}

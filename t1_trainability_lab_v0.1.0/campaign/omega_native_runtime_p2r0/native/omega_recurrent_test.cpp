#include "omega_recurrent.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
      kBatch * kSequenceLength * kSlots * kDimension, state, kSlots * kDimension * kDimension, kSlots * kDimension, 3 * kDimension * kDimension, 3 * kDimension,
      kDimension * kDimension, kDimension, four_dimension * kDimension, four_dimension, kDimension * four_dimension,
      kDimension, kDimension, rounds * kDimension, rounds * kDimension,
  };
}

bool contains(const std::string& value, const std::string& needle) {
  return value.find(needle) != std::string::npos;
}

bool load_golden(const std::filesystem::path& path, size_t rounds, GoldenCase* result, std::string* error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    *error = "cannot open " + path.string();
    return false;
  }
  std::uint32_t header_length = 0;
  input.read(reinterpret_cast<char*>(&header_length), sizeof(header_length));
  if (!input) {
    *error = "cannot read header length";
    return false;
  }
  std::string header(header_length, '\0');
  input.read(header.data(), static_cast<std::streamsize>(header.size()));
  if (!input || !contains(header, "\"magic\":\"OMEGA-P2R0-GOLDEN\"") || !contains(header, "\"format_version\":1")) {
    *error = "invalid P2-R0 JSON header";
    return false;
  }
  const std::vector<size_t> counts = tensor_counts(rounds);
  result->tensors.clear();
  result->tensors.reserve(counts.size());
  for (size_t count : counts) {
    std::vector<float> tensor(count);
    input.read(reinterpret_cast<char*>(tensor.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!input) {
      *error = "truncated tensor payload";
      return false;
    }
    result->tensors.push_back(std::move(tensor));
  }
  char extra = 0;
  if (input.read(&extra, 1)) {
    *error = "unexpected bytes after tensor payload";
    return false;
  }
  return true;
}

bool compare_tensor(const char* name, const std::vector<float>& actual, const std::vector<float>& expected, float tolerance) {
  if (actual.size() != expected.size()) {
    std::cerr << name << ": size mismatch\n";
    return false;
  }
  float maximum = 0.0F;
  size_t maximum_index = 0;
  for (size_t index = 0; index < actual.size(); ++index) {
    const float difference = std::fabs(actual[index] - expected[index]);
    if (difference > maximum) {
      maximum = difference;
      maximum_index = index;
    }
    if (!std::isfinite(actual[index]) || difference > tolerance) {
      std::cerr << name << ": mismatch at " << index << ", actual=" << actual[index] << ", expected=" << expected[index] << ", abs_diff=" << difference << "\n";
      return false;
    }
  }
  std::cout << "  " << name << ": max_abs_diff=" << maximum << " at " << maximum_index << "\n";
  return true;
}

struct GradientCheck {
  const char* name;
  std::vector<float>* actual;
  size_t expected_index;
};

struct GradientReport {
  double max_abs_error = 0.0;
  double max_relative_error = 0.0;
  double max_kappa = 0.0;
  double max_eta = 0.0;
  size_t normal_gate_fail_count = 0;
  size_t cancellation_fallback_count = 0;
  bool overall_pass = true;
};

constexpr double kGradientAtol = 1.0e-5;
constexpr double kGradientRtol = 1.0e-4;
constexpr double kFp32Epsilon = 1.1920929e-7;
constexpr double kCancellationKappaMinimum = 32.0;
constexpr double kCancellationEtaMaximum = 32.0 * kFp32Epsilon;

bool run_case(const std::filesystem::path& path, size_t rounds) {
  GoldenCase golden;
  std::string error;
  if (!load_golden(path, rounds, &golden, &error)) {
    std::cerr << "FAIL " << path.filename().string() << ": " << error << "\n";
    return false;
  }
  const OmegaRecurrentConfig config{kSequenceLength, kBatch, kSlots, kDimension, rounds, 0};
  const auto& t = golden.tensors;
  OmegaRecurrentParams params{
      t[2].data(), t[3].data(), t[4].data(), t[5].data(), t[6].data(), t[7].data(), t[8].data(),
      t[9].data(), t[10].data(), t[11].data(), t[12].data(), t[13].data(), t[14].data(),
  };
  const size_t workspace_bytes = omega_recurrent_workspace_bytes(config);
  if (workspace_bytes == 0) {
    std::cerr << "FAIL " << path.filename().string() << ": workspace size is zero\n";
    return false;
  }
  std::vector<unsigned char> workspace(workspace_bytes);
  std::vector<float> next_state(state_count());
  std::vector<float> readout_states(readout_count());
  const int status = omega_recurrent_forward(&config, &params, t[0].data(), t[1].data(), next_state.data(), readout_states.data(), workspace.data(), workspace.size());
  if (status != 0) {
    std::cerr << "FAIL " << path.filename().string() << ": forward returned " << status << "\n";
    return false;
  }
  bool passed = compare_tensor("next_state", next_state, t[15], 1.0e-5F) && compare_tensor("readout_states", readout_states, t[16], 1.0e-5F);

  const OmegaRecurrentConfig training_config{kSequenceLength, kBatch, kSlots, kDimension, rounds, 1};
  const size_t training_workspace_bytes = omega_recurrent_workspace_bytes(training_config);
  std::vector<unsigned char> training_workspace(training_workspace_bytes);
  std::vector<float> training_next_state(state_count());
  std::vector<float> training_readout_states(readout_count());
  const int training_status = omega_recurrent_forward(&training_config, &params, t[0].data(), t[1].data(), training_next_state.data(), training_readout_states.data(), training_workspace.data(), training_workspace.size());
  if (training_status != 0) {
    std::cerr << "FAIL " << path.filename().string() << ": training forward returned " << training_status << "\n";
    return false;
  }
  passed = compare_tensor("training_next_state", training_next_state, t[15], 1.0e-5F) && passed;
  passed = compare_tensor("training_readout_states", training_readout_states, t[16], 1.0e-5F) && passed;

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
  std::vector<float> sum_abs_d_token_part(t[0].size());
  std::vector<float> sum_abs_d_previous_state(t[1].size());
  std::vector<float> sum_abs_d_state_part_weight(t[2].size());
  std::vector<float> sum_abs_d_prelude_norm_weight(t[3].size());
  std::vector<float> sum_abs_d_block_qkv_weight(t[4].size());
  std::vector<float> sum_abs_d_block_qkv_bias(t[5].size());
  std::vector<float> sum_abs_d_block_out_weight(t[6].size());
  std::vector<float> sum_abs_d_block_out_bias(t[7].size());
  std::vector<float> sum_abs_d_block_fc1_weight(t[8].size());
  std::vector<float> sum_abs_d_block_fc1_bias(t[9].size());
  std::vector<float> sum_abs_d_block_fc2_weight(t[10].size());
  std::vector<float> sum_abs_d_block_fc2_bias(t[11].size());
  std::vector<float> sum_abs_d_block_norm_weight(t[12].size());
  std::vector<float> sum_abs_d_depth_embedding(t[13].size());
  std::vector<float> sum_abs_d_gate_logits(t[14].size());
  OmegaRecurrentGrads grads{
      d_token_part.data(), d_previous_state.data(), d_state_part_weight.data(), d_prelude_norm_weight.data(),
      d_block_qkv_weight.data(), d_block_qkv_bias.data(), d_block_out_weight.data(), d_block_out_bias.data(),
      d_block_fc1_weight.data(), d_block_fc1_bias.data(), d_block_fc2_weight.data(), d_block_fc2_bias.data(),
      d_block_norm_weight.data(), d_depth_embedding.data(), d_gate_logits.data(),
      sum_abs_d_token_part.data(), sum_abs_d_previous_state.data(), sum_abs_d_state_part_weight.data(), sum_abs_d_prelude_norm_weight.data(),
      sum_abs_d_block_qkv_weight.data(), sum_abs_d_block_qkv_bias.data(), sum_abs_d_block_out_weight.data(), sum_abs_d_block_out_bias.data(),
      sum_abs_d_block_fc1_weight.data(), sum_abs_d_block_fc1_bias.data(), sum_abs_d_block_fc2_weight.data(), sum_abs_d_block_fc2_bias.data(),
      sum_abs_d_block_norm_weight.data(), sum_abs_d_depth_embedding.data(), sum_abs_d_gate_logits.data(),
  };
  const int backward_status = omega_recurrent_backward(&training_config, &params, t[0].data(), t[17].data(), t[18].data(), training_workspace.data(), training_workspace.size(), &grads);
  if (backward_status != 0) {
    std::cerr << "FAIL " << path.filename().string() << ": backward returned " << backward_status << "\n";
    return false;
  }
  const std::vector<GradientCheck> gradient_checks = {
      {"d_token_part", &d_token_part, 19},
      {"d_previous_state", &d_previous_state, 20},
      {"d_state_part_weight", &d_state_part_weight, 21},
      {"d_prelude_norm_weight", &d_prelude_norm_weight, 22},
      {"d_block_qkv_weight", &d_block_qkv_weight, 23},
      {"d_block_qkv_bias", &d_block_qkv_bias, 24},
      {"d_block_out_weight", &d_block_out_weight, 25},
      {"d_block_out_bias", &d_block_out_bias, 26},
      {"d_block_fc1_weight", &d_block_fc1_weight, 27},
      {"d_block_fc1_bias", &d_block_fc1_bias, 28},
      {"d_block_fc2_weight", &d_block_fc2_weight, 29},
      {"d_block_fc2_bias", &d_block_fc2_bias, 30},
      {"d_block_norm_weight", &d_block_norm_weight, 31},
      {"d_depth_embedding", &d_depth_embedding, 32},
      {"d_gate_logits", &d_gate_logits, 33},
  };
  std::vector<GradientReport> reports(gradient_checks.size());
  std::vector<std::vector<size_t>> failed_indices(gradient_checks.size());
  for (size_t check_index = 0; check_index < gradient_checks.size(); ++check_index) {
    const GradientCheck& check = gradient_checks[check_index];
    const std::vector<float>& actual = *check.actual;
    const std::vector<float>& expected = t[check.expected_index];
    if (actual.size() != expected.size()) {
      reports[check_index].overall_pass = false;
      reports[check_index].normal_gate_fail_count = 1;
      continue;
    }
    for (size_t index = 0; index < actual.size(); ++index) {
      const double native_value = actual[index];
      const double reference_value = expected[index];
      const double absolute_error = std::fabs(native_value - reference_value);
      const double relative_error = absolute_error / std::max(std::fabs(reference_value), 1.0e-30);
      reports[check_index].max_abs_error = std::max(reports[check_index].max_abs_error, absolute_error);
      reports[check_index].max_relative_error = std::max(reports[check_index].max_relative_error, relative_error);
      const bool normal_pass = std::isfinite(native_value) &&
          absolute_error <= kGradientAtol + kGradientRtol * std::fabs(reference_value);
      if (!normal_pass) {
        ++reports[check_index].normal_gate_fail_count;
        failed_indices[check_index].push_back(index);
      }
    }
  }

  const std::array<std::vector<float>*, 15> sum_abs_contributions = {
      &sum_abs_d_token_part, &sum_abs_d_previous_state, &sum_abs_d_state_part_weight, &sum_abs_d_prelude_norm_weight,
      &sum_abs_d_block_qkv_weight, &sum_abs_d_block_qkv_bias, &sum_abs_d_block_out_weight, &sum_abs_d_block_out_bias,
      &sum_abs_d_block_fc1_weight, &sum_abs_d_block_fc1_bias, &sum_abs_d_block_fc2_weight, &sum_abs_d_block_fc2_bias,
      &sum_abs_d_block_norm_weight, &sum_abs_d_depth_embedding, &sum_abs_d_gate_logits,
  };

  std::cout << std::setprecision(9);
  for (size_t check_index = 0; check_index < gradient_checks.size(); ++check_index) {
    const GradientCheck& check = gradient_checks[check_index];
    GradientReport& report = reports[check_index];
    for (size_t index : failed_indices[check_index]) {
      const double reference_value = t[check.expected_index][index];
      const double native_value = (*check.actual)[index];
      const double absolute_error = std::fabs(native_value - reference_value);
      const double contribution_sum = (*sum_abs_contributions[check_index])[index];
      const double kappa = contribution_sum / std::max(std::fabs(reference_value), 1.0e-30);
      const double eta = absolute_error / std::max(contribution_sum, 1.0e-30);
      report.max_kappa = std::max(report.max_kappa, kappa);
      report.max_eta = std::max(report.max_eta, eta);
      const bool fallback_pass = kappa >= kCancellationKappaMinimum && eta <= kCancellationEtaMaximum;
      if (fallback_pass) {
        ++report.cancellation_fallback_count;
      } else {
        report.overall_pass = false;
      }
      std::cout << "    fallback " << check.name << " index=" << index << " reference=" << reference_value << " native=" << native_value
                << " abs_error=" << absolute_error << " sum_abs_contributions=" << contribution_sum << " kappa=" << kappa << " eta=" << eta
                << " pass=" << (fallback_pass ? "true" : "false") << "\n";
    }
    if (report.normal_gate_fail_count == 0) {
      report.overall_pass = true;
    }
    std::cout << "  backward " << check.name << ": max_abs_error=" << report.max_abs_error << " max_relative_error=" << report.max_relative_error
              << " normal_gate_fail_count=" << report.normal_gate_fail_count << " cancellation_fallback_count=" << report.cancellation_fallback_count
              << " max_kappa=" << report.max_kappa << " max_eta=" << report.max_eta << " overall_pass="
              << (report.overall_pass ? "true" : "false") << "\n";
    passed = report.overall_pass && passed;
  }
  std::cout << (passed ? "PASS " : "FAIL ") << path.filename().string() << "\n";
  return passed;
}

}  // namespace

int main(int argc, char** argv) {
  const std::filesystem::path root = argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path("golden_cases");
  const std::vector<std::pair<std::string, size_t>> cases = {
      {"golden_K1_window0.bin", 1}, {"golden_K1_window1.bin", 1}, {"golden_K4_window0.bin", 4}, {"golden_K4_window1.bin", 4},
  };
  bool passed = true;
  for (const auto& item : cases) {
    passed = run_case(root / item.first, item.second) && passed;
  }
  return passed ? 0 : 1;
}

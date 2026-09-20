#include "omega_recurrent.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
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

bool run_case(const std::filesystem::path& path, size_t rounds) {
  GoldenCase golden;
  std::string error;
  if (!load_golden(path, rounds, &golden, &error)) {
    std::cerr << "FAIL " << path.filename().string() << ": " << error << "\n";
    return false;
  }
  const OmegaRecurrentConfig config{kSequenceLength, kBatch, kSlots, kDimension, rounds};
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
  std::cout << "PASS " << path.filename().string() << "\n";
  return compare_tensor("next_state", next_state, t[15], 1.0e-5F) && compare_tensor("readout_states", readout_states, t[16], 1.0e-5F);
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

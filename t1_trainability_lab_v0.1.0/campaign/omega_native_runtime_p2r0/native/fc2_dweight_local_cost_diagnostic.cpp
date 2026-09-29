#include "omega_recurrent.cpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#if !OMEGA_HAS_AVX2
#error This diagnostic must compile with AVX2 enabled.
#endif

namespace {

constexpr size_t kGroupSlots = 8;
constexpr size_t kOutputs = 128;
constexpr size_t kHidden = 512;
constexpr size_t kMeasuredPairs = 30;
constexpr size_t kWarmupPairs = 5;
constexpr std::array<char, 8> kInputMagic = {'F', 'C', '2', 'M', '8', 'D', '1', '\0'};

using PreActivation = std::array<float, kGroupSlots * kHidden>;
using OutputGradient = std::array<float, kGroupSlots * kOutputs>;
using DWeight = std::array<float, kOutputs * kHidden>;
using Activated = std::array<float, kGroupSlots * kHidden>;

struct Group {
  std::string case_id;
  PreActivation preactivation{};
  Activated captured_activated{};
  OutputGradient output_gradient{};
  DWeight dweight_before{};
  DWeight expected_control{};
  DWeight expected_candidate{};
};

struct alignas(64) WorkBuffers {
  DWeight dweight{};
  std::array<float, kHidden> hidden_values{};
  Activated grouped_activated{};
  OutputGradient grouped_output_gradient{};
};

struct TimedPair {
  uint64_t control_ns = 0;
  uint64_t candidate_ns = 0;
  uint64_t control_hash = 0;
  uint64_t candidate_hash = 0;
  const char* order = nullptr;
};

template <size_t Count>
bool read_floats(std::ifstream& stream, std::array<float, Count>* values) {
  stream.read(reinterpret_cast<char*>(values->data()), static_cast<std::streamsize>(Count * sizeof(float)));
  return stream.good();
}

bool load_group(const std::string& path, Group* group) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) return false;
  std::array<char, kInputMagic.size()> magic{};
  stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!stream || magic != kInputMagic) return false;
  if (!read_floats(stream, &group->preactivation) || !read_floats(stream, &group->captured_activated) ||
      !read_floats(stream, &group->output_gradient) || !read_floats(stream, &group->dweight_before) ||
      !read_floats(stream, &group->expected_control) || !read_floats(stream, &group->expected_candidate)) {
    return false;
  }
  char trailing = 0;
  if (stream.read(&trailing, 1)) return false;
  const size_t marker = path.find_last_of("\\/");
  group->case_id = path.substr(marker == std::string::npos ? 0 : marker + 1);
  const size_t extension = group->case_id.find_last_of('.');
  if (extension != std::string::npos) group->case_id.resize(extension);
  return true;
}

uint64_t hash_dweight(const DWeight& values) {
  uint64_t hash = UINT64_C(14695981039346656037);
  const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
  for (size_t index = 0; index < values.size() * sizeof(float); ++index) {
    hash ^= bytes[index];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

bool equal_bits(const DWeight& left, const DWeight& right) {
  return std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) == 0;
}

void run_control_full(const Group& group, WorkBuffers* work) {
  for (size_t slot = 0; slot < kGroupSlots; ++slot) {
    const float* pre = group.preactivation.data() + slot * kHidden;
    const float* gradient = group.output_gradient.data() + slot * kOutputs;
    prepare_fc2_dweight_slot_avx2(pre, gradient, work->hidden_values.data(), nullptr, nullptr,
        slot, kOutputs, kHidden);
    accumulate_fc2_dweight_slot_avx2(work->hidden_values.data(), gradient, work->dweight.data(),
        nullptr, nullptr, kOutputs, kHidden);
  }
}

void run_candidate_full(const Group& group, WorkBuffers* work) {
  for (size_t slot = 0; slot < kGroupSlots; ++slot) {
    const float* pre = group.preactivation.data() + slot * kHidden;
    const float* gradient = group.output_gradient.data() + slot * kOutputs;
    prepare_fc2_dweight_slot_avx2(pre, gradient, work->hidden_values.data(),
        work->grouped_activated.data(), work->grouped_output_gradient.data(), slot, kOutputs, kHidden);
    accumulate_fc2_dweight_slot_diagnostics_avx2(work->hidden_values.data(), gradient,
        nullptr, nullptr, kOutputs, kHidden);
  }
  omega_fc2_local_reduction::accumulate_dweight_m8(
      work->grouped_activated.data(), work->grouped_output_gradient.data(), work->dweight.data(), kOutputs, kHidden);
}

void run_control_accumulation_only(const Group& group, WorkBuffers* work) {
  for (size_t slot = 0; slot < kGroupSlots; ++slot) {
    accumulate_fc2_dweight_slot_avx2(group.captured_activated.data() + slot * kHidden,
        group.output_gradient.data() + slot * kOutputs, work->dweight.data(), nullptr, nullptr, kOutputs, kHidden);
  }
}

void run_candidate_accumulation_only(const Group& group, WorkBuffers* work) {
  for (size_t slot = 0; slot < kGroupSlots; ++slot) {
    accumulate_fc2_dweight_slot_diagnostics_avx2(group.captured_activated.data() + slot * kHidden,
        group.output_gradient.data() + slot * kOutputs, nullptr, nullptr, kOutputs, kHidden);
  }
  omega_fc2_local_reduction::accumulate_dweight_m8(
      group.captured_activated.data(), group.output_gradient.data(), work->dweight.data(), kOutputs, kHidden);
}

template <typename Evaluation>
uint64_t measure_one(const Group& group, WorkBuffers* work, const DWeight& expected, Evaluation evaluation,
    uint64_t* output_hash) {
  std::memcpy(work->dweight.data(), group.dweight_before.data(), group.dweight_before.size() * sizeof(float));
  const auto started = std::chrono::steady_clock::now();
  evaluation();
  const auto ended = std::chrono::steady_clock::now();
  if (!equal_bits(work->dweight, expected)) throw std::runtime_error("timed kernel output differs from captured result");
  *output_hash = hash_dweight(work->dweight);
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(ended - started).count());
}

void validate_group(const Group& group) {
  bool initial_nonzero = false;
  for (float value : group.dweight_before) initial_nonzero = initial_nonzero || value != 0.0F;
  if (!initial_nonzero) throw std::runtime_error("dW_old must be nonzero for every capture group");

  WorkBuffers work{};
  std::memcpy(work.dweight.data(), group.dweight_before.data(), group.dweight_before.size() * sizeof(float));
  run_control_full(group, &work);
  if (!equal_bits(work.dweight, group.expected_control)) {
    throw std::runtime_error("control AVX2 helper does not reproduce captured production dW");
  }
  std::memcpy(work.dweight.data(), group.dweight_before.data(), group.dweight_before.size() * sizeof(float));
  run_candidate_full(group, &work);
  if (!equal_bits(work.dweight, group.expected_candidate)) {
    throw std::runtime_error("candidate full-cost path does not reproduce captured production dW");
  }
  if (!equal_bits(group.expected_control, group.expected_candidate)) {
    throw std::runtime_error("captured control/candidate results are not bitwise equal");
  }
  if (std::memcmp(work.grouped_activated.data(), group.captured_activated.data(),
          group.captured_activated.size() * sizeof(float)) != 0) {
    throw std::runtime_error("candidate GELU preparation differs from captured production activations");
  }
}

void run_mode(const Group& group, size_t group_index, size_t process_index, bool accumulation_only,
    std::vector<TimedPair>* measured) {
  WorkBuffers work{};
  const DWeight& expected = group.expected_control;
  auto control_eval = [&] {
    if (accumulation_only) run_control_accumulation_only(group, &work);
    else run_control_full(group, &work);
  };
  auto candidate_eval = [&] {
    if (accumulation_only) run_candidate_accumulation_only(group, &work);
    else run_candidate_full(group, &work);
  };

  for (size_t pair = 0; pair < kWarmupPairs; ++pair) {
    const bool control_first = ((pair + group_index + process_index) & 1U) == 0;
    uint64_t ignored_hash = 0;
    if (control_first) {
      (void)measure_one(group, &work, expected, control_eval, &ignored_hash);
      (void)measure_one(group, &work, expected, candidate_eval, &ignored_hash);
    } else {
      (void)measure_one(group, &work, expected, candidate_eval, &ignored_hash);
      (void)measure_one(group, &work, expected, control_eval, &ignored_hash);
    }
  }

  measured->clear();
  measured->reserve(kMeasuredPairs);
  for (size_t pair = 0; pair < kMeasuredPairs; ++pair) {
    const bool control_first = ((pair + group_index + process_index) & 1U) == 0;
    TimedPair sample{};
    if (control_first) {
      sample.order = "control_then_candidate";
      sample.control_ns = measure_one(group, &work, expected, control_eval, &sample.control_hash);
      sample.candidate_ns = measure_one(group, &work, expected, candidate_eval, &sample.candidate_hash);
    } else {
      sample.order = "candidate_then_control";
      sample.candidate_ns = measure_one(group, &work, expected, candidate_eval, &sample.candidate_hash);
      sample.control_ns = measure_one(group, &work, expected, control_eval, &sample.control_hash);
    }
    measured->push_back(sample);
  }
}

void write_samples(std::ostream& stream, const std::vector<TimedPair>& samples) {
  stream << '[';
  for (size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) stream << ',';
    const TimedPair& sample = samples[index];
    stream << "{\"pair\":" << (index + 1)
           << ",\"order\":\"" << sample.order
           << "\",\"control_ns\":" << sample.control_ns
           << ",\"candidate_ns\":" << sample.candidate_ns
           << ",\"control_output_hash\":\"" << std::hex << std::setw(16) << std::setfill('0') << sample.control_hash
           << "\",\"candidate_output_hash\":\"" << std::setw(16) << sample.candidate_hash << std::dec << "\"}";
  }
  stream << ']';
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: fc2_dweight_local_cost_diagnostic.exe process-index group1.bin group2.bin group3.bin group4.bin\n";
    return 2;
  }
  try {
    const size_t process_index = static_cast<size_t>(std::stoul(argv[1]));
    if (process_index >= 3) throw std::runtime_error("process-index must be 0, 1, or 2");
    std::vector<Group> groups(4);
    for (size_t index = 0; index < groups.size(); ++index) {
      if (!load_group(argv[index + 2], &groups[index])) {
        throw std::runtime_error(std::string("invalid input group: ") + argv[index + 2]);
      }
      validate_group(groups[index]);
    }

    struct GroupMeasurements {
      std::vector<TimedPair> full_cost;
      std::vector<TimedPair> accumulation_only;
    };
    std::vector<GroupMeasurements> measurements(groups.size());
    for (size_t sequence_index = 0; sequence_index < groups.size(); ++sequence_index) {
      const size_t group_index = (sequence_index + process_index) % groups.size();
      run_mode(groups[group_index], group_index, process_index, false, &measurements[group_index].full_cost);
      run_mode(groups[group_index], group_index, process_index, true, &measurements[group_index].accumulation_only);
    }

    std::cout << "{\"schema\":\"omega-fc2-dweight-local-cost-process-v1\","
              << "\"status\":\"LOCAL_COST_PROCESS_PASS\",\"process_index\":" << process_index
              << ",\"worker_threads\":1,\"warmup_pairs_per_group_modality\":" << kWarmupPairs
              << ",\"measured_pairs_per_group_modality\":" << kMeasuredPairs
              << ",\"clock\":\"steady_clock nanoseconds; timer encloses local function only\",\"groups\":[";
    for (size_t index = 0; index < groups.size(); ++index) {
      if (index != 0) std::cout << ',';
      const Group& group = groups[index];
      std::cout << "{\"case_id\":\"" << group.case_id << "\",\"group_index\":" << index
                << ",\"initial_dW_nonzero\":true,"
                << "\"full_cost\":{\"preparation_included\":true,\"warmup_pairs\":" << kWarmupPairs
                << ",\"measured_pairs\":" << kMeasuredPairs << ",\"samples\":";
      write_samples(std::cout, measurements[index].full_cost);
      std::cout << "},\"accumulation_only\":{\"diagnostic_only\":true,\"excludes_operand_preparation\":true,"
                << "\"warmup_pairs\":" << kWarmupPairs << ",\"measured_pairs\":" << kMeasuredPairs
                << ",\"samples\":";
      write_samples(std::cout, measurements[index].accumulation_only);
      std::cout << "}}";
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}

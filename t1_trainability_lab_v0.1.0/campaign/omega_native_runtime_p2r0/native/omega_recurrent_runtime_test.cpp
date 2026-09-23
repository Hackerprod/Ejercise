#include "omega_recurrent.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#ifdef NDEBUG
#undef assert
#define assert(condition) do { if (!(condition)) std::abort(); } while (false)
#endif

namespace {

constexpr size_t kSequence = 3;
constexpr size_t kBatch = 8;
constexpr size_t kSlots = 2;
constexpr size_t kDimension = 10;
constexpr size_t kRounds = 2;
constexpr size_t kParameterCount = 13;

size_t state_count(const OmegaRecurrentConfig& config) { return config.batch * config.slots * config.dimension; }
size_t readout_count(const OmegaRecurrentConfig& config) {
  return config.batch * config.sequence_length * config.slots * config.dimension;
}

std::array<size_t, kParameterCount> parameter_counts() {
  const size_t d2 = kDimension * kDimension;
  const size_t four_d = 4 * kDimension;
  return {kSlots * d2, kSlots * kDimension, 3 * d2, 3 * kDimension, d2, kDimension,
          4 * d2, four_d, kDimension * four_d, kDimension, kDimension, kRounds * kDimension,
          kRounds * kDimension};
}

void fill_values(std::vector<float>* values, float seed) {
  for (size_t index = 0; index < values->size(); ++index) {
    (*values)[index] = seed + static_cast<float>((index * 17) % 23) * 0.007F;
  }
}

struct Inputs {
  OmegaRecurrentConfig config{kSequence, kBatch, kSlots, kDimension, kRounds, 1, 1};
  std::vector<float> token = std::vector<float>(state_count(config) * kSequence);
  std::vector<float> previous = std::vector<float>(state_count(config));
  std::array<std::vector<float>, kParameterCount> parameters{};
  OmegaRecurrentParams view{};

  Inputs() {
    const auto counts = parameter_counts();
    for (size_t index = 0; index < kParameterCount; ++index) parameters[index].resize(counts[index]);
    fill_values(&token, 0.01F);
    fill_values(&previous, 0.02F);
    for (size_t index = 0; index < kParameterCount; ++index) fill_values(&parameters[index], 0.03F + 0.01F * static_cast<float>(index));
    view = {OmegaMatrixViewF32{parameters[0].data(), kSlots * kDimension, kDimension, static_cast<ptrdiff_t>(kDimension)},
            parameters[1].data(), parameters[2].data(), parameters[3].data(), parameters[4].data(), parameters[5].data(),
            parameters[6].data(), parameters[7].data(), parameters[8].data(), parameters[9].data(), parameters[10].data(),
            parameters[11].data(), parameters[12].data()};
  }
};

struct Gradients {
  std::vector<float> token;
  std::vector<float> previous;
  std::vector<float> token_sums;
  std::vector<float> previous_sums;
  std::vector<size_t> token_counts;
  std::vector<size_t> previous_counts;
  std::array<std::vector<float>, kParameterCount> primary{};
  std::array<std::vector<float>, kParameterCount> sums{};
  std::array<std::vector<size_t>, kParameterCount> counts{};
  std::vector<double> fp64;
  std::vector<size_t> depth_max;

  explicit Gradients(const OmegaRecurrentConfig& config)
      : token(config.batch * config.sequence_length * config.slots * config.dimension),
        previous(state_count(config)), token_sums(token.size()), previous_sums(previous.size()),
        token_counts(token.size()), previous_counts(previous.size()), fp64(kRounds * kDimension), depth_max(kRounds * kDimension) {
    const auto sizes = parameter_counts();
    for (size_t index = 0; index < kParameterCount; ++index) {
      primary[index].resize(sizes[index]);
      sums[index].resize(sizes[index]);
      counts[index].resize(sizes[index]);
    }
  }

  OmegaRecurrentGrads view() {
    return {token.data(), previous.data(), primary[0].data(), primary[1].data(), primary[2].data(), primary[3].data(),
            primary[4].data(), primary[5].data(), primary[6].data(), primary[7].data(), primary[8].data(), primary[9].data(),
            primary[10].data(), primary[11].data(), primary[12].data(), token_sums.data(), previous_sums.data(),
            sums[0].data(), sums[1].data(), sums[2].data(), sums[3].data(),
            sums[4].data(), sums[5].data(), sums[6].data(), sums[7].data(), sums[8].data(), sums[9].data(), sums[10].data(),
            sums[11].data(), sums[12].data(), token_counts.data(), previous_counts.data(), counts[0].data(), counts[1].data(), counts[2].data(),
            counts[3].data(), counts[4].data(), counts[5].data(), counts[6].data(), counts[7].data(), counts[8].data(),
            counts[9].data(), counts[10].data(), counts[11].data(), counts[12].data(), fp64.data(), depth_max.data()};
  }
};

void assert_close(const std::vector<float>& left, const std::vector<float>& right, float tolerance) {
  assert(left.size() == right.size());
  for (size_t index = 0; index < left.size(); ++index) {
    assert(std::fabs(left[index] - right[index]) <= tolerance);
  }
}

void assert_equal(const Gradients& left, const Gradients& right) {
  assert(left.token == right.token);
  assert(left.previous == right.previous);
  assert(left.token_sums == right.token_sums);
  assert(left.previous_sums == right.previous_sums);
  assert(left.token_counts == right.token_counts);
  assert(left.previous_counts == right.previous_counts);
  for (size_t index = 0; index < kParameterCount; ++index) {
    assert(left.primary[index] == right.primary[index]);
    assert(left.sums[index] == right.sums[index]);
    assert(left.counts[index] == right.counts[index]);
  }
  assert(left.fp64 == right.fp64);
  assert(left.depth_max == right.depth_max);
}

template <typename T>
void hash_vector(std::uint64_t* hash, const std::vector<T>& values) {
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(values.data());
  const size_t byte_count = values.size() * sizeof(T);
  for (size_t index = 0; index < byte_count; ++index) {
    *hash ^= bytes[index];
    *hash *= UINT64_C(1099511628211);
  }
}

std::uint64_t result_hash(const std::vector<float>& next, const std::vector<float>& readout, const Gradients& gradients) {
  std::uint64_t hash = UINT64_C(14695981039346656037);
  hash_vector(&hash, next);
  hash_vector(&hash, readout);
  hash_vector(&hash, gradients.token);
  hash_vector(&hash, gradients.previous);
  hash_vector(&hash, gradients.token_sums);
  hash_vector(&hash, gradients.previous_sums);
  hash_vector(&hash, gradients.token_counts);
  hash_vector(&hash, gradients.previous_counts);
  for (size_t index = 0; index < kParameterCount; ++index) {
    hash_vector(&hash, gradients.primary[index]);
    hash_vector(&hash, gradients.sums[index]);
    hash_vector(&hash, gradients.counts[index]);
  }
  hash_vector(&hash, gradients.fp64);
  hash_vector(&hash, gradients.depth_max);
  return hash;
}

void run_runtime_case(size_t workers, const Inputs& inputs, const std::vector<float>& d_readout,
    const std::vector<float>& d_next, const std::vector<float>& expected_next, const std::vector<float>& expected_readout,
    const Gradients& expected_gradients) {
  OmegaRuntime* runtime = omega_runtime_create(workers);
  assert(runtime != nullptr);
  const size_t workspace_size = omega_runtime_workspace_bytes(runtime, inputs.config);
  assert(workspace_size > 0);
  std::vector<unsigned char> workspace(workspace_size);
  std::vector<float> next(state_count(inputs.config));
  std::vector<float> readout(readout_count(inputs.config));
  assert(omega_runtime_forward(runtime, &inputs.config, &inputs.view, inputs.token.data(), inputs.previous.data(),
             next.data(), readout.data(), workspace.data(), workspace.size()) == 0);
#ifdef OMEGA_P2R_DIAGNOSTIC
  OmegaRuntimeDiagnosticSnapshot forward_timing{};
  assert(omega_runtime_diagnostic_snapshot(runtime, &forward_timing) == 0);
  assert(forward_timing.call_sequence > 0);
  assert(forward_timing.worker_count == workers);
  assert(forward_timing.dispatch_start_ns <= forward_timing.all_workers_done_ns);
  assert(forward_timing.all_workers_done_ns <= forward_timing.return_ns);
  for (size_t worker = 0; worker < workers; ++worker) {
    assert(forward_timing.worker_start_ns[worker] >= forward_timing.dispatch_start_ns);
    assert(forward_timing.worker_end_ns[worker] >= forward_timing.worker_start_ns[worker]);
    assert(forward_timing.worker_end_ns[worker] <= forward_timing.all_workers_done_ns);
  }
  assert(forward_timing.final_gradient_reduction_start_ns == 0);
  assert(forward_timing.final_gradient_reduction_end_ns == 0);
#endif
  assert_close(next, expected_next, 1.0e-6F);
  assert_close(readout, expected_readout, 1.0e-6F);

  Gradients first(inputs.config);
  OmegaRecurrentGrads first_view = first.view();
  assert(omega_runtime_backward(runtime, &inputs.config, &inputs.view, inputs.token.data(), d_readout.data(), d_next.data(),
             workspace.data(), workspace.size(), &first_view) == 0);
#ifdef OMEGA_P2R_DIAGNOSTIC
  OmegaRuntimeDiagnosticSnapshot backward_timing{};
  assert(omega_runtime_diagnostic_snapshot(runtime, &backward_timing) == 0);
  assert(backward_timing.call_sequence == forward_timing.call_sequence + 1);
  if (workers == 1) {
    assert(backward_timing.final_gradient_reduction_start_ns == 0);
    assert(backward_timing.final_gradient_reduction_end_ns == 0);
  } else {
    assert(backward_timing.final_gradient_reduction_start_ns > 0);
    assert(backward_timing.final_gradient_reduction_end_ns >= backward_timing.final_gradient_reduction_start_ns);
    assert(backward_timing.return_ns >= backward_timing.final_gradient_reduction_end_ns);
  }
#endif
  assert_close(first.token, expected_gradients.token, 1.0e-5F);
  assert_close(first.previous, expected_gradients.previous, 1.0e-5F);
  for (size_t index = 0; index < kParameterCount; ++index) {
    assert_close(first.primary[index], expected_gradients.primary[index], 1.0e-5F);
    assert_close(first.sums[index], expected_gradients.sums[index], 1.0e-5F);
    assert(first.counts[index] == expected_gradients.counts[index]);
  }
  assert(first.token_sums == expected_gradients.token_sums);
  assert(first.previous_sums == expected_gradients.previous_sums);
  assert(first.token_counts == expected_gradients.token_counts);
  assert(first.previous_counts == expected_gradients.previous_counts);
  for (size_t index = 0; index < first.fp64.size(); ++index) {
    assert(std::fabs(first.fp64[index] - expected_gradients.fp64[index]) <= 1.0e-10);
  }

  std::vector<unsigned char> second_workspace(workspace_size);
  std::vector<float> next_again(state_count(inputs.config));
  std::vector<float> readout_again(readout_count(inputs.config));
  assert(omega_runtime_forward(runtime, &inputs.config, &inputs.view, inputs.token.data(), inputs.previous.data(),
             next_again.data(), readout_again.data(), second_workspace.data(), second_workspace.size()) == 0);
  assert(next == next_again);
  assert(readout == readout_again);
  Gradients second(inputs.config);
  OmegaRecurrentGrads second_view = second.view();
  assert(omega_runtime_backward(runtime, &inputs.config, &inputs.view, inputs.token.data(), d_readout.data(), d_next.data(),
             second_workspace.data(), second_workspace.size(), &second_view) == 0);
  assert_equal(first, second);
  std::printf("determinism workers=%zu hash=%016llx\n", workers,
      static_cast<unsigned long long>(result_hash(next, readout, first)));
  omega_runtime_destroy(runtime);
}

}  // namespace

int main() {
  Inputs inputs;
  const std::array<OmegaRecurrentConfig, 4> t1_workspace_configs = {
      OmegaRecurrentConfig{256, 8, 8, 128, 1, 0, 0},
      OmegaRecurrentConfig{256, 8, 8, 128, 1, 1, 1},
      OmegaRecurrentConfig{256, 8, 8, 128, 4, 0, 0},
      OmegaRecurrentConfig{256, 8, 8, 128, 4, 1, 1},
  };
  const std::array<size_t, 4> expected_t1_workspace_bytes = {
      430152, 110039112, 528480, 388534368,
  };
  OmegaRuntime* workspace_runtime = omega_runtime_create(1);
  assert(workspace_runtime != nullptr);
  for (size_t index = 0; index < t1_workspace_configs.size(); ++index) {
    const OmegaRecurrentConfig& workspace_config = t1_workspace_configs[index];
    const size_t legacy_bytes = omega_recurrent_workspace_bytes(workspace_config);
    const size_t runtime_bytes = omega_runtime_workspace_bytes(workspace_runtime, workspace_config);
    assert(legacy_bytes == expected_t1_workspace_bytes[index]);
    assert(runtime_bytes == legacy_bytes);
  }
  omega_runtime_destroy(workspace_runtime);
  const OmegaRecurrentConfig inference{ kSequence, kBatch, kSlots, kDimension, kRounds, 0, 0 };
  const size_t serial_workspace = omega_recurrent_workspace_bytes(inference);
  OmegaRuntime* runtime_one = omega_runtime_create(1);
  assert(runtime_one != nullptr);
  assert(omega_runtime_workspace_bytes(runtime_one, inference) == serial_workspace);
  omega_runtime_destroy(runtime_one);
  assert(omega_runtime_create(3) == nullptr);
  OmegaRuntime* invalid_status_runtime = nullptr;
  assert(omega_runtime_create_status(3, &invalid_status_runtime) == OMEGA_RUNTIME_STATUS_INVALID_WORKER_COUNT);
  assert(invalid_status_runtime == nullptr);

  std::vector<unsigned char> serial_workspace_memory(omega_recurrent_workspace_bytes(inputs.config));
  std::vector<float> expected_next(state_count(inputs.config));
  std::vector<float> expected_readout(readout_count(inputs.config));
  assert(omega_recurrent_forward(&inputs.config, &inputs.view, inputs.token.data(), inputs.previous.data(), expected_next.data(),
             expected_readout.data(), serial_workspace_memory.data(), serial_workspace_memory.size()) == 0);
  std::vector<unsigned char> insufficient_serial_workspace(serial_workspace_memory.size() - 1);
  assert(omega_recurrent_forward(&inputs.config, &inputs.view, inputs.token.data(), inputs.previous.data(), expected_next.data(),
             expected_readout.data(), insufficient_serial_workspace.data(), insufficient_serial_workspace.size()) == 2);

  std::vector<float> d_readout(readout_count(inputs.config));
  std::vector<float> d_next(state_count(inputs.config));
  fill_values(&d_readout, 0.04F);
  fill_values(&d_next, 0.05F);
  Gradients expected_gradients(inputs.config);
  OmegaRecurrentGrads expected_view = expected_gradients.view();
  assert(omega_recurrent_backward(&inputs.config, &inputs.view, inputs.token.data(), d_readout.data(), d_next.data(),
             serial_workspace_memory.data(), serial_workspace_memory.size(), &expected_view) == 0);

  OmegaRuntime* invalid_partition_runtime = omega_runtime_create(2);
  assert(invalid_partition_runtime != nullptr);
  std::vector<unsigned char> insufficient_runtime_workspace(omega_runtime_workspace_bytes(invalid_partition_runtime, inputs.config) - 1);
  assert(omega_runtime_forward(invalid_partition_runtime, &inputs.config, &inputs.view, inputs.token.data(), inputs.previous.data(),
             expected_next.data(), expected_readout.data(), insufficient_runtime_workspace.data(), insufficient_runtime_workspace.size()) ==
         OMEGA_RUNTIME_STATUS_WORKSPACE);
  OmegaRecurrentConfig invalid_partition = inputs.config;
  invalid_partition.batch = 7;
  assert(omega_runtime_workspace_bytes(invalid_partition_runtime, invalid_partition) == 0);
  assert(omega_runtime_forward(invalid_partition_runtime, &invalid_partition, &inputs.view, inputs.token.data(), inputs.previous.data(),
             expected_next.data(), expected_readout.data(), serial_workspace_memory.data(), serial_workspace_memory.size()) ==
         OMEGA_RUNTIME_STATUS_INVALID_PARTITION);
  omega_runtime_destroy(invalid_partition_runtime);

  run_runtime_case(1, inputs, d_readout, d_next, expected_next, expected_readout, expected_gradients);
#ifdef OMEGA_PROFILE_INTERNAL
  OmegaRuntime* profile_runtime = omega_runtime_create(2);
  assert(profile_runtime != nullptr);
  assert(omega_runtime_forward(profile_runtime, &inputs.config, &inputs.view, inputs.token.data(), inputs.previous.data(),
             expected_next.data(), expected_readout.data(), serial_workspace_memory.data(), serial_workspace_memory.size()) ==
         OMEGA_RUNTIME_STATUS_PROFILE_GUARD);
  omega_runtime_destroy(profile_runtime);
#else
  run_runtime_case(2, inputs, d_readout, d_next, expected_next, expected_readout, expected_gradients);
  run_runtime_case(4, inputs, d_readout, d_next, expected_next, expected_readout, expected_gradients);
#endif
  return 0;
}

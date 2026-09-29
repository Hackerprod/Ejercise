#include "omega_recurrent.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr size_t kSequence = 3;
constexpr size_t kBatch = 8;
constexpr size_t kSlots = 8;
constexpr size_t kDimension = 128;
constexpr size_t kParameterCount = 13;

size_t state_count(const OmegaRecurrentConfig& config) {
  return config.batch * config.slots * config.dimension;
}

size_t readout_count(const OmegaRecurrentConfig& config) {
  return config.batch * config.sequence_length * config.slots * config.dimension;
}

std::array<size_t, kParameterCount> parameter_counts(const OmegaRecurrentConfig& config) {
  const size_t dimension2 = config.dimension * config.dimension;
  const size_t hidden = 4 * config.dimension;
  return {
      config.slots * dimension2,
      config.slots * config.dimension,
      3 * dimension2,
      3 * config.dimension,
      dimension2,
      config.dimension,
      hidden * config.dimension,
      hidden,
      config.dimension * hidden,
      config.dimension,
      config.dimension,
      config.rounds * config.dimension,
      config.rounds * config.dimension,
  };
}

void fill_values(std::vector<float>* values, float scale, size_t seed) {
  for (size_t index = 0; index < values->size(); ++index) {
    const int centered = static_cast<int>((index * 17 + seed * 11) % 29) - 14;
    (*values)[index] = scale * static_cast<float>(centered);
  }
}

struct Inputs {
  OmegaRecurrentConfig config;
  std::vector<float> token;
  std::vector<float> previous;
  std::vector<float> upstream_readout;
  std::vector<float> upstream_next;
  std::array<std::vector<float>, kParameterCount> parameters{};
  OmegaRecurrentParams view{};

  explicit Inputs(size_t rounds)
      : config{kSequence, kBatch, kSlots, kDimension, rounds, 1, 1},
        token(readout_count(config)),
        previous(state_count(config)),
        upstream_readout(readout_count(config)),
        upstream_next(state_count(config)) {
    fill_values(&token, 0.001F, 1);
    fill_values(&previous, 0.001F, 2);
    fill_values(&upstream_readout, 0.0005F, 3);
    fill_values(&upstream_next, 0.0005F, 4);

    const auto counts = parameter_counts(config);
    for (size_t index = 0; index < kParameterCount; ++index) {
      parameters[index].resize(counts[index]);
      const bool norm = index == 1 || index == 10;
      const bool bias = index == 3 || index == 5 || index == 7 || index == 9;
      const float scale = norm ? 0.0001F : (bias ? 0.0002F : 0.00005F);
      fill_values(&parameters[index], scale, index + 5);
      if (norm) {
        for (float& value : parameters[index]) value += 1.0F;
      }
    }
    view = {
        OmegaMatrixViewF32{parameters[0].data(), config.slots * config.dimension,
            config.dimension, static_cast<ptrdiff_t>(config.dimension)},
        parameters[1].data(), parameters[2].data(), parameters[3].data(), parameters[4].data(),
        parameters[5].data(), parameters[6].data(), parameters[7].data(), parameters[8].data(),
        parameters[9].data(), parameters[10].data(), parameters[11].data(), parameters[12].data(),
    };
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
  std::vector<double> fp64_depth;
  std::vector<size_t> depth_max;

  explicit Gradients(const OmegaRecurrentConfig& config)
      : token(readout_count(config)),
        previous(state_count(config)),
        token_sums(token.size()),
        previous_sums(previous.size()),
        token_counts(token.size()),
        previous_counts(previous.size()),
        fp64_depth(config.rounds * config.dimension),
        depth_max(config.rounds * config.dimension) {
    const auto sizes = parameter_counts(config);
    for (size_t index = 0; index < kParameterCount; ++index) {
      primary[index].resize(sizes[index]);
      sums[index].resize(sizes[index]);
      counts[index].resize(sizes[index]);
    }
  }

  OmegaRecurrentGrads view() {
    return {
        token.data(), previous.data(),
        primary[0].data(), primary[1].data(), primary[2].data(), primary[3].data(), primary[4].data(),
        primary[5].data(), primary[6].data(), primary[7].data(), primary[8].data(), primary[9].data(),
        primary[10].data(), primary[11].data(), primary[12].data(),
        token_sums.data(), previous_sums.data(),
        sums[0].data(), sums[1].data(), sums[2].data(), sums[3].data(), sums[4].data(), sums[5].data(),
        sums[6].data(), sums[7].data(), sums[8].data(), sums[9].data(), sums[10].data(), sums[11].data(),
        sums[12].data(),
        token_counts.data(), previous_counts.data(),
        counts[0].data(), counts[1].data(), counts[2].data(), counts[3].data(), counts[4].data(),
        counts[5].data(), counts[6].data(), counts[7].data(), counts[8].data(), counts[9].data(),
        counts[10].data(), counts[11].data(), counts[12].data(),
        fp64_depth.data(), depth_max.data(),
    };
  }
};

struct Run {
  std::vector<float> next;
  std::vector<float> readout;
  Gradients gradients;

  explicit Run(const OmegaRecurrentConfig& config)
      : next(state_count(config)), readout(readout_count(config)), gradients(config) {}
};

void require(bool condition, const char* label) {
  if (!condition) {
    std::fprintf(stderr, "FAIL %s\n", label);
    std::abort();
  }
}

bool all_finite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

double max_abs_difference(const std::vector<float>& left, const std::vector<float>& right) {
  require(left.size() == right.size(), "vector size mismatch");
  double maximum = 0.0;
  for (size_t index = 0; index < left.size(); ++index) {
    maximum = std::max(maximum, std::fabs(static_cast<double>(left[index]) - right[index]));
  }
  return maximum;
}

bool gradients_finite(const Gradients& gradients) {
  if (!all_finite(gradients.token) || !all_finite(gradients.previous) ||
      !all_finite(gradients.token_sums) || !all_finite(gradients.previous_sums)) return false;
  for (size_t index = 0; index < kParameterCount; ++index) {
    if (!all_finite(gradients.primary[index]) || !all_finite(gradients.sums[index])) return false;
  }
  return std::all_of(gradients.fp64_depth.begin(), gradients.fp64_depth.end(),
      [](double value) { return std::isfinite(value); });
}

bool gradients_equal(const Gradients& left, const Gradients& right) {
  if (left.token != right.token || left.previous != right.previous || left.token_sums != right.token_sums ||
      left.previous_sums != right.previous_sums || left.token_counts != right.token_counts ||
      left.previous_counts != right.previous_counts || left.fp64_depth != right.fp64_depth ||
      left.depth_max != right.depth_max) return false;
  for (size_t index = 0; index < kParameterCount; ++index) {
    if (left.primary[index] != right.primary[index] || left.sums[index] != right.sums[index] ||
        left.counts[index] != right.counts[index]) return false;
  }
  return true;
}

uint64_t hash_bytes(uint64_t hash, const void* data, size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (size_t index = 0; index < size; ++index) {
    hash ^= bytes[index];
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

uint64_t result_hash(const Run& run) {
  uint64_t hash = UINT64_C(14695981039346656037);
  hash = hash_bytes(hash, run.next.data(), run.next.size() * sizeof(float));
  hash = hash_bytes(hash, run.readout.data(), run.readout.size() * sizeof(float));
  hash = hash_bytes(hash, run.gradients.token.data(), run.gradients.token.size() * sizeof(float));
  hash = hash_bytes(hash, run.gradients.previous.data(), run.gradients.previous.size() * sizeof(float));
  hash = hash_bytes(hash, run.gradients.token_sums.data(), run.gradients.token_sums.size() * sizeof(float));
  hash = hash_bytes(hash, run.gradients.previous_sums.data(), run.gradients.previous_sums.size() * sizeof(float));
  hash = hash_bytes(hash, run.gradients.token_counts.data(), run.gradients.token_counts.size() * sizeof(size_t));
  hash = hash_bytes(hash, run.gradients.previous_counts.data(), run.gradients.previous_counts.size() * sizeof(size_t));
  for (size_t index = 0; index < kParameterCount; ++index) {
    hash = hash_bytes(hash, run.gradients.primary[index].data(), run.gradients.primary[index].size() * sizeof(float));
    hash = hash_bytes(hash, run.gradients.sums[index].data(), run.gradients.sums[index].size() * sizeof(float));
    hash = hash_bytes(hash, run.gradients.counts[index].data(), run.gradients.counts[index].size() * sizeof(size_t));
  }
  hash = hash_bytes(hash, run.gradients.fp64_depth.data(), run.gradients.fp64_depth.size() * sizeof(double));
  hash = hash_bytes(hash, run.gradients.depth_max.data(), run.gradients.depth_max.size() * sizeof(size_t));
  return hash;
}

void run_once(const Inputs& inputs, size_t workers, Run* result) {
  OmegaRuntime* runtime = omega_runtime_create(workers);
  require(runtime != nullptr, "runtime_create");
  const size_t workspace_bytes = omega_runtime_workspace_bytes(runtime, inputs.config);
  require(workspace_bytes > 0, "runtime_workspace_bytes");
  std::vector<unsigned char> workspace(workspace_bytes);
  require(omega_runtime_forward(runtime, &inputs.config, &inputs.view, inputs.token.data(), inputs.previous.data(),
      result->next.data(), result->readout.data(), workspace.data(), workspace.size()) == OMEGA_RUNTIME_STATUS_OK,
      "runtime_forward");
  OmegaRecurrentGrads gradients = result->gradients.view();
  require(omega_runtime_backward(runtime, &inputs.config, &inputs.view, inputs.token.data(),
      inputs.upstream_readout.data(), inputs.upstream_next.data(), workspace.data(), workspace.size(),
      &gradients) == OMEGA_RUNTIME_STATUS_OK, "runtime_backward");
  omega_runtime_destroy(runtime);
}

bool run_case(size_t rounds) {
  Inputs inputs(rounds);
  Run reference(inputs.config);
  run_once(inputs, 1, &reference);
  require(all_finite(reference.next) && all_finite(reference.readout) && gradients_finite(reference.gradients),
      "serial reference finite");

  bool passed = true;
  for (size_t workers : {size_t{1}, size_t{2}, size_t{4}}) {
    Run first(inputs.config);
    Run second(inputs.config);
    run_once(inputs, workers, &first);
    run_once(inputs, workers, &second);
    const bool finite = all_finite(first.next) && all_finite(first.readout) && gradients_finite(first.gradients);
    const bool repeat_exact = first.next == second.next && first.readout == second.readout &&
        gradients_equal(first.gradients, second.gradients);
    const double next_error = max_abs_difference(first.next, reference.next);
    const double readout_error = max_abs_difference(first.readout, reference.readout);
    bool counts_match_serial = true;
    double max_gradient_error = 0.0;
    double max_fp64_depth_error = 0.0;
    max_gradient_error = std::max(max_gradient_error,
        max_abs_difference(first.gradients.token, reference.gradients.token));
    max_gradient_error = std::max(max_gradient_error,
        max_abs_difference(first.gradients.previous, reference.gradients.previous));
    max_gradient_error = std::max(max_gradient_error,
        max_abs_difference(first.gradients.token_sums, reference.gradients.token_sums));
    max_gradient_error = std::max(max_gradient_error,
        max_abs_difference(first.gradients.previous_sums, reference.gradients.previous_sums));
    for (size_t index = 0; index < kParameterCount; ++index) {
      max_gradient_error = std::max(max_gradient_error,
          max_abs_difference(first.gradients.primary[index], reference.gradients.primary[index]));
      max_gradient_error = std::max(max_gradient_error,
          max_abs_difference(first.gradients.sums[index], reference.gradients.sums[index]));
      counts_match_serial = counts_match_serial && first.gradients.counts[index] == reference.gradients.counts[index];
    }
    counts_match_serial = counts_match_serial && first.gradients.token_counts == reference.gradients.token_counts &&
        first.gradients.previous_counts == reference.gradients.previous_counts;
    const bool depth_max_matches_serial = first.gradients.depth_max == reference.gradients.depth_max;
    for (size_t index = 0; index < first.gradients.fp64_depth.size(); ++index) {
      max_fp64_depth_error = std::max(max_fp64_depth_error,
          std::fabs(first.gradients.fp64_depth[index] - reference.gradients.fp64_depth[index]));
    }
    const bool matches_serial = next_error <= 1.0e-6 && readout_error <= 1.0e-6 &&
        max_gradient_error <= 1.0e-5 && max_fp64_depth_error <= 1.0e-10 && counts_match_serial;
    passed = passed && finite && repeat_exact && matches_serial;
    std::printf(
        "{\"record\":\"m8_runtime_case\",\"K\":%zu,\"workers\":%zu,\"local_batch\":%zu,"
        "\"sequence_length\":%zu,\"finite\":%s,\"repeat_exact\":%s,\"matches_serial_existing_tolerance\":%s,"
        "\"counts_match_serial\":%s,\"depth_max_matches_serial\":%s,"
        "\"max_next_error\":%.9g,\"max_readout_error\":%.9g,\"max_gradient_error\":%.9g,"
        "\"max_fp64_depth_error\":%.9g,\"result_hash\":\"%016llx\"}\n",
        rounds, workers, inputs.config.batch / workers, inputs.config.sequence_length,
        finite ? "true" : "false", repeat_exact ? "true" : "false", matches_serial ? "true" : "false",
        counts_match_serial ? "true" : "false", depth_max_matches_serial ? "true" : "false",
        next_error, readout_error, max_gradient_error, max_fp64_depth_error,
        static_cast<unsigned long long>(result_hash(first)));
  }
  return passed;
}

}  // namespace

int main() {
  bool passed = true;
  for (size_t rounds : {size_t{1}, size_t{4}}) passed = run_case(rounds) && passed;
  std::printf("{\"record\":\"m8_runtime_result\",\"status\":\"%s\"}\n", passed ? "PASS" : "FAIL");
  return passed ? 0 : 1;
}

#include "omega_recurrent.h"

#include "fc2_dweight_m8.h"

#include <algorithm>
#ifdef OMEGA_PROFILE_INTERNAL
#include <atomic>
#include <chrono>
#include <mutex>
#endif
#ifdef OMEGA_P2R_DIAGNOSTIC
#include <chrono>
#endif
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <array>
#include <condition_variable>
#include <exception>
#include <new>
#include <thread>

#if defined(OMEGA_FC2_GELU_COUNT_GUARD) && !defined(OMEGA_PROFILE_INTERNAL) && !defined(OMEGA_P2R_DIAGNOSTIC)
#include <cstdio>
#define OMEGA_FC2_GELU_COUNT_GUARD_ACTIVE 1
#else
#define OMEGA_FC2_GELU_COUNT_GUARD_ACTIVE 0
#endif

#if defined(__AVX2__) || defined(_M_AVX2)
#define OMEGA_HAS_AVX2 1
#include <immintrin.h>
#else
#define OMEGA_HAS_AVX2 0
#endif

namespace {

constexpr size_t kAlignment = 64;
constexpr float kEpsilon = 1.0e-6F;
constexpr float kSqrtTwo = 1.4142135623730950488F;
constexpr size_t kFc2DWeightGroupSlots = 8;
constexpr size_t kFc2DWeightGroupDimension = 128;

#ifdef OMEGA_FC2_REPLAY_CAPTURE
constexpr size_t kFc2ReplaySlots = 8;
constexpr size_t kFc2ReplayDimension = 128;
constexpr size_t kFc2ReplayHidden = 512;

const OmegaFc2ReplayGroupIdentity* g_fc2_replay_targets = nullptr;
size_t g_fc2_replay_target_count = 0;
OmegaFc2ReplayCaptureCallback g_fc2_replay_callback = nullptr;
void* g_fc2_replay_user_data = nullptr;
thread_local size_t g_fc2_replay_worker_index = 0;

struct Fc2ReplayCaptureScratch {
  bool active = false;
  size_t position = 0;
  size_t round = 0;
  size_t batch = 0;
  std::array<float, kFc2ReplaySlots * kFc2ReplayHidden> preactivation{};
  std::array<float, kFc2ReplaySlots * kFc2ReplayHidden> activated{};
  std::array<float, kFc2ReplaySlots * kFc2ReplayDimension> output_gradient{};
  std::array<float, kFc2ReplayDimension * kFc2ReplayHidden> dweight_before{};
  std::array<float, kFc2ReplayDimension * kFc2ReplayHidden> dweight_candidate{};
};

thread_local Fc2ReplayCaptureScratch g_fc2_replay_scratch{};

bool fc2_replay_group_selected(size_t position, size_t round, size_t batch) {
  if (g_fc2_replay_callback == nullptr || g_fc2_replay_targets == nullptr) return false;
  for (size_t index = 0; index < g_fc2_replay_target_count; ++index) {
    const OmegaFc2ReplayGroupIdentity& target = g_fc2_replay_targets[index];
    if (target.position == position && target.round == round && target.batch == batch) return true;
  }
  return false;
}
#endif

#ifdef OMEGA_P2R_DIAGNOSTIC
using DiagnosticClock = std::chrono::steady_clock;

uint64_t diagnostic_now_ns() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      DiagnosticClock::now().time_since_epoch()).count());
}
#endif

#ifdef OMEGA_PROFILE_INTERNAL

enum class ProfileDirection { kForward, kBackward };
enum class ProfileStage {
  kQkvProjection,
  kAttentionScoresSoftmaxMixing,
  kOutProjection,
  kFc1Gelu,
  kFc2,
  kRmsnormGates,
  kStatePrelude,
  kDepthEmbeddingPairwiseReduction,
  kDepthEmbeddingCarry,
  kHistoryBufferReadsWrites,
};

using ProfileClock = std::chrono::steady_clock;
std::atomic<bool> g_profile_enabled{false};
OmegaRecurrentProfileSnapshot g_profile_snapshot{};
std::mutex g_profile_mutex;

OmegaRecurrentProfileDirection& profile_direction(ProfileDirection direction) {
  return direction == ProfileDirection::kForward ? g_profile_snapshot.forward : g_profile_snapshot.backward;
}

void profile_record_call(ProfileDirection direction) {
  if (!g_profile_enabled.load(std::memory_order_relaxed)) return;
  std::lock_guard<std::mutex> lock(g_profile_mutex);
  ++profile_direction(direction).calls;
}

void profile_record_duration(ProfileDirection direction, ProfileStage stage, double seconds) {
  std::lock_guard<std::mutex> lock(g_profile_mutex);
  OmegaRecurrentProfileDirection& totals = profile_direction(direction);
  std::uint64_t* calls = nullptr;
  double* total_seconds = nullptr;
  switch (stage) {
    case ProfileStage::kQkvProjection:
      calls = &totals.qkv_projection_calls;
      total_seconds = &totals.qkv_projection_seconds;
      break;
    case ProfileStage::kAttentionScoresSoftmaxMixing:
      calls = &totals.attention_scores_softmax_mixing_calls;
      total_seconds = &totals.attention_scores_softmax_mixing_seconds;
      break;
    case ProfileStage::kOutProjection:
      calls = &totals.out_projection_calls;
      total_seconds = &totals.out_projection_seconds;
      break;
    case ProfileStage::kFc1Gelu:
      calls = &totals.fc1_gelu_calls;
      total_seconds = &totals.fc1_gelu_seconds;
      break;
    case ProfileStage::kFc2:
      calls = &totals.fc2_calls;
      total_seconds = &totals.fc2_seconds;
      break;
    case ProfileStage::kRmsnormGates:
      calls = &totals.rmsnorm_gates_calls;
      total_seconds = &totals.rmsnorm_gates_seconds;
      break;
    case ProfileStage::kStatePrelude:
      calls = &totals.state_prelude_calls;
      total_seconds = &totals.state_prelude_seconds;
      break;
    case ProfileStage::kDepthEmbeddingPairwiseReduction:
      calls = &totals.depth_embedding_pairwise_reduction_calls;
      total_seconds = &totals.depth_embedding_pairwise_reduction_seconds;
      break;
    case ProfileStage::kDepthEmbeddingCarry:
      calls = &totals.depth_embedding_carry_calls;
      total_seconds = &totals.depth_embedding_carry_seconds;
      break;
    case ProfileStage::kHistoryBufferReadsWrites:
      calls = &totals.history_buffer_reads_writes_calls;
      total_seconds = &totals.history_buffer_reads_writes_seconds;
      break;
  }
  ++*calls;
  *total_seconds += seconds;
}

class ProfileScope {
 public:
  ProfileScope(ProfileDirection direction, ProfileStage stage)
      : direction_(direction), stage_(stage), active_(g_profile_enabled.load(std::memory_order_relaxed)) {
    if (active_) started_ = ProfileClock::now();
  }

  ~ProfileScope() {
    if (!active_) return;
    const std::chrono::duration<double> elapsed = ProfileClock::now() - started_;
    profile_record_duration(direction_, stage_, elapsed.count());
  }

 private:
  ProfileDirection direction_;
  ProfileStage stage_;
  bool active_;
  ProfileClock::time_point started_{};
};

#define OMEGA_PROFILE_CALL(direction) profile_record_call(direction)
#define OMEGA_PROFILE_JOIN_IMPL(left, right) left##right
#define OMEGA_PROFILE_JOIN(left, right) OMEGA_PROFILE_JOIN_IMPL(left, right)
#define OMEGA_PROFILE_SCOPE(direction, stage) ProfileScope OMEGA_PROFILE_JOIN(omega_profile_scope_, __LINE__)((direction), (stage))

#else

#define OMEGA_PROFILE_CALL(direction) ((void)0)
#define OMEGA_PROFILE_SCOPE(direction, stage) ((void)0)

#endif

#ifdef OMEGA_P2R_DIAGNOSTIC
#ifndef OMEGA_DIAGNOSTIC_ELIDE_MASK
#define OMEGA_DIAGNOSTIC_ELIDE_MASK 0
#endif
constexpr unsigned int kDiagnosticElideMask = OMEGA_DIAGNOSTIC_ELIDE_MASK;

constexpr bool diagnostic_elide(unsigned int bit) {
  return (kDiagnosticElideMask & bit) != 0;
}

constexpr bool diagnostic_any_elide() {
  return kDiagnosticElideMask != 0;
}
#else
constexpr bool diagnostic_elide(unsigned int) {
  return false;
}
constexpr bool diagnostic_any_elide() {
  return false;
}
#define OMEGA_DIAGNOSTIC_ELIDE_ATTENTION_SCORES_SOFTMAX_MIXING 0u
#define OMEGA_DIAGNOSTIC_ELIDE_DEPTH_EMBEDDING_PAIRWISE 0u
#define OMEGA_DIAGNOSTIC_ELIDE_RMSNORM_GATES 0u
#define OMEGA_DIAGNOSTIC_ELIDE_STATE_PRELUDE 0u
#define OMEGA_DIAGNOSTIC_ELIDE_HISTORY_BUFFER 0u
#define OMEGA_DIAGNOSTIC_ELIDE_DEPTH_EMBEDDING_CARRY 0u
#endif

bool checked_add(size_t left, size_t right, size_t* result) {
  if (left > std::numeric_limits<size_t>::max() - right) {
    return false;
  }
  *result = left + right;
  return true;
}

bool checked_mul(size_t left, size_t right, size_t* result) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

bool align_size(size_t value, size_t* result) {
  const size_t remainder = value % kAlignment;
  const size_t padding = remainder == 0 ? 0 : kAlignment - remainder;
  return checked_add(value, padding, result);
}

bool append_floats(size_t count, size_t* offset) {
  size_t aligned = 0;
  size_t bytes = 0;
  if (!align_size(*offset, &aligned) || !checked_mul(count, sizeof(float), &bytes) || !checked_add(aligned, bytes, offset)) {
    return false;
  }
  return true;
}

bool append_uint64s(size_t count, size_t* offset) {
  size_t aligned = 0;
  size_t bytes = 0;
  if (!align_size(*offset, &aligned) || !checked_mul(count, sizeof(std::uint64_t), &bytes) || !checked_add(aligned, bytes, offset)) {
    return false;
  }
  return true;
}

bool append_size_ts(size_t count, size_t* offset) {
  size_t aligned = 0;
  size_t bytes = 0;
  if (!align_size(*offset, &aligned) || !checked_mul(count, sizeof(size_t), &bytes) || !checked_add(aligned, bytes, offset)) {
    return false;
  }
  return true;
}

bool append_doubles(size_t count, size_t* offset) {
  size_t aligned = 0;
  size_t bytes = 0;
  if (!align_size(*offset, &aligned) || !checked_mul(count, sizeof(double), &bytes) || !checked_add(aligned, bytes, offset)) {
    return false;
  }
  return true;
}

bool append_bytes(size_t count, size_t* offset) {
  size_t aligned = 0;
  if (!align_size(*offset, &aligned) || !checked_add(aligned, count, offset)) {
    return false;
  }
  return true;
}

constexpr size_t kParameterCount = 13;

bool parameter_counts(const OmegaRecurrentConfig& config, std::array<size_t, kParameterCount>* counts, size_t* total) {
  size_t d2 = 0;
  size_t four_d = 0;
  size_t rd = 0;
  if (!checked_mul(config.dimension, config.dimension, &d2) || !checked_mul(4, config.dimension, &four_d) ||
      !checked_mul(config.rounds, config.dimension, &rd) ||
      !checked_mul(config.slots, d2, &(*counts)[0]) || !checked_mul(config.slots, config.dimension, &(*counts)[1])) {
    return false;
  }
  if (!checked_mul(3, d2, &(*counts)[2]) || !checked_mul(3, config.dimension, &(*counts)[3]) ||
      !checked_mul(d2, 1, &(*counts)[4]) || !checked_mul(config.dimension, 1, &(*counts)[5]) ||
      !checked_mul(4, d2, &(*counts)[6]) || !checked_mul(4, config.dimension, &(*counts)[7]) ||
      !checked_mul(config.dimension, four_d, &(*counts)[8]) || !checked_mul(config.dimension, 1, &(*counts)[9]) ||
      !checked_mul(config.dimension, 1, &(*counts)[10])) {
    return false;
  }
  (*counts)[11] = rd;
  (*counts)[12] = rd;
  size_t sum = 0;
  for (size_t count : *counts) {
    if (!checked_add(sum, count, &sum)) return false;
  }
  *total = sum;
  return true;
}

size_t element_count(const OmegaRecurrentConfig& config) {
  size_t result = 0;
  size_t value = 0;
  if (!checked_mul(config.batch, config.slots, &value) || !checked_mul(value, config.dimension, &result)) {
    return 0;
  }
  return result;
}

bool training_counts(const OmegaRecurrentConfig& config, size_t state_count, size_t* history_state, size_t* round_state, size_t* round_probability, size_t* round_fc1) {
  size_t time_plus_one = 0;
  size_t round_count = 0;
  size_t batch_slots = 0;
  if (!checked_add(config.sequence_length, 1, &time_plus_one) || !checked_mul(time_plus_one, state_count, history_state) ||
      !checked_mul(config.sequence_length, config.rounds, &round_count) || !checked_mul(round_count, state_count, round_state) ||
      !checked_mul(config.batch, config.slots, &batch_slots) || !checked_mul(batch_slots, config.slots, &batch_slots) ||
      !checked_mul(round_count, batch_slots, round_probability) || !checked_mul(*round_state, 4, round_fc1)) {
    return false;
  }
  return true;
}

bool depth_vector_reduction_layout(const OmegaRecurrentConfig& config, size_t* hidden_dimension,
    size_t* levels, size_t* partial_count) {
  size_t contributions = 0;
  if (!checked_mul(config.batch, config.sequence_length, &contributions) ||
      !checked_mul(contributions, config.slots, &contributions) ||
      !checked_mul(config.dimension, 4, hidden_dimension)) {
    return false;
  }
  if (contributions == 0) return false;
  constexpr size_t mask_bits = std::numeric_limits<std::uint64_t>::digits;
  size_t capacity = 1;
  size_t computed_levels = 1;
  while (capacity < contributions) {
    if (computed_levels >= mask_bits || capacity > std::numeric_limits<size_t>::max() / 2) return false;
    capacity *= 2;
    ++computed_levels;
  }
  *levels = computed_levels;
  size_t round_levels = 0;
  return checked_mul(config.rounds, *levels, &round_levels) &&
      checked_mul(round_levels, *hidden_dimension, partial_count);
}

bool append_training_storage(const OmegaRecurrentConfig& config, size_t state_count, size_t* offset) {
  size_t history_state = 0;
  size_t round_state = 0;
  size_t round_probability = 0;
  size_t round_fc1 = 0;
  size_t anchor_state = 0;
  if (!training_counts(config, state_count, &history_state, &round_state, &round_probability, &round_fc1) ||
      !checked_mul(state_count, config.sequence_length, &anchor_state)) {
    return false;
  }
  return append_floats(history_state, offset) && append_floats(anchor_state, offset) && append_floats(round_state, offset) &&
      append_floats(round_state, offset) && append_floats(round_state, offset) && append_floats(round_state, offset) &&
      append_floats(round_probability, offset) && append_floats(round_state, offset) &&
      append_floats(round_state, offset) && append_floats(round_fc1, offset) && append_floats(round_state, offset);
}

size_t workspace_bytes_impl(const OmegaRecurrentConfig& config) {
  if (config.sequence_length == 0 || config.batch == 0 || config.slots == 0 || config.dimension == 0 || config.rounds == 0) {
    return 0;
  }
  const size_t state_count = element_count(config);
  if (state_count == 0) {
    return 0;
  }
  size_t slot_scores = 0;
  if (!checked_mul(config.batch, config.slots, &slot_scores) || !checked_mul(slot_scores, config.slots, &slot_scores)) {
    return 0;
  }
  size_t fc1_count = 0;
  if (!checked_mul(state_count, 4, &fc1_count)) {
    return 0;
  }
  size_t activation_count = 0;
  if (!checked_mul(4, config.dimension, &activation_count)) {
    return 0;
  }
  size_t depth_bias_count = 0;
  if (!checked_mul(config.rounds, config.dimension, &depth_bias_count) || !checked_mul(depth_bias_count, 4, &depth_bias_count)) {
    return 0;
  }
  size_t depth_hidden_dimension = 0;
  size_t depth_levels = 0;
  size_t depth_partial_count = 0;
  if (!depth_vector_reduction_layout(config, &depth_hidden_dimension, &depth_levels, &depth_partial_count)) {
    return 0;
  }
  /* Caller may provide ordinary heap storage, not a 64-byte-aligned pointer. */
  size_t total = kAlignment - 1;
  /* state, anchor, candidate, q, k, v, mixed, update */
  for (int index = 0; index < 8; ++index) {
    if (!append_floats(state_count, &total)) {
      return 0;
    }
  }
  if (!append_floats(slot_scores, &total) || !append_floats(fc1_count, &total) ||
      !append_floats(activation_count, &total) || !append_floats(depth_bias_count, &total)) {
    return 0;
  }
  if (config.training != 0 && !append_training_storage(config, state_count, &total)) {
    return 0;
  }
  if (config.training != 0 && config.slots == kFc2DWeightGroupSlots &&
      config.dimension == kFc2DWeightGroupDimension) {
    size_t grouped_activation_count = 0;
    size_t grouped_output_gradient_count = 0;
    if (!checked_mul(config.slots, activation_count, &grouped_activation_count) ||
        !checked_mul(config.slots, config.dimension, &grouped_output_gradient_count) ||
        !append_floats(grouped_activation_count, &total) ||
        !append_floats(grouped_output_gradient_count, &total)) {
      return 0;
    }
  }
  if (!append_floats(depth_partial_count, &total) || !append_uint64s(config.rounds, &total)) {
    return 0;
  }
  return total;
}

bool valid_runtime_worker_count(size_t count) {
  return count == 1 || count == 2 || count == 4;
}

size_t runtime_workspace_bytes_impl(size_t worker_count, const OmegaRecurrentConfig& config) {
  if (!valid_runtime_worker_count(worker_count) || worker_count == 1) {
    return worker_count == 1 ? workspace_bytes_impl(config) : 0;
  }
  if (config.batch == 0 || config.batch % worker_count != 0) return 0;
  OmegaRecurrentConfig local = config;
  local.batch = config.batch / worker_count;
  /* Reserve diagnostic-capable core storage; worker config activates it only when needed. */
  local.instrumentation = 1;
  const size_t core_bytes = workspace_bytes_impl(local);
  if (core_bytes == 0) return 0;
  std::array<size_t, kParameterCount> counts{};
  size_t parameter_count = 0;
  if (!parameter_counts(config, &counts, &parameter_count)) return 0;
  size_t depth_count = 0;
  if (!checked_mul(config.rounds, config.dimension, &depth_count)) return 0;
  size_t total = kAlignment - 1;
  for (size_t worker = 0; worker < worker_count; ++worker) {
    if (!append_bytes(core_bytes, &total) || !append_floats(parameter_count, &total) ||
        !append_floats(parameter_count, &total) || !append_size_ts(parameter_count, &total) ||
        !append_doubles(depth_count, &total) || !append_size_ts(depth_count, &total)) {
      return 0;
    }
  }
  return total;
}

struct WorkspaceCursor {
  unsigned char* base;
  size_t capacity;
  size_t used;

  unsigned char* take_bytes(size_t bytes) {
    if (used > capacity) return nullptr;
    const uintptr_t address = reinterpret_cast<uintptr_t>(base) + used;
    const size_t remainder = static_cast<size_t>(address % kAlignment);
    const size_t padding = remainder == 0 ? 0 : kAlignment - remainder;
    size_t required = 0;
    if (!checked_add(padding, bytes, &required) || required > capacity - used) {
      return nullptr;
    }
    used += padding;
    unsigned char* result = base + used;
    used += bytes;
    return result;
  }

  float* take(size_t count) {
    size_t bytes = 0;
    if (!checked_mul(count, sizeof(float), &bytes)) {
      return nullptr;
    }
    return reinterpret_cast<float*>(take_bytes(bytes));
  }

  std::uint64_t* take_uint64(size_t count) {
    size_t bytes = 0;
    if (!checked_mul(count, sizeof(std::uint64_t), &bytes)) return nullptr;
    return reinterpret_cast<std::uint64_t*>(take_bytes(bytes));
  }

};

struct TrainingBuffers {
  float* state_history = nullptr;
  float* anchor_history = nullptr;
  float* candidate_inputs = nullptr;
  float* q_history = nullptr;
  float* key_history = nullptr;
  float* value_history = nullptr;
  float* probability_history = nullptr;
  float* attention_history = nullptr;
  float* out_history = nullptr;
  float* fc1_pre_history = nullptr;
  float* pre_rms_history = nullptr;
};

bool bind_training_storage(const OmegaRecurrentConfig& config, size_t state_count, WorkspaceCursor* cursor, TrainingBuffers* buffers) {
  if (config.training == 0) {
    return true;
  }
  size_t history_state = 0;
  size_t round_state = 0;
  size_t round_probability = 0;
  size_t round_fc1 = 0;
  size_t anchor_state = 0;
  if (!training_counts(config, state_count, &history_state, &round_state, &round_probability, &round_fc1) ||
      !checked_mul(state_count, config.sequence_length, &anchor_state)) {
    return false;
  }
  buffers->state_history = cursor->take(history_state);
  buffers->anchor_history = cursor->take(anchor_state);
  buffers->candidate_inputs = cursor->take(round_state);
  buffers->q_history = cursor->take(round_state);
  buffers->key_history = cursor->take(round_state);
  buffers->value_history = cursor->take(round_state);
  buffers->probability_history = cursor->take(round_probability);
  buffers->attention_history = cursor->take(round_state);
  buffers->out_history = cursor->take(round_state);
  buffers->fc1_pre_history = cursor->take(round_fc1);
  buffers->pre_rms_history = cursor->take(round_state);
  return buffers->state_history != nullptr && buffers->anchor_history != nullptr && buffers->candidate_inputs != nullptr &&
      buffers->q_history != nullptr && buffers->key_history != nullptr && buffers->value_history != nullptr &&
      buffers->probability_history != nullptr && buffers->attention_history != nullptr && buffers->out_history != nullptr &&
      buffers->fc1_pre_history != nullptr && buffers->pre_rms_history != nullptr;
}

void clear_training_storage(const OmegaRecurrentConfig& config, size_t state_count, TrainingBuffers* buffers) {
  size_t history_state = 0;
  size_t round_state = 0;
  size_t round_probability = 0;
  size_t round_fc1 = 0;
  size_t anchor_state = 0;
  if (!training_counts(config, state_count, &history_state, &round_state, &round_probability, &round_fc1) ||
      !checked_mul(state_count, config.sequence_length, &anchor_state)) {
    return;
  }
  std::memset(buffers->state_history, 0, history_state * sizeof(float));
  std::memset(buffers->anchor_history, 0, anchor_state * sizeof(float));
  std::memset(buffers->candidate_inputs, 0, round_state * sizeof(float));
  std::memset(buffers->q_history, 0, round_state * sizeof(float));
  std::memset(buffers->key_history, 0, round_state * sizeof(float));
  std::memset(buffers->value_history, 0, round_state * sizeof(float));
  std::memset(buffers->probability_history, 0, round_probability * sizeof(float));
  std::memset(buffers->attention_history, 0, round_state * sizeof(float));
  std::memset(buffers->out_history, 0, round_state * sizeof(float));
  std::memset(buffers->fc1_pre_history, 0, round_fc1 * sizeof(float));
  std::memset(buffers->pre_rms_history, 0, round_state * sizeof(float));
}

inline size_t state_index(size_t batch, size_t slot, size_t dimension, size_t slots, size_t dimensions) {
  return (batch * slots + slot) * dimensions + dimension;
}

inline size_t readout_index(size_t batch, size_t position, size_t slot, size_t dimension, const OmegaRecurrentConfig& config) {
  return ((batch * config.sequence_length + position) * config.slots + slot) * config.dimension + dimension;
}

float sigmoid(float value) {
  return 1.0F / (1.0F + std::exp(-value));
}

float gelu(float value) {
  return 0.5F * value * (1.0F + std::erf(value / kSqrtTwo));
}

#if OMEGA_HAS_AVX2
void prepare_fc2_dweight_slot_avx2(
    const float* preactivation,
    const float* output_gradient,
    float* hidden_values,
    float* grouped_activated,
    float* grouped_output_gradient,
    size_t slot,
    size_t output_count,
    size_t hidden_count) {
  for (size_t hidden = 0; hidden < hidden_count; ++hidden) {
    hidden_values[hidden] = gelu(preactivation[hidden]);
  }
  if (grouped_activated != nullptr) {
    std::memcpy(grouped_activated + slot * hidden_count, hidden_values, hidden_count * sizeof(float));
  }
  if (grouped_output_gradient != nullptr) {
    std::memcpy(grouped_output_gradient + slot * output_count, output_gradient, output_count * sizeof(float));
  }
}
#endif

bool valid_state_part_weight(const OmegaMatrixViewF32& view, const OmegaRecurrentConfig& config) {
  size_t expected_rows = 0;
  if (!checked_mul(config.slots, config.dimension, &expected_rows) || view.data == nullptr ||
      view.rows != expected_rows || view.cols != config.dimension || view.row_stride < 0) {
    return false;
  }
  return static_cast<size_t>(view.row_stride) >= view.cols;
}

bool valid_params(const OmegaRecurrentParams& params, const OmegaRecurrentConfig& config) {
  return valid_state_part_weight(params.state_part_weight, config) && params.prelude_norm_weight != nullptr && params.block_qkv_weight != nullptr &&
      params.block_qkv_bias != nullptr && params.block_out_weight != nullptr && params.block_out_bias != nullptr &&
      params.block_fc1_weight != nullptr && params.block_fc1_bias != nullptr && params.block_fc2_weight != nullptr &&
      params.block_fc2_bias != nullptr && params.block_norm_weight != nullptr && params.depth_embedding != nullptr &&
      params.gate_logits != nullptr;
}

bool valid_grads(const OmegaRecurrentGrads& grads) {
  return grads.d_token_part != nullptr && grads.d_previous_state != nullptr && grads.d_state_part_weight != nullptr &&
      grads.d_prelude_norm_weight != nullptr && grads.d_block_qkv_weight != nullptr && grads.d_block_qkv_bias != nullptr &&
      grads.d_block_out_weight != nullptr && grads.d_block_out_bias != nullptr && grads.d_block_fc1_weight != nullptr &&
      grads.d_block_fc1_bias != nullptr && grads.d_block_fc2_weight != nullptr && grads.d_block_fc2_bias != nullptr &&
      grads.d_block_norm_weight != nullptr && grads.d_depth_embedding != nullptr && grads.d_gate_logits != nullptr;
}

float gelu_derivative(float value) {
  constexpr float kInvSqrtTwoPi = 0.39894228040143267794F;
  return 0.5F * (1.0F + std::erf(value / kSqrtTwo)) + value * std::exp(-0.5F * value * value) * kInvSqrtTwoPi;
}

#if OMEGA_HAS_AVX2
inline float horizontal_sum(__m256 value) {
  const __m128 low = _mm256_castps256_ps128(value);
  const __m128 high = _mm256_extractf128_ps(value, 1);
  __m128 sum = _mm_add_ps(low, high);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

inline float state_prelude_write(const float* token_part, const OmegaMatrixViewF32& state_part_weight,
    size_t flattened, const float* mean_state, size_t dimension) {
  float write = token_part[flattened];
  const float* state_part_row = state_part_weight.data + flattened * static_cast<size_t>(state_part_weight.row_stride);
  __m256 sum = _mm256_setzero_ps();
  size_t input = 0;
  for (; input + 8 <= dimension; input += 8) {
    sum = _mm256_add_ps(sum, _mm256_mul_ps(_mm256_loadu_ps(state_part_row + input), _mm256_loadu_ps(mean_state + input)));
  }
  write += horizontal_sum(sum);
  for (; input < dimension; ++input) write += state_part_row[input] * mean_state[input];
  return write;
}
#else
inline float state_prelude_write(const float* token_part, const OmegaMatrixViewF32& state_part_weight,
    size_t flattened, const float* mean_state, size_t dimension) {
  float write = token_part[flattened];
  const float* state_part_row = state_part_weight.data + flattened * static_cast<size_t>(state_part_weight.row_stride);
  for (size_t input = 0; input < dimension; ++input) write += state_part_row[input] * mean_state[input];
  return write;
}

#endif

void clear_optional_contribution_sums(const OmegaRecurrentConfig& config, const OmegaRecurrentGrads& grads, size_t state_count,
    size_t state_dimension, size_t dimension) {
  const size_t fc1_count = 4 * dimension * dimension;
  const size_t fc2_count = dimension * 4 * dimension;
  const size_t qkv_count = 3 * dimension * dimension;
  const size_t qkv_bias_count = 3 * dimension;
  const size_t matrix_count = state_dimension * dimension;
  if (grads.sum_abs_d_token_part != nullptr) std::memset(grads.sum_abs_d_token_part, 0, config.batch * config.sequence_length * state_dimension * sizeof(float));
  if (grads.sum_abs_d_previous_state != nullptr) std::memset(grads.sum_abs_d_previous_state, 0, state_count * sizeof(float));
  if (grads.sum_abs_d_state_part_weight != nullptr) std::memset(grads.sum_abs_d_state_part_weight, 0, matrix_count * sizeof(float));
  if (grads.sum_abs_d_prelude_norm_weight != nullptr) std::memset(grads.sum_abs_d_prelude_norm_weight, 0, state_dimension * sizeof(float));
  if (grads.sum_abs_d_block_qkv_weight != nullptr) std::memset(grads.sum_abs_d_block_qkv_weight, 0, qkv_count * sizeof(float));
  if (grads.sum_abs_d_block_qkv_bias != nullptr) std::memset(grads.sum_abs_d_block_qkv_bias, 0, qkv_bias_count * sizeof(float));
  if (grads.sum_abs_d_block_out_weight != nullptr) std::memset(grads.sum_abs_d_block_out_weight, 0, dimension * dimension * sizeof(float));
  if (grads.sum_abs_d_block_out_bias != nullptr) std::memset(grads.sum_abs_d_block_out_bias, 0, dimension * sizeof(float));
  if (grads.sum_abs_d_block_fc1_weight != nullptr) std::memset(grads.sum_abs_d_block_fc1_weight, 0, fc1_count * sizeof(float));
  if (grads.sum_abs_d_block_fc1_bias != nullptr) std::memset(grads.sum_abs_d_block_fc1_bias, 0, 4 * dimension * sizeof(float));
  if (grads.sum_abs_d_block_fc2_weight != nullptr) std::memset(grads.sum_abs_d_block_fc2_weight, 0, fc2_count * sizeof(float));
  if (grads.sum_abs_d_block_fc2_bias != nullptr) std::memset(grads.sum_abs_d_block_fc2_bias, 0, dimension * sizeof(float));
  if (grads.sum_abs_d_block_norm_weight != nullptr) std::memset(grads.sum_abs_d_block_norm_weight, 0, dimension * sizeof(float));
  if (grads.sum_abs_d_depth_embedding != nullptr) std::memset(grads.sum_abs_d_depth_embedding, 0, config.rounds * dimension * sizeof(float));
  if (grads.sum_abs_d_gate_logits != nullptr) std::memset(grads.sum_abs_d_gate_logits, 0, config.rounds * dimension * sizeof(float));
  if (grads.fp64_d_depth_embedding != nullptr) std::memset(grads.fp64_d_depth_embedding, 0, config.rounds * dimension * sizeof(double));
  if (grads.depth_max_level != nullptr) std::memset(grads.depth_max_level, 0, config.rounds * dimension * sizeof(size_t));
  if (grads.count_d_token_part != nullptr) std::memset(grads.count_d_token_part, 0, config.batch * config.sequence_length * state_dimension * sizeof(size_t));
  if (grads.count_d_previous_state != nullptr) std::memset(grads.count_d_previous_state, 0, state_count * sizeof(size_t));
  if (grads.count_d_state_part_weight != nullptr) std::memset(grads.count_d_state_part_weight, 0, matrix_count * sizeof(size_t));
  if (grads.count_d_prelude_norm_weight != nullptr) std::memset(grads.count_d_prelude_norm_weight, 0, state_dimension * sizeof(size_t));
  if (grads.count_d_block_qkv_weight != nullptr) std::memset(grads.count_d_block_qkv_weight, 0, qkv_count * sizeof(size_t));
  if (grads.count_d_block_qkv_bias != nullptr) std::memset(grads.count_d_block_qkv_bias, 0, qkv_bias_count * sizeof(size_t));
  if (grads.count_d_block_out_weight != nullptr) std::memset(grads.count_d_block_out_weight, 0, dimension * dimension * sizeof(size_t));
  if (grads.count_d_block_out_bias != nullptr) std::memset(grads.count_d_block_out_bias, 0, dimension * sizeof(size_t));
  if (grads.count_d_block_fc1_weight != nullptr) std::memset(grads.count_d_block_fc1_weight, 0, fc1_count * sizeof(size_t));
  if (grads.count_d_block_fc1_bias != nullptr) std::memset(grads.count_d_block_fc1_bias, 0, 4 * dimension * sizeof(size_t));
  if (grads.count_d_block_fc2_weight != nullptr) std::memset(grads.count_d_block_fc2_weight, 0, fc2_count * sizeof(size_t));
  if (grads.count_d_block_fc2_bias != nullptr) std::memset(grads.count_d_block_fc2_bias, 0, dimension * sizeof(size_t));
  if (grads.count_d_block_norm_weight != nullptr) std::memset(grads.count_d_block_norm_weight, 0, dimension * sizeof(size_t));
  if (grads.count_d_depth_embedding != nullptr) std::memset(grads.count_d_depth_embedding, 0, config.rounds * dimension * sizeof(size_t));
  if (grads.count_d_gate_logits != nullptr) std::memset(grads.count_d_gate_logits, 0, config.rounds * dimension * sizeof(size_t));
}

void add_abs_contribution(float* sums, size_t* counts, size_t index, float contribution) {
  if (sums != nullptr) sums[index] += std::fabs(contribution);
  if (counts != nullptr) ++counts[index];
}

#if OMEGA_HAS_AVX2
void accumulate_fc2_dweight_slot_avx2(
    const float* activated,
    const float* output_gradient,
    float* dweight,
    float* sum_abs,
    size_t* counts,
    size_t output_count,
    size_t hidden_count) {
  for (size_t output = 0; output < output_count; ++output) {
    const __m256 output_gradient_vector = _mm256_set1_ps(output_gradient[output]);
    size_t hidden = 0;
    for (; hidden + 8 <= hidden_count; hidden += 8) {
      const __m256 hidden_vector = _mm256_loadu_ps(activated + hidden);
      float* gradient = dweight + output * hidden_count + hidden;
      const __m256 contribution = _mm256_mul_ps(hidden_vector, output_gradient_vector);
      _mm256_storeu_ps(gradient, _mm256_add_ps(_mm256_loadu_ps(gradient), contribution));
      for (size_t lane = 0; lane < 8; ++lane) {
        add_abs_contribution(sum_abs, counts, output * hidden_count + hidden + lane,
            activated[hidden + lane] * output_gradient[output]);
      }
    }
    for (; hidden < hidden_count; ++hidden) {
      const float contribution = output_gradient[output] * activated[hidden];
      dweight[output * hidden_count + hidden] += contribution;
      add_abs_contribution(sum_abs, counts, output * hidden_count + hidden, contribution);
    }
  }
}

void accumulate_fc2_dweight_slot_diagnostics_avx2(
    const float* activated,
    const float* output_gradient,
    float* sum_abs,
    size_t* counts,
    size_t output_count,
    size_t hidden_count) {
  for (size_t output = 0; output < output_count; ++output) {
    const float gradient = output_gradient[output];
    size_t hidden = 0;
    for (; hidden + 8 <= hidden_count; hidden += 8) {
      for (size_t lane = 0; lane < 8; ++lane) {
        add_abs_contribution(sum_abs, counts, output * hidden_count + hidden + lane,
            activated[hidden + lane] * gradient);
      }
    }
    for (; hidden < hidden_count; ++hidden) {
      const float contribution = gradient * activated[hidden];
      add_abs_contribution(sum_abs, counts, output * hidden_count + hidden, contribution);
    }
  }
}
#endif

bool add_depth_hidden_vector(float* partials, std::uint64_t* masks, size_t round, size_t levels,
    size_t hidden_dimension, const float* source) {
  size_t round_base = 0;
  if (!checked_mul(round, levels, &round_base) || !checked_mul(round_base, hidden_dimension, &round_base)) {
    return false;
  }
  std::uint64_t mask = masks[round];
  const float* carry_source = source;
  for (size_t level = 0; level < levels; ++level) {
    const size_t level_base = round_base + level * hidden_dimension;
    const std::uint64_t bit = std::uint64_t{1} << level;
    if ((mask & bit) == 0) {
      std::memcpy(partials + level_base, carry_source, hidden_dimension * sizeof(float));
      masks[round] = mask | bit;
      return true;
    }
#if OMEGA_HAS_AVX2
    size_t index = 0;
    for (; index + 8 <= hidden_dimension; index += 8) {
      _mm256_storeu_ps(partials + level_base + index, _mm256_add_ps(
          _mm256_loadu_ps(partials + level_base + index), _mm256_loadu_ps(carry_source + index)));
    }
    for (; index < hidden_dimension; ++index) partials[level_base + index] += carry_source[index];
#else
    for (size_t index = 0; index < hidden_dimension; ++index) {
      partials[level_base + index] += carry_source[index];
    }
#endif
    carry_source = partials + level_base;
    mask &= ~bit;
  }
  return false;
}

}  // namespace

#ifdef OMEGA_PROFILE_INTERNAL
extern "C" void omega_recurrent_profile_set_enabled(int enabled) {
  g_profile_enabled.store(enabled != 0, std::memory_order_relaxed);
}

extern "C" void omega_recurrent_profile_reset(void) {
  std::lock_guard<std::mutex> lock(g_profile_mutex);
  g_profile_snapshot = OmegaRecurrentProfileSnapshot{};
}

extern "C" int omega_recurrent_profile_snapshot(OmegaRecurrentProfileSnapshot* snapshot) {
  if (snapshot == nullptr) return 1;
  {
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    *snapshot = g_profile_snapshot;
  }
  snapshot->enabled = g_profile_enabled.load(std::memory_order_relaxed) ? 1 : 0;
  snapshot->compiled = 1;
  return 0;
}
#endif

extern "C" size_t omega_recurrent_workspace_bytes(OmegaRecurrentConfig config) {
  return workspace_bytes_impl(config);
}

int recurrent_forward_impl(
    const OmegaRecurrentConfig* config,
    const OmegaRecurrentParams* params,
    const float* token_part,
    const float* previous_state,
    float* next_state,
    float* readout_states,
    void* workspace,
    size_t workspace_bytes) {
  if (config == nullptr || params == nullptr || token_part == nullptr || previous_state == nullptr || next_state == nullptr || readout_states == nullptr || workspace == nullptr || !valid_params(*params, *config)) {
    return 1;
  }
  const size_t required = workspace_bytes_impl(*config);
  if (required == 0 || workspace_bytes < required) {
    return 2;
  }
#if OMEGA_FC2_GELU_COUNT_GUARD_ACTIVE
  size_t expected_activations = 0;
  size_t expected_hidden_count = 0;
  if (!checked_mul(4, config->dimension, &expected_hidden_count) ||
      !checked_mul(config->batch, config->sequence_length, &expected_activations) ||
      !checked_mul(expected_activations, config->rounds, &expected_activations) ||
      !checked_mul(expected_activations, config->slots, &expected_activations) ||
      !checked_mul(expected_activations, expected_hidden_count, &expected_activations)) {
    std::fprintf(stderr,
        "FC2 GELU count guard batch=%zu sequence_length=%zu rounds=%zu slots=%zu dimension=%zu training=%d instrumentation=%d actual=0 expected=overflow pass=0\n",
        config->batch, config->sequence_length, config->rounds, config->slots, config->dimension,
        config->training, config->instrumentation);
    return 3;
  }
  size_t actual_activations = 0;
  bool actual_activations_overflow = false;
#endif
  OMEGA_PROFILE_CALL(ProfileDirection::kForward);

  const size_t state_count = element_count(*config);
  size_t slot_scores = 0;
  if (!checked_mul(config->batch, config->slots, &slot_scores) || !checked_mul(slot_scores, config->slots, &slot_scores)) {
    return 3;
  }
  size_t fc1_count = 0;
  if (!checked_mul(state_count, 4, &fc1_count)) {
    return 3;
  }
  size_t activation_count = 0;
  if (!checked_mul(4, config->dimension, &activation_count)) {
    return 3;
  }
  size_t depth_bias_count = 0;
  if (!checked_mul(config->rounds, config->dimension, &depth_bias_count) || !checked_mul(depth_bias_count, 4, &depth_bias_count)) {
    return 3;
  }
  WorkspaceCursor cursor{reinterpret_cast<unsigned char*>(workspace), workspace_bytes, 0};
  float* state = cursor.take(state_count);
  float* anchor = cursor.take(state_count);
  float* candidate = cursor.take(state_count);
  float* q = cursor.take(state_count);
  float* key = cursor.take(state_count);
  float* value = cursor.take(state_count);
  float* mixed = cursor.take(state_count);
  float* update = cursor.take(state_count);
  float* scores = cursor.take(slot_scores);
  float* fc1 = cursor.take(fc1_count);
  float* gelu_activated = cursor.take(activation_count);
  float* depth_bias = cursor.take(depth_bias_count);
  if (state == nullptr || anchor == nullptr || candidate == nullptr || q == nullptr || key == nullptr || value == nullptr || mixed == nullptr || update == nullptr || scores == nullptr || fc1 == nullptr || gelu_activated == nullptr || depth_bias == nullptr) {
    return 4;
  }
  TrainingBuffers training_buffers;
  if (!bind_training_storage(*config, state_count, &cursor, &training_buffers)) {
    return 5;
  }

  std::memcpy(state, previous_state, state_count * sizeof(float));
  const size_t dimension = config->dimension;
  const size_t slots = config->slots;
  const size_t state_dimension = slots * dimension;

  if (diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_DEPTH_EMBEDDING_PAIRWISE)) {
    std::memset(depth_bias, 0, depth_bias_count * sizeof(float));
  } else {
    OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kDepthEmbeddingPairwiseReduction);
    for (size_t round = 0; round < config->rounds; ++round) {
      for (size_t row = 0; row < 4 * dimension; ++row) {
        float sum = params->block_fc1_bias[row];
        for (size_t column = 0; column < dimension; ++column) {
          sum += params->block_fc1_weight[row * dimension + column] * params->depth_embedding[round * dimension + column];
        }
        depth_bias[round * 4 * dimension + row] = sum;
      }
    }
  }

  for (size_t position = 0; position < config->sequence_length; ++position) {
    if (config->training != 0 && !diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_HISTORY_BUFFER)) {
      OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kHistoryBufferReadsWrites);
      std::memcpy(training_buffers.state_history + position * state_count, state, state_count * sizeof(float));
    }
     for (size_t batch = 0; batch < config->batch; ++batch) {
      if (diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_STATE_PRELUDE)) {
        std::memcpy(anchor + batch * state_dimension,
            token_part + (batch * config->sequence_length + position) * state_dimension,
            state_dimension * sizeof(float));
      } else {
        OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kStatePrelude);
        float* mean_state = mixed + batch * state_dimension;
#if OMEGA_HAS_AVX2
        const __m256 inverse_slots = _mm256_set1_ps(1.0F / static_cast<float>(slots));
        size_t input = 0;
        for (; input + 8 <= dimension; input += 8) {
          __m256 mean = _mm256_setzero_ps();
          for (size_t slot = 0; slot < slots; ++slot) {
            mean = _mm256_add_ps(mean, _mm256_loadu_ps(state + state_index(batch, slot, input, slots, dimension)));
          }
          _mm256_storeu_ps(mean_state + input, _mm256_mul_ps(mean, inverse_slots));
        }
        for (; input < dimension; ++input) {
          float mean = 0.0F;
          for (size_t slot = 0; slot < slots; ++slot) {
            mean += state[state_index(batch, slot, input, slots, dimension)];
          }
          mean_state[input] = mean / static_cast<float>(slots);
        }
#else
        for (size_t input = 0; input < dimension; ++input) {
          float mean = 0.0F;
          for (size_t slot = 0; slot < slots; ++slot) {
            mean += state[state_index(batch, slot, input, slots, dimension)];
          }
          mean_state[input] = mean / static_cast<float>(slots);
        }
#endif
        for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
          const float* token_row = token_part + (batch * config->sequence_length + position) * state_dimension;
          anchor[batch * state_dimension + flattened] = state_prelude_write(
              token_row, params->state_part_weight, flattened, mean_state, dimension);
        }
      }

      if (diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_RMSNORM_GATES)) {
        for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
          anchor[batch * state_dimension + flattened] += state[batch * state_dimension + flattened];
        }
      } else {
        OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kRmsnormGates);
        float sum_squares = 0.0F;
        for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
          const float value_at = state[batch * state_dimension + flattened] + anchor[batch * state_dimension + flattened];
          sum_squares += value_at * value_at;
        }
        const float inverse_rms = 1.0F / std::sqrt(sum_squares / static_cast<float>(state_dimension) + kEpsilon);
        for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
          anchor[batch * state_dimension + flattened] =
              (state[batch * state_dimension + flattened] + anchor[batch * state_dimension + flattened]) * inverse_rms * params->prelude_norm_weight[flattened];
        }
      }
    }
    if (config->training != 0 && !diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_HISTORY_BUFFER)) {
      OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kHistoryBufferReadsWrites);
      std::memcpy(training_buffers.anchor_history + position * state_count, anchor, state_count * sizeof(float));
    }

    std::memcpy(candidate, anchor, state_count * sizeof(float));
    for (size_t round = 0; round < config->rounds; ++round) {
      if (config->training != 0 && !diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_HISTORY_BUFFER)) {
        const size_t round_offset = (position * config->rounds + round) * state_count;
        OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kHistoryBufferReadsWrites);
        std::memcpy(training_buffers.candidate_inputs + round_offset, candidate, state_count * sizeof(float));
      }
      for (size_t batch = 0; batch < config->batch; ++batch) {
          {
            OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kQkvProjection);
#if OMEGA_HAS_AVX2
            for (size_t slot = 0; slot < slots; ++slot) {
              const size_t base = state_index(batch, slot, 0, slots, dimension);
              float* input_values = mixed + base;
              for (size_t input = 0; input + 8 <= dimension; input += 8) {
                _mm256_storeu_ps(input_values + input, _mm256_add_ps(
                    _mm256_loadu_ps(candidate + base + input), _mm256_loadu_ps(anchor + base + input)));
              }
              for (size_t input = dimension & ~size_t{7}; input < dimension; ++input) {
                input_values[input] = candidate[base + input] + anchor[base + input];
              }
              for (size_t group = 0; group < 3; ++group) {
                size_t d = 0;
                for (; d + 4 <= dimension; d += 4) {
                  __m256 accumulators[4] = {
                      _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
                  size_t input = 0;
                  for (; input + 8 <= dimension; input += 8) {
                    const __m256 input_vector = _mm256_loadu_ps(input_values + input);
                    for (size_t output = 0; output < 4; ++output) {
                      const float* weight_row = params->block_qkv_weight +
                          (group * dimension + d + output) * dimension + input;
                      accumulators[output] = _mm256_add_ps(accumulators[output], _mm256_mul_ps(
                          _mm256_loadu_ps(weight_row), input_vector));
                    }
                  }
                  float values[4] = {
                      params->block_qkv_bias[group * dimension + d + 0],
                      params->block_qkv_bias[group * dimension + d + 1],
                      params->block_qkv_bias[group * dimension + d + 2],
                      params->block_qkv_bias[group * dimension + d + 3]};
                  for (size_t output = 0; output < 4; ++output) {
                    values[output] += horizontal_sum(accumulators[output]);
                  }
                  for (; input < dimension; ++input) {
                    const float input_value = input_values[input];
                    for (size_t output = 0; output < 4; ++output) {
                      values[output] += params->block_qkv_weight[
                          (group * dimension + d + output) * dimension + input] * input_value;
                    }
                  }
                  if (group == 0) {
                    q[base + d + 0] = values[0];
                    q[base + d + 1] = values[1];
                    q[base + d + 2] = values[2];
                    q[base + d + 3] = values[3];
                  } else if (group == 1) {
                    key[base + d + 0] = values[0];
                    key[base + d + 1] = values[1];
                    key[base + d + 2] = values[2];
                    key[base + d + 3] = values[3];
                  } else {
                    value[base + d + 0] = values[0];
                    value[base + d + 1] = values[1];
                    value[base + d + 2] = values[2];
                    value[base + d + 3] = values[3];
                  }
                }
                for (; d < dimension; ++d) {
                  float output_value = params->block_qkv_bias[group * dimension + d];
                  for (size_t input = 0; input < dimension; ++input) {
                    output_value += params->block_qkv_weight[(group * dimension + d) * dimension + input] * input_values[input];
                  }
                  if (group == 0) q[base + d] = output_value;
                  else if (group == 1) key[base + d] = output_value;
                  else value[base + d] = output_value;
                }
              }
            }
#else
            for (size_t slot = 0; slot < slots; ++slot) {
           for (size_t d = 0; d < dimension; ++d) {
            float q_value = params->block_qkv_bias[d];
            float k_value = params->block_qkv_bias[dimension + d];
            float v_value = params->block_qkv_bias[2 * dimension + d];
            for (size_t input_dimension = 0; input_dimension < dimension; ++input_dimension) {
              const float block_input = candidate[state_index(batch, slot, input_dimension, slots, dimension)] + anchor[state_index(batch, slot, input_dimension, slots, dimension)];
              q_value += params->block_qkv_weight[d * dimension + input_dimension] * block_input;
              k_value += params->block_qkv_weight[(dimension + d) * dimension + input_dimension] * block_input;
              v_value += params->block_qkv_weight[(2 * dimension + d) * dimension + input_dimension] * block_input;
            }
            q[state_index(batch, slot, d, slots, dimension)] = q_value;
            key[state_index(batch, slot, d, slots, dimension)] = k_value;
            value[state_index(batch, slot, d, slots, dimension)] = v_value;
            }
           }
#endif
           }

           if (diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_ATTENTION_SCORES_SOFTMAX_MIXING)) {
             std::memset(mixed + batch * state_dimension, 0, state_dimension * sizeof(float));
             std::memset(scores + batch * slots * slots, 0, slots * slots * sizeof(float));
           } else {
             OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kAttentionScoresSoftmaxMixing);
             const float scale = 1.0F / std::sqrt(static_cast<float>(dimension));
             for (size_t query_slot = 0; query_slot < slots; ++query_slot) {
               float maximum = -std::numeric_limits<float>::infinity();
               for (size_t key_slot = 0; key_slot < slots; ++key_slot) {
                 float score = 0.0F;
                 for (size_t d = 0; d < dimension; ++d) {
                   score += q[state_index(batch, query_slot, d, slots, dimension)] * key[state_index(batch, key_slot, d, slots, dimension)];
                 }
                 score *= scale;
                 scores[(batch * slots + query_slot) * slots + key_slot] = score;
                 maximum = std::max(maximum, score);
               }
               float denominator = 0.0F;
               for (size_t key_slot = 0; key_slot < slots; ++key_slot) {
                 float probability = std::exp(scores[(batch * slots + query_slot) * slots + key_slot] - maximum);
                 scores[(batch * slots + query_slot) * slots + key_slot] = probability;
                 denominator += probability;
               }
               for (size_t key_slot = 0; key_slot < slots; ++key_slot) {
                 scores[(batch * slots + query_slot) * slots + key_slot] /= denominator;
               }
               for (size_t d = 0; d < dimension; ++d) {
                 float attention = 0.0F;
                 for (size_t key_slot = 0; key_slot < slots; ++key_slot) {
                   attention += scores[(batch * slots + query_slot) * slots + key_slot] * value[state_index(batch, key_slot, d, slots, dimension)];
                 }
                 mixed[state_index(batch, query_slot, d, slots, dimension)] = attention;
               }
             }
           }

        for (size_t slot = 0; slot < slots; ++slot) {
          {
            OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kOutProjection);
#if OMEGA_HAS_AVX2
            size_t d = 0;
            for (; d + 4 <= dimension; d += 4) {
              __m256 accumulators[4] = {
                  _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
              size_t input_dimension = 0;
              for (; input_dimension + 8 <= dimension; input_dimension += 8) {
                const __m256 input_vector = _mm256_loadu_ps(
                    mixed + state_index(batch, slot, input_dimension, slots, dimension));
                for (size_t output = 0; output < 4; ++output) {
                  const float* weight_row = params->block_out_weight + (d + output) * dimension + input_dimension;
                  accumulators[output] = _mm256_add_ps(accumulators[output], _mm256_mul_ps(
                      _mm256_loadu_ps(weight_row), input_vector));
                }
              }
              float values[4] = {
                  params->block_out_bias[d], params->block_out_bias[d + 1],
                  params->block_out_bias[d + 2], params->block_out_bias[d + 3]};
              for (size_t output = 0; output < 4; ++output) {
                values[output] += horizontal_sum(accumulators[output]);
              }
              for (; input_dimension < dimension; ++input_dimension) {
                const float input_value = mixed[state_index(batch, slot, input_dimension, slots, dimension)];
                for (size_t output = 0; output < 4; ++output) {
                  values[output] += params->block_out_weight[(d + output) * dimension + input_dimension] * input_value;
                }
              }
              for (size_t output = 0; output < 4; ++output) {
                update[state_index(batch, slot, d + output, slots, dimension)] = values[output];
              }
            }
            for (; d < dimension; ++d) {
              float output = params->block_out_bias[d];
              for (size_t input_dimension = 0; input_dimension < dimension; ++input_dimension) {
                output += params->block_out_weight[d * dimension + input_dimension] *
                    mixed[state_index(batch, slot, input_dimension, slots, dimension)];
              }
              update[state_index(batch, slot, d, slots, dimension)] = output;
            }
#else
          for (size_t d = 0; d < dimension; ++d) {
            float output = params->block_out_bias[d];
            for (size_t input_dimension = 0; input_dimension < dimension; ++input_dimension) {
              output += params->block_out_weight[d * dimension + input_dimension] * mixed[state_index(batch, slot, input_dimension, slots, dimension)];
            }
            update[state_index(batch, slot, d, slots, dimension)] = output;
          }
#endif
          }
    if (config->training != 0 && !diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_HISTORY_BUFFER)) {
            const size_t round_offset = (position * config->rounds + round) * state_count;
            std::memcpy(training_buffers.out_history + round_offset + batch * state_dimension + slot * dimension,
                update + batch * state_dimension + slot * dimension, dimension * sizeof(float));
          }
          {
            OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kFc1Gelu);
#if OMEGA_HAS_AVX2
            const float* input_values = update + state_index(batch, slot, 0, slots, dimension);
            const size_t hidden_count = 4 * dimension;
            size_t row = 0;
            for (; row + 4 <= hidden_count; row += 4) {
              __m256 accumulators[4][4] = {
                  {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()},
                  {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()},
                  {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()},
                  {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()}};
              size_t input_dimension = 0;
              for (; input_dimension + 32 <= dimension; input_dimension += 32) {
                const __m256 input0 = _mm256_loadu_ps(input_values + input_dimension);
                const __m256 input1 = _mm256_loadu_ps(input_values + input_dimension + 8);
                const __m256 input2 = _mm256_loadu_ps(input_values + input_dimension + 16);
                const __m256 input3 = _mm256_loadu_ps(input_values + input_dimension + 24);
                for (size_t output = 0; output < 4; ++output) {
                  const float* weight_row = params->block_fc1_weight + (row + output) * dimension + input_dimension;
                  accumulators[output][0] = _mm256_add_ps(accumulators[output][0], _mm256_mul_ps(
                      _mm256_loadu_ps(weight_row), input0));
                  accumulators[output][1] = _mm256_add_ps(accumulators[output][1], _mm256_mul_ps(
                      _mm256_loadu_ps(weight_row + 8), input1));
                  accumulators[output][2] = _mm256_add_ps(accumulators[output][2], _mm256_mul_ps(
                      _mm256_loadu_ps(weight_row + 16), input2));
                  accumulators[output][3] = _mm256_add_ps(accumulators[output][3], _mm256_mul_ps(
                      _mm256_loadu_ps(weight_row + 24), input3));
                }
              }
              for (; input_dimension + 8 <= dimension; input_dimension += 8) {
                const __m256 input_vector = _mm256_loadu_ps(input_values + input_dimension);
                for (size_t output = 0; output < 4; ++output) {
                  accumulators[output][0] = _mm256_add_ps(accumulators[output][0], _mm256_mul_ps(
                      _mm256_loadu_ps(params->block_fc1_weight + (row + output) * dimension + input_dimension), input_vector));
                }
              }
              float hidden_values[4] = {
                  depth_bias[round * hidden_count + row], depth_bias[round * hidden_count + row + 1],
                  depth_bias[round * hidden_count + row + 2], depth_bias[round * hidden_count + row + 3]};
              for (size_t output = 0; output < 4; ++output) {
                hidden_values[output] += horizontal_sum(accumulators[output][0]) + horizontal_sum(accumulators[output][1]) +
                    horizontal_sum(accumulators[output][2]) + horizontal_sum(accumulators[output][3]);
              }
              for (; input_dimension < dimension; ++input_dimension) {
                const float input_value = input_values[input_dimension];
                hidden_values[0] += params->block_fc1_weight[row * dimension + input_dimension] * input_value;
                hidden_values[1] += params->block_fc1_weight[(row + 1) * dimension + input_dimension] * input_value;
                hidden_values[2] += params->block_fc1_weight[(row + 2) * dimension + input_dimension] * input_value;
                hidden_values[3] += params->block_fc1_weight[(row + 3) * dimension + input_dimension] * input_value;
              }
              fc1[(batch * slots + slot) * hidden_count + row] = hidden_values[0];
              fc1[(batch * slots + slot) * hidden_count + row + 1] = hidden_values[1];
              fc1[(batch * slots + slot) * hidden_count + row + 2] = hidden_values[2];
              fc1[(batch * slots + slot) * hidden_count + row + 3] = hidden_values[3];
            }
            for (; row < hidden_count; ++row) {
              float hidden = depth_bias[round * hidden_count + row];
              for (size_t input_dimension = 0; input_dimension < dimension; ++input_dimension) {
                hidden += params->block_fc1_weight[row * dimension + input_dimension] * input_values[input_dimension];
              }
              fc1[(batch * slots + slot) * hidden_count + row] = hidden;
            }
#else
            for (size_t row = 0; row < 4 * dimension; ++row) {
              float hidden = depth_bias[round * 4 * dimension + row];
              for (size_t input_dimension = 0; input_dimension < dimension; ++input_dimension) {
                hidden += params->block_fc1_weight[row * dimension + input_dimension] * update[state_index(batch, slot, input_dimension, slots, dimension)];
              }
              fc1[(batch * slots + slot) * 4 * dimension + row] = hidden;
            }
#endif
          }
          float sum_squares_update = 0.0F;
          {
            OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kFc2);
            const size_t hidden_count = activation_count;
            const size_t fc1_base = (batch * slots + slot) * hidden_count;
            const float* hidden_input = fc1 + fc1_base;
            for (size_t row = 0; row < hidden_count; ++row) {
              gelu_activated[row] = gelu(hidden_input[row]);
#if OMEGA_FC2_GELU_COUNT_GUARD_ACTIVE
              size_t next_actual_activations = 0;
              if (!checked_add(actual_activations, 1, &next_actual_activations)) {
                actual_activations_overflow = true;
              } else {
                actual_activations = next_actual_activations;
              }
#endif
            }
#if OMEGA_HAS_AVX2
            size_t d = 0;
            for (; d + 4 <= dimension; d += 4) {
              __m256 accumulators[4][4] = {
                  {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()},
                  {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()},
                  {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()},
                  {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()}};
              size_t row = 0;
              for (; row + 32 <= hidden_count; row += 32) {
                const __m256 input0 = _mm256_loadu_ps(gelu_activated + row);
                const __m256 input1 = _mm256_loadu_ps(gelu_activated + row + 8);
                const __m256 input2 = _mm256_loadu_ps(gelu_activated + row + 16);
                const __m256 input3 = _mm256_loadu_ps(gelu_activated + row + 24);
                for (size_t output = 0; output < 4; ++output) {
                  const float* weight = params->block_fc2_weight + (d + output) * hidden_count + row;
                  accumulators[output][0] = _mm256_add_ps(accumulators[output][0], _mm256_mul_ps(
                      _mm256_loadu_ps(weight), input0));
                  accumulators[output][1] = _mm256_add_ps(accumulators[output][1], _mm256_mul_ps(
                      _mm256_loadu_ps(weight + 8), input1));
                  accumulators[output][2] = _mm256_add_ps(accumulators[output][2], _mm256_mul_ps(
                      _mm256_loadu_ps(weight + 16), input2));
                  accumulators[output][3] = _mm256_add_ps(accumulators[output][3], _mm256_mul_ps(
                      _mm256_loadu_ps(weight + 24), input3));
                }
              }
              for (; row + 8 <= hidden_count; row += 8) {
                const __m256 input = _mm256_loadu_ps(gelu_activated + row);
                for (size_t output = 0; output < 4; ++output) {
                  accumulators[output][0] = _mm256_add_ps(accumulators[output][0], _mm256_mul_ps(
                      _mm256_loadu_ps(params->block_fc2_weight + (d + output) * hidden_count + row), input));
                }
              }
              for (size_t output = 0; output < 4; ++output) {
                float transformed = params->block_fc2_bias[d + output];
                transformed += horizontal_sum(accumulators[output][0]) + horizontal_sum(accumulators[output][1]) +
                    horizontal_sum(accumulators[output][2]) + horizontal_sum(accumulators[output][3]);
                const float gate = diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_RMSNORM_GATES)
                    ? 1.0F : sigmoid(params->gate_logits[round * dimension + d + output]);
                 const float value_at = candidate[state_index(batch, slot, d + output, slots, dimension)] + gate * transformed;
                 update[state_index(batch, slot, d + output, slots, dimension)] = value_at;
              }
            }
            for (; d < dimension; ++d) {
              float transformed = params->block_fc2_bias[d];
              for (size_t row = 0; row < hidden_count; ++row) {
                transformed += params->block_fc2_weight[d * hidden_count + row] * gelu_activated[row];
              }
                const float gate = diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_RMSNORM_GATES)
                    ? 1.0F : sigmoid(params->gate_logits[round * dimension + d]);
               const float value_at = candidate[state_index(batch, slot, d, slots, dimension)] + gate * transformed;
               update[state_index(batch, slot, d, slots, dimension)] = value_at;
            }
#else
            for (size_t d = 0; d < dimension; ++d) {
              float transformed = params->block_fc2_bias[d];
              for (size_t row = 0; row < 4 * dimension; ++row) {
                transformed += params->block_fc2_weight[d * 4 * dimension + row] * gelu_activated[row];
              }
              const float gate = sigmoid(params->gate_logits[round * dimension + d]);
              const float value_at = candidate[state_index(batch, slot, d, slots, dimension)] + gate * transformed;
              update[state_index(batch, slot, d, slots, dimension)] = value_at;
            }
#endif
          }
          for (size_t d = 0; d < dimension; ++d) {
            const float value_at = update[state_index(batch, slot, d, slots, dimension)];
            sum_squares_update += value_at * value_at;
          }
          if (config->training != 0 && !diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_HISTORY_BUFFER)) {
            const size_t round_offset = (position * config->rounds + round) * state_count;
            OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kHistoryBufferReadsWrites);
            std::memcpy(training_buffers.pre_rms_history + round_offset + batch * state_dimension + slot * dimension,
                update + batch * state_dimension + slot * dimension, dimension * sizeof(float));
          }
          if (diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_RMSNORM_GATES)) {
            std::memcpy(candidate + batch * state_dimension + slot * dimension,
                update + batch * state_dimension + slot * dimension, dimension * sizeof(float));
          } else {
            OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kRmsnormGates);
            const float inverse_rms_update = 1.0F / std::sqrt(sum_squares_update / static_cast<float>(dimension) + kEpsilon);
            for (size_t d = 0; d < dimension; ++d) {
              candidate[state_index(batch, slot, d, slots, dimension)] = update[state_index(batch, slot, d, slots, dimension)] * inverse_rms_update * params->block_norm_weight[d];
            }
          }
        }
      }
      if (config->training != 0) {
        const size_t round_offset = (position * config->rounds + round) * state_count;
        const size_t probability_offset = (position * config->rounds + round) * slot_scores;
       if (!diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_HISTORY_BUFFER)) {
         OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kHistoryBufferReadsWrites);
         std::memcpy(training_buffers.q_history + round_offset, q, state_count * sizeof(float));
         std::memcpy(training_buffers.key_history + round_offset, key, state_count * sizeof(float));
         std::memcpy(training_buffers.value_history + round_offset, value, state_count * sizeof(float));
         std::memcpy(training_buffers.probability_history + probability_offset, scores, slot_scores * sizeof(float));
         std::memcpy(training_buffers.attention_history + round_offset, mixed, state_count * sizeof(float));
         std::memcpy(training_buffers.fc1_pre_history + round_offset * 4, fc1, fc1_count * sizeof(float));
       }
      }
    }

    for (size_t batch = 0; batch < config->batch; ++batch) {
      std::memcpy(state + batch * state_dimension, candidate + batch * state_dimension, state_dimension * sizeof(float));
      std::memcpy(readout_states + (batch * config->sequence_length + position) * state_dimension, candidate + batch * state_dimension, state_dimension * sizeof(float));
    }
    if (config->training != 0 && !diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_HISTORY_BUFFER)) {
      OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kHistoryBufferReadsWrites);
      std::memcpy(training_buffers.state_history + (position + 1) * state_count, state, state_count * sizeof(float));
    }
  }
  std::memcpy(next_state, state, state_count * sizeof(float));
#if OMEGA_FC2_GELU_COUNT_GUARD_ACTIVE
  if (actual_activations_overflow) {
    std::fprintf(stderr,
        "FC2 GELU count guard batch=%zu sequence_length=%zu rounds=%zu slots=%zu dimension=%zu training=%d instrumentation=%d actual=overflow expected=%zu pass=0\n",
        config->batch, config->sequence_length, config->rounds, config->slots, config->dimension,
        config->training, config->instrumentation, expected_activations);
    return 6;
  }
  const bool activation_count_pass = actual_activations == expected_activations;
  std::fprintf(stderr,
      "FC2 GELU count guard batch=%zu sequence_length=%zu rounds=%zu slots=%zu dimension=%zu training=%d instrumentation=%d actual=%zu expected=%zu pass=%d\n",
      config->batch, config->sequence_length, config->rounds, config->slots, config->dimension,
      config->training, config->instrumentation, actual_activations, expected_activations, activation_count_pass ? 1 : 0);
  if (!activation_count_pass) return 6;
#endif
  return 0;
}

template <bool CollectDiagnostics>
int recurrent_backward_impl_body(
    const OmegaRecurrentConfig* config,
    const OmegaRecurrentParams* params,
    const float* token_part,
    const float* d_readout_states,
    const float* d_next_state,
    void* workspace,
    size_t workspace_bytes,
    OmegaRecurrentGrads* grads) {
  if (config == nullptr || params == nullptr || token_part == nullptr || d_readout_states == nullptr || d_next_state == nullptr || workspace == nullptr || grads == nullptr || !valid_params(*params, *config) || !valid_grads(*grads) || config->training == 0) {
    return 10;
  }
  const size_t required = workspace_bytes_impl(*config);
  if (required == 0 || workspace_bytes < required) {
    return 11;
  }
  OMEGA_PROFILE_CALL(ProfileDirection::kBackward);
  const size_t state_count = element_count(*config);
  const size_t dimension = config->dimension;
  const size_t slots = config->slots;
  const size_t state_dimension = slots * dimension;
  size_t slot_scores = 0;
  size_t fc1_count = 0;
  size_t activation_count = 0;
  size_t depth_bias_count = 0;
  size_t depth_output_count = 0;
  size_t depth_hidden_dimension = 0;
  size_t depth_levels = 0;
  size_t depth_partial_count = 0;
  size_t grouped_activation_count = 0;
  size_t grouped_output_gradient_count = 0;
  const bool fc2_m8_workspace_enabled = slots == kFc2DWeightGroupSlots &&
      dimension == kFc2DWeightGroupDimension;
  if (!checked_mul(config->batch, slots, &slot_scores) || !checked_mul(slot_scores, slots, &slot_scores) ||
      !checked_mul(state_count, 4, &fc1_count) || !checked_mul(4, dimension, &activation_count) ||
      (fc2_m8_workspace_enabled &&
          (!checked_mul(slots, activation_count, &grouped_activation_count) ||
           !checked_mul(slots, dimension, &grouped_output_gradient_count))) ||
      !checked_mul(config->rounds, dimension, &depth_bias_count) ||
      !checked_mul(depth_bias_count, 4, &depth_bias_count) ||
      !checked_mul(config->rounds, dimension, &depth_output_count) ||
      !depth_vector_reduction_layout(*config, &depth_hidden_dimension, &depth_levels, &depth_partial_count)) {
    return 12;
  }
  WorkspaceCursor cursor{reinterpret_cast<unsigned char*>(workspace), workspace_bytes, 0};
  float* state = cursor.take(state_count);       /* d_state carry */
  float* anchor = cursor.take(state_count);      /* d_anchor */
  float* candidate = cursor.take(state_count);   /* d_candidate */
  float* q = cursor.take(state_count);           /* d_q */
  float* key = cursor.take(state_count);         /* d_key */
  float* value = cursor.take(state_count);       /* d_value */
  float* mixed = cursor.take(state_count);       /* d_out / d_attention */
  float* update = cursor.take(state_count);      /* d_u */
  float* scores = cursor.take(slot_scores);      /* d_scores */
  float* fc1 = cursor.take(fc1_count);           /* d_fc1_pre */
  float* gelu_activated = cursor.take(activation_count);
  float* depth_bias = cursor.take(depth_bias_count);
  if (state == nullptr || anchor == nullptr || candidate == nullptr || q == nullptr || key == nullptr || value == nullptr || mixed == nullptr || update == nullptr || scores == nullptr || fc1 == nullptr || gelu_activated == nullptr || depth_bias == nullptr) {
    return 13;
  }
  TrainingBuffers training_buffers;
  if (!bind_training_storage(*config, state_count, &cursor, &training_buffers)) {
    return 14;
  }
  if (diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_HISTORY_BUFFER)) {
    clear_training_storage(*config, state_count, &training_buffers);
  }
  float* fc2_group_activated = nullptr;
  float* fc2_group_output_gradient = nullptr;
  if (fc2_m8_workspace_enabled) {
    fc2_group_activated = cursor.take(grouped_activation_count);
    fc2_group_output_gradient = cursor.take(grouped_output_gradient_count);
    if (fc2_group_activated == nullptr || fc2_group_output_gradient == nullptr) return 14;
  }
  float* depth_partials = cursor.take(depth_partial_count);
  std::uint64_t* depth_masks = cursor.take_uint64(config->rounds);
  if (depth_partials == nullptr || depth_masks == nullptr) {
    return 15;
  }
  std::memset(depth_partials, 0, depth_partial_count * sizeof(float));
  std::memset(depth_masks, 0, config->rounds * sizeof(std::uint64_t));

#if OMEGA_HAS_AVX2
  bool use_fc2_m8_candidate = fc2_m8_workspace_enabled;
#ifdef OMEGA_FC2_REPLAY_CAPTURE
  /* Keep the Step-A capture DLL on the original per-slot control route. */
  use_fc2_m8_candidate = false;
#endif
#endif

  std::memset(grads->d_token_part, 0, config->batch * config->sequence_length * state_dimension * sizeof(float));
  std::memset(grads->d_previous_state, 0, state_count * sizeof(float));
  std::memset(grads->d_state_part_weight, 0, slots * dimension * dimension * sizeof(float));
  std::memset(grads->d_prelude_norm_weight, 0, state_dimension * sizeof(float));
  std::memset(grads->d_block_qkv_weight, 0, 3 * dimension * dimension * sizeof(float));
  std::memset(grads->d_block_qkv_bias, 0, 3 * dimension * sizeof(float));
  std::memset(grads->d_block_out_weight, 0, dimension * dimension * sizeof(float));
  std::memset(grads->d_block_out_bias, 0, dimension * sizeof(float));
  std::memset(grads->d_block_fc1_weight, 0, 4 * dimension * dimension * sizeof(float));
  std::memset(grads->d_block_fc1_bias, 0, 4 * dimension * sizeof(float));
  std::memset(grads->d_block_fc2_weight, 0, dimension * 4 * dimension * sizeof(float));
  std::memset(grads->d_block_fc2_bias, 0, dimension * sizeof(float));
  std::memset(grads->d_block_norm_weight, 0, dimension * sizeof(float));
  std::memset(grads->d_depth_embedding, 0, config->rounds * dimension * sizeof(float));
  std::memset(grads->d_gate_logits, 0, config->rounds * dimension * sizeof(float));
  clear_optional_contribution_sums(*config, *grads, state_count, state_dimension, dimension);
  std::memcpy(state, d_next_state, state_count * sizeof(float));

  const float scale = 1.0F / std::sqrt(static_cast<float>(dimension));
  for (size_t reverse_position = config->sequence_length; reverse_position-- > 0;) {
    const size_t position = reverse_position;
    for (size_t index = 0; index < state_count; ++index) {
      const size_t batch = index / state_dimension;
      const size_t local = index % state_dimension;
      candidate[index] = state[index] + d_readout_states[(batch * config->sequence_length + position) * state_dimension + local];
      anchor[index] = 0.0F;
    }
    for (size_t reverse_round = config->rounds; reverse_round-- > 0;) {
      const size_t round = reverse_round;
      const size_t round_offset = (position * config->rounds + round) * state_count;
      const size_t probability_offset = (position * config->rounds + round) * slot_scores;
      OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kHistoryBufferReadsWrites);
      const float* saved_candidate = training_buffers.candidate_inputs + round_offset;
      const float* saved_q = training_buffers.q_history + round_offset;
      const float* saved_key = training_buffers.key_history + round_offset;
      const float* saved_value = training_buffers.value_history + round_offset;
      const float* saved_probability = training_buffers.probability_history + probability_offset;
      const float* saved_attention = training_buffers.attention_history + round_offset;
      const float* saved_out = training_buffers.out_history + round_offset;
      const float* saved_fc1_pre = training_buffers.fc1_pre_history + round_offset * 4;
      const float* saved_pre_rms = training_buffers.pre_rms_history + round_offset;
       std::memset(q, 0, state_count * sizeof(float));
       std::memset(key, 0, state_count * sizeof(float));
       std::memset(value, 0, state_count * sizeof(float));

       if (diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_RMSNORM_GATES)) {
         std::memcpy(update, candidate, state_count * sizeof(float));
       } else {
       for (size_t batch = 0; batch < config->batch; ++batch) {
        for (size_t slot = 0; slot < slots; ++slot) {
           const size_t base = state_index(batch, slot, 0, slots, dimension);
           float rms_dot = 0.0F;
#if OMEGA_HAS_AVX2
           __m256 rms_dot_vector = _mm256_setzero_ps();
           size_t dot_dimension = 0;
           for (; dot_dimension + 8 <= dimension; dot_dimension += 8) {
             const __m256 candidate_vector = _mm256_loadu_ps(candidate + base + dot_dimension);
             const __m256 norm_weight_vector = _mm256_loadu_ps(params->block_norm_weight + dot_dimension);
             const __m256 saved_pre_vector = _mm256_loadu_ps(saved_pre_rms + base + dot_dimension);
             rms_dot_vector = _mm256_add_ps(rms_dot_vector, _mm256_mul_ps(
                 _mm256_mul_ps(candidate_vector, norm_weight_vector), saved_pre_vector));
           }
           rms_dot = horizontal_sum(rms_dot_vector);
           for (; dot_dimension < dimension; ++dot_dimension) {
             const size_t index = base + dot_dimension;
             rms_dot += candidate[index] * params->block_norm_weight[dot_dimension] * saved_pre_rms[index];
           }
#else
           for (size_t d = 0; d < dimension; ++d) {
             const size_t index = base + d;
             rms_dot += candidate[index] * params->block_norm_weight[d] * saved_pre_rms[index];
           }
#endif
           float pre_rms_sum = 0.0F;
#if OMEGA_HAS_AVX2
           __m256 pre_rms_vector = _mm256_setzero_ps();
           size_t sum_dimension = 0;
           for (; sum_dimension + 8 <= dimension; sum_dimension += 8) {
             const __m256 saved_pre_vector = _mm256_loadu_ps(saved_pre_rms + base + sum_dimension);
             pre_rms_vector = _mm256_add_ps(pre_rms_vector, _mm256_mul_ps(saved_pre_vector, saved_pre_vector));
           }
           pre_rms_sum = horizontal_sum(pre_rms_vector);
           for (; sum_dimension < dimension; ++sum_dimension) {
             const float pre = saved_pre_rms[base + sum_dimension];
             pre_rms_sum += pre * pre;
           }
#else
           for (size_t d = 0; d < dimension; ++d) {
             const float pre = saved_pre_rms[base + d];
             pre_rms_sum += pre * pre;
           }
#endif
           const float inverse_rms = 1.0F / std::sqrt(pre_rms_sum / static_cast<float>(dimension) + kEpsilon);
           const float inverse_rms_cubed_over_dimension = inverse_rms * inverse_rms * inverse_rms / static_cast<float>(dimension);
            for (size_t d = 0; d < dimension; ++d) {
             const size_t index = base + d;
             const float d_pre = candidate[index] * params->block_norm_weight[d] * inverse_rms - saved_pre_rms[index] * inverse_rms_cubed_over_dimension * rms_dot;
             const float gate = sigmoid(params->gate_logits[round * dimension + d]);
            const float transformed = (saved_pre_rms[index] - saved_candidate[index]) / gate;
            update[index] = d_pre * gate;
            const float block_norm_contribution = candidate[index] * saved_pre_rms[index] * inverse_rms;
            const float gate_contribution = d_pre * transformed * gate * (1.0F - gate);
            grads->d_block_norm_weight[d] += block_norm_contribution;
            grads->d_gate_logits[round * dimension + d] += gate_contribution;
             add_abs_contribution(grads->sum_abs_d_block_norm_weight, grads->count_d_block_norm_weight, d, block_norm_contribution);
              add_abs_contribution(grads->sum_abs_d_gate_logits, grads->count_d_gate_logits, round * dimension + d, gate_contribution);
            }

            {
              OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kFc2);
#if OMEGA_HAS_AVX2
              const size_t hidden_count = 4 * dimension;
              const size_t fc1_base = (batch * slots + slot) * hidden_count;
              float* hidden_values = fc1 + fc1_base;
              prepare_fc2_dweight_slot_avx2(saved_fc1_pre + fc1_base, update + base, hidden_values,
                  fc2_m8_workspace_enabled ? fc2_group_activated : nullptr,
                  fc2_m8_workspace_enabled ? fc2_group_output_gradient : nullptr,
                  slot, dimension, hidden_count);
#ifdef OMEGA_FC2_REPLAY_CAPTURE
              const size_t global_batch = g_fc2_replay_worker_index * config->batch + batch;
              if (slot == 0) {
                Fc2ReplayCaptureScratch& capture = g_fc2_replay_scratch;
                capture.active = slots == kFc2ReplaySlots && dimension == kFc2ReplayDimension &&
                    fc2_replay_group_selected(position, round, global_batch);
                if (capture.active) {
                  capture.position = position;
                  capture.round = round;
                  capture.batch = global_batch;
                  std::memcpy(capture.dweight_before.data(), grads->d_block_fc2_weight,
                      capture.dweight_before.size() * sizeof(float));
                }
              }
              if (g_fc2_replay_scratch.active) {
                Fc2ReplayCaptureScratch& capture = g_fc2_replay_scratch;
                std::memcpy(capture.preactivation.data() + slot * kFc2ReplayHidden,
                    saved_fc1_pre + fc1_base, kFc2ReplayHidden * sizeof(float));
                std::memcpy(capture.activated.data() + slot * kFc2ReplayHidden,
                    hidden_values, kFc2ReplayHidden * sizeof(float));
                std::memcpy(capture.output_gradient.data() + slot * kFc2ReplayDimension,
                    update + base, kFc2ReplayDimension * sizeof(float));
              }
#endif
              if (use_fc2_m8_candidate) {
                accumulate_fc2_dweight_slot_diagnostics_avx2(hidden_values, update + base,
                    grads->sum_abs_d_block_fc2_weight, grads->count_d_block_fc2_weight, dimension, hidden_count);
              } else {
                accumulate_fc2_dweight_slot_avx2(hidden_values, update + base,
                    grads->d_block_fc2_weight, grads->sum_abs_d_block_fc2_weight,
                    grads->count_d_block_fc2_weight, dimension, hidden_count);
              }
              if (use_fc2_m8_candidate && slot + 1 == slots) {
                omega_fc2_local_reduction::accumulate_dweight_m8(
                    fc2_group_activated, fc2_group_output_gradient, grads->d_block_fc2_weight,
                    dimension, hidden_count);
              }
#ifdef OMEGA_FC2_REPLAY_CAPTURE
              if (g_fc2_replay_scratch.active && slot + 1 == slots) {
                Fc2ReplayCaptureScratch& capture = g_fc2_replay_scratch;
                std::memcpy(capture.dweight_candidate.data(), capture.dweight_before.data(),
                    capture.dweight_before.size() * sizeof(float));
                omega_fc2_local_reduction::accumulate_dweight_m8(
                    capture.activated.data(), capture.output_gradient.data(), capture.dweight_candidate.data(),
                    kFc2ReplayDimension, kFc2ReplayHidden);
                g_fc2_replay_callback(g_fc2_replay_worker_index, capture.position, capture.round, capture.batch,
                    capture.preactivation.data(), capture.activated.data(), capture.output_gradient.data(),
                    capture.dweight_before.data(), grads->d_block_fc2_weight, capture.dweight_candidate.data(),
                    g_fc2_replay_user_data);
                capture.active = false;
              }
#endif
              size_t d = 0;
              for (; d + 8 <= dimension; d += 8) {
                const __m256 contribution = _mm256_loadu_ps(update + base + d);
                _mm256_storeu_ps(grads->d_block_fc2_bias + d, _mm256_add_ps(
                    _mm256_loadu_ps(grads->d_block_fc2_bias + d), contribution));
                for (size_t lane = 0; lane < 8; ++lane) {
                  add_abs_contribution(grads->sum_abs_d_block_fc2_bias, grads->count_d_block_fc2_bias,
                      d + lane, update[base + d + lane]);
                }
              }
              for (; d < dimension; ++d) {
                grads->d_block_fc2_bias[d] += update[base + d];
                add_abs_contribution(grads->sum_abs_d_block_fc2_bias, grads->count_d_block_fc2_bias,
                    d, update[base + d]);
              }
#else
              for (size_t d = 0; d < dimension; ++d) {
                for (size_t row = 0; row < 4 * dimension; ++row) {
                  const float hidden = gelu(saved_fc1_pre[(batch * slots + slot) * 4 * dimension + row]);
                  const float contribution = update[base + d] * hidden;
                  grads->d_block_fc2_weight[d * 4 * dimension + row] += contribution;
                  add_abs_contribution(grads->sum_abs_d_block_fc2_weight, grads->count_d_block_fc2_weight,
                      d * 4 * dimension + row, contribution);
                }
                grads->d_block_fc2_bias[d] += update[base + d];
                add_abs_contribution(grads->sum_abs_d_block_fc2_bias, grads->count_d_block_fc2_bias,
                    d, update[base + d]);
              }
#endif
            }
            {
             OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kFc1Gelu);
             for (size_t d = 0; d < dimension; ++d) {
               mixed[base + d] = 0.0F;
              }
  #if OMEGA_HAS_AVX2
               const size_t hidden_count = 4 * dimension;
               const size_t fc1_base = (batch * slots + slot) * hidden_count;
               float* d_hidden_values = fc1 + fc1_base;
               std::memset(d_hidden_values, 0, hidden_count * sizeof(float));
                size_t row = 0;
                for (; row + 8 <= hidden_count; row += 8) {
                  __m256 accumulator0 = _mm256_setzero_ps();
                  __m256 accumulator1 = _mm256_setzero_ps();
                  __m256 accumulator2 = _mm256_setzero_ps();
                  __m256 accumulator3 = _mm256_setzero_ps();
                  size_t d = 0;
                  for (; d + 4 <= dimension; d += 4) {
                    accumulator0 = _mm256_add_ps(accumulator0, _mm256_mul_ps(
                        _mm256_loadu_ps(params->block_fc2_weight + (d + 0) * hidden_count + row),
                        _mm256_set1_ps(update[base + d + 0])));
                    accumulator1 = _mm256_add_ps(accumulator1, _mm256_mul_ps(
                        _mm256_loadu_ps(params->block_fc2_weight + (d + 1) * hidden_count + row),
                        _mm256_set1_ps(update[base + d + 1])));
                    accumulator2 = _mm256_add_ps(accumulator2, _mm256_mul_ps(
                        _mm256_loadu_ps(params->block_fc2_weight + (d + 2) * hidden_count + row),
                        _mm256_set1_ps(update[base + d + 2])));
                    accumulator3 = _mm256_add_ps(accumulator3, _mm256_mul_ps(
                        _mm256_loadu_ps(params->block_fc2_weight + (d + 3) * hidden_count + row),
                        _mm256_set1_ps(update[base + d + 3])));
                  }
                  float* target = d_hidden_values + row;
                  _mm256_storeu_ps(target, _mm256_add_ps(
                      _mm256_add_ps(accumulator0, accumulator1),
                      _mm256_add_ps(accumulator2, accumulator3)));
                  for (; d < dimension; ++d) {
                    const __m256 contribution = _mm256_mul_ps(
                        _mm256_loadu_ps(params->block_fc2_weight + d * hidden_count + row),
                        _mm256_set1_ps(update[base + d]));
                    _mm256_storeu_ps(target, _mm256_add_ps(_mm256_loadu_ps(target), contribution));
                  }
                }
                for (; row < hidden_count; ++row) {
                  float d_hidden = 0.0F;
                  for (size_t d = 0; d < dimension; ++d) {
                    d_hidden += params->block_fc2_weight[d * hidden_count + row] * update[base + d];
                  }
                  d_hidden_values[row] = d_hidden;
                }

                row = 0;
               for (; row + 8 <= hidden_count; row += 8) {
                 float fc1_values[8];
                  for (size_t lane = 0; lane < 8; ++lane) {
                    const size_t current_row = row + lane;
                    const size_t fc_index = fc1_base + current_row;
                    fc1_values[lane] = d_hidden_values[current_row] * gelu_derivative(saved_fc1_pre[fc_index]);
                    fc1[fc_index] = fc1_values[lane];
                  }
                  const __m256 fc1_vector = _mm256_loadu_ps(fc1_values);
                 // FC1 dBias is a contiguous eight-output vector update.
                 const __m256 bias_vector = _mm256_add_ps(
                    _mm256_loadu_ps(grads->d_block_fc1_bias + row), fc1_vector);
                _mm256_storeu_ps(grads->d_block_fc1_bias + row, bias_vector);
                for (size_t lane = 0; lane < 8; ++lane) {
                  add_abs_contribution(grads->sum_abs_d_block_fc1_bias, grads->count_d_block_fc1_bias,
                      row + lane, fc1_values[lane]);
                }

                {
                  OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kDepthEmbeddingPairwiseReduction);
                   for (size_t input = 0; input < dimension; input += 8) {
                    const size_t width = std::min<size_t>(8, dimension - input);
                     if (width == 8) {
                       // FC1 dWeight outer product: load x[i:i+8] once, then broadcast each dY row.
                       const __m256 input_values = _mm256_add_ps(
                          _mm256_loadu_ps(saved_out + base + input),
                          _mm256_loadu_ps(params->depth_embedding + round * dimension + input));
                      for (size_t lane = 0; lane < 8; ++lane) {
                         const size_t current_row = row + lane;
                         const float fc1_value = fc1_values[lane];
                         const __m256 contribution = _mm256_mul_ps(input_values, _mm256_set1_ps(fc1_value));
                        float contribution_values[8];
                        if constexpr (CollectDiagnostics) {
                          _mm256_storeu_ps(contribution_values, contribution);
                        }
                        float* gradient_row = grads->d_block_fc1_weight + current_row * dimension + input;
                        _mm256_storeu_ps(gradient_row, _mm256_add_ps(_mm256_loadu_ps(gradient_row), contribution));
                        if constexpr (CollectDiagnostics) {
                          const float* weight_row = params->block_fc1_weight + current_row * dimension + input;
                          for (size_t offset = 0; offset < 8; ++offset) {
                            add_abs_contribution(grads->sum_abs_d_block_fc1_weight, grads->count_d_block_fc1_weight,
                                current_row * dimension + input + offset, contribution_values[offset]);
                            const size_t depth_index = round * dimension + input + offset;
                            const float depth_contribution = weight_row[offset] * fc1_value;
                            if (grads->sum_abs_d_depth_embedding != nullptr) {
                              grads->sum_abs_d_depth_embedding[depth_index] += std::fabs(depth_contribution);
                            }
                            if (grads->count_d_depth_embedding != nullptr) ++grads->count_d_depth_embedding[depth_index];
                            if (grads->fp64_d_depth_embedding != nullptr) {
                              grads->fp64_d_depth_embedding[depth_index] += static_cast<double>(depth_contribution);
                            }
                          }
                        }
                      }
                    } else {
                      for (size_t offset = 0; offset < width; ++offset) {
                        const size_t input_index = input + offset;
                        const float input_value = saved_out[base + input_index] +
                            params->depth_embedding[round * dimension + input_index];
                        for (size_t lane = 0; lane < 8; ++lane) {
                          const size_t current_row = row + lane;
                          const float contribution = fc1_values[lane] * input_value;
                          grads->d_block_fc1_weight[current_row * dimension + input_index] += contribution;
                          if constexpr (CollectDiagnostics) {
                            add_abs_contribution(grads->sum_abs_d_block_fc1_weight, grads->count_d_block_fc1_weight,
                                current_row * dimension + input_index, contribution);
                            const size_t depth_index = round * dimension + input_index;
                            const float depth_contribution = params->block_fc1_weight[current_row * dimension + input_index] * fc1_values[lane];
                            if (grads->sum_abs_d_depth_embedding != nullptr) {
                              grads->sum_abs_d_depth_embedding[depth_index] += std::fabs(depth_contribution);
                            }
                            if (grads->count_d_depth_embedding != nullptr) ++grads->count_d_depth_embedding[depth_index];
                            if (grads->fp64_d_depth_embedding != nullptr) {
                              grads->fp64_d_depth_embedding[depth_index] += static_cast<double>(depth_contribution);
                            }
                          }
                        }
                      }
                    }
                  }

                  for (size_t input = 0; input < dimension; input += 32) {
                    const size_t width = std::min<size_t>(32, dimension - input);
                    if (width == 32) {
                      __m256 mixed0 = _mm256_loadu_ps(mixed + base + input);
                      __m256 mixed1 = _mm256_loadu_ps(mixed + base + input + 8);
                      __m256 mixed2 = _mm256_loadu_ps(mixed + base + input + 16);
                      __m256 mixed3 = _mm256_loadu_ps(mixed + base + input + 24);
                      for (size_t lane = 0; lane < 8; ++lane) {
                        const float* weight_row = params->block_fc1_weight + (row + lane) * dimension + input;
                        const __m256 fc1_scale = _mm256_set1_ps(fc1_values[lane]);
                        mixed0 = _mm256_add_ps(mixed0, _mm256_mul_ps(_mm256_loadu_ps(weight_row), fc1_scale));
                        mixed1 = _mm256_add_ps(mixed1, _mm256_mul_ps(_mm256_loadu_ps(weight_row + 8), fc1_scale));
                        mixed2 = _mm256_add_ps(mixed2, _mm256_mul_ps(_mm256_loadu_ps(weight_row + 16), fc1_scale));
                        mixed3 = _mm256_add_ps(mixed3, _mm256_mul_ps(_mm256_loadu_ps(weight_row + 24), fc1_scale));
                      }
                      _mm256_storeu_ps(mixed + base + input, mixed0);
                      _mm256_storeu_ps(mixed + base + input + 8, mixed1);
                      _mm256_storeu_ps(mixed + base + input + 16, mixed2);
                      _mm256_storeu_ps(mixed + base + input + 24, mixed3);
                    } else {
                      for (size_t offset = 0; offset < width; ++offset) {
                        const size_t input_index = input + offset;
                        for (size_t lane = 0; lane < 8; ++lane) {
                          mixed[base + input_index] += params->block_fc1_weight[(row + lane) * dimension + input_index] * fc1_values[lane];
                        }
                      }
                    }
                  }
                }
               }
                for (; row < hidden_count; ++row) {
                  const size_t fc_index = fc1_base + row;
                  fc1[fc_index] = d_hidden_values[row] * gelu_derivative(saved_fc1_pre[fc_index]);
                  grads->d_block_fc1_bias[row] += fc1[fc_index];
                add_abs_contribution(grads->sum_abs_d_block_fc1_bias, grads->count_d_block_fc1_bias, row, fc1[fc_index]);
                {
                  OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kDepthEmbeddingPairwiseReduction);
                   for (size_t input = 0; input < dimension; ++input) {
                     const float fc1_weight_contribution = fc1[fc_index] * (saved_out[base + input] + params->depth_embedding[round * dimension + input]);
                     float depth_contribution = 0.0F;
                     if constexpr (CollectDiagnostics) {
                       depth_contribution = params->block_fc1_weight[row * dimension + input] * fc1[fc_index];
                     }
                     grads->d_block_fc1_weight[row * dimension + input] += fc1_weight_contribution;
                     if constexpr (CollectDiagnostics) {
                       add_abs_contribution(grads->sum_abs_d_block_fc1_weight, grads->count_d_block_fc1_weight, row * dimension + input, fc1_weight_contribution);
                       const size_t depth_index = round * dimension + input;
                       if (grads->sum_abs_d_depth_embedding != nullptr) {
                         grads->sum_abs_d_depth_embedding[depth_index] += std::fabs(depth_contribution);
                       }
                       if (grads->count_d_depth_embedding != nullptr) ++grads->count_d_depth_embedding[depth_index];
                       if (grads->fp64_d_depth_embedding != nullptr) {
                         grads->fp64_d_depth_embedding[depth_index] += static_cast<double>(depth_contribution);
                       }
                     }
                    mixed[base + input] += params->block_fc1_weight[row * dimension + input] * fc1[fc_index];
                  }
                }
              }
#else
              for (size_t row = 0; row < 4 * dimension; ++row) {
                float d_hidden = 0.0F;
                for (size_t d = 0; d < dimension; ++d) {
                  d_hidden += params->block_fc2_weight[d * 4 * dimension + row] * update[base + d];
                }
                const size_t fc_index = (batch * slots + slot) * 4 * dimension + row;
                fc1[fc_index] = d_hidden * gelu_derivative(saved_fc1_pre[fc_index]);
                grads->d_block_fc1_bias[row] += fc1[fc_index];
                add_abs_contribution(grads->sum_abs_d_block_fc1_bias, grads->count_d_block_fc1_bias, row, fc1[fc_index]);
                {
                  OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kDepthEmbeddingPairwiseReduction);
                  for (size_t input = 0; input < dimension; ++input) {
                    const float fc1_weight_contribution = fc1[fc_index] * (saved_out[base + input] + params->depth_embedding[round * dimension + input]);
                    float depth_contribution = 0.0F;
                    if constexpr (CollectDiagnostics) {
                      depth_contribution = params->block_fc1_weight[row * dimension + input] * fc1[fc_index];
                    }
                    grads->d_block_fc1_weight[row * dimension + input] += fc1_weight_contribution;
                    if constexpr (CollectDiagnostics) {
                      add_abs_contribution(grads->sum_abs_d_block_fc1_weight, grads->count_d_block_fc1_weight, row * dimension + input, fc1_weight_contribution);
                      const size_t depth_index = round * dimension + input;
                      if (grads->sum_abs_d_depth_embedding != nullptr) {
                        grads->sum_abs_d_depth_embedding[depth_index] += std::fabs(depth_contribution);
                      }
                      if (grads->count_d_depth_embedding != nullptr) ++grads->count_d_depth_embedding[depth_index];
                      if (grads->fp64_d_depth_embedding != nullptr) {
                        grads->fp64_d_depth_embedding[depth_index] += static_cast<double>(depth_contribution);
                      }
                    }
                    mixed[base + input] += params->block_fc1_weight[row * dimension + input] * fc1[fc_index];
                  }
                }
              }
#endif
            if (!diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_DEPTH_EMBEDDING_PAIRWISE) &&
                !add_depth_hidden_vector(depth_partials, depth_masks, round, depth_levels,
                    depth_hidden_dimension, fc1 + (batch * slots + slot) * (4 * dimension))) {
              return 15;
            }
            {
              OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kRmsnormGates);
#if OMEGA_HAS_AVX2
              size_t d = 0;
              for (; d + 8 <= dimension; d += 8) {
                const __m256 contribution = _mm256_loadu_ps(mixed + base + d);
                _mm256_storeu_ps(grads->d_block_out_bias + d, _mm256_add_ps(
                    _mm256_loadu_ps(grads->d_block_out_bias + d), contribution));
                for (size_t lane = 0; lane < 8; ++lane) {
                  add_abs_contribution(grads->sum_abs_d_block_out_bias, grads->count_d_block_out_bias,
                      d + lane, mixed[base + d + lane]);
                }
              }
              for (; d < dimension; ++d) {
                grads->d_block_out_bias[d] += mixed[base + d];
                add_abs_contribution(grads->sum_abs_d_block_out_bias, grads->count_d_block_out_bias,
                    d, mixed[base + d]);
              }
#else
              for (size_t d = 0; d < dimension; ++d) {
                grads->d_block_out_bias[d] += mixed[base + d];
               add_abs_contribution(grads->sum_abs_d_block_out_bias, grads->count_d_block_out_bias, d, mixed[base + d]);
              }
#endif
            }
            {
              OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kOutProjection);
#if OMEGA_HAS_AVX2
               for (size_t input = 0; input + 8 <= dimension; input += 8) {
                 _mm256_storeu_ps(q + base + input, _mm256_setzero_ps());
               }
               for (size_t input = dimension & ~size_t{7}; input < dimension; ++input) {
                 q[base + input] = 0.0F;
               }
               for (size_t d = 0; d < dimension; ++d) {
                const float output_gradient = mixed[base + d];
                const __m256 output_gradient_vector = _mm256_set1_ps(output_gradient);
                const float* weight_row = params->block_out_weight + d * dimension;
                float* gradient_row = grads->d_block_out_weight + d * dimension;
                size_t input = 0;
                for (; input + 8 <= dimension; input += 8) {
                  const __m256 weight_vector = _mm256_loadu_ps(weight_row + input);
                  _mm256_storeu_ps(q + base + input, _mm256_add_ps(
                      _mm256_loadu_ps(q + base + input), _mm256_mul_ps(weight_vector, output_gradient_vector)));
                  const __m256 contribution = _mm256_mul_ps(
                      _mm256_loadu_ps(saved_attention + base + input), output_gradient_vector);
                  _mm256_storeu_ps(gradient_row + input, _mm256_add_ps(
                      _mm256_loadu_ps(gradient_row + input), contribution));
                  for (size_t lane = 0; lane < 8; ++lane) {
                    const float contribution_value = output_gradient * saved_attention[base + input + lane];
                    add_abs_contribution(grads->sum_abs_d_block_out_weight, grads->count_d_block_out_weight,
                        d * dimension + input + lane, contribution_value);
                  }
                }
                for (; input < dimension; ++input) {
                  const float contribution = output_gradient * saved_attention[base + input];
                  q[base + input] += weight_row[input] * output_gradient;
                  gradient_row[input] += contribution;
                  add_abs_contribution(grads->sum_abs_d_block_out_weight, grads->count_d_block_out_weight,
                      d * dimension + input, contribution);
                }
              }
#else
              for (size_t input = 0; input < dimension; ++input) {
                float d_attention = 0.0F;
               for (size_t d = 0; d < dimension; ++d) {
                d_attention += params->block_out_weight[d * dimension + input] * mixed[base + d];
                const float out_weight_contribution = mixed[base + d] * saved_attention[base + input];
                grads->d_block_out_weight[d * dimension + input] += out_weight_contribution;
                add_abs_contribution(grads->sum_abs_d_block_out_weight, grads->count_d_block_out_weight, d * dimension + input, out_weight_contribution);
              }
               q[base + input] = d_attention;
               }
#endif
             }
           for (size_t input = 0; input < dimension; ++input) {
             mixed[base + input] = q[base + input];
           }
           }
          }
        }

           std::memset(q, 0, state_count * sizeof(float));
        std::memset(key, 0, state_count * sizeof(float));
        std::memset(value, 0, state_count * sizeof(float));

        if (!diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_ATTENTION_SCORES_SOFTMAX_MIXING)) {
          OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kAttentionScoresSoftmaxMixing);
          for (size_t batch = 0; batch < config->batch; ++batch) {
            for (size_t query_slot = 0; query_slot < slots; ++query_slot) {
              float probability_dot = 0.0F;
              for (size_t key_slot = 0; key_slot < slots; ++key_slot) {
                float d_probability = 0.0F;
                for (size_t d = 0; d < dimension; ++d) {
                  d_probability += mixed[state_index(batch, query_slot, d, slots, dimension)] * saved_value[state_index(batch, key_slot, d, slots, dimension)];
                  value[state_index(batch, key_slot, d, slots, dimension)] += saved_probability[(batch * slots + query_slot) * slots + key_slot] * mixed[state_index(batch, query_slot, d, slots, dimension)];
                }
                scores[(batch * slots + query_slot) * slots + key_slot] = d_probability;
                probability_dot += saved_probability[(batch * slots + query_slot) * slots + key_slot] * d_probability;
              }
              for (size_t key_slot = 0; key_slot < slots; ++key_slot) {
                const size_t score_index = (batch * slots + query_slot) * slots + key_slot;
                scores[score_index] = saved_probability[score_index] * (scores[score_index] - probability_dot);
                for (size_t d = 0; d < dimension; ++d) {
                  q[state_index(batch, query_slot, d, slots, dimension)] += scores[score_index] * saved_key[state_index(batch, key_slot, d, slots, dimension)] * scale;
                  key[state_index(batch, key_slot, d, slots, dimension)] += scores[score_index] * saved_q[state_index(batch, query_slot, d, slots, dimension)] * scale;
                }
              }
            }
          }
        }

        {
          OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kQkvProjection);
#if OMEGA_HAS_AVX2
          for (size_t batch = 0; batch < config->batch; ++batch) {
            for (size_t slot = 0; slot < slots; ++slot) {
              const size_t base = state_index(batch, slot, 0, slots, dimension);
              for (size_t input = 0; input + 8 <= dimension; input += 8) {
                _mm256_storeu_ps(candidate + base + input, _mm256_add_ps(
                    _mm256_loadu_ps(saved_candidate + base + input),
                    _mm256_loadu_ps(training_buffers.anchor_history + position * state_count + base + input)));
                _mm256_storeu_ps(mixed + base + input, _mm256_setzero_ps());
              }
              for (size_t input = dimension & ~size_t{7}; input < dimension; ++input) {
                candidate[base + input] = saved_candidate[base + input] +
                    training_buffers.anchor_history[position * state_count + base + input];
                mixed[base + input] = 0.0F;
              }

              for (size_t group = 0; group < 3; ++group) {
                const float* d_output = group == 0 ? q + base : (group == 1 ? key + base : value + base);
                for (size_t output = 0; output < dimension; ++output) {
                  const float output_gradient = d_output[output];
                  const __m256 output_gradient_vector = _mm256_set1_ps(output_gradient);
                  const float* weight_row = params->block_qkv_weight +
                      (group * dimension + output) * dimension;
                  float* gradient_row = grads->d_block_qkv_weight +
                      (group * dimension + output) * dimension;
                  for (size_t input = 0; input + 8 <= dimension; input += 8) {
                    const __m256 z_vector = _mm256_loadu_ps(candidate + base + input);
                    const __m256 weight_vector = _mm256_loadu_ps(weight_row + input);
                    _mm256_storeu_ps(mixed + base + input, _mm256_add_ps(
                        _mm256_loadu_ps(mixed + base + input),
                        _mm256_mul_ps(weight_vector, output_gradient_vector)));
                    _mm256_storeu_ps(gradient_row + input, _mm256_add_ps(
                        _mm256_loadu_ps(gradient_row + input),
                        _mm256_mul_ps(z_vector, output_gradient_vector)));
                    for (size_t lane = 0; lane < 8; ++lane) {
                      const float contribution = output_gradient * candidate[base + input + lane];
                      add_abs_contribution(grads->sum_abs_d_block_qkv_weight, grads->count_d_block_qkv_weight,
                          group * dimension * dimension + output * dimension + input + lane, contribution);
                    }
                  }
                  for (size_t input = dimension & ~size_t{7}; input < dimension; ++input) {
                    const float contribution = output_gradient * candidate[base + input];
                    mixed[base + input] += params->block_qkv_weight[
                        (group * dimension + output) * dimension + input] * output_gradient;
                    gradient_row[input] += contribution;
                    add_abs_contribution(grads->sum_abs_d_block_qkv_weight, grads->count_d_block_qkv_weight,
                        group * dimension * dimension + output * dimension + input, contribution);
                  }
                }
              }

              for (size_t group = 0; group < 3; ++group) {
                const float* d_output = group == 0 ? q + base : (group == 1 ? key + base : value + base);
                float* gradient_bias = grads->d_block_qkv_bias + group * dimension;
                size_t output = 0;
                for (; output + 8 <= dimension; output += 8) {
                  _mm256_storeu_ps(gradient_bias + output, _mm256_add_ps(
                      _mm256_loadu_ps(gradient_bias + output), _mm256_loadu_ps(d_output + output)));
                  for (size_t lane = 0; lane < 8; ++lane) {
                    add_abs_contribution(grads->sum_abs_d_block_qkv_bias, grads->count_d_block_qkv_bias,
                        group * dimension + output + lane, d_output[output + lane]);
                  }
                }
                for (; output < dimension; ++output) {
                  gradient_bias[output] += d_output[output];
                  add_abs_contribution(grads->sum_abs_d_block_qkv_bias, grads->count_d_block_qkv_bias,
                      group * dimension + output, d_output[output]);
                }
              }

              for (size_t input = 0; input + 8 <= dimension; input += 8) {
                const __m256 d_input = _mm256_loadu_ps(mixed + base + input);
                _mm256_storeu_ps(anchor + base + input, _mm256_add_ps(
                    _mm256_loadu_ps(anchor + base + input), d_input));
                for (size_t lane = 0; lane < 8; ++lane) {
                  candidate[base + input + lane] = mixed[base + input + lane] + update[base + input + lane] /
                      sigmoid(params->gate_logits[round * dimension + input + lane]);
                }
              }
              for (size_t input = dimension & ~size_t{7}; input < dimension; ++input) {
                const float d_input = mixed[base + input];
                candidate[base + input] = d_input + update[base + input] /
                    sigmoid(params->gate_logits[round * dimension + input]);
                anchor[base + input] += d_input;
              }
            }
#else
           for (size_t batch = 0; batch < config->batch; ++batch) {
            for (size_t slot = 0; slot < slots; ++slot) {
              const size_t base = state_index(batch, slot, 0, slots, dimension);
              for (size_t input = 0; input < dimension; ++input) {
                float d_z = 0.0F;
                const float z = saved_candidate[base + input] + training_buffers.anchor_history[position * state_count + base + input];
                for (size_t output = 0; output < dimension; ++output) {
                  d_z += q[base + output] * params->block_qkv_weight[output * dimension + input];
                  d_z += key[base + output] * params->block_qkv_weight[(dimension + output) * dimension + input];
                  d_z += value[base + output] * params->block_qkv_weight[(2 * dimension + output) * dimension + input];
                  const float q_weight_contribution = q[base + output] * z;
                  const float key_weight_contribution = key[base + output] * z;
                  const float value_weight_contribution = value[base + output] * z;
                  grads->d_block_qkv_weight[output * dimension + input] += q_weight_contribution;
                  grads->d_block_qkv_weight[(dimension + output) * dimension + input] += key_weight_contribution;
                  grads->d_block_qkv_weight[(2 * dimension + output) * dimension + input] += value_weight_contribution;
                  add_abs_contribution(grads->sum_abs_d_block_qkv_weight, grads->count_d_block_qkv_weight, output * dimension + input, q_weight_contribution);
                  add_abs_contribution(grads->sum_abs_d_block_qkv_weight, grads->count_d_block_qkv_weight, (dimension + output) * dimension + input, key_weight_contribution);
                  add_abs_contribution(grads->sum_abs_d_block_qkv_weight, grads->count_d_block_qkv_weight, (2 * dimension + output) * dimension + input, value_weight_contribution);
                }
                candidate[base + input] = d_z + update[base + input] / sigmoid(params->gate_logits[round * dimension + input]);
                anchor[base + input] += d_z;
              }
              for (size_t output = 0; output < dimension; ++output) {
                grads->d_block_qkv_bias[output] += q[base + output];
                grads->d_block_qkv_bias[dimension + output] += key[base + output];
                grads->d_block_qkv_bias[2 * dimension + output] += value[base + output];
                add_abs_contribution(grads->sum_abs_d_block_qkv_bias, grads->count_d_block_qkv_bias, output, q[base + output]);
                add_abs_contribution(grads->sum_abs_d_block_qkv_bias, grads->count_d_block_qkv_bias, dimension + output, key[base + output]);
                add_abs_contribution(grads->sum_abs_d_block_qkv_bias, grads->count_d_block_qkv_bias, 2 * dimension + output, value[base + output]);
             }
           }
#endif
         }
       }
        }
      }
       for (size_t index = 0; index < state_count; ++index) {
        anchor[index] += candidate[index];
      }

      if (diagnostic_elide(OMEGA_DIAGNOSTIC_ELIDE_STATE_PRELUDE)) {
        for (size_t index = 0; index < state_count; ++index) {
          const size_t batch = index / state_dimension;
          const size_t position_index = (batch * config->sequence_length + position) * state_dimension + (index % state_dimension);
          grads->d_token_part[position_index] += anchor[index];
          add_abs_contribution(grads->sum_abs_d_token_part, grads->count_d_token_part, position_index, anchor[index]);
        }
        std::memset(state, 0, state_count * sizeof(float));
       } else {
       {
         OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kStatePrelude);
      const float* saved_state = training_buffers.state_history + position * state_count;
      for (size_t batch = 0; batch < config->batch; ++batch) {
        const float* token_row = token_part + (batch * config->sequence_length + position) * state_dimension;
        float* mean_state = mixed + batch * state_dimension;
#if OMEGA_HAS_AVX2
        const __m256 inverse_slots = _mm256_set1_ps(1.0F / static_cast<float>(slots));
        size_t input = 0;
        for (; input + 8 <= dimension; input += 8) {
          __m256 mean = _mm256_setzero_ps();
          for (size_t slot = 0; slot < slots; ++slot) {
            mean = _mm256_add_ps(mean, _mm256_loadu_ps(saved_state + state_index(batch, slot, input, slots, dimension)));
          }
          _mm256_storeu_ps(mean_state + input, _mm256_mul_ps(mean, inverse_slots));
        }
        for (; input < dimension; ++input) {
          float mean = 0.0F;
          for (size_t slot = 0; slot < slots; ++slot) {
            mean += saved_state[state_index(batch, slot, input, slots, dimension)];
          }
          mean_state[input] = mean / static_cast<float>(slots);
        }
#else
        for (size_t input = 0; input < dimension; ++input) {
          float mean = 0.0F;
          for (size_t slot = 0; slot < slots; ++slot) {
            mean += saved_state[state_index(batch, slot, input, slots, dimension)];
          }
          mean_state[input] = mean / static_cast<float>(slots);
        }
#endif
        float pre_norm_sum = 0.0F;
        for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
         const float write = state_prelude_write(token_row, params->state_part_weight, flattened, mean_state, dimension);
         const float pre = saved_state[batch * state_dimension + flattened] + write;
          pre_norm_sum += pre * pre;
        }
        const float inverse_rms = 1.0F / std::sqrt(pre_norm_sum / static_cast<float>(state_dimension) + kEpsilon);
        float norm_dot = 0.0F;
        for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
         const float write = state_prelude_write(token_row, params->state_part_weight, flattened, mean_state, dimension);
         const float pre = saved_state[batch * state_dimension + flattened] + write;
          norm_dot += anchor[batch * state_dimension + flattened] * params->prelude_norm_weight[flattened] * pre;
        }
        const float inverse_rms_cubed_over_dimension = inverse_rms * inverse_rms * inverse_rms / static_cast<float>(state_dimension);
#if OMEGA_HAS_AVX2
        std::memset(update + batch * state_dimension, 0, dimension * sizeof(float));
        for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
         const float write = state_prelude_write(token_row, params->state_part_weight, flattened, mean_state, dimension);
         const float pre = saved_state[batch * state_dimension + flattened] + write;
         const float d_pre = anchor[batch * state_dimension + flattened] * params->prelude_norm_weight[flattened] * inverse_rms - pre * inverse_rms_cubed_over_dimension * norm_dot;
         const float prelude_norm_contribution = anchor[batch * state_dimension + flattened] * pre * inverse_rms;
        grads->d_prelude_norm_weight[flattened] += prelude_norm_contribution;
        add_abs_contribution(grads->sum_abs_d_prelude_norm_weight, grads->count_d_prelude_norm_weight, flattened, prelude_norm_contribution);
        grads->d_token_part[(batch * config->sequence_length + position) * state_dimension + flattened] += d_pre;
         add_abs_contribution(grads->sum_abs_d_token_part, grads->count_d_token_part, (batch * config->sequence_length + position) * state_dimension + flattened, d_pre);
         state[batch * state_dimension + flattened] = d_pre;
         const float* state_part_row = params->state_part_weight.data +
             flattened * static_cast<size_t>(params->state_part_weight.row_stride);
         float* gradient_row = grads->d_state_part_weight + flattened * dimension;
         const __m256 d_pre_vector = _mm256_set1_ps(d_pre);
          size_t input_index = 0;
          for (; input_index + 8 <= dimension; input_index += 8) {
            const __m256 mean_vector = _mm256_loadu_ps(mean_state + input_index);
            const __m256 contribution = _mm256_mul_ps(d_pre_vector, mean_vector);
            _mm256_storeu_ps(gradient_row + input_index, _mm256_add_ps(_mm256_loadu_ps(gradient_row + input_index), contribution));
            _mm256_storeu_ps(update + batch * state_dimension + input_index, _mm256_add_ps(
                _mm256_loadu_ps(update + batch * state_dimension + input_index),
                _mm256_mul_ps(_mm256_loadu_ps(state_part_row + input_index), d_pre_vector)));
            for (size_t lane = 0; lane < 8; ++lane) {
              const float state_part_contribution = d_pre * mean_state[input_index + lane];
              add_abs_contribution(grads->sum_abs_d_state_part_weight, grads->count_d_state_part_weight,
                  flattened * dimension + input_index + lane, state_part_contribution);
            }
          }
          for (; input_index < dimension; ++input_index) {
            const float state_part_contribution = d_pre * mean_state[input_index];
            gradient_row[input_index] += state_part_contribution;
            update[batch * state_dimension + input_index] += d_pre * state_part_row[input_index];
            add_abs_contribution(grads->sum_abs_d_state_part_weight, grads->count_d_state_part_weight,
                flattened * dimension + input_index, state_part_contribution);
          }
        }
        for (size_t tail_input = dimension & ~size_t{7}; tail_input < dimension; ++tail_input) {
          update[batch * state_dimension + tail_input] /= static_cast<float>(slots);
        }
        for (size_t vector_input = 0; vector_input + 8 <= dimension; vector_input += 8) {
          _mm256_storeu_ps(update + batch * state_dimension + vector_input, _mm256_mul_ps(
              _mm256_loadu_ps(update + batch * state_dimension + vector_input), inverse_slots));
        }
#else
        for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
         const float write = state_prelude_write(token_row, params->state_part_weight, flattened, mean_state, dimension);
         const float pre = saved_state[batch * state_dimension + flattened] + write;
         const float d_pre = anchor[batch * state_dimension + flattened] * params->prelude_norm_weight[flattened] * inverse_rms - pre * inverse_rms_cubed_over_dimension * norm_dot;
         const float prelude_norm_contribution = anchor[batch * state_dimension + flattened] * pre * inverse_rms;
         grads->d_prelude_norm_weight[flattened] += prelude_norm_contribution;
         add_abs_contribution(grads->sum_abs_d_prelude_norm_weight, grads->count_d_prelude_norm_weight, flattened, prelude_norm_contribution);
         grads->d_token_part[(batch * config->sequence_length + position) * state_dimension + flattened] += d_pre;
          add_abs_contribution(grads->sum_abs_d_token_part, grads->count_d_token_part, (batch * config->sequence_length + position) * state_dimension + flattened, d_pre);
          state[batch * state_dimension + flattened] = d_pre;
          for (size_t input = 0; input < dimension; ++input) {
           const float state_part_contribution = d_pre * mean_state[input];
           grads->d_state_part_weight[flattened * dimension + input] += state_part_contribution;
           add_abs_contribution(grads->sum_abs_d_state_part_weight, grads->count_d_state_part_weight,
               flattened * dimension + input, state_part_contribution);
         }
        }
        for (size_t input = 0; input < dimension; ++input) {
         float d_mean = 0.0F;
         for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
           const float* state_part_row = params->state_part_weight.data +
               flattened * static_cast<size_t>(params->state_part_weight.row_stride);
           d_mean += state[batch * state_dimension + flattened] * state_part_row[input];
         }
         update[batch * state_dimension + input] = d_mean / static_cast<float>(slots);
       }
#endif
       for (size_t slot = 0; slot < slots; ++slot) {
#if OMEGA_HAS_AVX2
          const size_t base = state_index(batch, slot, 0, slots, dimension);
          size_t distribution_input = 0;
          for (; distribution_input + 8 <= dimension; distribution_input += 8) {
            _mm256_storeu_ps(state + base + distribution_input, _mm256_add_ps(
                _mm256_loadu_ps(state + base + distribution_input),
                _mm256_loadu_ps(update + batch * state_dimension + distribution_input)));
          }
          for (; distribution_input < dimension; ++distribution_input) {
            state[base + distribution_input] += update[batch * state_dimension + distribution_input];
          }
#else
         for (size_t input = 0; input < dimension; ++input) {
           state[state_index(batch, slot, input, slots, dimension)] += update[batch * state_dimension + input];
         }
#endif
       }
      }
   }
   }
  }
  if (!diagnostic_any_elide()) {
    OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kDepthEmbeddingPairwiseReduction);
    for (size_t round = 0; round < config->rounds; ++round) {
      const std::uint64_t mask = depth_masks[round];
      const size_t round_base = round * depth_levels * depth_hidden_dimension;
      size_t highest_level = 0;
      for (size_t level = 1; level < depth_levels; ++level) {
        if ((mask & (std::uint64_t{1} << level)) != 0) {
          highest_level = level;
        }
      }
      for (size_t input = 0; input < dimension; ++input) {
        float depth_gradient = 0.0F;
        bool initialized = false;
        for (size_t level = 0; level < depth_levels; ++level) {
          if ((mask & (std::uint64_t{1} << level)) == 0) continue;
          const float* partial = depth_partials + round_base + level * depth_hidden_dimension;
          float partial_gradient = 0.0F;
          for (size_t row = 0; row < depth_hidden_dimension; ++row) {
            partial_gradient += params->block_fc1_weight[row * dimension + input] * partial[row];
          }
          depth_gradient = initialized ? depth_gradient + partial_gradient : partial_gradient;
          initialized = true;
        }
        const size_t depth_index = round * dimension + input;
        grads->d_depth_embedding[depth_index] = depth_gradient;
        if (grads->depth_max_level != nullptr) grads->depth_max_level[depth_index] = highest_level;
      }
    }
  }
  if (grads->sum_abs_d_previous_state != nullptr || grads->count_d_previous_state != nullptr) {
    for (size_t index = 0; index < state_count; ++index) {
      if (grads->sum_abs_d_previous_state != nullptr) grads->sum_abs_d_previous_state[index] = std::fabs(state[index]);
      if (grads->count_d_previous_state != nullptr) grads->count_d_previous_state[index] = 1;
    }
  }
  std::memcpy(grads->d_previous_state, state, state_count * sizeof(float));
  return 0;
}

int recurrent_backward_impl(
    const OmegaRecurrentConfig* config,
    const OmegaRecurrentParams* params,
    const float* token_part,
    const float* d_readout_states,
    const float* d_next_state,
    void* workspace,
    size_t workspace_bytes,
    OmegaRecurrentGrads* grads) {
  const bool collect_diagnostics = grads != nullptr &&
      (grads->sum_abs_d_block_fc1_weight != nullptr || grads->count_d_block_fc1_weight != nullptr ||
       grads->sum_abs_d_depth_embedding != nullptr || grads->count_d_depth_embedding != nullptr ||
       grads->fp64_d_depth_embedding != nullptr);
  if (collect_diagnostics) {
    return recurrent_backward_impl_body<true>(config, params, token_part, d_readout_states, d_next_state,
        workspace, workspace_bytes, grads);
  }
  return recurrent_backward_impl_body<false>(config, params, token_part, d_readout_states, d_next_state,
      workspace, workspace_bytes, grads);
}

extern "C" int omega_recurrent_forward(
    const OmegaRecurrentConfig* config,
    const OmegaRecurrentParams* params,
    const float* token_part,
    const float* previous_state,
    float* next_state,
    float* readout_states,
    void* workspace,
    size_t workspace_bytes) {
  return recurrent_forward_impl(config, params, token_part, previous_state, next_state, readout_states, workspace, workspace_bytes);
}

extern "C" int omega_recurrent_backward(
    const OmegaRecurrentConfig* config,
    const OmegaRecurrentParams* params,
    const float* token_part,
    const float* d_readout_states,
    const float* d_next_state,
    void* workspace,
    size_t workspace_bytes,
    OmegaRecurrentGrads* grads) {
  return recurrent_backward_impl(config, params, token_part, d_readout_states, d_next_state, workspace, workspace_bytes, grads);
}

namespace {

struct RuntimeArena {
  unsigned char* core = nullptr;
  size_t core_bytes = 0;
  float* primary = nullptr;
  float* sum_abs = nullptr;
  size_t* counts = nullptr;
  double* fp64 = nullptr;
  size_t* depth_max = nullptr;
};

void bind_parameter_pointers(const OmegaRecurrentConfig& config, float* base, std::array<float*, kParameterCount>* pointers) {
  std::array<size_t, kParameterCount> counts{};
  size_t total = 0;
  if (!parameter_counts(config, &counts, &total)) {
    pointers->fill(nullptr);
    return;
  }
  float* cursor = base;
  for (size_t index = 0; index < kParameterCount; ++index) {
    (*pointers)[index] = cursor;
    cursor += counts[index];
  }
}

void bind_parameter_pointers(const OmegaRecurrentConfig& config, const float* base, std::array<const float*, kParameterCount>* pointers) {
  std::array<size_t, kParameterCount> counts{};
  size_t total = 0;
  if (!parameter_counts(config, &counts, &total)) {
    pointers->fill(nullptr);
    return;
  }
  const float* cursor = base;
  for (size_t index = 0; index < kParameterCount; ++index) {
    (*pointers)[index] = cursor;
    cursor += counts[index];
  }
}

void bind_external_parameter_pointers(OmegaRecurrentGrads* grads, std::array<float*, kParameterCount>* pointers) {
  *pointers = {grads->d_state_part_weight, grads->d_prelude_norm_weight, grads->d_block_qkv_weight,
               grads->d_block_qkv_bias, grads->d_block_out_weight, grads->d_block_out_bias,
               grads->d_block_fc1_weight, grads->d_block_fc1_bias, grads->d_block_fc2_weight,
               grads->d_block_fc2_bias, grads->d_block_norm_weight, grads->d_depth_embedding,
               grads->d_gate_logits};
}

void bind_external_optional_float_pointers(OmegaRecurrentGrads* grads, std::array<float*, kParameterCount>* pointers) {
  *pointers = {grads->sum_abs_d_state_part_weight, grads->sum_abs_d_prelude_norm_weight,
               grads->sum_abs_d_block_qkv_weight, grads->sum_abs_d_block_qkv_bias,
               grads->sum_abs_d_block_out_weight, grads->sum_abs_d_block_out_bias,
               grads->sum_abs_d_block_fc1_weight, grads->sum_abs_d_block_fc1_bias,
               grads->sum_abs_d_block_fc2_weight, grads->sum_abs_d_block_fc2_bias,
               grads->sum_abs_d_block_norm_weight, grads->sum_abs_d_depth_embedding,
               grads->sum_abs_d_gate_logits};
}

void bind_external_optional_count_pointers(OmegaRecurrentGrads* grads, std::array<size_t*, kParameterCount>* pointers) {
  *pointers = {grads->count_d_state_part_weight, grads->count_d_prelude_norm_weight,
               grads->count_d_block_qkv_weight, grads->count_d_block_qkv_bias,
               grads->count_d_block_out_weight, grads->count_d_block_out_bias,
               grads->count_d_block_fc1_weight, grads->count_d_block_fc1_bias,
               grads->count_d_block_fc2_weight, grads->count_d_block_fc2_bias,
               grads->count_d_block_norm_weight, grads->count_d_depth_embedding,
               grads->count_d_gate_logits};
}

bool bind_runtime_arenas(const OmegaRecurrentConfig& config, size_t worker_count, void* workspace,
    size_t workspace_bytes, RuntimeArena* arenas, size_t* core_bytes_out) {
  OmegaRecurrentConfig local = config;
  local.batch = config.batch / worker_count;
  local.instrumentation = 1;
  const size_t core_bytes = workspace_bytes_impl(local);
  std::array<size_t, kParameterCount> counts{};
  size_t parameter_count = 0;
  size_t depth_count = 0;
  if (core_bytes == 0 || !parameter_counts(config, &counts, &parameter_count) ||
      !checked_mul(config.rounds, config.dimension, &depth_count)) return false;
  WorkspaceCursor cursor{reinterpret_cast<unsigned char*>(workspace), workspace_bytes, 0};
  for (size_t worker = 0; worker < worker_count; ++worker) {
    RuntimeArena& arena = arenas[worker];
    arena.core = cursor.take_bytes(core_bytes);
    arena.primary = cursor.take(parameter_count);
    arena.sum_abs = cursor.take(parameter_count);
    size_t parameter_bytes = 0;
    size_t depth_double_bytes = 0;
    size_t depth_size_bytes = 0;
    if (!checked_mul(parameter_count, sizeof(size_t), &parameter_bytes) ||
        !checked_mul(depth_count, sizeof(double), &depth_double_bytes) ||
        !checked_mul(depth_count, sizeof(size_t), &depth_size_bytes)) return false;
    arena.counts = reinterpret_cast<size_t*>(cursor.take_bytes(parameter_bytes));
    arena.fp64 = reinterpret_cast<double*>(cursor.take_bytes(depth_double_bytes));
    arena.depth_max = reinterpret_cast<size_t*>(cursor.take_bytes(depth_size_bytes));
    arena.core_bytes = core_bytes;
    if (arena.core == nullptr || arena.primary == nullptr || arena.sum_abs == nullptr || arena.counts == nullptr ||
        arena.fp64 == nullptr || arena.depth_max == nullptr) return false;
  }
  *core_bytes_out = core_bytes;
  return true;
}

}  // namespace

struct OmegaRuntime {
  enum class Operation { kNone, kForward, kBackward };

  explicit OmegaRuntime(size_t worker_count) : worker_count_(worker_count) {
    if (worker_count_ == 1) return;
    size_t started = 0;
    try {
      for (; started < worker_count_; ++started) {
        workers_[started] = std::thread(&OmegaRuntime::worker_loop, this, started);
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        ++dispatch_id_;
      }
      work_cv_.notify_all();
      for (size_t index = 0; index < started; ++index) {
        if (workers_[index].joinable()) workers_[index].join();
      }
      throw;
    }
  }

  ~OmegaRuntime() {
    std::unique_lock<std::mutex> lock(mutex_);
    done_cv_.wait(lock, [this] { return !busy_; });
    stopping_ = true;
    ++dispatch_id_;
    lock.unlock();
    work_cv_.notify_all();
    for (size_t index = 0; index < worker_count_; ++index) {
      if (workers_[index].joinable()) workers_[index].join();
    }
  }

  size_t worker_count() const { return worker_count_; }

#ifdef OMEGA_P2R_DIAGNOSTIC
  void diagnostic_begin() {
    std::lock_guard<std::mutex> lock(mutex_);
    diagnostic_begin_locked();
  }

  void diagnostic_worker_start(size_t worker_index) {
    diagnostic_snapshot_.worker_start_ns[worker_index] = diagnostic_now_ns();
  }

  void diagnostic_worker_end(size_t worker_index) {
    diagnostic_snapshot_.worker_end_ns[worker_index] = diagnostic_now_ns();
  }

  void diagnostic_all_workers_done() {
    diagnostic_snapshot_.all_workers_done_ns = diagnostic_now_ns();
  }

  void diagnostic_reduction_start() {
    diagnostic_snapshot_.final_gradient_reduction_start_ns = diagnostic_now_ns();
  }

  void diagnostic_reduction_end() {
    diagnostic_snapshot_.final_gradient_reduction_end_ns = diagnostic_now_ns();
  }

  void diagnostic_return() {
    diagnostic_snapshot_.return_ns = diagnostic_now_ns();
  }

  int diagnostic_snapshot(OmegaRuntimeDiagnosticSnapshot* snapshot) const {
    if (snapshot == nullptr) return OMEGA_RUNTIME_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(mutex_);
    if (diagnostic_snapshot_.call_sequence == 0 || diagnostic_snapshot_.return_ns == 0) {
      return OMEGA_RUNTIME_STATUS_BUSY;
    }
    *snapshot = diagnostic_snapshot_;
    return OMEGA_RUNTIME_STATUS_OK;
  }
#endif

  int run(Operation operation, const OmegaRecurrentConfig& config, const OmegaRecurrentParams* params,
      const float* token_part, const float* previous_state, float* next_state, float* readout_states,
      const float* d_readout_states, const float* d_next_state, void* workspace, size_t workspace_bytes,
      OmegaRecurrentGrads* grads) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (busy_) return OMEGA_RUNTIME_STATUS_BUSY;
    size_t bound_core_bytes = 0;
    if (!bind_runtime_arenas(config, worker_count_, workspace, workspace_bytes, arenas_, &bound_core_bytes)) {
      return OMEGA_RUNTIME_STATUS_WORKSPACE;
    }
    busy_ = true;
#ifdef OMEGA_P2R_DIAGNOSTIC
    diagnostic_begin_locked();
#endif
    operation_ = operation;
    config_ = config;
    params_ = params;
    token_part_ = token_part;
    previous_state_ = previous_state;
    next_state_ = next_state;
    readout_states_ = readout_states;
    d_readout_states_ = d_readout_states;
    d_next_state_ = d_next_state;
    grads_ = grads;
    (void)bound_core_bytes;
    completed_ = 0;
    ++dispatch_id_;
    work_cv_.notify_all();
    done_cv_.wait(lock, [this] { return completed_ == worker_count_; });
#ifdef OMEGA_P2R_DIAGNOSTIC
    diagnostic_snapshot_.all_workers_done_ns = diagnostic_now_ns();
#endif
    busy_ = false;
    int status = OMEGA_RUNTIME_STATUS_OK;
    for (size_t index = 0; index < worker_count_; ++index) {
      if (statuses_[index] != 0 && status == OMEGA_RUNTIME_STATUS_OK) status = statuses_[index];
    }
    return status;
  }

  RuntimeArena arenas_[4]{};

 private:
#ifdef OMEGA_P2R_DIAGNOSTIC
  void diagnostic_begin_locked() {
    const uint64_t sequence = diagnostic_snapshot_.call_sequence + 1;
    diagnostic_snapshot_ = OmegaRuntimeDiagnosticSnapshot{};
    diagnostic_snapshot_.call_sequence = sequence;
    diagnostic_snapshot_.worker_count = worker_count_;
    diagnostic_snapshot_.dispatch_start_ns = diagnostic_now_ns();
  }
#endif

  void worker_loop(size_t worker_index) {
    size_t observed_dispatch = 0;
    for (;;) {
      Operation operation = Operation::kNone;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        work_cv_.wait(lock, [this, observed_dispatch] { return stopping_ || dispatch_id_ != observed_dispatch; });
        if (stopping_) return;
        observed_dispatch = dispatch_id_;
        operation = operation_;
      }
      int status = OMEGA_RUNTIME_STATUS_WORKER_FAILURE;
#ifdef OMEGA_P2R_DIAGNOSTIC
      diagnostic_worker_start(worker_index);
#endif
      try {
        status = run_worker(worker_index, operation);
      } catch (...) {
        status = OMEGA_RUNTIME_STATUS_WORKER_FAILURE;
      }
#ifdef OMEGA_P2R_DIAGNOSTIC
      diagnostic_worker_end(worker_index);
#endif
      {
        std::lock_guard<std::mutex> lock(mutex_);
        statuses_[worker_index] = status;
        ++completed_;
        if (completed_ == worker_count_) done_cv_.notify_one();
      }
    }
  }

  int run_worker(size_t worker_index, Operation operation) {
    OmegaRecurrentConfig local = config_;
    local.batch = config_.batch / worker_count_;
    size_t state_dimension = 0;
    size_t sequence_state = 0;
    size_t batch_state = 0;
    size_t batch_offset = 0;
    if (!checked_mul(config_.slots, config_.dimension, &state_dimension) ||
        !checked_mul(config_.sequence_length, state_dimension, &sequence_state) ||
        !checked_mul(local.batch, state_dimension, &batch_state) ||
        !checked_mul(worker_index, batch_state, &batch_offset)) {
      return OMEGA_RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    size_t worker_batch = 0;
    size_t token_offset = 0;
    if (!checked_mul(worker_index, local.batch, &worker_batch) || !checked_mul(worker_batch, sequence_state, &token_offset)) {
      return OMEGA_RUNTIME_STATUS_INVALID_ARGUMENT;
    }
    RuntimeArena& arena = arenas_[worker_index];
    if (operation == Operation::kForward) {
      return recurrent_forward_impl(&local, params_, token_part_ + token_offset, previous_state_ + batch_offset,
          next_state_ + batch_offset, readout_states_ + token_offset, arena.core, arena.core_bytes);
    }
    if (operation != Operation::kBackward || grads_ == nullptr) return OMEGA_RUNTIME_STATUS_INVALID_ARGUMENT;
    if (grads_->sum_abs_d_depth_embedding != nullptr) local.instrumentation = 1;

    std::array<float*, kParameterCount> primary{};
    std::array<float*, kParameterCount> sum_abs{};
    std::array<size_t*, kParameterCount> counts{};
    bind_parameter_pointers(config_, arena.primary, &primary);
    bind_parameter_pointers(config_, arena.sum_abs, &sum_abs);
    std::array<size_t, kParameterCount> parameter_counts_for_config{};
    size_t parameter_count = 0;
    if (!parameter_counts(config_, &parameter_counts_for_config, &parameter_count)) return OMEGA_RUNTIME_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < kParameterCount; ++index) {
      counts[index] = arena.counts + (primary[index] - arena.primary);
    }
    OmegaRecurrentGrads local_grads{};
    local_grads.d_token_part = grads_->d_token_part + token_offset;
    local_grads.d_previous_state = grads_->d_previous_state + batch_offset;
    local_grads.d_state_part_weight = primary[0];
    local_grads.d_prelude_norm_weight = primary[1];
    local_grads.d_block_qkv_weight = primary[2];
    local_grads.d_block_qkv_bias = primary[3];
    local_grads.d_block_out_weight = primary[4];
    local_grads.d_block_out_bias = primary[5];
    local_grads.d_block_fc1_weight = primary[6];
    local_grads.d_block_fc1_bias = primary[7];
    local_grads.d_block_fc2_weight = primary[8];
    local_grads.d_block_fc2_bias = primary[9];
    local_grads.d_block_norm_weight = primary[10];
    local_grads.d_depth_embedding = primary[11];
    local_grads.d_gate_logits = primary[12];
    local_grads.sum_abs_d_token_part = grads_->sum_abs_d_token_part == nullptr ? nullptr : grads_->sum_abs_d_token_part + token_offset;
    local_grads.sum_abs_d_previous_state = grads_->sum_abs_d_previous_state == nullptr ? nullptr : grads_->sum_abs_d_previous_state + batch_offset;
    local_grads.count_d_token_part = grads_->count_d_token_part == nullptr ? nullptr : grads_->count_d_token_part + token_offset;
    local_grads.count_d_previous_state = grads_->count_d_previous_state == nullptr ? nullptr : grads_->count_d_previous_state + batch_offset;
    local_grads.sum_abs_d_state_part_weight = grads_->sum_abs_d_state_part_weight == nullptr ? nullptr : sum_abs[0];
    local_grads.sum_abs_d_prelude_norm_weight = grads_->sum_abs_d_prelude_norm_weight == nullptr ? nullptr : sum_abs[1];
    local_grads.sum_abs_d_block_qkv_weight = grads_->sum_abs_d_block_qkv_weight == nullptr ? nullptr : sum_abs[2];
    local_grads.sum_abs_d_block_qkv_bias = grads_->sum_abs_d_block_qkv_bias == nullptr ? nullptr : sum_abs[3];
    local_grads.sum_abs_d_block_out_weight = grads_->sum_abs_d_block_out_weight == nullptr ? nullptr : sum_abs[4];
    local_grads.sum_abs_d_block_out_bias = grads_->sum_abs_d_block_out_bias == nullptr ? nullptr : sum_abs[5];
    local_grads.sum_abs_d_block_fc1_weight = grads_->sum_abs_d_block_fc1_weight == nullptr ? nullptr : sum_abs[6];
    local_grads.sum_abs_d_block_fc1_bias = grads_->sum_abs_d_block_fc1_bias == nullptr ? nullptr : sum_abs[7];
    local_grads.sum_abs_d_block_fc2_weight = grads_->sum_abs_d_block_fc2_weight == nullptr ? nullptr : sum_abs[8];
    local_grads.sum_abs_d_block_fc2_bias = grads_->sum_abs_d_block_fc2_bias == nullptr ? nullptr : sum_abs[9];
    local_grads.sum_abs_d_block_norm_weight = grads_->sum_abs_d_block_norm_weight == nullptr ? nullptr : sum_abs[10];
    local_grads.sum_abs_d_depth_embedding = grads_->sum_abs_d_depth_embedding == nullptr ? nullptr : sum_abs[11];
    local_grads.sum_abs_d_gate_logits = grads_->sum_abs_d_gate_logits == nullptr ? nullptr : sum_abs[12];
    local_grads.count_d_state_part_weight = grads_->count_d_state_part_weight == nullptr ? nullptr : counts[0];
    local_grads.count_d_prelude_norm_weight = grads_->count_d_prelude_norm_weight == nullptr ? nullptr : counts[1];
    local_grads.count_d_block_qkv_weight = grads_->count_d_block_qkv_weight == nullptr ? nullptr : counts[2];
    local_grads.count_d_block_qkv_bias = grads_->count_d_block_qkv_bias == nullptr ? nullptr : counts[3];
    local_grads.count_d_block_out_weight = grads_->count_d_block_out_weight == nullptr ? nullptr : counts[4];
    local_grads.count_d_block_out_bias = grads_->count_d_block_out_bias == nullptr ? nullptr : counts[5];
    local_grads.count_d_block_fc1_weight = grads_->count_d_block_fc1_weight == nullptr ? nullptr : counts[6];
    local_grads.count_d_block_fc1_bias = grads_->count_d_block_fc1_bias == nullptr ? nullptr : counts[7];
    local_grads.count_d_block_fc2_weight = grads_->count_d_block_fc2_weight == nullptr ? nullptr : counts[8];
    local_grads.count_d_block_fc2_bias = grads_->count_d_block_fc2_bias == nullptr ? nullptr : counts[9];
    local_grads.count_d_block_norm_weight = grads_->count_d_block_norm_weight == nullptr ? nullptr : counts[10];
    local_grads.count_d_depth_embedding = grads_->count_d_depth_embedding == nullptr ? nullptr : counts[11];
    local_grads.count_d_gate_logits = grads_->count_d_gate_logits == nullptr ? nullptr : counts[12];
    local_grads.fp64_d_depth_embedding = grads_->fp64_d_depth_embedding == nullptr ? nullptr : arena.fp64;
    local_grads.depth_max_level = grads_->depth_max_level == nullptr ? nullptr : arena.depth_max;
#ifdef OMEGA_FC2_REPLAY_CAPTURE
    const size_t prior_worker_index = g_fc2_replay_worker_index;
    g_fc2_replay_worker_index = worker_index;
#endif
    const int status = recurrent_backward_impl(&local, params_, token_part_ + token_offset,
        d_readout_states_ + token_offset, d_next_state_ + batch_offset, arena.core, arena.core_bytes, &local_grads);
#ifdef OMEGA_FC2_REPLAY_CAPTURE
    g_fc2_replay_worker_index = prior_worker_index;
#endif
    return status;
  }

  size_t worker_count_;
  std::array<std::thread, 4> workers_{};
  mutable std::mutex mutex_;
  std::condition_variable work_cv_;
  std::condition_variable done_cv_;
  bool stopping_ = false;
  bool busy_ = false;
  size_t dispatch_id_ = 0;
  size_t completed_ = 0;
  Operation operation_ = Operation::kNone;
  int statuses_[4]{};
  OmegaRecurrentConfig config_{};
  const OmegaRecurrentParams* params_ = nullptr;
  const float* token_part_ = nullptr;
  const float* previous_state_ = nullptr;
  float* next_state_ = nullptr;
  float* readout_states_ = nullptr;
  const float* d_readout_states_ = nullptr;
  const float* d_next_state_ = nullptr;
  OmegaRecurrentGrads* grads_ = nullptr;
#ifdef OMEGA_P2R_DIAGNOSTIC
  OmegaRuntimeDiagnosticSnapshot diagnostic_snapshot_{};
#endif
};

namespace {

void zero_and_reduce_float(float* destination, const std::array<float*, 4>& sources, size_t count, size_t workers) {
  std::memset(destination, 0, count * sizeof(float));
  for (size_t worker = 0; worker < workers; ++worker) {
    for (size_t index = 0; index < count; ++index) destination[index] += sources[worker][index];
  }
}

void zero_and_reduce_counts(size_t* destination, const std::array<size_t*, 4>& sources, size_t count, size_t workers) {
  std::memset(destination, 0, count * sizeof(size_t));
  for (size_t worker = 0; worker < workers; ++worker) {
    for (size_t index = 0; index < count; ++index) destination[index] += sources[worker][index];
  }
}

void zero_and_reduce_doubles(double* destination, const std::array<double*, 4>& sources, size_t count, size_t workers) {
  std::memset(destination, 0, count * sizeof(double));
  for (size_t worker = 0; worker < workers; ++worker) {
    for (size_t index = 0; index < count; ++index) destination[index] += sources[worker][index];
  }
}

void zero_and_reduce_max(size_t* destination, const std::array<size_t*, 4>& sources, size_t count, size_t workers) {
  std::memset(destination, 0, count * sizeof(size_t));
  for (size_t worker = 0; worker < workers; ++worker) {
    for (size_t index = 0; index < count; ++index) destination[index] = std::max(destination[index], sources[worker][index]);
  }
}

void reduce_runtime_gradients(OmegaRuntime* runtime, OmegaRecurrentGrads* grads, const OmegaRecurrentConfig& config) {
  std::array<size_t, kParameterCount> counts{};
  size_t total = 0;
  parameter_counts(config, &counts, &total);
  std::array<float*, kParameterCount> destinations{};
  std::array<float*, kParameterCount> sum_destinations{};
  std::array<size_t*, kParameterCount> count_destinations{};
  bind_external_parameter_pointers(grads, &destinations);
  bind_external_optional_float_pointers(grads, &sum_destinations);
  bind_external_optional_count_pointers(grads, &count_destinations);
  std::array<float*, 4> primary_sources{};
  std::array<float*, 4> sum_sources{};
  std::array<size_t*, 4> count_sources{};
  for (size_t parameter = 0; parameter < kParameterCount; ++parameter) {
    for (size_t worker = 0; worker < runtime->worker_count(); ++worker) {
      std::array<float*, kParameterCount> primary{};
      std::array<float*, kParameterCount> sums{};
      std::array<size_t, kParameterCount> local_counts{};
      size_t parameter_count = 0;
      parameter_counts(config, &local_counts, &parameter_count);
      bind_parameter_pointers(config, runtime->arenas_[worker].primary, &primary);
      bind_parameter_pointers(config, runtime->arenas_[worker].sum_abs, &sums);
      primary_sources[worker] = primary[parameter];
      sum_sources[worker] = sums[parameter];
      count_sources[worker] = runtime->arenas_[worker].counts + (primary[parameter] - runtime->arenas_[worker].primary);
    }
    if (destinations[parameter] != nullptr) zero_and_reduce_float(destinations[parameter], primary_sources, counts[parameter], runtime->worker_count());
    if (sum_destinations[parameter] != nullptr) zero_and_reduce_float(sum_destinations[parameter], sum_sources, counts[parameter], runtime->worker_count());
    if (count_destinations[parameter] != nullptr) zero_and_reduce_counts(count_destinations[parameter], count_sources, counts[parameter], runtime->worker_count());
  }
  size_t depth_count = 0;
  checked_mul(config.rounds, config.dimension, &depth_count);
  if (grads->fp64_d_depth_embedding != nullptr || grads->depth_max_level != nullptr) {
    std::array<double*, 4> fp64_sources{};
    std::array<size_t*, 4> max_sources{};
    for (size_t worker = 0; worker < runtime->worker_count(); ++worker) {
      fp64_sources[worker] = runtime->arenas_[worker].fp64;
      max_sources[worker] = runtime->arenas_[worker].depth_max;
    }
    if (grads->fp64_d_depth_embedding != nullptr) zero_and_reduce_doubles(grads->fp64_d_depth_embedding, fp64_sources, depth_count, runtime->worker_count());
    if (grads->depth_max_level != nullptr) zero_and_reduce_max(grads->depth_max_level, max_sources, depth_count, runtime->worker_count());
  }
}

bool runtime_common_valid(const OmegaRuntime* runtime, const OmegaRecurrentConfig* config,
    const OmegaRecurrentParams* params) {
  return runtime != nullptr && config != nullptr && params != nullptr && valid_params(*params, *config);
}

}  // namespace

#ifdef OMEGA_FC2_REPLAY_CAPTURE
extern "C" int omega_fc2_replay_capture_set_targets(
    const OmegaFc2ReplayGroupIdentity* targets,
    size_t target_count,
    OmegaFc2ReplayCaptureCallback callback,
    void* user_data) {
  if (target_count == 0 || targets == nullptr || callback == nullptr) return OMEGA_RUNTIME_STATUS_INVALID_ARGUMENT;
  g_fc2_replay_targets = targets;
  g_fc2_replay_target_count = target_count;
  g_fc2_replay_callback = callback;
  g_fc2_replay_user_data = user_data;
  return OMEGA_RUNTIME_STATUS_OK;
}
#endif

extern "C" OmegaRuntime* omega_runtime_create(size_t num_threads) {
  OmegaRuntime* runtime = nullptr;
  if (omega_runtime_create_status(num_threads, &runtime) != OMEGA_RUNTIME_STATUS_OK) return nullptr;
  return runtime;
}

extern "C" int omega_runtime_create_status(size_t num_threads, OmegaRuntime** runtime) {
  if (runtime == nullptr) return OMEGA_RUNTIME_STATUS_INVALID_ARGUMENT;
  *runtime = nullptr;
  if (!valid_runtime_worker_count(num_threads)) return OMEGA_RUNTIME_STATUS_INVALID_WORKER_COUNT;
  try {
    *runtime = new OmegaRuntime(num_threads);
    return OMEGA_RUNTIME_STATUS_OK;
  } catch (const std::exception&) {
    return OMEGA_RUNTIME_STATUS_WORKER_FAILURE;
  } catch (...) {
    return OMEGA_RUNTIME_STATUS_WORKER_FAILURE;
  }
}

extern "C" void omega_runtime_destroy(OmegaRuntime* runtime) {
  delete runtime;
}

#ifdef OMEGA_P2R_DIAGNOSTIC
extern "C" int omega_runtime_diagnostic_snapshot(
    const OmegaRuntime* runtime, OmegaRuntimeDiagnosticSnapshot* snapshot) {
  if (runtime == nullptr) return OMEGA_RUNTIME_STATUS_INVALID_HANDLE;
  return runtime->diagnostic_snapshot(snapshot);
}
#endif

extern "C" size_t omega_runtime_workspace_bytes(const OmegaRuntime* runtime, OmegaRecurrentConfig config) {
  if (runtime == nullptr) return 0;
  return runtime_workspace_bytes_impl(runtime->worker_count(), config);
}

extern "C" int omega_runtime_forward(
    OmegaRuntime* runtime,
    const OmegaRecurrentConfig* config,
    const OmegaRecurrentParams* params,
    const float* token_part,
    const float* previous_state,
    float* next_state,
    float* readout_states,
    void* workspace,
    size_t workspace_bytes) {
  if (runtime == nullptr) return OMEGA_RUNTIME_STATUS_INVALID_HANDLE;
  if (config == nullptr || params == nullptr || token_part == nullptr || previous_state == nullptr || next_state == nullptr ||
      readout_states == nullptr || workspace == nullptr) return OMEGA_RUNTIME_STATUS_INVALID_ARGUMENT;
  if (!runtime_common_valid(runtime, config, params)) return OMEGA_RUNTIME_STATUS_INVALID_ARGUMENT;
  if (runtime->worker_count() == 1) {
#ifdef OMEGA_P2R_DIAGNOSTIC
    runtime->diagnostic_begin();
    runtime->diagnostic_worker_start(0);
     const int status = recurrent_forward_impl(config, params, token_part, previous_state, next_state, readout_states, workspace, workspace_bytes);
    runtime->diagnostic_worker_end(0);
    runtime->diagnostic_all_workers_done();
    runtime->diagnostic_return();
    return status;
#else
  return recurrent_forward_impl(config, params, token_part, previous_state, next_state, readout_states, workspace, workspace_bytes);
#endif
  }
  if (config->batch == 0 || config->batch % runtime->worker_count() != 0) return OMEGA_RUNTIME_STATUS_INVALID_PARTITION;
#ifdef OMEGA_PROFILE_INTERNAL
  return OMEGA_RUNTIME_STATUS_PROFILE_GUARD;
#else
  const size_t required = runtime_workspace_bytes_impl(runtime->worker_count(), *config);
  if (required == 0 || workspace_bytes < required) return OMEGA_RUNTIME_STATUS_WORKSPACE;
  const int status = runtime->run(OmegaRuntime::Operation::kForward, *config, params, token_part, previous_state, next_state,
      readout_states, nullptr, nullptr, workspace, workspace_bytes, nullptr);
#ifdef OMEGA_P2R_DIAGNOSTIC
  runtime->diagnostic_return();
#endif
  return status;
#endif
}

extern "C" int omega_runtime_backward(
    OmegaRuntime* runtime,
    const OmegaRecurrentConfig* config,
    const OmegaRecurrentParams* params,
    const float* token_part,
    const float* d_readout_states,
    const float* d_next_state,
    void* workspace,
    size_t workspace_bytes,
    OmegaRecurrentGrads* grads) {
  if (runtime == nullptr) return OMEGA_RUNTIME_STATUS_INVALID_HANDLE;
  if (config == nullptr || params == nullptr || token_part == nullptr || d_readout_states == nullptr || d_next_state == nullptr ||
      workspace == nullptr || grads == nullptr) return OMEGA_RUNTIME_STATUS_INVALID_ARGUMENT;
  if (!runtime_common_valid(runtime, config, params) || !valid_grads(*grads) || config->training == 0) {
    return OMEGA_RUNTIME_STATUS_INVALID_ARGUMENT;
  }
  if (runtime->worker_count() == 1) {
#ifdef OMEGA_P2R_DIAGNOSTIC
    runtime->diagnostic_begin();
    runtime->diagnostic_worker_start(0);
     const int status = recurrent_backward_impl(config, params, token_part, d_readout_states, d_next_state, workspace, workspace_bytes, grads);
    runtime->diagnostic_worker_end(0);
    runtime->diagnostic_all_workers_done();
    runtime->diagnostic_return();
    return status;
#else
  return recurrent_backward_impl(config, params, token_part, d_readout_states, d_next_state, workspace, workspace_bytes, grads);
#endif
  }
  if (config->batch == 0 || config->batch % runtime->worker_count() != 0) return OMEGA_RUNTIME_STATUS_INVALID_PARTITION;
#ifdef OMEGA_PROFILE_INTERNAL
  return OMEGA_RUNTIME_STATUS_PROFILE_GUARD;
#else
  const size_t required = runtime_workspace_bytes_impl(runtime->worker_count(), *config);
  if (required == 0 || workspace_bytes < required) return OMEGA_RUNTIME_STATUS_WORKSPACE;
  const int status = runtime->run(OmegaRuntime::Operation::kBackward, *config, params, token_part, nullptr, nullptr, nullptr,
      d_readout_states, d_next_state, workspace, workspace_bytes, grads);
  if (status != OMEGA_RUNTIME_STATUS_OK) return status;
#ifdef OMEGA_P2R_DIAGNOSTIC
  runtime->diagnostic_reduction_start();
#endif
  reduce_runtime_gradients(runtime, grads, *config);
#ifdef OMEGA_P2R_DIAGNOSTIC
  runtime->diagnostic_reduction_end();
  runtime->diagnostic_return();
#endif
  return OMEGA_RUNTIME_STATUS_OK;
#endif
}

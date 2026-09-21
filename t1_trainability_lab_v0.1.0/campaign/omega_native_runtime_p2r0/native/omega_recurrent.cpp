#include "omega_recurrent.h"

#include <algorithm>
#ifdef OMEGA_PROFILE_INTERNAL
#include <atomic>
#include <chrono>
#include <mutex>
#endif
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr size_t kAlignment = 64;
constexpr float kEpsilon = 1.0e-6F;
constexpr float kSqrtTwo = 1.4142135623730950488F;

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

bool depth_reduction_layout(const OmegaRecurrentConfig& config, size_t* output_count, size_t* levels, size_t* partial_count) {
  size_t contributions = 0;
  size_t four_dimension = 0;
  if (!checked_mul(4, config.dimension, &four_dimension) || !checked_mul(config.batch, config.sequence_length, &contributions) ||
      !checked_mul(contributions, config.slots, &contributions) || !checked_mul(contributions, four_dimension, &contributions) ||
      !checked_mul(config.rounds, config.dimension, output_count)) {
    return false;
  }
  if (contributions == 0) {
    return false;
  }
  constexpr size_t mask_bits = std::numeric_limits<std::uint64_t>::digits;
  size_t capacity = 1;
  size_t computed_levels = 1;
  while (capacity < contributions) {
    if (computed_levels >= mask_bits || capacity > std::numeric_limits<size_t>::max() / 2) {
      return false;
    }
    capacity *= 2;
    ++computed_levels;
  }
  *levels = computed_levels;
  return checked_mul(*output_count, *levels, partial_count);
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
  size_t depth_bias_count = 0;
  if (!checked_mul(config.rounds, config.dimension, &depth_bias_count) || !checked_mul(depth_bias_count, 4, &depth_bias_count)) {
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
  if (!append_floats(slot_scores, &total) || !append_floats(fc1_count, &total) || !append_floats(depth_bias_count, &total)) {
    return 0;
  }
  if (config.training != 0 && !append_training_storage(config, state_count, &total)) {
    return 0;
  }
  size_t output_count = 0;
  size_t levels = 0;
  size_t partial_count = 0;
  if (!depth_reduction_layout(config, &output_count, &levels, &partial_count) ||
      !append_floats(partial_count, &total) ||
      (config.instrumentation != 0 && !append_floats(partial_count, &total)) ||
      !append_uint64s(output_count, &total) ||
      (config.instrumentation != 0 && !append_uint64s(output_count, &total))) {
    return 0;
  }
  return total;
}

struct WorkspaceCursor {
  unsigned char* base;
  size_t capacity;
  size_t used;

  float* take(size_t count) {
    size_t bytes = 0;
    if (!checked_mul(count, sizeof(float), &bytes)) {
      return nullptr;
    }
    const uintptr_t address = reinterpret_cast<uintptr_t>(base) + used;
    const size_t remainder = static_cast<size_t>(address % kAlignment);
    const size_t padding = remainder == 0 ? 0 : kAlignment - remainder;
    size_t required = 0;
    if (!checked_add(padding, bytes, &required) || required > capacity - used) {
      return nullptr;
    }
    used += padding;
    float* result = reinterpret_cast<float*>(base + used);
    used += bytes;
    return result;
  }

  std::uint64_t* take_uint64(size_t count) {
    size_t bytes = 0;
    if (!checked_mul(count, sizeof(std::uint64_t), &bytes) || used > capacity) {
      return nullptr;
    }
    const uintptr_t address = reinterpret_cast<uintptr_t>(base) + used;
    const size_t remainder = static_cast<size_t>(address % kAlignment);
    const size_t padding = remainder == 0 ? 0 : kAlignment - remainder;
    size_t required = 0;
    if (!checked_add(padding, bytes, &required) || required > capacity - used) {
      return nullptr;
    }
    used += padding;
    std::uint64_t* result = reinterpret_cast<std::uint64_t*>(base + used);
    used += bytes;
    return result;
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

bool carry_add(float* partials, std::uint64_t* masks, size_t output_index, size_t levels, float value) {
  size_t base = 0;
  if (!checked_mul(output_index, levels, &base)) {
    return false;
  }
  std::uint64_t mask = masks[output_index];
  for (size_t level = 0; level < levels; ++level) {
    const std::uint64_t bit = std::uint64_t{1} << level;
    if ((mask & bit) == 0) {
      partials[base + level] = value;
      masks[output_index] = mask | bit;
      return true;
    }
    value = partials[base + level] + value;
    mask &= ~bit;
  }
  return false;
}

bool carry_flush(float* partials, std::uint64_t* masks, size_t output_index, size_t levels, float* output) {
  size_t base = 0;
  if (!checked_mul(output_index, levels, &base)) {
    return false;
  }
  const std::uint64_t mask = masks[output_index];
  bool initialized = false;
  float total = 0.0F;
  for (size_t level = 0; level < levels; ++level) {
    const std::uint64_t bit = std::uint64_t{1} << level;
    if ((mask & bit) != 0) {
      total = initialized ? total + partials[base + level] : partials[base + level];
      initialized = true;
    }
  }
  if (!initialized) {
    return false;
  }
  *output = total;
  return true;
}

bool add_depth_contribution(const OmegaRecurrentGrads& grads, float* gradient_partials, std::uint64_t* gradient_masks,
    float* abs_partials, std::uint64_t* abs_masks, size_t output_index, size_t levels, float contribution) {
  if (!carry_add(gradient_partials, gradient_masks, output_index, levels, contribution)) {
    return false;
  }
  if (grads.sum_abs_d_depth_embedding != nullptr &&
      !carry_add(abs_partials, abs_masks, output_index, levels, std::fabs(contribution))) {
    return false;
  }
  if (grads.count_d_depth_embedding != nullptr) {
    ++grads.count_d_depth_embedding[output_index];
  }
  if (grads.fp64_d_depth_embedding != nullptr) {
    grads.fp64_d_depth_embedding[output_index] += static_cast<double>(contribution);
  }
  return true;
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

extern "C" int omega_recurrent_forward(
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
  float* depth_bias = cursor.take(depth_bias_count);
  if (state == nullptr || anchor == nullptr || candidate == nullptr || q == nullptr || key == nullptr || value == nullptr || mixed == nullptr || update == nullptr || scores == nullptr || fc1 == nullptr || depth_bias == nullptr) {
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

  {
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
    if (config->training != 0) {
      OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kHistoryBufferReadsWrites);
      std::memcpy(training_buffers.state_history + position * state_count, state, state_count * sizeof(float));
    }
     for (size_t batch = 0; batch < config->batch; ++batch) {
      OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kStatePrelude);
      for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
        float write = token_part[(batch * config->sequence_length + position) * state_dimension + flattened];
             for (size_t input = 0; input < dimension; ++input) {
          float mean = 0.0F;
          for (size_t slot = 0; slot < slots; ++slot) {
            mean += state[state_index(batch, slot, input, slots, dimension)];
          }
          mean /= static_cast<float>(slots);
          const float* state_part_row = params->state_part_weight.data +
              flattened * static_cast<size_t>(params->state_part_weight.row_stride);
          write += state_part_row[input] * mean;
        }
        anchor[batch * state_dimension + flattened] = write;
      }

      {
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
    if (config->training != 0) {
      OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kHistoryBufferReadsWrites);
      std::memcpy(training_buffers.anchor_history + position * state_count, anchor, state_count * sizeof(float));
    }

    std::memcpy(candidate, anchor, state_count * sizeof(float));
    for (size_t round = 0; round < config->rounds; ++round) {
      if (config->training != 0) {
        const size_t round_offset = (position * config->rounds + round) * state_count;
        OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kHistoryBufferReadsWrites);
        std::memcpy(training_buffers.candidate_inputs + round_offset, candidate, state_count * sizeof(float));
      }
      for (size_t batch = 0; batch < config->batch; ++batch) {
        {
          OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kQkvProjection);
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
           }

           {
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
          for (size_t d = 0; d < dimension; ++d) {
            float output = params->block_out_bias[d];
            for (size_t input_dimension = 0; input_dimension < dimension; ++input_dimension) {
              output += params->block_out_weight[d * dimension + input_dimension] * mixed[state_index(batch, slot, input_dimension, slots, dimension)];
            }
            update[state_index(batch, slot, d, slots, dimension)] = output;
          }
          }
          if (config->training != 0) {
            const size_t round_offset = (position * config->rounds + round) * state_count;
            std::memcpy(training_buffers.out_history + round_offset + batch * state_dimension + slot * dimension,
                update + batch * state_dimension + slot * dimension, dimension * sizeof(float));
          }
          {
            OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kFc1Gelu);
            for (size_t row = 0; row < 4 * dimension; ++row) {
              float hidden = depth_bias[round * 4 * dimension + row];
              for (size_t input_dimension = 0; input_dimension < dimension; ++input_dimension) {
                hidden += params->block_fc1_weight[row * dimension + input_dimension] * update[state_index(batch, slot, input_dimension, slots, dimension)];
              }
              fc1[(batch * slots + slot) * 4 * dimension + row] = hidden;
            }
          }
          float sum_squares_update = 0.0F;
          {
            OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kFc2);
            for (size_t d = 0; d < dimension; ++d) {
              float transformed = params->block_fc2_bias[d];
              for (size_t row = 0; row < 4 * dimension; ++row) {
                transformed += params->block_fc2_weight[d * 4 * dimension + row] * gelu(fc1[(batch * slots + slot) * 4 * dimension + row]);
              }
              const float gate = sigmoid(params->gate_logits[round * dimension + d]);
              const float value_at = candidate[state_index(batch, slot, d, slots, dimension)] + gate * transformed;
              update[state_index(batch, slot, d, slots, dimension)] = value_at;
              sum_squares_update += value_at * value_at;
            }
          }
          if (config->training != 0) {
            const size_t round_offset = (position * config->rounds + round) * state_count;
            OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kHistoryBufferReadsWrites);
            std::memcpy(training_buffers.pre_rms_history + round_offset + batch * state_dimension + slot * dimension,
                update + batch * state_dimension + slot * dimension, dimension * sizeof(float));
          }
          {
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
        OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kHistoryBufferReadsWrites);
        std::memcpy(training_buffers.q_history + round_offset, q, state_count * sizeof(float));
        std::memcpy(training_buffers.key_history + round_offset, key, state_count * sizeof(float));
        std::memcpy(training_buffers.value_history + round_offset, value, state_count * sizeof(float));
        std::memcpy(training_buffers.probability_history + probability_offset, scores, slot_scores * sizeof(float));
        std::memcpy(training_buffers.attention_history + round_offset, mixed, state_count * sizeof(float));
        std::memcpy(training_buffers.fc1_pre_history + round_offset * 4, fc1, fc1_count * sizeof(float));
      }
    }

    for (size_t batch = 0; batch < config->batch; ++batch) {
      std::memcpy(state + batch * state_dimension, candidate + batch * state_dimension, state_dimension * sizeof(float));
      std::memcpy(readout_states + (batch * config->sequence_length + position) * state_dimension, candidate + batch * state_dimension, state_dimension * sizeof(float));
    }
    if (config->training != 0) {
      OMEGA_PROFILE_SCOPE(ProfileDirection::kForward, ProfileStage::kHistoryBufferReadsWrites);
      std::memcpy(training_buffers.state_history + (position + 1) * state_count, state, state_count * sizeof(float));
    }
  }
  std::memcpy(next_state, state, state_count * sizeof(float));
  return 0;
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
  size_t depth_bias_count = 0;
  size_t depth_output_count = 0;
  size_t depth_levels = 0;
  size_t depth_partial_count = 0;
  if (!checked_mul(config->batch, slots, &slot_scores) || !checked_mul(slot_scores, slots, &slot_scores) ||
      !checked_mul(state_count, 4, &fc1_count) || !checked_mul(config->rounds, dimension, &depth_bias_count) ||
      !checked_mul(depth_bias_count, 4, &depth_bias_count) ||
      !depth_reduction_layout(*config, &depth_output_count, &depth_levels, &depth_partial_count)) {
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
  float* depth_bias = cursor.take(depth_bias_count);
  if (state == nullptr || anchor == nullptr || candidate == nullptr || q == nullptr || key == nullptr || value == nullptr || mixed == nullptr || update == nullptr || scores == nullptr || fc1 == nullptr || depth_bias == nullptr) {
    return 13;
  }
  TrainingBuffers training_buffers;
  if (!bind_training_storage(*config, state_count, &cursor, &training_buffers)) {
    return 14;
  }
  float* depth_gradient_partials = cursor.take(depth_partial_count);
  float* depth_abs_partials = config->instrumentation != 0 ? cursor.take(depth_partial_count) : nullptr;
  std::uint64_t* depth_gradient_masks = cursor.take_uint64(depth_output_count);
  std::uint64_t* depth_abs_masks = config->instrumentation != 0 ? cursor.take_uint64(depth_output_count) : nullptr;
  if (depth_gradient_partials == nullptr || depth_gradient_masks == nullptr ||
      (config->instrumentation != 0 && (depth_abs_partials == nullptr || depth_abs_masks == nullptr))) {
    return 15;
  }
  std::memset(depth_gradient_partials, 0, depth_partial_count * sizeof(float));
  std::memset(depth_gradient_masks, 0, depth_output_count * sizeof(std::uint64_t));
  if (config->instrumentation != 0) {
    std::memset(depth_abs_partials, 0, depth_partial_count * sizeof(float));
    std::memset(depth_abs_masks, 0, depth_output_count * sizeof(std::uint64_t));
  }

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

       for (size_t batch = 0; batch < config->batch; ++batch) {
        for (size_t slot = 0; slot < slots; ++slot) {
          const size_t base = state_index(batch, slot, 0, slots, dimension);
          float rms_dot = 0.0F;
          for (size_t d = 0; d < dimension; ++d) {
            const size_t index = base + d;
            rms_dot += candidate[index] * params->block_norm_weight[d] * saved_pre_rms[index];
          }
          float pre_rms_sum = 0.0F;
          for (size_t d = 0; d < dimension; ++d) {
            const float pre = saved_pre_rms[base + d];
            pre_rms_sum += pre * pre;
          }
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
              for (size_t d = 0; d < dimension; ++d) {
              for (size_t row = 0; row < 4 * dimension; ++row) {
               const float hidden = gelu(saved_fc1_pre[(batch * slots + slot) * 4 * dimension + row]);
               const float contribution = update[base + d] * hidden;
               grads->d_block_fc2_weight[d * 4 * dimension + row] += contribution;
               add_abs_contribution(grads->sum_abs_d_block_fc2_weight, grads->count_d_block_fc2_weight, d * 4 * dimension + row, contribution);
              }
             grads->d_block_fc2_bias[d] += update[base + d];
             add_abs_contribution(grads->sum_abs_d_block_fc2_bias, grads->count_d_block_fc2_bias, d, update[base + d]);
            }
            }
            {
             OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kFc1Gelu);
             for (size_t d = 0; d < dimension; ++d) {
               mixed[base + d] = 0.0F;
              }
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
               const float depth_contribution = params->block_fc1_weight[row * dimension + input] * fc1[fc_index];
               grads->d_block_fc1_weight[row * dimension + input] += fc1_weight_contribution;
                add_abs_contribution(grads->sum_abs_d_block_fc1_weight, grads->count_d_block_fc1_weight, row * dimension + input, fc1_weight_contribution);
               const size_t depth_index = round * dimension + input;
               if (!add_depth_contribution(*grads, depth_gradient_partials, depth_gradient_masks, depth_abs_partials, depth_abs_masks,
                        depth_index, depth_levels, depth_contribution)) {
                  return 15;
               }
               mixed[base + input] += params->block_fc1_weight[row * dimension + input] * fc1[fc_index];
            }
           }
           }
           {
             OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kRmsnormGates);
             for (size_t d = 0; d < dimension; ++d) {
            grads->d_block_out_bias[d] += mixed[base + d];
              add_abs_contribution(grads->sum_abs_d_block_out_bias, grads->count_d_block_out_bias, d, mixed[base + d]);
           }
           }
           {
             OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kOutProjection);
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

        {
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
          }
        }
      }
       for (size_t index = 0; index < state_count; ++index) {
        anchor[index] += candidate[index];
      }

     {
       OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kStatePrelude);
     const float* saved_state = training_buffers.state_history + position * state_count;
    for (size_t batch = 0; batch < config->batch; ++batch) {
      float pre_norm_sum = 0.0F;
      for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
        float write = token_part[(batch * config->sequence_length + position) * state_dimension + flattened];
        for (size_t input = 0; input < dimension; ++input) {
          float mean = 0.0F;
          for (size_t slot = 0; slot < slots; ++slot) {
            mean += saved_state[state_index(batch, slot, input, slots, dimension)];
          }
          mean /= static_cast<float>(slots);
          const float* state_part_row = params->state_part_weight.data +
              flattened * static_cast<size_t>(params->state_part_weight.row_stride);
          write += state_part_row[input] * mean;
        }
        const float pre = saved_state[batch * state_dimension + flattened] + write;
        pre_norm_sum += pre * pre;
      }
      const float inverse_rms = 1.0F / std::sqrt(pre_norm_sum / static_cast<float>(state_dimension) + kEpsilon);
      float norm_dot = 0.0F;
      for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
        float write = token_part[(batch * config->sequence_length + position) * state_dimension + flattened];
        for (size_t input = 0; input < dimension; ++input) {
          float mean = 0.0F;
          for (size_t slot = 0; slot < slots; ++slot) {
            mean += saved_state[state_index(batch, slot, input, slots, dimension)];
          }
          mean /= static_cast<float>(slots);
          const float* state_part_row = params->state_part_weight.data +
              flattened * static_cast<size_t>(params->state_part_weight.row_stride);
          write += state_part_row[input] * mean;
        }
        const float pre = saved_state[batch * state_dimension + flattened] + write;
        norm_dot += anchor[batch * state_dimension + flattened] * params->prelude_norm_weight[flattened] * pre;
      }
      const float inverse_rms_cubed_over_dimension = inverse_rms * inverse_rms * inverse_rms / static_cast<float>(state_dimension);
      for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
        float write = token_part[(batch * config->sequence_length + position) * state_dimension + flattened];
        for (size_t input = 0; input < dimension; ++input) {
          float mean = 0.0F;
          for (size_t slot = 0; slot < slots; ++slot) {
            mean += saved_state[state_index(batch, slot, input, slots, dimension)];
          }
          mean /= static_cast<float>(slots);
          const float* state_part_row = params->state_part_weight.data +
              flattened * static_cast<size_t>(params->state_part_weight.row_stride);
          write += state_part_row[input] * mean;
        }
        const float pre = saved_state[batch * state_dimension + flattened] + write;
        const float d_pre = anchor[batch * state_dimension + flattened] * params->prelude_norm_weight[flattened] * inverse_rms - pre * inverse_rms_cubed_over_dimension * norm_dot;
        const float prelude_norm_contribution = anchor[batch * state_dimension + flattened] * pre * inverse_rms;
        grads->d_prelude_norm_weight[flattened] += prelude_norm_contribution;
        add_abs_contribution(grads->sum_abs_d_prelude_norm_weight, grads->count_d_prelude_norm_weight, flattened, prelude_norm_contribution);
        grads->d_token_part[(batch * config->sequence_length + position) * state_dimension + flattened] += d_pre;
        add_abs_contribution(grads->sum_abs_d_token_part, grads->count_d_token_part, (batch * config->sequence_length + position) * state_dimension + flattened, d_pre);
        state[batch * state_dimension + flattened] = d_pre;
        for (size_t input = 0; input < dimension; ++input) {
          float mean = 0.0F;
          for (size_t slot = 0; slot < slots; ++slot) {
            mean += saved_state[state_index(batch, slot, input, slots, dimension)];
          }
          mean /= static_cast<float>(slots);
          const float state_part_contribution = d_pre * mean;
          grads->d_state_part_weight[flattened * dimension + input] += state_part_contribution;
           add_abs_contribution(grads->sum_abs_d_state_part_weight, grads->count_d_state_part_weight, flattened * dimension + input, state_part_contribution);
         }
       }
       for (size_t input = 0; input < dimension; ++input) {
        float d_mean = 0.0F;
        for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
          const float* state_part_row = params->state_part_weight.data +
              flattened * static_cast<size_t>(params->state_part_weight.row_stride);
          d_mean += state[batch * state_dimension + flattened] * state_part_row[input];
        }
        anchor[batch * state_dimension + input] = d_mean / static_cast<float>(slots);
      }
      for (size_t slot = 0; slot < slots; ++slot) {
        for (size_t input = 0; input < dimension; ++input) {
          state[state_index(batch, slot, input, slots, dimension)] += anchor[batch * state_dimension + input];
       }
     }
     }
   }
  }
  {
    OMEGA_PROFILE_SCOPE(ProfileDirection::kBackward, ProfileStage::kDepthEmbeddingCarry);
    for (size_t depth_index = 0; depth_index < depth_output_count; ++depth_index) {
      if (grads->depth_max_level != nullptr) {
        const std::uint64_t mask = depth_gradient_masks[depth_index];
        for (size_t level = depth_levels; level-- > 0;) {
          if ((mask & (std::uint64_t{1} << level)) != 0) {
            grads->depth_max_level[depth_index] = level;
            break;
          }
        }
      }
      if (!carry_flush(depth_gradient_partials, depth_gradient_masks, depth_index, depth_levels, &grads->d_depth_embedding[depth_index])) {
        return 15;
      }
      if (grads->sum_abs_d_depth_embedding != nullptr &&
          !carry_flush(depth_abs_partials, depth_abs_masks, depth_index, depth_levels, &grads->sum_abs_d_depth_embedding[depth_index])) {
        return 15;
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

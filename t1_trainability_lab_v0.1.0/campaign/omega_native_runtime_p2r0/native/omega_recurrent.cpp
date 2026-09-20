#include "omega_recurrent.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr size_t kAlignment = 64;
constexpr float kEpsilon = 1.0e-6F;
constexpr float kSqrtTwo = 1.4142135623730950488F;

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

size_t element_count(const OmegaRecurrentConfig& config) {
  size_t result = 0;
  size_t value = 0;
  if (!checked_mul(config.batch, config.slots, &value) || !checked_mul(value, config.dimension, &result)) {
    return 0;
  }
  return result;
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
};

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

bool valid_params(const OmegaRecurrentParams& params) {
  return params.state_part_weight != nullptr && params.prelude_norm_weight != nullptr && params.block_qkv_weight != nullptr &&
      params.block_qkv_bias != nullptr && params.block_out_weight != nullptr && params.block_out_bias != nullptr &&
      params.block_fc1_weight != nullptr && params.block_fc1_bias != nullptr && params.block_fc2_weight != nullptr &&
      params.block_fc2_bias != nullptr && params.block_norm_weight != nullptr && params.depth_embedding != nullptr &&
      params.gate_logits != nullptr;
}

}  // namespace

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
  if (config == nullptr || params == nullptr || token_part == nullptr || previous_state == nullptr || next_state == nullptr || readout_states == nullptr || workspace == nullptr || !valid_params(*params)) {
    return 1;
  }
  const size_t required = workspace_bytes_impl(*config);
  if (required == 0 || workspace_bytes < required) {
    return 2;
  }

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

  std::memcpy(state, previous_state, state_count * sizeof(float));
  const size_t dimension = config->dimension;
  const size_t slots = config->slots;
  const size_t state_dimension = slots * dimension;

  for (size_t round = 0; round < config->rounds; ++round) {
    for (size_t row = 0; row < 4 * dimension; ++row) {
      float sum = params->block_fc1_bias[row];
      for (size_t column = 0; column < dimension; ++column) {
        sum += params->block_fc1_weight[row * dimension + column] * params->depth_embedding[round * dimension + column];
      }
      depth_bias[round * 4 * dimension + row] = sum;
    }
  }

  for (size_t position = 0; position < config->sequence_length; ++position) {
    for (size_t batch = 0; batch < config->batch; ++batch) {
      for (size_t flattened = 0; flattened < state_dimension; ++flattened) {
        float write = token_part[(batch * config->sequence_length + position) * state_dimension + flattened];
        for (size_t input = 0; input < dimension; ++input) {
          float mean = 0.0F;
          for (size_t slot = 0; slot < slots; ++slot) {
            mean += state[state_index(batch, slot, input, slots, dimension)];
          }
          mean /= static_cast<float>(slots);
          write += params->state_part_weight[flattened * dimension + input] * mean;
        }
        anchor[batch * state_dimension + flattened] = write;
      }

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

    std::memcpy(candidate, anchor, state_count * sizeof(float));
    for (size_t round = 0; round < config->rounds; ++round) {
      for (size_t batch = 0; batch < config->batch; ++batch) {
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

        for (size_t slot = 0; slot < slots; ++slot) {
          for (size_t d = 0; d < dimension; ++d) {
            float output = params->block_out_bias[d];
            for (size_t input_dimension = 0; input_dimension < dimension; ++input_dimension) {
              output += params->block_out_weight[d * dimension + input_dimension] * mixed[state_index(batch, slot, input_dimension, slots, dimension)];
            }
            update[state_index(batch, slot, d, slots, dimension)] = output;
          }
          for (size_t row = 0; row < 4 * dimension; ++row) {
            float hidden = depth_bias[round * 4 * dimension + row];
            for (size_t input_dimension = 0; input_dimension < dimension; ++input_dimension) {
              hidden += params->block_fc1_weight[row * dimension + input_dimension] * update[state_index(batch, slot, input_dimension, slots, dimension)];
            }
            fc1[(batch * slots + slot) * 4 * dimension + row] = gelu(hidden);
          }
          float sum_squares_update = 0.0F;
          for (size_t d = 0; d < dimension; ++d) {
            float transformed = params->block_fc2_bias[d];
            for (size_t row = 0; row < 4 * dimension; ++row) {
              transformed += params->block_fc2_weight[d * 4 * dimension + row] * fc1[(batch * slots + slot) * 4 * dimension + row];
            }
            const float gate = sigmoid(params->gate_logits[round * dimension + d]);
            const float value_at = candidate[state_index(batch, slot, d, slots, dimension)] + gate * transformed;
            update[state_index(batch, slot, d, slots, dimension)] = value_at;
            sum_squares_update += value_at * value_at;
          }
          const float inverse_rms_update = 1.0F / std::sqrt(sum_squares_update / static_cast<float>(dimension) + kEpsilon);
          for (size_t d = 0; d < dimension; ++d) {
            candidate[state_index(batch, slot, d, slots, dimension)] = update[state_index(batch, slot, d, slots, dimension)] * inverse_rms_update * params->block_norm_weight[d];
          }
        }
      }
    }

    for (size_t batch = 0; batch < config->batch; ++batch) {
      std::memcpy(state + batch * state_dimension, candidate + batch * state_dimension, state_dimension * sizeof(float));
      std::memcpy(readout_states + (batch * config->sequence_length + position) * state_dimension, candidate + batch * state_dimension, state_dimension * sizeof(float));
    }
  }
  std::memcpy(next_state, state, state_count * sizeof(float));
  return 0;
}

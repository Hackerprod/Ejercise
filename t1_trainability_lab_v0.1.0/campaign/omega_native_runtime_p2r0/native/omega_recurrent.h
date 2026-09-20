#ifndef OMEGA_RECURRENT_H
#define OMEGA_RECURRENT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OmegaRecurrentConfig {
  size_t sequence_length;
  size_t batch;
  size_t slots;
  size_t dimension;
  size_t rounds;
} OmegaRecurrentConfig;

typedef struct OmegaRecurrentParams {
  const float* state_part_weight;   /* [slots * dimension, dimension] */
  const float* prelude_norm_weight; /* [slots * dimension] */
  const float* block_qkv_weight;    /* [3 * dimension, dimension] */
  const float* block_qkv_bias;      /* [3 * dimension] */
  const float* block_out_weight;    /* [dimension, dimension] */
  const float* block_out_bias;      /* [dimension] */
  const float* block_fc1_weight;    /* [4 * dimension, dimension] */
  const float* block_fc1_bias;      /* [4 * dimension] */
  const float* block_fc2_weight;    /* [dimension, 4 * dimension] */
  const float* block_fc2_bias;      /* [dimension] */
  const float* block_norm_weight;   /* [dimension] */
  const float* depth_embedding;     /* [rounds, dimension] */
  const float* gate_logits;         /* [rounds, dimension] */
} OmegaRecurrentParams;

/* Returns zero for invalid dimensions or size overflow. */
size_t omega_recurrent_workspace_bytes(OmegaRecurrentConfig config);

/* Returns 0 on success; nonzero on invalid arguments or insufficient workspace. */
int omega_recurrent_forward(
    const OmegaRecurrentConfig* config,
    const OmegaRecurrentParams* params,
    const float* token_part,
    const float* previous_state,
    float* next_state,
    float* readout_states,
    void* workspace,
    size_t workspace_bytes);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OMEGA_RECURRENT_H */

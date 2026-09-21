#ifndef OMEGA_RECURRENT_H
#define OMEGA_RECURRENT_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(OMEGA_RECURRENT_BUILD_SHARED)
#define OMEGA_RECURRENT_API __declspec(dllexport)
#else
#define OMEGA_RECURRENT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct OmegaRecurrentConfig {
  size_t sequence_length;
  size_t batch;
  size_t slots;
  size_t dimension;
  size_t rounds;
  int training; /* 0: inference workspace; 1: full-BPTT checkpoint workspace */
#ifdef __cplusplus
  int instrumentation = 1; /* 1: certification buffers; 0: clean production workspace */
#else
  int instrumentation; /* 1: certification buffers; 0: clean production workspace */
#endif
} OmegaRecurrentConfig;

typedef struct OmegaMatrixViewF32 {
  const float* data;
  size_t rows;
  size_t cols;
  ptrdiff_t row_stride;
} OmegaMatrixViewF32;

typedef struct OmegaRecurrentParams {
  OmegaMatrixViewF32 state_part_weight; /* [slots * dimension, dimension], column stride 1 */
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

typedef struct OmegaRecurrentGrads {
  float* d_token_part;
  float* d_previous_state;
  float* d_state_part_weight;
  float* d_prelude_norm_weight;
  float* d_block_qkv_weight;
  float* d_block_qkv_bias;
  float* d_block_out_weight;
  float* d_block_out_bias;
  float* d_block_fc1_weight;
  float* d_block_fc1_bias;
  float* d_block_fc2_weight;
  float* d_block_fc2_bias;
  float* d_block_norm_weight;
  float* d_depth_embedding;
  float* d_gate_logits;
  /* Optional per-element sums of absolute accumulation contributions. */
  float* sum_abs_d_token_part;
  float* sum_abs_d_previous_state;
  float* sum_abs_d_state_part_weight;
  float* sum_abs_d_prelude_norm_weight;
  float* sum_abs_d_block_qkv_weight;
  float* sum_abs_d_block_qkv_bias;
  float* sum_abs_d_block_out_weight;
  float* sum_abs_d_block_out_bias;
  float* sum_abs_d_block_fc1_weight;
  float* sum_abs_d_block_fc1_bias;
  float* sum_abs_d_block_fc2_weight;
  float* sum_abs_d_block_fc2_bias;
  float* sum_abs_d_block_norm_weight;
  float* sum_abs_d_depth_embedding;
  float* sum_abs_d_gate_logits;
  /* Optional exact counts of accumulated contributions per element. */
  size_t* count_d_token_part;
  size_t* count_d_previous_state;
  size_t* count_d_state_part_weight;
  size_t* count_d_prelude_norm_weight;
  size_t* count_d_block_qkv_weight;
  size_t* count_d_block_qkv_bias;
  size_t* count_d_block_out_weight;
  size_t* count_d_block_out_bias;
  size_t* count_d_block_fc1_weight;
  size_t* count_d_block_fc1_bias;
  size_t* count_d_block_fc2_weight;
  size_t* count_d_block_fc2_bias;
  size_t* count_d_block_norm_weight;
  size_t* count_d_depth_embedding;
  size_t* count_d_gate_logits;
  /* Optional diagnostic-only FP64 flat accumulation for depth embedding. */
  double* fp64_d_depth_embedding;
  /* Optional diagnostic-only maximum occupied carry level per depth element. */
  size_t* depth_max_level;
} OmegaRecurrentGrads;

/* Temporary high-level regression profile. Present only in profile builds. */
#ifdef OMEGA_PROFILE_INTERNAL
typedef struct OmegaRecurrentProfileDirection {
  uint64_t calls;
  uint64_t qkv_projection_calls;
  double qkv_projection_seconds;
  uint64_t attention_scores_softmax_mixing_calls;
  double attention_scores_softmax_mixing_seconds;
  uint64_t out_projection_calls;
  double out_projection_seconds;
  uint64_t fc1_gelu_calls;
  double fc1_gelu_seconds;
  uint64_t fc2_calls;
  double fc2_seconds;
  uint64_t rmsnorm_gates_calls;
  double rmsnorm_gates_seconds;
  uint64_t state_prelude_calls;
  double state_prelude_seconds;
  uint64_t depth_embedding_pairwise_reduction_calls;
  double depth_embedding_pairwise_reduction_seconds;
  uint64_t depth_embedding_carry_calls;
  double depth_embedding_carry_seconds;
  uint64_t history_buffer_reads_writes_calls;
  double history_buffer_reads_writes_seconds;
} OmegaRecurrentProfileDirection;

typedef struct OmegaRecurrentProfileSnapshot {
  int enabled;
  int compiled;
  OmegaRecurrentProfileDirection forward;
  OmegaRecurrentProfileDirection backward;
} OmegaRecurrentProfileSnapshot;
#endif

/* Returns zero for invalid dimensions or size overflow. */
OMEGA_RECURRENT_API size_t omega_recurrent_workspace_bytes(OmegaRecurrentConfig config);

/* Returns 0 on success; nonzero on invalid arguments or insufficient workspace. */
OMEGA_RECURRENT_API int omega_recurrent_forward(
    const OmegaRecurrentConfig* config,
    const OmegaRecurrentParams* params,
    const float* token_part,
    const float* previous_state,
    float* next_state,
    float* readout_states,
    void* workspace,
    size_t workspace_bytes);

/* Computes full-BPTT gradients from a prior forward with config.training=1. */
OMEGA_RECURRENT_API int omega_recurrent_backward(
    const OmegaRecurrentConfig* config,
    const OmegaRecurrentParams* params,
    const float* token_part,
    const float* d_readout_states,
    const float* d_next_state,
    void* workspace,
    size_t workspace_bytes,
    OmegaRecurrentGrads* grads);

/* Temporary profiling ABI. Exported only from OMEGA_PROFILE_INTERNAL builds. */
#ifdef OMEGA_PROFILE_INTERNAL
OMEGA_RECURRENT_API void omega_recurrent_profile_set_enabled(int enabled);
OMEGA_RECURRENT_API void omega_recurrent_profile_reset(void);
OMEGA_RECURRENT_API int omega_recurrent_profile_snapshot(OmegaRecurrentProfileSnapshot* snapshot);
#endif

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OMEGA_RECURRENT_H */

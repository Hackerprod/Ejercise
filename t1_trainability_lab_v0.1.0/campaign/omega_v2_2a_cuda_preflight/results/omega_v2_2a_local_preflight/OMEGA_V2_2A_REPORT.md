# OMEGA-V2-2A Local CUDA Preflight

- terminal_status: `OMEGA_V2_2A_LOCAL_PREFLIGHT_FAIL`
- source_seal_sha256: `c57e534c184702cb5d1f9bc476b0c7af586b810778e6539736da8eed25c1c490`
- wall_seconds: `6.970898`
- residency_verdict: `NOT_ESTABLISHED`

## Ordered gate results

| Gate | Status | Pass |
|---|---|---:|
| D4_parameter_storage_cpu | PASS | True |
| fixed_tensor_copy | PASS | True |
| cpu_to_cuda_weight_copy | PASS | True |
| D4_parameter_storage_cuda | PASS | True |
| D1_R4_K1 | PASS | True |
| D1_R4_K4 | PASS | True |
| D2_init_trace_parity | PASS | True |
| D3_gradient_sharing | FAIL | False |
| D5_iso_flop | PASS | True |
| D6_k_flex | PASS | True |
| D7_R4_optimizer_smoke | PASS | True |
| D7_U4_optimizer_smoke | PASS | True |
| D8_vram_wall_contract | PASS | True |

## Gate reports

```json
{
  "R4_smoke": {
    "L0": 7.3987135887146,
    "L0_finite": true,
    "L20": 1.3690094947814941,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "gradient_clipping": false,
    "loss_decrease_pass": true,
    "lr": 0.0003,
    "optimizer": "AdamW",
    "pass": true,
    "scheduler": null,
    "schema": "omega-v2-2a-optimizer-smoke-v1",
    "steps": [
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 1
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 2
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 3
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 4
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 5
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 6
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 7
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 8
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 9
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 10
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 11
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 12
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 13
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 14
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 15
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 16
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 17
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 18
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 19
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 20
      }
    ],
    "updates": 20,
    "variant": "R4",
    "weight_decay": 0
  },
  "U4_smoke": {
    "L0": 7.3987135887146,
    "L0_finite": true,
    "L20": 1.017160415649414,
    "betas": [
      0.9,
      0.999
    ],
    "eps": 1e-08,
    "gradient_clipping": false,
    "loss_decrease_pass": true,
    "lr": 0.0003,
    "optimizer": "AdamW",
    "pass": true,
    "scheduler": null,
    "schema": "omega-v2-2a-optimizer-smoke-v1",
    "steps": [
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 1
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 2
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 3
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 4
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 5
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 6
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 7
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 8
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 9
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 10
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 11
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 12
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 13
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 14
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 15
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 16
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 17
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 18
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 19
      },
      {
        "adam_exp_avg_exp_avg_sq_finite": true,
        "gradients_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameters_finite": true,
        "pass": true,
        "step": 20
      }
    ],
    "updates": 20,
    "variant": "U4",
    "weight_decay": 0
  },
  "cpu_to_cuda_weight_copy": {
    "all_bitwise_equal": true,
    "families": [
      {
        "R4_cpu_to_cuda_bitwise_equal": true,
        "U4_cpu_to_cuda_bitwise_equal": true,
        "family": "W_Q"
      },
      {
        "R4_cpu_to_cuda_bitwise_equal": true,
        "U4_cpu_to_cuda_bitwise_equal": true,
        "family": "W_K"
      },
      {
        "R4_cpu_to_cuda_bitwise_equal": true,
        "U4_cpu_to_cuda_bitwise_equal": true,
        "family": "W_V"
      },
      {
        "R4_cpu_to_cuda_bitwise_equal": true,
        "U4_cpu_to_cuda_bitwise_equal": true,
        "family": "W_O"
      },
      {
        "R4_cpu_to_cuda_bitwise_equal": true,
        "U4_cpu_to_cuda_bitwise_equal": true,
        "family": "W_gate"
      },
      {
        "R4_cpu_to_cuda_bitwise_equal": true,
        "U4_cpu_to_cuda_bitwise_equal": true,
        "family": "W_up"
      },
      {
        "R4_cpu_to_cuda_bitwise_equal": true,
        "U4_cpu_to_cuda_bitwise_equal": true,
        "family": "W_down"
      }
    ],
    "pass": true,
    "status": "PASS"
  },
  "cuda_correctness": {
    "cells": [
      {
        "K": 1,
        "L2_pass": true,
        "L2_relative_error": 3.049385099984647e-07,
        "elementwise_pass": true,
        "item": "final_output",
        "max_abs": 2.1457672119140625e-06,
        "max_scaled_error": 0.09781026840209961,
        "pass": true
      },
      {
        "K": 4,
        "L2_pass": true,
        "L2_relative_error": 3.049385099984647e-07,
        "elementwise_pass": true,
        "item": "round_trace_1",
        "max_abs": 2.1457672119140625e-06,
        "max_scaled_error": 0.09781026840209961,
        "pass": true
      },
      {
        "K": 4,
        "L2_pass": true,
        "L2_relative_error": 3.90067953048856e-07,
        "elementwise_pass": true,
        "item": "round_trace_2",
        "max_abs": 2.7418136596679688e-06,
        "max_scaled_error": 0.1466994285583496,
        "pass": true
      },
      {
        "K": 4,
        "L2_pass": true,
        "L2_relative_error": 4.1137752759823343e-07,
        "elementwise_pass": true,
        "item": "round_trace_3",
        "max_abs": 3.5762786865234375e-06,
        "max_scaled_error": 0.1882624328136444,
        "pass": true
      },
      {
        "K": 4,
        "L2_pass": true,
        "L2_relative_error": 4.13041647107093e-07,
        "elementwise_pass": true,
        "item": "round_trace_4",
        "max_abs": 4.410743713378906e-06,
        "max_scaled_error": 0.26584193110466003,
        "pass": true
      },
      {
        "K": 4,
        "L2_pass": true,
        "L2_relative_error": 4.13041647107093e-07,
        "elementwise_pass": true,
        "item": "final_output",
        "max_abs": 4.410743713378906e-06,
        "max_scaled_error": 0.26584193110466003,
        "pass": true
      }
    ],
    "pass": true,
    "schema": "omega-v2-2a-cuda-correctness-gate-v1"
  },
  "environment_config": {
    "pass": true,
    "source_seal_sha256": "c57e534c184702cb5d1f9bc476b0c7af586b810778e6539736da8eed25c1c490",
    "status": "PASS"
  },
  "fixed_tensor_copy": {
    "input": true,
    "loss_weights": true,
    "pass": true,
    "status": "PASS",
    "target": true
  },
  "gradient_sharing": {
    "families": [
      {
        "L2_relative_error": 5.372153566440829e-08,
        "family": "W_Q",
        "finite": true,
        "max_abs": 7.62939453125e-06,
        "max_rel": 0.00036101083969697356,
        "pass": false
      },
      {
        "L2_relative_error": 5.370466382714767e-08,
        "family": "W_K",
        "finite": true,
        "max_abs": 7.62939453125e-06,
        "max_rel": 0.0034722222480922937,
        "pass": false
      },
      {
        "L2_relative_error": 5.404850611512302e-08,
        "family": "W_V",
        "finite": true,
        "max_abs": 1.52587890625e-05,
        "max_rel": 0.00023239600704982877,
        "pass": false
      },
      {
        "L2_relative_error": 5.3697743140901366e-08,
        "family": "W_O",
        "finite": true,
        "max_abs": 1.52587890625e-05,
        "max_rel": 0.00012130034156143665,
        "pass": true
      },
      {
        "L2_relative_error": 5.3564853885745833e-08,
        "family": "W_gate",
        "finite": true,
        "max_abs": 7.62939453125e-06,
        "max_rel": 0.005988024175167084,
        "pass": false
      },
      {
        "L2_relative_error": 5.346880982415314e-08,
        "family": "W_up",
        "finite": true,
        "max_abs": 7.62939453125e-06,
        "max_rel": 0.00042698546894825995,
        "pass": false
      },
      {
        "L2_relative_error": 5.3162100499548615e-08,
        "family": "W_down",
        "finite": true,
        "max_abs": 7.62939453125e-06,
        "max_rel": 0.002994012087583542,
        "pass": false
      }
    ],
    "loss": "sum(y * loss_weights)",
    "pass": false,
    "schema": "omega-v2-2a-gradient-sharing-v1"
  },
  "init_trace_parity": {
    "bitwise_pass": true,
    "cpu_to_cuda_weights_bitwise_equal": true,
    "fallback_used": false,
    "final_output_torch_equal": true,
    "initial_values_bitwise_equal": true,
    "pass": true,
    "round_trace_torch_equal": [
      true,
      true,
      true,
      true
    ],
    "schema": "omega-v2-2a-r4-u4-init-parity-v1"
  },
  "iso_flop": {
    "R4_non_gemm_counts_total": {
      "residual_vector_additions": 131072,
      "rmsnorm_applications": 8,
      "rmsnorm_elements": 131072,
      "silu_activations": 262144,
      "softmax_elements": 2048,
      "softmax_rows": 256,
      "swiglu_hadamard_multiplies": 262144
    },
    "U4_non_gemm_counts_total": {
      "residual_vector_additions": 131072,
      "rmsnorm_applications": 8,
      "rmsnorm_elements": 131072,
      "silu_activations": 262144,
      "softmax_elements": 2048,
      "softmax_rows": 256,
      "swiglu_hadamard_multiplies": 262144
    },
    "exact_integer_flop_equality": true,
    "k4_b1_flops": 67371008,
    "k4_b8_R4_forward_flops": 538968064,
    "k4_b8_U4_forward_flops": 538968064,
    "mac_convention": "1 MAC = 2 FLOPs",
    "non_gemm_counts_identical": true,
    "pass": true,
    "per_round_b1_flops": 16842752,
    "schema": "omega-v2-2a-flop-ledger-v1"
  },
  "k_flex": {
    "SCHEMA_SHA256_final": "31524d657d1797148e6af48066164bbc23bf9a4ba3b30570901ad7f19b6baa59",
    "VALUE_SHA256_final": "62a873bb27850523757a5f44aa008eb5a863d605aee52cfd34c94e466713baf2",
    "parameter_count_final": 1048576,
    "pass": true,
    "rows": [
      {
        "K": 1,
        "output_finite": true,
        "parameter_count_unchanged": true,
        "pass": true,
        "phase": "forward",
        "schema_unchanged": true,
        "value_unchanged": true
      },
      {
        "K": 2,
        "output_finite": true,
        "parameter_count_unchanged": true,
        "pass": true,
        "phase": "forward",
        "schema_unchanged": true,
        "value_unchanged": true
      },
      {
        "K": 4,
        "output_finite": true,
        "parameter_count_unchanged": true,
        "pass": true,
        "phase": "forward",
        "schema_unchanged": true,
        "value_unchanged": true
      },
      {
        "K": 8,
        "output_finite": true,
        "parameter_count_unchanged": true,
        "pass": true,
        "phase": "forward",
        "schema_unchanged": true,
        "value_unchanged": true
      },
      {
        "K": 16,
        "output_finite": true,
        "parameter_count_unchanged": true,
        "pass": true,
        "phase": "forward",
        "schema_unchanged": true,
        "value_unchanged": true
      },
      {
        "K": 1,
        "finite": true,
        "input_gradient_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameter_count_unchanged": true,
        "parameter_gradients_finite": true,
        "pass": true,
        "phase": "backward",
        "schema_unchanged": true,
        "value_unchanged": true
      },
      {
        "K": 4,
        "finite": true,
        "input_gradient_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameter_count_unchanged": true,
        "parameter_gradients_finite": true,
        "pass": true,
        "phase": "backward",
        "schema_unchanged": true,
        "value_unchanged": true
      },
      {
        "K": 8,
        "finite": true,
        "input_gradient_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameter_count_unchanged": true,
        "parameter_gradients_finite": true,
        "pass": true,
        "phase": "backward",
        "schema_unchanged": true,
        "value_unchanged": true
      },
      {
        "K": 16,
        "finite": true,
        "input_gradient_finite": true,
        "loss_finite": true,
        "output_finite": true,
        "parameter_count_unchanged": true,
        "parameter_gradients_finite": true,
        "pass": true,
        "phase": "backward",
        "schema_unchanged": true,
        "value_unchanged": true
      }
    ],
    "schema": "omega-v2-2a-k-flex-v1"
  },
  "parameter_storage": {
    "R4_one_storage_per_family": true,
    "R4_unique_parameters": 1048576,
    "U4_four_storages_per_family": true,
    "U4_storage_disjoint_from_R4": true,
    "U4_storages_pairwise_disjoint": true,
    "U4_unique_parameters": 4194304,
    "pass": true,
    "schema": "omega-v2-2a-parameter-storage-v1"
  },
  "parameter_storage_cuda": {
    "R4_one_storage_per_family": true,
    "R4_unique_parameters": 1048576,
    "U4_four_storages_per_family": true,
    "U4_storage_disjoint_from_R4": true,
    "U4_storages_pairwise_disjoint": true,
    "U4_unique_parameters": 4194304,
    "pass": true,
    "schema": "omega-v2-2a-parameter-storage-v1"
  },
  "vram_runtime": {
    "allocated_budget_bytes": 3221225472,
    "capacity_diagnostic_count": 0,
    "cells": [
      {
        "cell": "D1_R4_K1",
        "peak_memory_allocated_bytes": 56433152,
        "peak_memory_reserved_bytes": 56623104,
        "status": "OK",
        "wall_seconds": 2.4258179999887943,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D1_R4_K4",
        "peak_memory_allocated_bytes": 61357568,
        "peak_memory_reserved_bytes": 62914560,
        "status": "OK",
        "wall_seconds": 0.012242100026924163,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D2_R4_U4_K4_PARITY",
        "peak_memory_allocated_bytes": 57022464,
        "peak_memory_reserved_bytes": 58720256,
        "status": "OK",
        "wall_seconds": 0.2522962000221014,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D3_GRADIENT_IDENTITY_K4",
        "peak_memory_allocated_bytes": 122498048,
        "peak_memory_reserved_bytes": 142606336,
        "status": "OK",
        "wall_seconds": 1.4457036000094377,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D6_KFLEX_FWD_K1",
        "peak_memory_allocated_bytes": 89987072,
        "peak_memory_reserved_bytes": 90177536,
        "status": "OK",
        "wall_seconds": 0.008573800034355372,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D6_KFLEX_FWD_K2",
        "peak_memory_allocated_bytes": 90052608,
        "peak_memory_reserved_bytes": 90177536,
        "status": "OK",
        "wall_seconds": 0.008440900011919439,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D6_KFLEX_FWD_K4",
        "peak_memory_allocated_bytes": 90052608,
        "peak_memory_reserved_bytes": 90177536,
        "status": "OK",
        "wall_seconds": 0.009944899997208267,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D6_KFLEX_FWD_K8",
        "peak_memory_allocated_bytes": 90052608,
        "peak_memory_reserved_bytes": 90177536,
        "status": "OK",
        "wall_seconds": 0.013062899990472943,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D6_KFLEX_FWD_K16",
        "peak_memory_allocated_bytes": 90052608,
        "peak_memory_reserved_bytes": 90177536,
        "status": "OK",
        "wall_seconds": 0.01362779998453334,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D6_KFLEX_BWD_K1",
        "peak_memory_allocated_bytes": 94503424,
        "peak_memory_reserved_bytes": 96468992,
        "status": "OK",
        "wall_seconds": 0.0685494999634102,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D6_KFLEX_BWD_K4",
        "peak_memory_allocated_bytes": 99428352,
        "peak_memory_reserved_bytes": 100663296,
        "status": "OK",
        "wall_seconds": 0.015093700028955936,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D6_KFLEX_BWD_K8",
        "peak_memory_allocated_bytes": 105732096,
        "peak_memory_reserved_bytes": 106954752,
        "status": "OK",
        "wall_seconds": 0.025872200028970838,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D6_KFLEX_BWD_K16",
        "peak_memory_allocated_bytes": 118339584,
        "peak_memory_reserved_bytes": 119537664,
        "status": "OK",
        "wall_seconds": 0.032653899979777634,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D7_R4_SMOKE_20_UPDATES",
        "peak_memory_allocated_bytes": 112011264,
        "peak_memory_reserved_bytes": 113246208,
        "status": "OK",
        "wall_seconds": 1.03739690000657,
        "within_3_gib_allocated_budget": true
      },
      {
        "cell": "D7_U4_SMOKE_20_UPDATES",
        "peak_memory_allocated_bytes": 180683776,
        "peak_memory_reserved_bytes": 182452224,
        "status": "OK",
        "wall_seconds": 0.335993500018958,
        "within_3_gib_allocated_budget": true
      }
    ],
    "status": "PASS",
    "total_wall_limit_seconds": 1800,
    "total_wall_seconds": 6.9708981000003405,
    "within_wall_limit": true
  }
}
```

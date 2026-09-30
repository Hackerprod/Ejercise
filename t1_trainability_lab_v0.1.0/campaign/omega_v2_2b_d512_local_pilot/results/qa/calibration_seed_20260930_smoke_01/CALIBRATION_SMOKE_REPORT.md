# OMEGA-V2-2B calibration-seed harness smoke

- classification: `CALIBRATION_QA_ONLY`
- V2_2B_verdict: `null`
- terminal_status: `V2_2B_CALIBRATION_QA_HOLD`
- master_seed: `20260930`
- structural gates: `False`
- D3 A/B/C gates: `False`
- official held-out seeds used: `False`
- scientific verdict: `NONE`

```json
{
  "D1": {
    "cells": [
      {
        "D8": {
          "cell_id": "D1_20260930_K1",
          "gate": "D1",
          "master_seed": 20260930,
          "peak_allocated": 53876736,
          "peak_reserved": 62914560,
          "status": "PASS",
          "wall_seconds": 0.2069609999889508
        },
        "K": 1,
        "cell_id": "D1_20260930_K1",
        "copy_checks": {
          "input_bitwise_equal": true,
          "weights_bitwise_equal": true
        },
        "gate": "D1",
        "outputs": [
          {
            "dtype_cpu": "torch.float32",
            "dtype_cuda": "torch.float32",
            "elementwise_gate": {
              "abs_tolerance": 1e-05,
              "cpu_floor": 1e-06,
              "max_abs": 1.817941665649414e-06,
              "max_scaled_error": 0.12370575796108405,
              "pass": true,
              "rel_tolerance": 0.0001
            },
            "finite": true,
            "normwise_gate": {
              "E_L2": 2.890960680760423e-07,
              "denominator_floor": 1e-06,
              "limit": 1e-05,
              "pass": true
            },
            "output": "K1_final",
            "shape": [
              8,
              8,
              512
            ]
          }
        ],
        "pass": true,
        "raw_tensor_sha256": {
          "cpu_final": "41a64d3a69a4aa9e0dd7232a1e20c60f78879ceb1d6f6bc7c3ccbe1a2076d95d",
          "cuda_final": "f6ceb65038b48b3e7de2de35e9bc32a53864457a32acafc01f061cd0368ee800"
        },
        "seed_plan": {
          "input_seed": 20261938,
          "loss_w_seed": 20262938,
          "master_seed": 20260930,
          "mode": "smoke",
          "target_seed": 20263938,
          "weight_seed": 20260930
        },
        "status": "PASS"
      },
      {
        "D8": {
          "cell_id": "D1_20260930_K4",
          "gate": "D1",
          "master_seed": 20260930,
          "peak_allocated": 54401024,
          "peak_reserved": 62914560,
          "status": "PASS",
          "wall_seconds": 0.05216899997321889
        },
        "K": 4,
        "cell_id": "D1_20260930_K4",
        "copy_checks": {
          "input_bitwise_equal": true,
          "weights_bitwise_equal": true
        },
        "gate": "D1",
        "outputs": [
          {
            "dtype_cpu": "torch.float32",
            "dtype_cuda": "torch.float32",
            "elementwise_gate": {
              "abs_tolerance": 1e-05,
              "cpu_floor": 1e-06,
              "max_abs": 1.817941665649414e-06,
              "max_scaled_error": 0.12370575796108405,
              "pass": true,
              "rel_tolerance": 0.0001
            },
            "finite": true,
            "normwise_gate": {
              "E_L2": 2.890960680760423e-07,
              "denominator_floor": 1e-06,
              "limit": 1e-05,
              "pass": true
            },
            "output": "K4_round_1",
            "shape": [
              8,
              8,
              512
            ]
          },
          {
            "dtype_cpu": "torch.float32",
            "dtype_cuda": "torch.float32",
            "elementwise_gate": {
              "abs_tolerance": 1e-05,
              "cpu_floor": 1e-06,
              "max_abs": 2.6226043701171875e-06,
              "max_scaled_error": 0.16806494280882997,
              "pass": true,
              "rel_tolerance": 0.0001
            },
            "finite": true,
            "normwise_gate": {
              "E_L2": 3.812825636877123e-07,
              "denominator_floor": 1e-06,
              "limit": 1e-05,
              "pass": true
            },
            "output": "K4_round_2",
            "shape": [
              8,
              8,
              512
            ]
          },
          {
            "dtype_cpu": "torch.float32",
            "dtype_cuda": "torch.float32",
            "elementwise_gate": {
              "abs_tolerance": 1e-05,
              "cpu_floor": 1e-06,
              "max_abs": 3.635883331298828e-06,
              "max_scaled_error": 0.24872527055215665,
              "pass": true,
              "rel_tolerance": 0.0001
            },
            "finite": true,
            "normwise_gate": {
              "E_L2": 4.122292586140952e-07,
              "denominator_floor": 1e-06,
              "limit": 1e-05,
              "pass": true
            },
            "output": "K4_round_3",
            "shape": [
              8,
              8,
              512
            ]
          },
          {
            "dtype_cpu": "torch.float32",
            "dtype_cuda": "torch.float32",
            "elementwise_gate": {
              "abs_tolerance": 1e-05,
              "cpu_floor": 1e-06,
              "max_abs": 5.245208740234375e-06,
              "max_scaled_error": 0.26676360612280403,
              "pass": true,
              "rel_tolerance": 0.0001
            },
            "finite": true,
            "normwise_gate": {
              "E_L2": 4.229900935633447e-07,
              "denominator_floor": 1e-06,
              "limit": 1e-05,
              "pass": true
            },
            "output": "K4_round_4",
            "shape": [
              8,
              8,
              512
            ]
          },
          {
            "dtype_cpu": "torch.float32",
            "dtype_cuda": "torch.float32",
            "elementwise_gate": {
              "abs_tolerance": 1e-05,
              "cpu_floor": 1e-06,
              "max_abs": 5.245208740234375e-06,
              "max_scaled_error": 0.26676360612280403,
              "pass": true,
              "rel_tolerance": 0.0001
            },
            "finite": true,
            "normwise_gate": {
              "E_L2": 4.229900935633447e-07,
              "denominator_floor": 1e-06,
              "limit": 1e-05,
              "pass": true
            },
            "output": "K4_final",
            "shape": [
              8,
              8,
              512
            ]
          }
        ],
        "pass": true,
        "raw_tensor_sha256": {
          "cpu_final": "2d3b167bda821dd7276446e2a7281512037a6976a6cbee7a0568784ca409be10",
          "cpu_trace_1": "1408d23cebe5fa27c4282d406639cce35b06eaefcd67cb89891659b24fc8a96f",
          "cpu_trace_2": "e3c3a5caf5b8a056f509bf208b038892ce959be0d29ae3aa8bf8ceba51f0a112",
          "cpu_trace_3": "bdf877c8b2399a9c4a45eb0ca052594a858dcd0e267cde4c0ceb282a3744b6e4",
          "cpu_trace_4": "0b6cbfd388aa94b4446e64ea0629bed7a8444a8b77060705c16f45a1d8e31cb3",
          "cuda_final": "9fccfccbfe7873180e33a3ecf16e5096e5f6a9d0fb2e0f0fc94d1acdf1c65b3e",
          "cuda_trace_1": "73de805777866e0744a048fa4356573e21fe6a2cca0edda13c70f4d6e9c6a49b",
          "cuda_trace_2": "0c30b3e1db27a251c7a895aeab73e9f69e2c188805acb6edada69c06e8f9956b",
          "cuda_trace_3": "369b5a6dcbdd921a1fd9c5a559f05b32e940100c55cca1e4e156fecab364b4bf",
          "cuda_trace_4": "a24718639f9b1c7cc0a458efce6b71e859ef238ce391b8d36e0a1fd439083a68"
        },
        "seed_plan": {
          "input_seed": 20261938,
          "loss_w_seed": 20262938,
          "master_seed": 20260930,
          "mode": "smoke",
          "target_seed": 20263938,
          "weight_seed": 20260930
        },
        "status": "PASS"
      }
    ],
    "pass": true
  },
  "D2": {
    "D8": {
      "cell_id": "D2_20260930",
      "gate": "D2",
      "master_seed": 20260930,
      "peak_allocated": 122165248,
      "peak_reserved": 123731968,
      "status": "PASS",
      "wall_seconds": 0.1713768999907188
    },
    "clone_hashes": {
      "W_K": {
        "R4": "1670798a038d1bd88696b5b1cb76c8372ca467aba2aad8ba35570c5d89dd2fad",
        "U4_0": "04bbaabb1d6d96c42ac5e5ad061107024a099a9d401ebe26543c2907c2cc7739",
        "U4_1": "36e33b38aff72565c3809912aeaa357c17b5a784da06827180c32a595226a978",
        "U4_2": "b50a049f57ffa0eed16121c2519bab5bcfa0f992daf5ea0d5b527a9360377d04",
        "U4_3": "b0521d134b44aec13de4c2fe5100ba2f66f72af3d0127796973db4dd2c2c3cb8"
      },
      "W_O": {
        "R4": "dc46023885cde020dec11c4ef3ce6cebb90fac3655e0436fc59d14145630a549",
        "U4_0": "e75106464c77c5aa54a8bbb40b0dd5f5e0937d9fe5017d6a74aeae977d56a38a",
        "U4_1": "b0a269e9314f37e588280b6d9879cac5972125e1bd58c9149597bc54b917a5e8",
        "U4_2": "3d7aeab06f26d65506dbce60a8b1bf5c28de863456482eaf5a833c993b8afb53",
        "U4_3": "ab71dc72bccf1208ab8849beb24fec0182c24f5b4d39c0ae66d822161cb24aee"
      },
      "W_Q": {
        "R4": "800a0591114fc5b28e95fd747ccb0d5e0ec78e40661b87660b9fd4cb7305a766",
        "U4_0": "60323164ebbf9ac1b2c23c7a917f0968ea344f5be471d4af0a44667efd98f292",
        "U4_1": "4801b4e1f38b02aa7286eba694a86b917f18a0afbecb349ef7b3cdc3ac2f19b6",
        "U4_2": "6184dbb1120022ca791bdd019a26459ac2134413c15e8e27395ce17ce3ead374",
        "U4_3": "d6e506fac92dd47de0ccbaaa6aa7a10f0f9412698eacaa28b477fb3ba4764659"
      },
      "W_V": {
        "R4": "46d39156d411407ddc63a7c103c0d9b9803d6fc541533d7f0e3121b29e2da462",
        "U4_0": "b4ad9913f40c8d2c1008cdf1a2e1e57fe73b658f3771661cb52b05576546be60",
        "U4_1": "548c58e35e2fcd961a7ed679b3a445f1512982f58b2bd6b3018d30518b66a1b7",
        "U4_2": "59c86bdd4fe33f4629eb246a7c20934159f5b69786dace0b58e5ff825b2c4656",
        "U4_3": "19eea5761a506a6527a25b6647757bfe3a993fee45c71ea7403a18951e1b0e28"
      },
      "W_down": {
        "R4": "f43b7b659e101308ac7483abe2c9c5b41206e6e493a8f31f29a83e75662af485",
        "U4_0": "3925d711fed064ab4d57f305798ba93fce898dc6f052f5e2b5115d51f76fe43b",
        "U4_1": "d937e23b1cbbc94a5a0051af7f982a7e1be82c7f68de85b0da9aacf092fb1361",
        "U4_2": "8f8e9cd2324bef28ae050622da993a389935bdc1489b588082550f736b806294",
        "U4_3": "050d0552b81729b5f3a05fd763b3f26c080c40c166f827bc9592c4c1161db34a"
      },
      "W_gate": {
        "R4": "86ba2cd555eb1c18d913c2d6b65cb9979f1e6e203296936b149c13098c8c73d9",
        "U4_0": "1d34ffdc9a82892f07860bb93d6420733a7bb16954bb13893a03f369e185f225",
        "U4_1": "838de78458b92c80e25caff3b5afb27d9cbf2db329025081d3316a633bacb2c9",
        "U4_2": "d200aadbc91d14fdaef6246c67c8972fa3d8126b226135b49b7dcfc8544a9e01",
        "U4_3": "5ef9ba46318d6cc4e68dc1711ba3fffe96791d7d13973d9fca5c5a010f98b24c"
      },
      "W_up": {
        "R4": "4237c18450ea7331e990c3f3112c7323e10d647fbc625dfe418b3cf1a6f0deab",
        "U4_0": "c7fff5be0691455f1a7d7a5e91674673fe476af9b5ec2405f5ed85499afc8a82",
        "U4_1": "bf349b994c185493e8b4ead849ebf3a189c5d69a92b909ae292ef135aca26b2f",
        "U4_2": "23a57f36f0aa7cb43dcd961cab1de6a7380d6451c0bd71b26c405f1907d30d65",
        "U4_3": "36623edb535fe471e9e214b5bbc95c2fe1f01509efc7cb60a4a1eb485503ac7f"
      }
    },
    "copy_checks": {
      "R4_weights_bitwise_equal": true,
      "U4_weights_bitwise_equal": true,
      "x_bitwise_equal": true
    },
    "gate": "D2",
    "pass": true,
    "seed_plan": {
      "input_seed": 20261938,
      "loss_w_seed": 20262938,
      "master_seed": 20260930,
      "mode": "smoke",
      "target_seed": 20263938,
      "weight_seed": 20260930
    },
    "structural": {
      "final_equal": true,
      "initial_clone_report": {
        "initial_values_bitwise_equal": true,
        "r4_reuses_one_block_object": true,
        "u4_block_storages_pairwise_disjoint": true,
        "u4_has_four_block_objects": true,
        "u4_storage_disjoint_from_r4": true
      },
      "initial_weights_equal": true,
      "non_gate_fallback_used": false,
      "pass": true,
      "round_trace_equalities": [
        {
          "round": 1,
          "torch_equal": true
        },
        {
          "round": 2,
          "torch_equal": true
        },
        {
          "round": 3,
          "torch_equal": true
        },
        {
          "round": 4,
          "torch_equal": true
        }
      ]
    },
    "trace_raw_hashes": {
      "R4_final": "8fe80f44fa81a38cdefdcd674296d3eccecbd962477cc8f9063b588ad369f5ff",
      "R4_trace_1": "f5eb2900712200d53d8e7bfcf947fe8c885d5ad6561571399a1dbb1b5397195a",
      "R4_trace_2": "01b9d02dfbfb5d131d215031f9d6534f7de8eb923be31206d39dd870738c0dae",
      "R4_trace_3": "0543e78e7d4b3b2c845dbfc49912b2f65d7ccf210737d5d542730d200c25dc70",
      "R4_trace_4": "c1a65e4b44f1a63fda8cd22e99966301ff46f05c77a4aa9ed645a1f58088f705",
      "U4_final": "b384dbd134cb9a7e3fb081bbcffedb983a4d094d96d0d784d3467d002e0f3ab9",
      "U4_trace_1": "0631e4a0c495fdbde798141ca22be1a8019219ece1d543e63aa553e019a57e11",
      "U4_trace_2": "ac3ade785f1195d6a6f696ec9f595e34e08e59a58e1f164411a131453e7c8844",
      "U4_trace_3": "2ecf6e25eecddab4b1014efb45cf3b5b19719a788499fca752faabaa2b7c2e71",
      "U4_trace_4": "2ea75c29eb4287a3bab5219f34655d934da633053483291646bc7ea979847e0c"
    }
  },
  "D5": {
    "cuda_cell": false,
    "ledger": {
      "R4_U4_K4_B8_flops": {
        "R4": 2151677952,
        "U4": 2151677952
      },
      "R4_non_gemm_counts_K4_B8": {
        "residual_vector_additions": 262144,
        "rmsnorm_applications": 8,
        "rmsnorm_elements": 262144,
        "silu_activations": 524288,
        "softmax_elements": 2048,
        "softmax_rows": 256,
        "swiglu_hadamard_multiplies": 524288
      },
      "R4_unique_parameters": 4194304,
      "U4_non_gemm_counts_K4_B8": {
        "residual_vector_additions": 262144,
        "rmsnorm_applications": 8,
        "rmsnorm_elements": 262144,
        "silu_activations": 524288,
        "softmax_elements": 2048,
        "softmax_rows": 256,
        "swiglu_hadamard_multiplies": 524288
      },
      "U4_unique_parameters": 16777216,
      "actual_flops": {
        "K4_B1": 268959744,
        "K4_B128": 34426847232,
        "K4_B8": 2151677952,
        "per_round_B1": 67239936
      },
      "comparisons": {
        "K4_B1": true,
        "K4_B128": true,
        "K4_B8": true,
        "R4_U4_K4_B8_flops_equal": true,
        "R4_U4_non_gemm_counts_equal": true,
        "R4_unique_parameter_count_equal": true,
        "U4_unique_parameter_count_equal": true,
        "per_round_B1": true,
        "seven_families": true
      },
      "dimension": 512,
      "exact_match": true,
      "expected_flops": {
        "K4_B1": 268959744,
        "K4_B128": 34426847232,
        "K4_B8": 2151677952,
        "per_round_B1": 67239936
      },
      "family_parameters": {
        "W_K": 262144,
        "W_O": 262144,
        "W_Q": 262144,
        "W_V": 262144,
        "W_down": 1048576,
        "W_gate": 1048576,
        "W_up": 1048576
      },
      "m": 8,
      "schema": "omega-v2-2b-d512-ledger-crosscheck-v1",
      "terminal_on_mismatch": "FLOP_LEDGER_PRESEAL_HOLD"
    },
    "pass": true
  }
}
```

# OMEGA-V2-2B calibration-seed harness smoke

- classification: `CALIBRATION_QA_ONLY`
- V2_2B_verdict: `null`
- terminal_status: `V2_2B_CALIBRATION_SMOKE_COMPLETE`
- master_seed: `20260930`
- structural gates: `True`
- D3 A/B/C gates: `True`
- external launch log hash manifest: `external_launch_logs_artifact_hashes.json`
- wall_gate_seconds: `13.505253500014078`
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
          "wall_seconds": 0.11850599996978417
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
          "wall_seconds": 0.05155560001730919
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
      "wall_seconds": 0.16987309994874522
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
  "D3": {
    "D8": {
      "cell_id": "D3_20260930",
      "gate": "D3",
      "master_seed": 20260930,
      "peak_allocated": 478680064,
      "peak_reserved": 515899392,
      "status": "PASS",
      "wall_seconds": 0.2078369000228122
    },
    "copy_checks": {
      "R4_weights_bitwise_equal": true,
      "U4_weights_bitwise_equal": true,
      "w_bitwise_equal": true,
      "x_bitwise_equal": true
    },
    "families": {
      "W_K": {
        "D3Q_gate_source": "omega_v2_2a_d3q.metrics; unchanged limits",
        "diagnostics_NON_GATE": {
          "old_max_rel": {
            "S_reverse_equal_gR_NON_GATE": true,
            "abs_error": 6.146728992462158e-08,
            "abs_error_over_ulp_gR": 33792.0,
            "abs_error_over_ulp_sum_gU_old": 33792.0,
            "argmax_flat_index": 225112,
            "argmax_multi_index_row_major": [
              439,
              344
            ],
            "floor_1e_6_active": false,
            "gR_value": -3.0217692255973816e-05,
            "max_rel_old": 0.002030019648373127,
            "old_denominator": 3.0279159545898438e-05,
            "sum_gU_old_value": -3.0279159545898438e-05,
            "ulp_gR": 1.8189894035458565e-12,
            "ulp_sum_gU_old": 1.8189894035458565e-12
          },
          "sum_variant_names": [
            "S_forward",
            "S_reverse",
            "S_pairwise",
            "S_stack"
          ]
        },
        "pass": true,
        "primary_S64": {
          "finite": true,
          "gates": {
            "A": {
              "limit": 2e-07,
              "name": "normwise_L2",
              "pass": true,
              "value": 3.5376250416639134e-08
            },
            "B": {
              "limit": 5e-07,
              "name": "normwise_Linf",
              "pass": true,
              "value": 7.104031692695628e-08
            },
            "C": {
              "limit_ulps": 8.0,
              "limit_value": 3.0517578125e-05,
              "name": "max_abs_scale_aware",
              "pass": true,
              "value": 4.291534423828125e-06
            }
          },
          "metrics": {
            "E_L2": 3.5376250416639134e-08,
            "E_inf": 7.104031692695628e-08,
            "max_abs": 4.291534423828125e-06,
            "max_rel_old": 0.003945016334833262,
            "norm_S64_inf": 60.409842014312744,
            "norm_S64_l2": 6022.331528066186,
            "norm_gR64_inf": 60.40984344482422,
            "norm_gR64_l2": 6022.331527747317,
            "shape": [
              512,
              512
            ]
          },
          "scale_context": {
            "M32": 60.40984344482422,
            "M64": 60.40984344482422,
            "ULP_M32": 3.814697265625e-06
          },
          "scientific_gates_pass": true
        }
      },
      "W_O": {
        "D3Q_gate_source": "omega_v2_2a_d3q.metrics; unchanged limits",
        "diagnostics_NON_GATE": {
          "old_max_rel": {
            "S_reverse_equal_gR_NON_GATE": true,
            "abs_error": 4.76837158203125e-07,
            "abs_error_over_ulp_gR": 32768.0,
            "abs_error_over_ulp_sum_gU_old": 32768.0,
            "argmax_flat_index": 257763,
            "argmax_multi_index_row_major": [
              503,
              227
            ],
            "floor_1e_6_active": false,
            "gR_value": -0.00012969970703125,
            "max_rel_old": 0.0036764706019312143,
            "old_denominator": 0.00012969970703125,
            "sum_gU_old_value": -0.00012922286987304688,
            "ulp_gR": 1.4551915228366852e-11,
            "ulp_sum_gU_old": 1.4551915228366852e-11
          },
          "sum_variant_names": [
            "S_forward",
            "S_reverse",
            "S_pairwise",
            "S_stack"
          ]
        },
        "pass": true,
        "primary_S64": {
          "finite": true,
          "gates": {
            "A": {
              "limit": 2e-07,
              "name": "normwise_L2",
              "pass": true,
              "value": 3.7436473662083946e-08
            },
            "B": {
              "limit": 5e-07,
              "name": "normwise_Linf",
              "pass": true,
              "value": 7.035375797201296e-08
            },
            "C": {
              "limit_ulps": 8.0,
              "limit_value": 0.0001220703125,
              "name": "max_abs_scale_aware",
              "pass": true,
              "value": 9.5367431640625e-06
            }
          },
          "metrics": {
            "E_L2": 3.7436473662083946e-08,
            "E_inf": 7.035375797201296e-08,
            "max_abs": 9.5367431640625e-06,
            "max_rel_old": 0.004280821917808219,
            "norm_S64_inf": 135.55414009094238,
            "norm_S64_l2": 12140.59626079686,
            "norm_gR64_inf": 135.55413818359375,
            "norm_gR64_l2": 12140.596262157456,
            "shape": [
              512,
              512
            ]
          },
          "scale_context": {
            "M32": 135.55413818359375,
            "M64": 135.55414009094238,
            "ULP_M32": 1.52587890625e-05
          },
          "scientific_gates_pass": true
        }
      },
      "W_Q": {
        "D3Q_gate_source": "omega_v2_2a_d3q.metrics; unchanged limits",
        "diagnostics_NON_GATE": {
          "old_max_rel": {
            "S_reverse_equal_gR_NON_GATE": true,
            "abs_error": 8.940696716308594e-08,
            "abs_error_over_ulp_gR": 49152.0,
            "abs_error_over_ulp_sum_gU_old": 49152.0,
            "argmax_flat_index": 123953,
            "argmax_multi_index_row_major": [
              242,
              49
            ],
            "floor_1e_6_active": false,
            "gR_value": 2.7418136596679688e-05,
            "max_rel_old": 0.003250270849093795,
            "old_denominator": 2.7507543563842773e-05,
            "sum_gU_old_value": 2.7507543563842773e-05,
            "ulp_gR": 1.8189894035458565e-12,
            "ulp_sum_gU_old": 1.8189894035458565e-12
          },
          "sum_variant_names": [
            "S_forward",
            "S_reverse",
            "S_pairwise",
            "S_stack"
          ]
        },
        "pass": true,
        "primary_S64": {
          "finite": true,
          "gates": {
            "A": {
              "limit": 2e-07,
              "name": "normwise_L2",
              "pass": true,
              "value": 3.518036765670778e-08
            },
            "B": {
              "limit": 5e-07,
              "name": "normwise_Linf",
              "pass": true,
              "value": 5.063560707463341e-08
            },
            "C": {
              "limit_ulps": 8.0,
              "limit_value": 6.103515625e-05,
              "name": "max_abs_scale_aware",
              "pass": true,
              "value": 3.337860107421875e-06
            }
          },
          "metrics": {
            "E_L2": 3.518036765670778e-08,
            "E_inf": 5.063560707463341e-08,
            "max_abs": 3.337860107421875e-06,
            "max_rel_old": 0.0032502708559046588,
            "norm_S64_inf": 65.91922545433044,
            "norm_S64_l2": 6034.608095477019,
            "norm_gR64_inf": 65.91922760009766,
            "norm_gR64_l2": 6034.608095455464,
            "shape": [
              512,
              512
            ]
          },
          "scale_context": {
            "M32": 65.91922760009766,
            "M64": 65.91922760009766,
            "ULP_M32": 7.62939453125e-06
          },
          "scientific_gates_pass": true
        }
      },
      "W_V": {
        "D3Q_gate_source": "omega_v2_2a_d3q.metrics; unchanged limits",
        "diagnostics_NON_GATE": {
          "old_max_rel": {
            "S_reverse_equal_gR_NON_GATE": true,
            "abs_error": 5.960464477539063e-08,
            "abs_error_over_ulp_gR": 32768.0,
            "abs_error_over_ulp_sum_gU_old": 32768.0,
            "argmax_flat_index": 118132,
            "argmax_multi_index_row_major": [
              230,
              372
            ],
            "floor_1e_6_active": false,
            "gR_value": -2.384185791015625e-05,
            "max_rel_old": 0.0024999999441206455,
            "old_denominator": 2.384185791015625e-05,
            "sum_gU_old_value": -2.378225326538086e-05,
            "ulp_gR": 1.8189894035458565e-12,
            "ulp_sum_gU_old": 1.8189894035458565e-12
          },
          "sum_variant_names": [
            "S_forward",
            "S_reverse",
            "S_pairwise",
            "S_stack"
          ]
        },
        "pass": true,
        "primary_S64": {
          "finite": true,
          "gates": {
            "A": {
              "limit": 2e-07,
              "name": "normwise_L2",
              "pass": true,
              "value": 3.718606794721979e-08
            },
            "B": {
              "limit": 5e-07,
              "name": "normwise_Linf",
              "pass": true,
              "value": 7.073775059838479e-08
            },
            "C": {
              "limit_ulps": 8.0,
              "limit_value": 0.0001220703125,
              "name": "max_abs_scale_aware",
              "pass": true,
              "value": 9.5367431640625e-06
            }
          },
          "metrics": {
            "E_L2": 3.718606794721979e-08,
            "E_inf": 7.073775059838479e-08,
            "max_abs": 9.5367431640625e-06,
            "max_rel_old": 0.0025,
            "norm_S64_inf": 134.81829452514648,
            "norm_S64_l2": 12187.485159970835,
            "norm_gR64_inf": 134.81829833984375,
            "norm_gR64_l2": 12187.485164903439,
            "shape": [
              512,
              512
            ]
          },
          "scale_context": {
            "M32": 134.81829833984375,
            "M64": 134.81829833984375,
            "ULP_M32": 1.52587890625e-05
          },
          "scientific_gates_pass": true
        }
      },
      "W_down": {
        "D3Q_gate_source": "omega_v2_2a_d3q.metrics; unchanged limits",
        "diagnostics_NON_GATE": {
          "old_max_rel": {
            "S_reverse_equal_gR_NON_GATE": true,
            "abs_error": 5.960464477539063e-08,
            "abs_error_over_ulp_gR": 65536.0,
            "abs_error_over_ulp_sum_gU_old": 65536.0,
            "argmax_flat_index": 988250,
            "argmax_multi_index_row_major": [
              482,
              1114
            ],
            "floor_1e_6_active": false,
            "gR_value": 9.59634780883789e-06,
            "max_rel_old": 0.006172839552164078,
            "old_denominator": 9.655952453613281e-06,
            "sum_gU_old_value": 9.655952453613281e-06,
            "ulp_gR": 9.094947017729282e-13,
            "ulp_sum_gU_old": 9.094947017729282e-13
          },
          "sum_variant_names": [
            "S_forward",
            "S_reverse",
            "S_pairwise",
            "S_stack"
          ]
        },
        "pass": true,
        "primary_S64": {
          "finite": true,
          "gates": {
            "A": {
              "limit": 2e-07,
              "name": "normwise_L2",
              "pass": true,
              "value": 3.664947262976842e-08
            },
            "B": {
              "limit": 5e-07,
              "name": "normwise_Linf",
              "pass": true,
              "value": 8.488277543199513e-08
            },
            "C": {
              "limit_ulps": 8.0,
              "limit_value": 6.103515625e-05,
              "name": "max_abs_scale_aware",
              "pass": true,
              "value": 6.67572021484375e-06
            }
          },
          "metrics": {
            "E_L2": 3.664947262976842e-08,
            "E_inf": 8.488277543199513e-08,
            "max_abs": 6.67572021484375e-06,
            "max_rel_old": 0.0032948929159802307,
            "norm_S64_inf": 78.64634323120117,
            "norm_S64_l2": 8237.221203266949,
            "norm_gR64_inf": 78.64634704589844,
            "norm_gR64_l2": 8237.22120283195,
            "shape": [
              512,
              2048
            ]
          },
          "scale_context": {
            "M32": 78.64634704589844,
            "M64": 78.64634704589844,
            "ULP_M32": 7.62939453125e-06
          },
          "scientific_gates_pass": true
        }
      },
      "W_gate": {
        "D3Q_gate_source": "omega_v2_2a_d3q.metrics; unchanged limits",
        "diagnostics_NON_GATE": {
          "old_max_rel": {
            "S_reverse_equal_gR_NON_GATE": true,
            "abs_error": 4.76837158203125e-07,
            "abs_error_over_ulp_gR": 32768.0,
            "abs_error_over_ulp_sum_gU_old": 32768.0,
            "argmax_flat_index": 1035702,
            "argmax_multi_index_row_major": [
              2022,
              438
            ],
            "floor_1e_6_active": false,
            "gR_value": -0.00018525123596191406,
            "max_rel_old": 0.002574002603068948,
            "old_denominator": 0.00018525123596191406,
            "sum_gU_old_value": -0.00018477439880371094,
            "ulp_gR": 1.4551915228366852e-11,
            "ulp_sum_gU_old": 1.4551915228366852e-11
          },
          "sum_variant_names": [
            "S_forward",
            "S_reverse",
            "S_pairwise",
            "S_stack"
          ]
        },
        "pass": true,
        "primary_S64": {
          "finite": true,
          "gates": {
            "A": {
              "limit": 2e-07,
              "name": "normwise_L2",
              "pass": true,
              "value": 3.664359093845514e-08
            },
            "B": {
              "limit": 5e-07,
              "name": "normwise_Linf",
              "pass": true,
              "value": 7.886788619553305e-08
            },
            "C": {
              "limit_ulps": 8.0,
              "limit_value": 3.0517578125e-05,
              "name": "max_abs_scale_aware",
              "pass": true,
              "value": 4.76837158203125e-06
            }
          },
          "metrics": {
            "E_L2": 3.664359093845514e-08,
            "E_inf": 7.886788619553305e-08,
            "max_abs": 4.76837158203125e-06,
            "max_rel_old": 0.002270319358256395,
            "norm_S64_inf": 60.46024131774902,
            "norm_S64_l2": 8319.9889662662,
            "norm_gR64_inf": 60.460243225097656,
            "norm_gR64_l2": 8319.988965766564,
            "shape": [
              2048,
              512
            ]
          },
          "scale_context": {
            "M32": 60.460243225097656,
            "M64": 60.460243225097656,
            "ULP_M32": 3.814697265625e-06
          },
          "scientific_gates_pass": true
        }
      },
      "W_up": {
        "D3Q_gate_source": "omega_v2_2a_d3q.metrics; unchanged limits",
        "diagnostics_NON_GATE": {
          "old_max_rel": {
            "S_reverse_equal_gR_NON_GATE": true,
            "abs_error": 1.1920928955078125e-07,
            "abs_error_over_ulp_gR": 65536.0,
            "abs_error_over_ulp_sum_gU_old": 65536.0,
            "argmax_flat_index": 461414,
            "argmax_multi_index_row_major": [
              901,
              102
            ],
            "floor_1e_6_active": false,
            "gR_value": 2.396106719970703e-05,
            "max_rel_old": 0.004950494971126318,
            "old_denominator": 2.4080276489257812e-05,
            "sum_gU_old_value": 2.4080276489257812e-05,
            "ulp_gR": 1.8189894035458565e-12,
            "ulp_sum_gU_old": 1.8189894035458565e-12
          },
          "sum_variant_names": [
            "S_forward",
            "S_reverse",
            "S_pairwise",
            "S_stack"
          ]
        },
        "pass": true,
        "primary_S64": {
          "finite": true,
          "gates": {
            "A": {
              "limit": 2e-07,
              "name": "normwise_L2",
              "pass": true,
              "value": 3.668869362619809e-08
            },
            "B": {
              "limit": 5e-07,
              "name": "normwise_Linf",
              "pass": true,
              "value": 6.190390956793902e-08
            },
            "C": {
              "limit_ulps": 8.0,
              "limit_value": 6.103515625e-05,
              "name": "max_abs_scale_aware",
              "pass": true,
              "value": 4.172325134277344e-06
            }
          },
          "metrics": {
            "E_L2": 3.668869362619809e-08,
            "E_inf": 6.190390956793902e-08,
            "max_abs": 4.172325134277344e-06,
            "max_rel_old": 0.0016811708860759494,
            "norm_S64_inf": 67.40002632141113,
            "norm_S64_l2": 8228.915610488126,
            "norm_gR64_inf": 67.4000244140625,
            "norm_gR64_l2": 8228.915609946047,
            "shape": [
              2048,
              512
            ]
          },
          "scale_context": {
            "M32": 67.4000244140625,
            "M64": 67.40002632141113,
            "ULP_M32": 7.62939453125e-06
          },
          "scientific_gates_pass": true
        }
      }
    },
    "gate": "D3",
    "pass": true,
    "seed_plan": {
      "input_seed": 20261938,
      "loss_w_seed": 20262938,
      "master_seed": 20260930,
      "mode": "smoke",
      "target_seed": 20263938,
      "weight_seed": 20260930
    }
  },
  "D4": {
    "R4_state_dict_schema": [
      [
        "recurrent.block.W_Q",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "recurrent.block.W_K",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "recurrent.block.W_V",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "recurrent.block.W_O",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "recurrent.block.W_gate",
        "torch.float32",
        [
          2048,
          512
        ]
      ],
      [
        "recurrent.block.W_up",
        "torch.float32",
        [
          2048,
          512
        ]
      ],
      [
        "recurrent.block.W_down",
        "torch.float32",
        [
          512,
          2048
        ]
      ]
    ],
    "U4_state_dict_schema": [
      [
        "blocks.0.W_Q",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.0.W_K",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.0.W_V",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.0.W_O",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.0.W_gate",
        "torch.float32",
        [
          2048,
          512
        ]
      ],
      [
        "blocks.0.W_up",
        "torch.float32",
        [
          2048,
          512
        ]
      ],
      [
        "blocks.0.W_down",
        "torch.float32",
        [
          512,
          2048
        ]
      ],
      [
        "blocks.1.W_Q",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.1.W_K",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.1.W_V",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.1.W_O",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.1.W_gate",
        "torch.float32",
        [
          2048,
          512
        ]
      ],
      [
        "blocks.1.W_up",
        "torch.float32",
        [
          2048,
          512
        ]
      ],
      [
        "blocks.1.W_down",
        "torch.float32",
        [
          512,
          2048
        ]
      ],
      [
        "blocks.2.W_Q",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.2.W_K",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.2.W_V",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.2.W_O",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.2.W_gate",
        "torch.float32",
        [
          2048,
          512
        ]
      ],
      [
        "blocks.2.W_up",
        "torch.float32",
        [
          2048,
          512
        ]
      ],
      [
        "blocks.2.W_down",
        "torch.float32",
        [
          512,
          2048
        ]
      ],
      [
        "blocks.3.W_Q",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.3.W_K",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.3.W_V",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.3.W_O",
        "torch.float32",
        [
          512,
          512
        ]
      ],
      [
        "blocks.3.W_gate",
        "torch.float32",
        [
          2048,
          512
        ]
      ],
      [
        "blocks.3.W_up",
        "torch.float32",
        [
          2048,
          512
        ]
      ],
      [
        "blocks.3.W_down",
        "torch.float32",
        [
          512,
          2048
        ]
      ]
    ],
    "clone_report": {
      "initial_values_bitwise_equal": true,
      "r4_reuses_one_block_object": true,
      "u4_block_storages_pairwise_disjoint": true,
      "u4_has_four_block_objects": true,
      "u4_storage_disjoint_from_r4": true
    },
    "gate": "D4",
    "pass": true,
    "seed_plan": {
      "input_seed": 20261938,
      "loss_w_seed": 20262938,
      "master_seed": 20260930,
      "mode": "smoke",
      "target_seed": 20263938,
      "weight_seed": 20260930
    },
    "state_dict_schema_conformant": true,
    "storage": {
      "R4_one_storage_per_family": true,
      "R4_parameter_count_pass": true,
      "R4_unique_parameters": 4194304,
      "U4_disjoint_from_R4": true,
      "U4_four_pairwise_disjoint_storages_per_family": true,
      "U4_parameter_count_pass": true,
      "U4_unique_parameters": 16777216,
      "families": [
        {
          "family": "W_Q",
          "initial_bitwise_equal": true,
          "u4_disjoint_from_r4": true,
          "u4_four_disjoint_storages": true
        },
        {
          "family": "W_K",
          "initial_bitwise_equal": true,
          "u4_disjoint_from_r4": true,
          "u4_four_disjoint_storages": true
        },
        {
          "family": "W_V",
          "initial_bitwise_equal": true,
          "u4_disjoint_from_r4": true,
          "u4_four_disjoint_storages": true
        },
        {
          "family": "W_O",
          "initial_bitwise_equal": true,
          "u4_disjoint_from_r4": true,
          "u4_four_disjoint_storages": true
        },
        {
          "family": "W_gate",
          "initial_bitwise_equal": true,
          "u4_disjoint_from_r4": true,
          "u4_four_disjoint_storages": true
        },
        {
          "family": "W_up",
          "initial_bitwise_equal": true,
          "u4_disjoint_from_r4": true,
          "u4_four_disjoint_storages": true
        },
        {
          "family": "W_down",
          "initial_bitwise_equal": true,
          "u4_disjoint_from_r4": true,
          "u4_four_disjoint_storages": true
        }
      ],
      "initial_bitwise_equal": true,
      "pass": true
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
  },
  "D6": {
    "applicable": true,
    "cells": [
      {
        "D8": {
          "cell_id": "D6_20260930_forward_K1",
          "gate": "D6",
          "master_seed": 20260930,
          "peak_allocated": 87431680,
          "peak_reserved": 96468992,
          "status": "PASS",
          "wall_seconds": 0.05227290000766516
        },
        "K": 1,
        "cell_id": "D6_20260930_forward_K1",
        "copy_checks": {
          "R4_weights_bitwise_equal": true,
          "x_bitwise_equal": true
        },
        "equality": {
          "SCHEMA_SHA256_after": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "SCHEMA_SHA256_before": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "VALUE_SHA256_after": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "VALUE_SHA256_before": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "parameter_count_after": 4194304,
          "parameter_count_before": 4194304,
          "pass": true
        },
        "equality_pass": true,
        "finite": true,
        "metrics": {
          "output_finite": true
        },
        "mode": "forward",
        "pass": true
      },
      {
        "D8": {
          "cell_id": "D6_20260930_forward_K2",
          "gate": "D6",
          "master_seed": 20260930,
          "peak_allocated": 90580480,
          "peak_reserved": 100663296,
          "status": "PASS",
          "wall_seconds": 0.05245020001893863
        },
        "K": 2,
        "cell_id": "D6_20260930_forward_K2",
        "copy_checks": {
          "R4_weights_bitwise_equal": true,
          "x_bitwise_equal": true
        },
        "equality": {
          "SCHEMA_SHA256_after": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "SCHEMA_SHA256_before": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "VALUE_SHA256_after": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "VALUE_SHA256_before": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "parameter_count_after": 4194304,
          "parameter_count_before": 4194304,
          "pass": true
        },
        "equality_pass": true,
        "finite": true,
        "metrics": {
          "output_finite": true
        },
        "mode": "forward",
        "pass": true
      },
      {
        "D8": {
          "cell_id": "D6_20260930_forward_K4",
          "gate": "D6",
          "master_seed": 20260930,
          "peak_allocated": 96878080,
          "peak_reserved": 106954752,
          "status": "PASS",
          "wall_seconds": 0.05475080001633614
        },
        "K": 4,
        "cell_id": "D6_20260930_forward_K4",
        "copy_checks": {
          "R4_weights_bitwise_equal": true,
          "x_bitwise_equal": true
        },
        "equality": {
          "SCHEMA_SHA256_after": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "SCHEMA_SHA256_before": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "VALUE_SHA256_after": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "VALUE_SHA256_before": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "parameter_count_after": 4194304,
          "parameter_count_before": 4194304,
          "pass": true
        },
        "equality_pass": true,
        "finite": true,
        "metrics": {
          "output_finite": true
        },
        "mode": "forward",
        "pass": true
      },
      {
        "D8": {
          "cell_id": "D6_20260930_forward_K8",
          "gate": "D6",
          "master_seed": 20260930,
          "peak_allocated": 109473280,
          "peak_reserved": 119537664,
          "status": "PASS",
          "wall_seconds": 0.057872099976520985
        },
        "K": 8,
        "cell_id": "D6_20260930_forward_K8",
        "copy_checks": {
          "R4_weights_bitwise_equal": true,
          "x_bitwise_equal": true
        },
        "equality": {
          "SCHEMA_SHA256_after": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "SCHEMA_SHA256_before": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "VALUE_SHA256_after": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "VALUE_SHA256_before": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "parameter_count_after": 4194304,
          "parameter_count_before": 4194304,
          "pass": true
        },
        "equality_pass": true,
        "finite": true,
        "metrics": {
          "output_finite": true
        },
        "mode": "forward",
        "pass": true
      },
      {
        "D8": {
          "cell_id": "D6_20260930_forward_K16",
          "gate": "D6",
          "master_seed": 20260930,
          "peak_allocated": 134663680,
          "peak_reserved": 144703488,
          "status": "PASS",
          "wall_seconds": 0.06744889996480197
        },
        "K": 16,
        "cell_id": "D6_20260930_forward_K16",
        "copy_checks": {
          "R4_weights_bitwise_equal": true,
          "x_bitwise_equal": true
        },
        "equality": {
          "SCHEMA_SHA256_after": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "SCHEMA_SHA256_before": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "VALUE_SHA256_after": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "VALUE_SHA256_before": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "parameter_count_after": 4194304,
          "parameter_count_before": 4194304,
          "pass": true
        },
        "equality_pass": true,
        "finite": true,
        "metrics": {
          "output_finite": true
        },
        "mode": "forward",
        "pass": true
      },
      {
        "D8": {
          "cell_id": "D6_20260930_backward_K1",
          "gate": "D6",
          "master_seed": 20260930,
          "peak_allocated": 108528128,
          "peak_reserved": 123731968,
          "status": "PASS",
          "wall_seconds": 0.058990999998059124
        },
        "K": 1,
        "cell_id": "D6_20260930_backward_K1",
        "copy_checks": {
          "R4_weights_bitwise_equal": true,
          "w_bitwise_equal": true,
          "x_bitwise_equal": true
        },
        "equality": {
          "SCHEMA_SHA256_after": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "SCHEMA_SHA256_before": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "VALUE_SHA256_after": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "VALUE_SHA256_before": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "parameter_count_after": 4194304,
          "parameter_count_before": 4194304,
          "pass": true
        },
        "equality_pass": true,
        "finite": true,
        "metrics": {
          "input_gradient_finite": true,
          "loss": 235.60104370117188,
          "loss_finite": true,
          "output_finite": true,
          "parameter_gradients_finite": true
        },
        "mode": "backward",
        "pass": true
      },
      {
        "D8": {
          "cell_id": "D6_20260930_backward_K4",
          "gate": "D6",
          "master_seed": 20260930,
          "peak_allocated": 118892544,
          "peak_reserved": 130023424,
          "status": "PASS",
          "wall_seconds": 0.06408469995949417
        },
        "K": 4,
        "cell_id": "D6_20260930_backward_K4",
        "copy_checks": {
          "R4_weights_bitwise_equal": true,
          "w_bitwise_equal": true,
          "x_bitwise_equal": true
        },
        "equality": {
          "SCHEMA_SHA256_after": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "SCHEMA_SHA256_before": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "VALUE_SHA256_after": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "VALUE_SHA256_before": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "parameter_count_after": 4194304,
          "parameter_count_before": 4194304,
          "pass": true
        },
        "equality_pass": true,
        "finite": true,
        "metrics": {
          "input_gradient_finite": true,
          "loss": -84.44166564941406,
          "loss_finite": true,
          "output_finite": true,
          "parameter_gradients_finite": true
        },
        "mode": "backward",
        "pass": true
      },
      {
        "D8": {
          "cell_id": "D6_20260930_backward_K8",
          "gate": "D6",
          "master_seed": 20260930,
          "peak_allocated": 131487744,
          "peak_reserved": 142606336,
          "status": "PASS",
          "wall_seconds": 0.0713559000287205
        },
        "K": 8,
        "cell_id": "D6_20260930_backward_K8",
        "copy_checks": {
          "R4_weights_bitwise_equal": true,
          "w_bitwise_equal": true,
          "x_bitwise_equal": true
        },
        "equality": {
          "SCHEMA_SHA256_after": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "SCHEMA_SHA256_before": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "VALUE_SHA256_after": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "VALUE_SHA256_before": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "parameter_count_after": 4194304,
          "parameter_count_before": 4194304,
          "pass": true
        },
        "equality_pass": true,
        "finite": true,
        "metrics": {
          "input_gradient_finite": true,
          "loss": -708.5091552734375,
          "loss_finite": true,
          "output_finite": true,
          "parameter_gradients_finite": true
        },
        "mode": "backward",
        "pass": true
      },
      {
        "D8": {
          "cell_id": "D6_20260930_backward_K16",
          "gate": "D6",
          "master_seed": 20260930,
          "peak_allocated": 156678144,
          "peak_reserved": 167772160,
          "status": "PASS",
          "wall_seconds": 0.08946230000583455
        },
        "K": 16,
        "cell_id": "D6_20260930_backward_K16",
        "copy_checks": {
          "R4_weights_bitwise_equal": true,
          "w_bitwise_equal": true,
          "x_bitwise_equal": true
        },
        "equality": {
          "SCHEMA_SHA256_after": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "SCHEMA_SHA256_before": "ad7c4554878ed17e137645f57114bb1683aacf2a52eadd36e7e6fa6785e615e4",
          "VALUE_SHA256_after": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "VALUE_SHA256_before": "dcb356896757b5341df781bb3589cc8b16d87a0425fdfb06663cb37b6a3d1958",
          "parameter_count_after": 4194304,
          "parameter_count_before": 4194304,
          "pass": true
        },
        "equality_pass": true,
        "finite": true,
        "metrics": {
          "input_gradient_finite": true,
          "loss": -1089.4296875,
          "loss_finite": true,
          "output_finite": true,
          "parameter_gradients_finite": true
        },
        "mode": "backward",
        "pass": true
      }
    ],
    "pass": true
  },
  "D7": {
    "pass": true,
    "variants": [
      {
        "D8": {
          "cell_id": "D7_20260930_R4",
          "gate": "D7",
          "master_seed": 20260930,
          "peak_allocated": 362968064,
          "peak_reserved": 385875968,
          "status": "PASS",
          "wall_seconds": 1.2872987000155263
        },
        "L0": 7.384119510650635,
        "L20": 2.067394733428955,
        "L20_over_L0": 0.2799785039295486,
        "all_steps_finite": true,
        "cell_id": "D7_20260930_R4",
        "copy_checks": {
          "target_bitwise_equal": true,
          "weights_bitwise_equal": true,
          "x_bitwise_equal": true
        },
        "final_optimizer_state_canonical_sha256": "b6c41607ef728e49e4962e4ffefe067871927eeaedc597ff4f09ecf3a47790b4",
        "final_parameter_canonical_sha256": "c41bf47e848ebeec3e81a1c632192cfb0ffd3b92933c141670b252d9bb40c56d",
        "forward_count": 21,
        "gate": "D7",
        "initial_R4_U4_clone_report": {
          "initial_values_bitwise_equal": true,
          "r4_reuses_one_block_object": true,
          "u4_block_storages_pairwise_disjoint": true,
          "u4_has_four_block_objects": true,
          "u4_storage_disjoint_from_r4": true
        },
        "loss_reduction_pass": true,
        "optimizer_step_count": 20,
        "seed_plan": {
          "input_seed": 20261938,
          "loss_w_seed": 20262938,
          "master_seed": 20260930,
          "mode": "smoke",
          "target_seed": 20263938,
          "weight_seed": 20260930
        },
        "step_finiteness": [
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 7.384119510650635,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 1
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 6.287840366363525,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 2
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 5.529880523681641,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 3
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 4.963674545288086,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 4
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 4.515572547912598,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 5
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 4.151425361633301,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 6
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 3.85245943069458,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 7
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 3.604036808013916,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 8
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 3.3944196701049805,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 9
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 3.2145938873291016,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 10
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 3.0576446056365967,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 11
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.9183034896850586,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 12
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.792738676071167,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 13
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.678323268890381,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 14
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.5731201171875,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 15
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.475465774536133,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 16
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.3840343952178955,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 17
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.2979800701141357,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 18
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.2167937755584717,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 19
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.140061378479004,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 20
          }
        ],
        "updates": 20,
        "variant": "R4"
      },
      {
        "D8": {
          "cell_id": "D7_20260930_U4",
          "gate": "D7",
          "master_seed": 20260930,
          "peak_allocated": 564294656,
          "peak_reserved": 593494016,
          "status": "PASS",
          "wall_seconds": 1.830977100005839
        },
        "L0": 7.384119510650635,
        "L20": 1.7884756326675415,
        "L20_over_L0": 0.24220567260428238,
        "all_steps_finite": true,
        "cell_id": "D7_20260930_U4",
        "copy_checks": {
          "target_bitwise_equal": true,
          "weights_bitwise_equal": true,
          "x_bitwise_equal": true
        },
        "final_optimizer_state_canonical_sha256": "474b906cfad20e4c9c226792c48e21f81c5d1537406aee62fee723dfcbec68a0",
        "final_parameter_canonical_sha256": "663ae384a9e6dad8505f7d4f6e623415bb18e25eeac9c26b7be75fd130f4f2a1",
        "forward_count": 21,
        "gate": "D7",
        "initial_R4_U4_clone_report": {
          "initial_values_bitwise_equal": true,
          "r4_reuses_one_block_object": true,
          "u4_block_storages_pairwise_disjoint": true,
          "u4_has_four_block_objects": true,
          "u4_storage_disjoint_from_r4": true
        },
        "loss_reduction_pass": true,
        "optimizer_step_count": 20,
        "seed_plan": {
          "input_seed": 20261938,
          "loss_w_seed": 20262938,
          "master_seed": 20260930,
          "mode": "smoke",
          "target_seed": 20263938,
          "weight_seed": 20260930
        },
        "step_finiteness": [
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 7.384119510650635,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 1
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 6.090288162231445,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 2
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 5.2517805099487305,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 3
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 4.642855644226074,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 4
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 4.175604820251465,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 5
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 3.8106255531311035,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 6
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 3.5210509300231934,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 7
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 3.2857441902160645,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 8
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 3.0891008377075195,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 9
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.920257568359375,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 10
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.771773338317871,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 11
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.6385955810546875,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 12
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.517324924468994,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 13
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.4054646492004395,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 14
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.301229953765869,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 15
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.203470230102539,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 16
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.1113533973693848,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 17
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 2.0242350101470947,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 18
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 1.9416637420654297,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 19
          },
          {
            "exp_avg_finite": true,
            "exp_avg_sq_finite": true,
            "gradients_finite": true,
            "loss": 1.863234519958496,
            "loss_finite": true,
            "output_finite": true,
            "parameters_finite": true,
            "step": 20
          }
        ],
        "updates": 20,
        "variant": "U4"
      }
    ]
  }
}
```

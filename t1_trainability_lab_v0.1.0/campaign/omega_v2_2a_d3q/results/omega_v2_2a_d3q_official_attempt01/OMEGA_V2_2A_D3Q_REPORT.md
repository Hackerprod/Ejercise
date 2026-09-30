# OMEGA-V2-2A-D3Q

- terminal_status: `OMEGA_V2_2A_D3Q_PASS`
- classification: `INDEPENDENT_HELDOUT_VALIDATION`
- calibration seed 20260930 included in official verdict: `False`
- official seed slots completed: `5` / 5
- hard_stop: `None`

## Per-seed decisions

| Seed slot | Structural | Finite | Oracle | Seed result | Duration (s) | Peak allocated (bytes) | Peak reserved (bytes) |
|---|---:|---:|---:|---|---:|---:|---:|
| seed_20261001 | True | True | True | SEED_PASS | 0.452106 | 184026112 | 205520896 |
| seed_20261002 | True | True | True | SEED_PASS | 0.20627 | 184026112 | 205520896 |
| seed_20261003 | True | True | True | SEED_PASS | 0.1893723 | 184026112 | 205520896 |
| seed_20261004 | True | True | True | SEED_PASS | 0.2213986 | 184026112 | 205520896 |
| seed_20261005 | True | True | True | SEED_PASS | 0.4037727 | 184026112 | 205520896 |

## Seed seed_20261001

### Structural trace checks

```json
{
  "final_output_equal": true,
  "initial_clone_report": {
    "initial_values_bitwise_equal": true,
    "r4_reuses_one_block_object": true,
    "u4_block_storages_pairwise_disjoint": true,
    "u4_has_four_block_objects": true,
    "u4_storage_disjoint_from_r4": true
  },
  "input_copy_bitwise_equal": true,
  "loss_weight_copy_bitwise_equal": true,
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
  ],
  "weight_copy_report": {
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
    ]
  }
}
```

### Per-family scientific and oracle decisions

```json
{
  "W_K": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 1.0125679168707752e-16
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.3860443693396086e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 1.0125679168707752e-16,
        "E_inf": 1.3860443693396086e-16,
        "max_abs": 7.105427357601002e-15,
        "max_rel_old": 9.971536482643678e-12,
        "norm_S64_inf": 51.2640685592658,
        "norm_S64_l2": 2674.668244969742,
        "norm_gR64_inf": 51.264068559265795,
        "norm_gR64_l2": 2674.668244969742,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 1.1920928955078125e-07,
        "abs_error_over_ulp_gR": 32768.0,
        "abs_error_over_ulp_sum_gU_old": 32768.0,
        "argmax_flat_index": 32330,
        "argmax_multi_index_row_major": [
          126,
          74
        ],
        "floor_1e_6_active": false,
        "gR_value": -4.1961669921875e-05,
        "max_rel_old": 0.0028409091755747795,
        "old_denominator": 4.1961669921875e-05,
        "sum_gU_old_value": -4.184246063232422e-05,
        "ulp_gR": 3.637978807091713e-12,
        "ulp_sum_gU_old": 3.637978807091713e-12
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.3651105783590985e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 6.046030167058186e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 3.0517578125e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.0994415283203125e-06
        }
      },
      "metrics": {
        "E_L2": 3.3651105783590985e-08,
        "E_inf": 6.046030167058186e-08,
        "max_abs": 3.0994415283203125e-06,
        "max_rel_old": 0.0001312163758037003,
        "norm_S64_inf": 51.264076471328735,
        "norm_S64_l2": 2674.668235489723,
        "norm_gR64_inf": 51.264076232910156,
        "norm_gR64_l2": 2674.668235905495,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 51.264076232910156,
        "M64": 51.264076471328735,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_O": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 1.0054667232343397e-16
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.9261432181814226e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 1.0054667232343397e-16,
        "E_inf": 1.9261432181814226e-16,
        "max_abs": 2.842170943040401e-14,
        "max_rel_old": 1.696921595234148e-12,
        "norm_S64_inf": 147.5576123422354,
        "norm_S64_l2": 6093.871460148873,
        "norm_gR64_inf": 147.5576123422354,
        "norm_gR64_l2": 6093.871460148873,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 2.384185791015625e-07,
        "abs_error_over_ulp_gR": 262144.0,
        "abs_error_over_ulp_sum_gU_old": 262144.0,
        "argmax_flat_index": 44971,
        "argmax_multi_index_row_major": [
          175,
          171
        ],
        "floor_1e_6_active": false,
        "gR_value": -1.1205673217773438e-05,
        "max_rel_old": 0.021276595070958138,
        "old_denominator": 1.1205673217773438e-05,
        "sum_gU_old_value": -1.0967254638671875e-05,
        "ulp_gR": 9.094947017729282e-13,
        "ulp_sum_gU_old": 9.094947017729282e-13
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.745998055857121e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.998530919787538e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 0.0001220703125,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 8.851289749145508e-06
        }
      },
      "metrics": {
        "E_L2": 3.745998055857121e-08,
        "E_inf": 5.998530919787538e-08,
        "max_abs": 8.851289749145508e-06,
        "max_rel_old": 0.005291005291005291,
        "norm_S64_inf": 147.55762481689453,
        "norm_S64_l2": 6093.871652365623,
        "norm_gR64_inf": 147.5576171875,
        "norm_gR64_l2": 6093.871653036215,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 147.5576171875,
        "M64": 147.55762481689453,
        "ULP_M32": 1.52587890625e-05
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_Q": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 1.0132826779462512e-16
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.8559712514042934e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 1.0132826779462512e-16,
        "E_inf": 2.8559712514042934e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 1.467455353655146e-12,
        "norm_S64_inf": 49.758395530817985,
        "norm_S64_l2": 2659.1638798350227,
        "norm_gR64_inf": 49.75839553081798,
        "norm_gR64_l2": 2659.163879835023,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 2.384185791015625e-07,
        "abs_error_over_ulp_gR": 32768.0,
        "abs_error_over_ulp_sum_gU_old": 32768.0,
        "argmax_flat_index": 63435,
        "argmax_multi_index_row_major": [
          247,
          203
        ],
        "floor_1e_6_active": false,
        "gR_value": 7.152557373046875e-05,
        "max_rel_old": 0.003322259057313204,
        "old_denominator": 7.176399230957031e-05,
        "sum_gU_old_value": 7.176399230957031e-05,
        "ulp_gR": 7.275957614183426e-12,
        "ulp_sum_gU_old": 7.275957614183426e-12
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.4054932839816545e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 6.708133890730066e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 3.0517578125e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.337860107421875e-06
        }
      },
      "metrics": {
        "E_L2": 3.4054932839816545e-08,
        "E_inf": 6.708133890730066e-08,
        "max_abs": 3.337860107421875e-06,
        "max_rel_old": 0.0033222591362126247,
        "norm_S64_inf": 49.758400201797485,
        "norm_S64_l2": 2659.163887247338,
        "norm_gR64_inf": 49.758399963378906,
        "norm_gR64_l2": 2659.1638876592756,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 49.758399963378906,
        "M64": 49.758400201797485,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_V": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 1.0012462715968825e-16
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.0828540336910261e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 1.0012462715968825e-16,
        "E_inf": 1.0828540336910261e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 6.167187081546642e-13,
        "norm_S64_inf": 131.23518288760263,
        "norm_S64_l2": 6040.9690553371265,
        "norm_gR64_inf": 131.23518288760263,
        "norm_gR64_l2": 6040.9690553371265,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 1.1920928955078125e-07,
        "abs_error_over_ulp_gR": 8192.0,
        "abs_error_over_ulp_sum_gU_old": 8192.0,
        "argmax_flat_index": 39850,
        "argmax_multi_index_row_major": [
          155,
          170
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.0001800060272216797,
        "max_rel_old": 0.0006622516666539013,
        "old_denominator": 0.0001800060272216797,
        "sum_gU_old_value": -0.0001798868179321289,
        "ulp_gR": 1.4551915228366852e-11,
        "ulp_sum_gU_old": 1.4551915228366852e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.777226640671061e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 7.266909517093515e-08
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
        "E_L2": 3.777226640671061e-08,
        "E_inf": 7.266909517093515e-08,
        "max_abs": 9.5367431640625e-06,
        "max_rel_old": 0.0006622516556291391,
        "norm_S64_inf": 131.2351894378662,
        "norm_S64_l2": 6040.969269341832,
        "norm_gR64_inf": 131.23519897460938,
        "norm_gR64_l2": 6040.969264981963,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 131.23519897460938,
        "M64": 131.23519897460938,
        "ULP_M32": 1.52587890625e-05
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_down": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.964563000250281e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.4828820755890263e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.964563000250281e-17,
        "E_inf": 2.4828820755890263e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 8.944498578462473e-12,
        "norm_S64_inf": 57.235318805186076,
        "norm_S64_l2": 3940.6984380339713,
        "norm_gR64_inf": 57.235318805186076,
        "norm_gR64_l2": 3940.698438033971,
        "shape": [
          256,
          1024
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 1.1920928955078125e-07,
        "abs_error_over_ulp_gR": 8192.0,
        "abs_error_over_ulp_sum_gU_old": 8192.0,
        "argmax_flat_index": 8761,
        "argmax_multi_index_row_major": [
          8,
          569
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.00012540817260742188,
        "max_rel_old": 0.0009496675920672715,
        "old_denominator": 0.00012552738189697266,
        "sum_gU_old_value": -0.00012552738189697266,
        "ulp_gR": 1.4551915228366852e-11,
        "ulp_sum_gU_old": 1.4551915228366852e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.690862609340559e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 6.664936116004147e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 3.0517578125e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.814697265625e-06
        }
      },
      "metrics": {
        "E_L2": 3.690862609340559e-08,
        "E_inf": 6.664936116004147e-08,
        "max_abs": 3.814697265625e-06,
        "max_rel_old": 0.0017125445743474167,
        "norm_S64_inf": 57.23531627655029,
        "norm_S64_l2": 3940.6985926887864,
        "norm_gR64_inf": 57.235313415527344,
        "norm_gR64_l2": 3940.698592875122,
        "shape": [
          256,
          1024
        ]
      },
      "scale_context": {
        "M32": 57.23531723022461,
        "M64": 57.23531627655029,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_gate": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.904127313655654e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.0846799743200165e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.904127313655654e-17,
        "E_inf": 1.0846799743200165e-16,
        "max_abs": 7.105427357601002e-15,
        "max_rel_old": 9.21917002041354e-12,
        "norm_S64_inf": 65.50713137352221,
        "norm_S64_l2": 4062.467721550588,
        "norm_gR64_inf": 65.50713137352221,
        "norm_gR64_l2": 4062.467721550588,
        "shape": [
          1024,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 5.960464477539062e-07,
        "abs_error_over_ulp_gR": 81920.0,
        "abs_error_over_ulp_sum_gU_old": 81920.0,
        "argmax_flat_index": 30740,
        "argmax_multi_index_row_major": [
          120,
          20
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.0001049041748046875,
        "max_rel_old": 0.005681818351149559,
        "old_denominator": 0.0001049041748046875,
        "sum_gU_old_value": -0.0001043081283569336,
        "ulp_gR": 7.275957614183426e-12,
        "ulp_sum_gU_old": 7.275957614183426e-12
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.681502523963278e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.823329048139598e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 6.103515625e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.814697265625e-06
        }
      },
      "metrics": {
        "E_L2": 3.681502523963278e-08,
        "E_inf": 5.823329048139598e-08,
        "max_abs": 3.814697265625e-06,
        "max_rel_old": 0.005376344086021506,
        "norm_S64_inf": 65.50715416669846,
        "norm_S64_l2": 4062.467893651419,
        "norm_gR64_inf": 65.50715637207031,
        "norm_gR64_l2": 4062.467894200514,
        "shape": [
          1024,
          256
        ]
      },
      "scale_context": {
        "M32": 65.50715637207031,
        "M64": 65.50715637207031,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_up": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.869916553476866e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.7289858520544533e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.869916553476866e-17,
        "E_inf": 1.7289858520544533e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 1.932411428488744e-12,
        "norm_S64_inf": 82.19185077955422,
        "norm_S64_l2": 4042.0841182113486,
        "norm_gR64_inf": 82.19185077955422,
        "norm_gR64_l2": 4042.0841182113486,
        "shape": [
          1024,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 3.5762786865234375e-07,
        "abs_error_over_ulp_gR": 98304.0,
        "abs_error_over_ulp_sum_gU_old": 98304.0,
        "argmax_flat_index": 138766,
        "argmax_multi_index_row_major": [
          542,
          14
        ],
        "floor_1e_6_active": false,
        "gR_value": -4.220008850097656e-05,
        "max_rel_old": 0.008403361774981022,
        "old_denominator": 4.2557716369628906e-05,
        "sum_gU_old_value": -4.2557716369628906e-05,
        "ulp_gR": 3.637978807091713e-12,
        "ulp_sum_gU_old": 3.637978807091713e-12
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.7038688780020774e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.80151207826082e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 6.103515625e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 4.76837158203125e-06
        }
      },
      "metrics": {
        "E_L2": 3.7038688780020774e-08,
        "E_inf": 5.80151207826082e-08,
        "max_abs": 4.76837158203125e-06,
        "max_rel_old": 0.0056179775280898875,
        "norm_S64_inf": 82.19187545776367,
        "norm_S64_l2": 4042.084294198244,
        "norm_gR64_inf": 82.1918716430664,
        "norm_gR64_l2": 4042.08429398628,
        "shape": [
          1024,
          256
        ]
      },
      "scale_context": {
        "M32": 82.19187927246094,
        "M64": 82.19187545776367,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  }
}
```

### Persisted gradient bundle

```json
{
  "bundle_path": "seed_20261001_raw_gradients.pt",
  "bundle_sha256": "9ce624d958845accae51a9ece88d1977bd8203ef1ab0651b610050786c7968a5",
  "bundle_size_bytes": 105685741,
  "tensor_metadata": {
    "seed_20261001/cpu_fp64_oracle/W_K/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "ffc3b8458b68570b6398159ccfd35ad0a54f124cd0a0f44dd85dfe4de35d4c30",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_K/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "55a0d66b8f827b1096400cab5b8fdca9133d73552eab9c601dc9c626f1f97c4d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_K/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "0c0275f854ff3b58f3bf7dcf15b34633de5404515e4e49cb5aa11cf84abc50ce",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_K/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "c7efcbba113e43004543245571eb100d769475c98d78ca65b01b4e24d8fdbf0c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_K/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "7e1a0c097d22d8d299b41dbdb59387bdb077bfd4f222f9335231f16a02153310",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_K/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "550538f9d09c3c727f9be39d66b6c077b3e09e287e0dac9d57eab57c0de40d79",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_O/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "1a1de703c4c9c3b6a70e2f78bf6d7b5f3aca727eabf8c56c3c2d79195131d53b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_O/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "059a2f62406a4e44da539ac6b286d27dabf13b388a535ff14f897a8800e158be",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_O/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "0766abfac816c7ac36b68ec0642f672fd6e9d61736470a1d5671b7bc8c9c3ee0",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_O/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "1983f55359c9806d98122dc4efe4a3f5954cc8b38e348ae55d07266bafaa0800",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_O/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "79181f0bf49c338fef560fe332a931be9999854aa8b74bd12d71eb965f38fd3d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_O/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "5f0b931fedf75d1826a5de74bdd6869137369de7b4a1ac1428f1338c1bd78e9a",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_Q/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "af7670d6e8badd2b20d76e752822e87a8e9e1d4880bc82bb35227dd79ded9e59",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_Q/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "ac81107fe5bb05253fb9b4e5798ce64813d9c13a258518c66055d6bef71902a9",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_Q/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "94c434749ecb5df95ff2b377d13212760f8690f607383e6687902800be6811f0",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_Q/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "e24419977612a76146159db7db246547ea53422929f8e7c9ddce159e4be0553c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_Q/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "32e9a5591fc9f24242dad433203b415e650d99a4d0e02cbf5517ed8d1b0c5f83",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_Q/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "01486a86397ebc84ab04e29a0a8fbe9e31a10263185979b4f2ea6bc989d1a252",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_V/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "40c0852c846f7dac419ece8a50cff4098612e3f09304d842cccb2c8e7a1e6ba7",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_V/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "713a0ac00fadadb093d9f9e65b3221e0c0179b4b39edc0774dae5a8f4d947324",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_V/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "b7093854a8193add3fe2d00a4f73aab1e68adac84ee66529dcf1dc5e31d563cc",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_V/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "b80081d47cbacd5c25a28f0ca0b631b8453082deb24c8c6db63efbbbb5a73c0f",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_V/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "956dc1fb111e015fd1f97f22cefc8f5b630868b30c206cacc81c9bacf83b8711",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_V/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "c59644de21513f887598335a8b2ce89205b282ed174185a1f7f44c5611e9edaa",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_down/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "71e8c1242ddf314f53bc6304a4ac9be49befe2197cc46167739e5f8385f8e774",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_down/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "1e54f6ad76dd2d45afd42aa878ea89a423e04809265f44a89439e475bcd46ce3",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_down/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "20a245afb1a9936129deeb68789aaefeda269f44705e1e41e93ba9ed71eb6fa4",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_down/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "fda5eec8178237ce15e1f63a6d8653063279d9cb5aaafbe260e789845ffa247d",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_down/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "de59fa87dc676becc25be485e4b8174246adbf26f314b73974c8ba87556d20bd",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_down/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "ddab034b21ec704d2562df3b67fab2480511353d1a9cc0bd62f20cbed6947772",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_gate/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "223d1e6ddec01178295e99654a93c4eefb88901477e8430fa84503045017999f",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_gate/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "4f6d62cd6e8f597516b2d333cbfebfd0e2e7f424d6a5461bf3b2c271b86f43ad",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_gate/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "36c5a1d6f8b3a9b1fc1c5dcb0adea2b439fcf2e783df9a45ff78a54c14898465",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_gate/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "6065ef54c81707b227836a6337f5e73f6a8a379b5d73053839265f734e76cefb",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_gate/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "38a9bc7e320b306c76a86b40350783207516bfbd46530d71fe41dae34c8298d3",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_gate/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "eeb457d442d2543f7be384bafbe2e30fe1163b78217b2b243754d9e0673c62c7",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_up/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "141f7e5e3f6131a63e6cf8a97258f52a89ae976ec32d9bd984f0393e4146a355",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_up/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "2ca37f333e415b9cf09c0c1ecdbec1a5f4a3b844e30b34f9f3f6273981eedd91",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_up/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "c92bf37d32d3ad67e7c2fad2aede3bfb7da95f4c2f2766bc7c5f53d9ed09a620",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_up/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "846290512986dac9ec0588b4d80e5185dc2d97a5d35ce8d82a515ae7a81ed145",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_up/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "3864a6fab144c86b57d03b2fc29e0936874f1edcdd38dd3b364071dbf1451461",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cpu_fp64_oracle/W_up/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "28b690056df1c29ee44ff8eafebec7e10c2c2667022f3c47a80d465fc4dee107",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_K/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "bf7bbd9c9cfcd7c493c4b4ef3d0976170f0e34bbcc48da295a95ae548f632a5b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_K/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "4736dad519d3f2cf9c3bbfb0cb1985006d5b1b782c6877f865913fab1f16db16",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_K/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "f38a4db09d816ff227f003fd600088f7e7475923b33f2f050e717902e9c358e9",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_K/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "d025754417ed85e55b8f614b2f2d543bc1d9931cbfc269d2933a96f06280c384",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_K/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "ee0dde6208bd59d873274deaed3d42a335d422f934ea0adabfdf39452635c2cf",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_K/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "240c0c478833495bbd6788a187c74832048819baa551204a3577ad0ecf96dab5",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_K/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "4bb18fda4ab93a6874572ba2141d1fa74076e4d6ee6468cd2796b5c482ecfb04",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_K/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "14a32efbf02d8b45411aad352b05e591cc1daef215a6737ef055b4a9ea13b586",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_K/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "071acbf65cea9f4fc6003b9493abb1d4b3d11a7ea187bfdae0f2b214684731d4",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_K/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "174ee1bade343ef8b2e479e4d404f164a85ebc111a7045b6d4895a6197b096ed",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_K/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "4b296d48c25181abcb4f79efe24acf8e4d1d3cc85112d06795d8e4a5d47d3e5c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_O/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "7d40b4b7ea658360265cd1f420c4f3ad9cae455f6da8d19db24267193d546330",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_O/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "9564e2249303ef46e0e1a4bc2731442188789cdad12f7a0672ca0ac899253788",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_O/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "d3e5aa301c2f4d9f5004f1cdb09af1727e7bd0af12192573136724def77fd74d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_O/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "9763505bf1afb218cd60a5a5dbd2ad4c9f91f09ae19805c787f6ab8e0c519f44",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_O/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "c9bc79287a5edc4474e7c5f52054f8dc8d84714d75da0bb137476c5cc4d1fa2e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_O/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "23d3f87beb1a817ea0b120b35c0a84b2f3d2660a8c8ca7c770b293dd966d4d8b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_O/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "743552164bd9c0c6830c460e44b2606f58d2f837b1c56d3381aeb0e8cf4b3835",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_O/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "0864a3bb7a0d5b3d0e11a0981b9f74fc03043872225c1801a9ba32677bcaf938",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_O/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "e90a0cd080704817ed192fb58b12e7484e72370348b011101cbe082d0f987997",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_O/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "e6daa5bd1f94dfc39689549b8f9408e0eee86d907f5a86e95cc58e8dae7089d6",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_O/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "9448a3dbec9988e57aebdcaae325a008635d73b6e707ff833455de9cdbc8a8b2",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_Q/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "6ff405b6c4ded696f2e5a344c3a4f987e157561a57ed7ced956061bae1bdf8cd",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_Q/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "b226052fe38415c99663f9201ca5139eb4d739a304fdfef694a469fde2fdb2df",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_Q/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "86c5f67f83a2739b501d843fad35792d1fe2b038133790ab5666091966a1102a",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_Q/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "52101e0b842566902a040c2b1e6cd537e3305d22e0860b28336af5209ff42138",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_Q/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "982487039e3e8eb99be1808ad8b6088e2b04ae7e6340d7bab6bd1587dde70fd2",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_Q/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "57d83dfd311de9215478bc03eb5b934be280b64240aee9808650a5a19fdfabca",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_Q/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "7dc86ec6db38e1c7f3ab82f9e5f7d28aba07e3a2d9e9e27f6456fed52ff78afd",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_Q/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "b967285796d03f5a939f5ea47a811d7d72fb7d4c499a2bbe0fb265d5077f76cf",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_Q/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "0834c43c66f00df4667a45366bb4cf07cf5de022561bb0dd4d7d05f1fa56cadb",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_Q/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "900922de1d91fc48842144492a33a71f58a810c8fe4c50c072684a5aa8a14e7e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_Q/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "a8c888532da25f79ba72ed43b8c59e1f3fa9b450caba0550d4874486ce5e8487",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_V/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "b5b5ea3a74737a46c737a4815905082491871bc40fb7909565e3e1c6edf6aa41",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_V/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "209d12f06ff49bec5c7a79c6f4e3eb25d62d66e7fd45e254d2cb694ce9af6390",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_V/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "0d2a1f8945ab92c55273c11f641c94029a73758211fe972bf746370faec0da7f",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_V/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "9ecad747410be5d16a04a13b385a952bd413b1250f57cd88114a52e2259e9181",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_V/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "cddaa7abf2afa0f957608cdcbb7a58ea30ff7d1208b31cafa09928811bd13f5e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_V/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "cec5d6984960a9edea52b05a874bbbf8eff97b2ce194028f25b486c7a98e50f1",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_V/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "a22d7dc627fc145392cd975f099eb95837550cbb1e932143e09717190616772d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_V/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "a13c2a85b151a0a38c8a54fba4311ef56abc589c31cb0872f5689660f7d8743c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_V/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "7cef74639851e4044469b4b7ce1ccf6ece19a3090edbcafae169c313d6b3002c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_V/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "ab0fe771e11d53cbce6a65515638b7b83c914bfdf1d3c15f70f493ba382d9a73",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_V/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "b295f5b5485541b65862cb433f248a2367857c8fce7efa5c596a22789d0d7803",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_down/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "1e67a94a741f4bf2428867bd46d7658216de51826c58a84b86a595643882e958",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cuda_fp32/W_down/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "4649e8ba3b95319fe63c531c8b806873cdfdbb9eef460b62644261d2a68c3647",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cuda_fp32/W_down/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "0229ff595b09cb6d9b955c8dc54afe568a8ab66849b6311b7e9b60933057641b",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cuda_fp32/W_down/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "849cf340eee4dc5b686965202fb4e615090b0a47c00e5229a11d222a05ad1a70",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cuda_fp32/W_down/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "e9da9e64134b5ffc0af610e355d0aea711855a9ff624749786ad4a4229ad08b5",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cuda_fp32/W_down/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "a97b6cc06951e3df22664e96507307ce30c34d9f11d7903f5a6ec4570e699a33",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cuda_fp32/W_down/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "9447570a5045c93b6bd26aaf8b5d77255009529a9cca13f049782801b8aa0b53",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cuda_fp32/W_down/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "b22c7276fdf297a4905d1bec0e1589b7828ad22e40c86594254932609263d4d4",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cuda_fp32/W_down/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "65e3dbc7bcf7a3d19646822cb8f9e937c1675e4b6a565e32d3a8100d0455a8e9",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cuda_fp32/W_down/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "6afff7a4e26b19b9eee78cedc06dc0e6642c59f8012c4722845c9efe2cba4432",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cuda_fp32/W_down/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "fd4d166176b2c52f4e8cfa98214815f1e44bfcaa8381bd81bbaf329f8120dc9c",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261001/cuda_fp32/W_gate/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "014e047b9353ae914643592152e450ca68c0ec4bb00ac7c4e1b65c2ca75f2f4d",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_gate/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "0c2a01fee088e5356603845cdabe453096425ae2744acad469743eeca21ef12f",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_gate/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "4c597bcdfc48709b3abf97c325fde6270a1f98c14a9a97f1b1ba4ab8d603b32b",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_gate/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "01e2b9ba614fa83195fbf052b7653d68e20e6b194fa708dc47d4e28033e4ee9e",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_gate/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "0a2ed877e1257fc1e458957e36c6238203579b38cc33fa00d480fbd60e04d307",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_gate/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "bb251cd70452134e8f05fe5166f8ca0a67d05458023f238c4727a3419b1a69eb",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_gate/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "fb0fda1388217f359aedcc054905cb8e713242ab4aabe5c20d2b2779f1ca0d1f",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_gate/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "b18b834f2d8d2d94899a2e80f188019bc476a6fd31d58c8da701da99c0065b28",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_gate/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "4a42d5b652677a45d3cd12c2544fa92925d28ecfe79ea1443cbfa5ceacf68bf3",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_gate/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "bd50e106c3c2438be8a18a16d184d53d770ad2f6e034281627a7df9884d5186f",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_gate/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "8555b353fae740075ab36a63261b14d7dbd89ebd9ec6cfc76a9b9ee1125a0811",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_up/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "f1f0efde052276ef44ac40db40787ddf9633d8a308a886b94e23c02e426c20a0",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_up/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "81e449c245b9fa4fe64304e5c6dff55f2e00726560884cc6a2956c94e934108b",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_up/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "f066ce54d68a6c35c9cca7681292b4694b4a2a7fe7229d9d58f22963728b8793",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_up/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "f72d1a585e920d8e779330f8f6f10a329e820bc59513d91aa6f09dd2c09cb7d8",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_up/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "17ac5ba1c1437a3811e162a869135b7192de78e6178799811de1204a0d609bac",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_up/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "60dc570220a2714b234484212833ab5ff321f9012f82d04b128d4431962f465d",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_up/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "ec636a2dbd68c8b48a48264c087171d4c995998bab950bdc5a0a313853d2e069",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_up/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "fb5f7150b7a046178a414b6db4868842fb0e1f2f8a74052638daa5a4fdd351d3",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_up/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "98c11bcf6e15ca538af4e51828cf23036e72c7f422b8c37b12e523bcc609b1df",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_up/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "b6a2e56d095f14c2d8e08d747d59049da0ad66bab5a622b110127cf26c034f4f",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/cuda_fp32/W_up/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "6d71114c3b6b3aebabfa45dcdb8440d3ced2f7131d9335a67ee60fe7fc8456fc",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261001/inputs/w_fp32_cpu": {
      "dtype": "torch.float32",
      "raw_sha256": "8700ea8868c318145c9c3fe14e8b0eedd381099fb677866b9a25a896c57b4d5b",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261001/inputs/x_fp32_cpu": {
      "dtype": "torch.float32",
      "raw_sha256": "bebd6a57cf82c695c3b222ddde3f53c185896e2871e49debf483599369f85591",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261001/structural_traces/R4/round_1": {
      "dtype": "torch.float32",
      "raw_sha256": "3012fd933265ed64938c252be8f4bf63d27dc5cec1386e9ff9f297404842757e",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261001/structural_traces/R4/round_2": {
      "dtype": "torch.float32",
      "raw_sha256": "6725c3ac7ac3e18cf191ff13fa9c0f8547916383c2c618e45d4a8ba64701dc6c",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261001/structural_traces/R4/round_3": {
      "dtype": "torch.float32",
      "raw_sha256": "92e1bf1dba5f44ae3f80825e01acfd29a0a66cf38d280e07e7ff40b7512ce00e",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261001/structural_traces/R4/round_4": {
      "dtype": "torch.float32",
      "raw_sha256": "936a488353474609719764d255dc49c0274c326b95894300acad161ed1681180",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261001/structural_traces/R4_final": {
      "dtype": "torch.float32",
      "raw_sha256": "20db9aa804088311fae6cbe309c8cfe504149122d657725dde54f63c514b51a3",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261001/structural_traces/U4/round_1": {
      "dtype": "torch.float32",
      "raw_sha256": "f87bf9cbdac71be4d2ce74c3ea583e052eeecc4806a46d65055a079503db38f4",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261001/structural_traces/U4/round_2": {
      "dtype": "torch.float32",
      "raw_sha256": "e26bf8ef42d7850d71a598a39782c3a7cb7424c95e64bd3bc291c85984159ffa",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261001/structural_traces/U4/round_3": {
      "dtype": "torch.float32",
      "raw_sha256": "498cc2d870312829c7cf3dd908ed158fe1e9e09a6c46bb4ade2ca20aaa0efa04",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261001/structural_traces/U4/round_4": {
      "dtype": "torch.float32",
      "raw_sha256": "a7c65644a6160041d9b55ed5705b99c84eefec95fad176ac02b5b2b65327eb9b",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261001/structural_traces/U4_final": {
      "dtype": "torch.float32",
      "raw_sha256": "a9a87c5e11b204e644737579b84375d9e21c4597854b54160b404fa5517ce8ef",
      "shape": [
        8,
        8,
        256
      ]
    }
  },
  "tensor_raw_sha256": {
    "seed_20261001/cpu_fp64_oracle/W_K/S64_cpu": "ffc3b8458b68570b6398159ccfd35ad0a54f124cd0a0f44dd85dfe4de35d4c30",
    "seed_20261001/cpu_fp64_oracle/W_K/gR64_cpu": "55a0d66b8f827b1096400cab5b8fdca9133d73552eab9c601dc9c626f1f97c4d",
    "seed_20261001/cpu_fp64_oracle/W_K/gU0_64_cpu": "0c0275f854ff3b58f3bf7dcf15b34633de5404515e4e49cb5aa11cf84abc50ce",
    "seed_20261001/cpu_fp64_oracle/W_K/gU1_64_cpu": "c7efcbba113e43004543245571eb100d769475c98d78ca65b01b4e24d8fdbf0c",
    "seed_20261001/cpu_fp64_oracle/W_K/gU2_64_cpu": "7e1a0c097d22d8d299b41dbdb59387bdb077bfd4f222f9335231f16a02153310",
    "seed_20261001/cpu_fp64_oracle/W_K/gU3_64_cpu": "550538f9d09c3c727f9be39d66b6c077b3e09e287e0dac9d57eab57c0de40d79",
    "seed_20261001/cpu_fp64_oracle/W_O/S64_cpu": "1a1de703c4c9c3b6a70e2f78bf6d7b5f3aca727eabf8c56c3c2d79195131d53b",
    "seed_20261001/cpu_fp64_oracle/W_O/gR64_cpu": "059a2f62406a4e44da539ac6b286d27dabf13b388a535ff14f897a8800e158be",
    "seed_20261001/cpu_fp64_oracle/W_O/gU0_64_cpu": "0766abfac816c7ac36b68ec0642f672fd6e9d61736470a1d5671b7bc8c9c3ee0",
    "seed_20261001/cpu_fp64_oracle/W_O/gU1_64_cpu": "1983f55359c9806d98122dc4efe4a3f5954cc8b38e348ae55d07266bafaa0800",
    "seed_20261001/cpu_fp64_oracle/W_O/gU2_64_cpu": "79181f0bf49c338fef560fe332a931be9999854aa8b74bd12d71eb965f38fd3d",
    "seed_20261001/cpu_fp64_oracle/W_O/gU3_64_cpu": "5f0b931fedf75d1826a5de74bdd6869137369de7b4a1ac1428f1338c1bd78e9a",
    "seed_20261001/cpu_fp64_oracle/W_Q/S64_cpu": "af7670d6e8badd2b20d76e752822e87a8e9e1d4880bc82bb35227dd79ded9e59",
    "seed_20261001/cpu_fp64_oracle/W_Q/gR64_cpu": "ac81107fe5bb05253fb9b4e5798ce64813d9c13a258518c66055d6bef71902a9",
    "seed_20261001/cpu_fp64_oracle/W_Q/gU0_64_cpu": "94c434749ecb5df95ff2b377d13212760f8690f607383e6687902800be6811f0",
    "seed_20261001/cpu_fp64_oracle/W_Q/gU1_64_cpu": "e24419977612a76146159db7db246547ea53422929f8e7c9ddce159e4be0553c",
    "seed_20261001/cpu_fp64_oracle/W_Q/gU2_64_cpu": "32e9a5591fc9f24242dad433203b415e650d99a4d0e02cbf5517ed8d1b0c5f83",
    "seed_20261001/cpu_fp64_oracle/W_Q/gU3_64_cpu": "01486a86397ebc84ab04e29a0a8fbe9e31a10263185979b4f2ea6bc989d1a252",
    "seed_20261001/cpu_fp64_oracle/W_V/S64_cpu": "40c0852c846f7dac419ece8a50cff4098612e3f09304d842cccb2c8e7a1e6ba7",
    "seed_20261001/cpu_fp64_oracle/W_V/gR64_cpu": "713a0ac00fadadb093d9f9e65b3221e0c0179b4b39edc0774dae5a8f4d947324",
    "seed_20261001/cpu_fp64_oracle/W_V/gU0_64_cpu": "b7093854a8193add3fe2d00a4f73aab1e68adac84ee66529dcf1dc5e31d563cc",
    "seed_20261001/cpu_fp64_oracle/W_V/gU1_64_cpu": "b80081d47cbacd5c25a28f0ca0b631b8453082deb24c8c6db63efbbbb5a73c0f",
    "seed_20261001/cpu_fp64_oracle/W_V/gU2_64_cpu": "956dc1fb111e015fd1f97f22cefc8f5b630868b30c206cacc81c9bacf83b8711",
    "seed_20261001/cpu_fp64_oracle/W_V/gU3_64_cpu": "c59644de21513f887598335a8b2ce89205b282ed174185a1f7f44c5611e9edaa",
    "seed_20261001/cpu_fp64_oracle/W_down/S64_cpu": "71e8c1242ddf314f53bc6304a4ac9be49befe2197cc46167739e5f8385f8e774",
    "seed_20261001/cpu_fp64_oracle/W_down/gR64_cpu": "1e54f6ad76dd2d45afd42aa878ea89a423e04809265f44a89439e475bcd46ce3",
    "seed_20261001/cpu_fp64_oracle/W_down/gU0_64_cpu": "20a245afb1a9936129deeb68789aaefeda269f44705e1e41e93ba9ed71eb6fa4",
    "seed_20261001/cpu_fp64_oracle/W_down/gU1_64_cpu": "fda5eec8178237ce15e1f63a6d8653063279d9cb5aaafbe260e789845ffa247d",
    "seed_20261001/cpu_fp64_oracle/W_down/gU2_64_cpu": "de59fa87dc676becc25be485e4b8174246adbf26f314b73974c8ba87556d20bd",
    "seed_20261001/cpu_fp64_oracle/W_down/gU3_64_cpu": "ddab034b21ec704d2562df3b67fab2480511353d1a9cc0bd62f20cbed6947772",
    "seed_20261001/cpu_fp64_oracle/W_gate/S64_cpu": "223d1e6ddec01178295e99654a93c4eefb88901477e8430fa84503045017999f",
    "seed_20261001/cpu_fp64_oracle/W_gate/gR64_cpu": "4f6d62cd6e8f597516b2d333cbfebfd0e2e7f424d6a5461bf3b2c271b86f43ad",
    "seed_20261001/cpu_fp64_oracle/W_gate/gU0_64_cpu": "36c5a1d6f8b3a9b1fc1c5dcb0adea2b439fcf2e783df9a45ff78a54c14898465",
    "seed_20261001/cpu_fp64_oracle/W_gate/gU1_64_cpu": "6065ef54c81707b227836a6337f5e73f6a8a379b5d73053839265f734e76cefb",
    "seed_20261001/cpu_fp64_oracle/W_gate/gU2_64_cpu": "38a9bc7e320b306c76a86b40350783207516bfbd46530d71fe41dae34c8298d3",
    "seed_20261001/cpu_fp64_oracle/W_gate/gU3_64_cpu": "eeb457d442d2543f7be384bafbe2e30fe1163b78217b2b243754d9e0673c62c7",
    "seed_20261001/cpu_fp64_oracle/W_up/S64_cpu": "141f7e5e3f6131a63e6cf8a97258f52a89ae976ec32d9bd984f0393e4146a355",
    "seed_20261001/cpu_fp64_oracle/W_up/gR64_cpu": "2ca37f333e415b9cf09c0c1ecdbec1a5f4a3b844e30b34f9f3f6273981eedd91",
    "seed_20261001/cpu_fp64_oracle/W_up/gU0_64_cpu": "c92bf37d32d3ad67e7c2fad2aede3bfb7da95f4c2f2766bc7c5f53d9ed09a620",
    "seed_20261001/cpu_fp64_oracle/W_up/gU1_64_cpu": "846290512986dac9ec0588b4d80e5185dc2d97a5d35ce8d82a515ae7a81ed145",
    "seed_20261001/cpu_fp64_oracle/W_up/gU2_64_cpu": "3864a6fab144c86b57d03b2fc29e0936874f1edcdd38dd3b364071dbf1451461",
    "seed_20261001/cpu_fp64_oracle/W_up/gU3_64_cpu": "28b690056df1c29ee44ff8eafebec7e10c2c2667022f3c47a80d465fc4dee107",
    "seed_20261001/cuda_fp32/W_K/S64_primary_cuda": "bf7bbd9c9cfcd7c493c4b4ef3d0976170f0e34bbcc48da295a95ae548f632a5b",
    "seed_20261001/cuda_fp32/W_K/S_forward_cuda": "4736dad519d3f2cf9c3bbfb0cb1985006d5b1b782c6877f865913fab1f16db16",
    "seed_20261001/cuda_fp32/W_K/S_pairwise_cuda": "f38a4db09d816ff227f003fd600088f7e7475923b33f2f050e717902e9c358e9",
    "seed_20261001/cuda_fp32/W_K/S_reverse_cuda": "d025754417ed85e55b8f614b2f2d543bc1d9931cbfc269d2933a96f06280c384",
    "seed_20261001/cuda_fp32/W_K/S_stack_cuda": "ee0dde6208bd59d873274deaed3d42a335d422f934ea0adabfdf39452635c2cf",
    "seed_20261001/cuda_fp32/W_K/gR64_primary_cuda": "240c0c478833495bbd6788a187c74832048819baa551204a3577ad0ecf96dab5",
    "seed_20261001/cuda_fp32/W_K/gR_cuda_fp32": "4bb18fda4ab93a6874572ba2141d1fa74076e4d6ee6468cd2796b5c482ecfb04",
    "seed_20261001/cuda_fp32/W_K/gU0_cuda_fp32": "14a32efbf02d8b45411aad352b05e591cc1daef215a6737ef055b4a9ea13b586",
    "seed_20261001/cuda_fp32/W_K/gU1_cuda_fp32": "071acbf65cea9f4fc6003b9493abb1d4b3d11a7ea187bfdae0f2b214684731d4",
    "seed_20261001/cuda_fp32/W_K/gU2_cuda_fp32": "174ee1bade343ef8b2e479e4d404f164a85ebc111a7045b6d4895a6197b096ed",
    "seed_20261001/cuda_fp32/W_K/gU3_cuda_fp32": "4b296d48c25181abcb4f79efe24acf8e4d1d3cc85112d06795d8e4a5d47d3e5c",
    "seed_20261001/cuda_fp32/W_O/S64_primary_cuda": "7d40b4b7ea658360265cd1f420c4f3ad9cae455f6da8d19db24267193d546330",
    "seed_20261001/cuda_fp32/W_O/S_forward_cuda": "9564e2249303ef46e0e1a4bc2731442188789cdad12f7a0672ca0ac899253788",
    "seed_20261001/cuda_fp32/W_O/S_pairwise_cuda": "d3e5aa301c2f4d9f5004f1cdb09af1727e7bd0af12192573136724def77fd74d",
    "seed_20261001/cuda_fp32/W_O/S_reverse_cuda": "9763505bf1afb218cd60a5a5dbd2ad4c9f91f09ae19805c787f6ab8e0c519f44",
    "seed_20261001/cuda_fp32/W_O/S_stack_cuda": "c9bc79287a5edc4474e7c5f52054f8dc8d84714d75da0bb137476c5cc4d1fa2e",
    "seed_20261001/cuda_fp32/W_O/gR64_primary_cuda": "23d3f87beb1a817ea0b120b35c0a84b2f3d2660a8c8ca7c770b293dd966d4d8b",
    "seed_20261001/cuda_fp32/W_O/gR_cuda_fp32": "743552164bd9c0c6830c460e44b2606f58d2f837b1c56d3381aeb0e8cf4b3835",
    "seed_20261001/cuda_fp32/W_O/gU0_cuda_fp32": "0864a3bb7a0d5b3d0e11a0981b9f74fc03043872225c1801a9ba32677bcaf938",
    "seed_20261001/cuda_fp32/W_O/gU1_cuda_fp32": "e90a0cd080704817ed192fb58b12e7484e72370348b011101cbe082d0f987997",
    "seed_20261001/cuda_fp32/W_O/gU2_cuda_fp32": "e6daa5bd1f94dfc39689549b8f9408e0eee86d907f5a86e95cc58e8dae7089d6",
    "seed_20261001/cuda_fp32/W_O/gU3_cuda_fp32": "9448a3dbec9988e57aebdcaae325a008635d73b6e707ff833455de9cdbc8a8b2",
    "seed_20261001/cuda_fp32/W_Q/S64_primary_cuda": "6ff405b6c4ded696f2e5a344c3a4f987e157561a57ed7ced956061bae1bdf8cd",
    "seed_20261001/cuda_fp32/W_Q/S_forward_cuda": "b226052fe38415c99663f9201ca5139eb4d739a304fdfef694a469fde2fdb2df",
    "seed_20261001/cuda_fp32/W_Q/S_pairwise_cuda": "86c5f67f83a2739b501d843fad35792d1fe2b038133790ab5666091966a1102a",
    "seed_20261001/cuda_fp32/W_Q/S_reverse_cuda": "52101e0b842566902a040c2b1e6cd537e3305d22e0860b28336af5209ff42138",
    "seed_20261001/cuda_fp32/W_Q/S_stack_cuda": "982487039e3e8eb99be1808ad8b6088e2b04ae7e6340d7bab6bd1587dde70fd2",
    "seed_20261001/cuda_fp32/W_Q/gR64_primary_cuda": "57d83dfd311de9215478bc03eb5b934be280b64240aee9808650a5a19fdfabca",
    "seed_20261001/cuda_fp32/W_Q/gR_cuda_fp32": "7dc86ec6db38e1c7f3ab82f9e5f7d28aba07e3a2d9e9e27f6456fed52ff78afd",
    "seed_20261001/cuda_fp32/W_Q/gU0_cuda_fp32": "b967285796d03f5a939f5ea47a811d7d72fb7d4c499a2bbe0fb265d5077f76cf",
    "seed_20261001/cuda_fp32/W_Q/gU1_cuda_fp32": "0834c43c66f00df4667a45366bb4cf07cf5de022561bb0dd4d7d05f1fa56cadb",
    "seed_20261001/cuda_fp32/W_Q/gU2_cuda_fp32": "900922de1d91fc48842144492a33a71f58a810c8fe4c50c072684a5aa8a14e7e",
    "seed_20261001/cuda_fp32/W_Q/gU3_cuda_fp32": "a8c888532da25f79ba72ed43b8c59e1f3fa9b450caba0550d4874486ce5e8487",
    "seed_20261001/cuda_fp32/W_V/S64_primary_cuda": "b5b5ea3a74737a46c737a4815905082491871bc40fb7909565e3e1c6edf6aa41",
    "seed_20261001/cuda_fp32/W_V/S_forward_cuda": "209d12f06ff49bec5c7a79c6f4e3eb25d62d66e7fd45e254d2cb694ce9af6390",
    "seed_20261001/cuda_fp32/W_V/S_pairwise_cuda": "0d2a1f8945ab92c55273c11f641c94029a73758211fe972bf746370faec0da7f",
    "seed_20261001/cuda_fp32/W_V/S_reverse_cuda": "9ecad747410be5d16a04a13b385a952bd413b1250f57cd88114a52e2259e9181",
    "seed_20261001/cuda_fp32/W_V/S_stack_cuda": "cddaa7abf2afa0f957608cdcbb7a58ea30ff7d1208b31cafa09928811bd13f5e",
    "seed_20261001/cuda_fp32/W_V/gR64_primary_cuda": "cec5d6984960a9edea52b05a874bbbf8eff97b2ce194028f25b486c7a98e50f1",
    "seed_20261001/cuda_fp32/W_V/gR_cuda_fp32": "a22d7dc627fc145392cd975f099eb95837550cbb1e932143e09717190616772d",
    "seed_20261001/cuda_fp32/W_V/gU0_cuda_fp32": "a13c2a85b151a0a38c8a54fba4311ef56abc589c31cb0872f5689660f7d8743c",
    "seed_20261001/cuda_fp32/W_V/gU1_cuda_fp32": "7cef74639851e4044469b4b7ce1ccf6ece19a3090edbcafae169c313d6b3002c",
    "seed_20261001/cuda_fp32/W_V/gU2_cuda_fp32": "ab0fe771e11d53cbce6a65515638b7b83c914bfdf1d3c15f70f493ba382d9a73",
    "seed_20261001/cuda_fp32/W_V/gU3_cuda_fp32": "b295f5b5485541b65862cb433f248a2367857c8fce7efa5c596a22789d0d7803",
    "seed_20261001/cuda_fp32/W_down/S64_primary_cuda": "1e67a94a741f4bf2428867bd46d7658216de51826c58a84b86a595643882e958",
    "seed_20261001/cuda_fp32/W_down/S_forward_cuda": "4649e8ba3b95319fe63c531c8b806873cdfdbb9eef460b62644261d2a68c3647",
    "seed_20261001/cuda_fp32/W_down/S_pairwise_cuda": "0229ff595b09cb6d9b955c8dc54afe568a8ab66849b6311b7e9b60933057641b",
    "seed_20261001/cuda_fp32/W_down/S_reverse_cuda": "849cf340eee4dc5b686965202fb4e615090b0a47c00e5229a11d222a05ad1a70",
    "seed_20261001/cuda_fp32/W_down/S_stack_cuda": "e9da9e64134b5ffc0af610e355d0aea711855a9ff624749786ad4a4229ad08b5",
    "seed_20261001/cuda_fp32/W_down/gR64_primary_cuda": "a97b6cc06951e3df22664e96507307ce30c34d9f11d7903f5a6ec4570e699a33",
    "seed_20261001/cuda_fp32/W_down/gR_cuda_fp32": "9447570a5045c93b6bd26aaf8b5d77255009529a9cca13f049782801b8aa0b53",
    "seed_20261001/cuda_fp32/W_down/gU0_cuda_fp32": "b22c7276fdf297a4905d1bec0e1589b7828ad22e40c86594254932609263d4d4",
    "seed_20261001/cuda_fp32/W_down/gU1_cuda_fp32": "65e3dbc7bcf7a3d19646822cb8f9e937c1675e4b6a565e32d3a8100d0455a8e9",
    "seed_20261001/cuda_fp32/W_down/gU2_cuda_fp32": "6afff7a4e26b19b9eee78cedc06dc0e6642c59f8012c4722845c9efe2cba4432",
    "seed_20261001/cuda_fp32/W_down/gU3_cuda_fp32": "fd4d166176b2c52f4e8cfa98214815f1e44bfcaa8381bd81bbaf329f8120dc9c",
    "seed_20261001/cuda_fp32/W_gate/S64_primary_cuda": "014e047b9353ae914643592152e450ca68c0ec4bb00ac7c4e1b65c2ca75f2f4d",
    "seed_20261001/cuda_fp32/W_gate/S_forward_cuda": "0c2a01fee088e5356603845cdabe453096425ae2744acad469743eeca21ef12f",
    "seed_20261001/cuda_fp32/W_gate/S_pairwise_cuda": "4c597bcdfc48709b3abf97c325fde6270a1f98c14a9a97f1b1ba4ab8d603b32b",
    "seed_20261001/cuda_fp32/W_gate/S_reverse_cuda": "01e2b9ba614fa83195fbf052b7653d68e20e6b194fa708dc47d4e28033e4ee9e",
    "seed_20261001/cuda_fp32/W_gate/S_stack_cuda": "0a2ed877e1257fc1e458957e36c6238203579b38cc33fa00d480fbd60e04d307",
    "seed_20261001/cuda_fp32/W_gate/gR64_primary_cuda": "bb251cd70452134e8f05fe5166f8ca0a67d05458023f238c4727a3419b1a69eb",
    "seed_20261001/cuda_fp32/W_gate/gR_cuda_fp32": "fb0fda1388217f359aedcc054905cb8e713242ab4aabe5c20d2b2779f1ca0d1f",
    "seed_20261001/cuda_fp32/W_gate/gU0_cuda_fp32": "b18b834f2d8d2d94899a2e80f188019bc476a6fd31d58c8da701da99c0065b28",
    "seed_20261001/cuda_fp32/W_gate/gU1_cuda_fp32": "4a42d5b652677a45d3cd12c2544fa92925d28ecfe79ea1443cbfa5ceacf68bf3",
    "seed_20261001/cuda_fp32/W_gate/gU2_cuda_fp32": "bd50e106c3c2438be8a18a16d184d53d770ad2f6e034281627a7df9884d5186f",
    "seed_20261001/cuda_fp32/W_gate/gU3_cuda_fp32": "8555b353fae740075ab36a63261b14d7dbd89ebd9ec6cfc76a9b9ee1125a0811",
    "seed_20261001/cuda_fp32/W_up/S64_primary_cuda": "f1f0efde052276ef44ac40db40787ddf9633d8a308a886b94e23c02e426c20a0",
    "seed_20261001/cuda_fp32/W_up/S_forward_cuda": "81e449c245b9fa4fe64304e5c6dff55f2e00726560884cc6a2956c94e934108b",
    "seed_20261001/cuda_fp32/W_up/S_pairwise_cuda": "f066ce54d68a6c35c9cca7681292b4694b4a2a7fe7229d9d58f22963728b8793",
    "seed_20261001/cuda_fp32/W_up/S_reverse_cuda": "f72d1a585e920d8e779330f8f6f10a329e820bc59513d91aa6f09dd2c09cb7d8",
    "seed_20261001/cuda_fp32/W_up/S_stack_cuda": "17ac5ba1c1437a3811e162a869135b7192de78e6178799811de1204a0d609bac",
    "seed_20261001/cuda_fp32/W_up/gR64_primary_cuda": "60dc570220a2714b234484212833ab5ff321f9012f82d04b128d4431962f465d",
    "seed_20261001/cuda_fp32/W_up/gR_cuda_fp32": "ec636a2dbd68c8b48a48264c087171d4c995998bab950bdc5a0a313853d2e069",
    "seed_20261001/cuda_fp32/W_up/gU0_cuda_fp32": "fb5f7150b7a046178a414b6db4868842fb0e1f2f8a74052638daa5a4fdd351d3",
    "seed_20261001/cuda_fp32/W_up/gU1_cuda_fp32": "98c11bcf6e15ca538af4e51828cf23036e72c7f422b8c37b12e523bcc609b1df",
    "seed_20261001/cuda_fp32/W_up/gU2_cuda_fp32": "b6a2e56d095f14c2d8e08d747d59049da0ad66bab5a622b110127cf26c034f4f",
    "seed_20261001/cuda_fp32/W_up/gU3_cuda_fp32": "6d71114c3b6b3aebabfa45dcdb8440d3ced2f7131d9335a67ee60fe7fc8456fc",
    "seed_20261001/inputs/w_fp32_cpu": "8700ea8868c318145c9c3fe14e8b0eedd381099fb677866b9a25a896c57b4d5b",
    "seed_20261001/inputs/x_fp32_cpu": "bebd6a57cf82c695c3b222ddde3f53c185896e2871e49debf483599369f85591",
    "seed_20261001/structural_traces/R4/round_1": "3012fd933265ed64938c252be8f4bf63d27dc5cec1386e9ff9f297404842757e",
    "seed_20261001/structural_traces/R4/round_2": "6725c3ac7ac3e18cf191ff13fa9c0f8547916383c2c618e45d4a8ba64701dc6c",
    "seed_20261001/structural_traces/R4/round_3": "92e1bf1dba5f44ae3f80825e01acfd29a0a66cf38d280e07e7ff40b7512ce00e",
    "seed_20261001/structural_traces/R4/round_4": "936a488353474609719764d255dc49c0274c326b95894300acad161ed1681180",
    "seed_20261001/structural_traces/R4_final": "20db9aa804088311fae6cbe309c8cfe504149122d657725dde54f63c514b51a3",
    "seed_20261001/structural_traces/U4/round_1": "f87bf9cbdac71be4d2ce74c3ea583e052eeecc4806a46d65055a079503db38f4",
    "seed_20261001/structural_traces/U4/round_2": "e26bf8ef42d7850d71a598a39782c3a7cb7424c95e64bd3bc291c85984159ffa",
    "seed_20261001/structural_traces/U4/round_3": "498cc2d870312829c7cf3dd908ed158fe1e9e09a6c46bb4ade2ca20aaa0efa04",
    "seed_20261001/structural_traces/U4/round_4": "a7c65644a6160041d9b55ed5705b99c84eefec95fad176ac02b5b2b65327eb9b",
    "seed_20261001/structural_traces/U4_final": "a9a87c5e11b204e644737579b84375d9e21c4597854b54160b404fa5517ce8ef"
  }
}
```

## Seed seed_20261002

### Structural trace checks

```json
{
  "final_output_equal": true,
  "initial_clone_report": {
    "initial_values_bitwise_equal": true,
    "r4_reuses_one_block_object": true,
    "u4_block_storages_pairwise_disjoint": true,
    "u4_has_four_block_objects": true,
    "u4_storage_disjoint_from_r4": true
  },
  "input_copy_bitwise_equal": true,
  "loss_weight_copy_bitwise_equal": true,
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
  ],
  "weight_copy_report": {
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
    ]
  }
}
```

### Per-family scientific and oracle decisions

```json
{
  "W_K": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 1.0017575271274672e-16
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.1490999786530455e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 1.0017575271274672e-16,
        "E_inf": 1.1490999786530455e-16,
        "max_abs": 7.105427357601002e-15,
        "max_rel_old": 1.37025819439933e-12,
        "norm_S64_inf": 61.83471838481676,
        "norm_S64_l2": 3289.1758673378054,
        "norm_gR64_inf": 61.83471838481676,
        "norm_gR64_l2": 3289.1758673378054,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 2.384185791015625e-07,
        "abs_error_over_ulp_gR": 8192.0,
        "abs_error_over_ulp_sum_gU_old": 8192.0,
        "argmax_flat_index": 2449,
        "argmax_multi_index_row_major": [
          9,
          145
        ],
        "floor_1e_6_active": false,
        "gR_value": 0.000324249267578125,
        "max_rel_old": 0.000735294132027775,
        "old_denominator": 0.000324249267578125,
        "sum_gU_old_value": 0.00032401084899902344,
        "ulp_gR": 2.9103830456733704e-11,
        "ulp_sum_gU_old": 2.9103830456733704e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.432434369355692e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 6.169183883907804e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 3.0517578125e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.814697265625e-06
        }
      },
      "metrics": {
        "E_L2": 3.432434369355692e-08,
        "E_inf": 6.169183883907804e-08,
        "max_abs": 3.814697265625e-06,
        "max_rel_old": 0.0007352941176470588,
        "norm_S64_inf": 61.83471488952637,
        "norm_S64_l2": 3289.175758505272,
        "norm_gR64_inf": 61.834712982177734,
        "norm_gR64_l2": 3289.1757595051513,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 61.834716796875,
        "M64": 61.83471488952637,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_O": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.955393335218661e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.8774659474649771e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.955393335218661e-17,
        "E_inf": 1.8774659474649771e-16,
        "max_abs": 2.842170943040401e-14,
        "max_rel_old": 3.639120199260065e-13,
        "norm_S64_inf": 151.38335514835853,
        "norm_S64_l2": 6315.924740252597,
        "norm_gR64_inf": 151.38335514835853,
        "norm_gR64_l2": 6315.924740252597,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 4.76837158203125e-07,
        "abs_error_over_ulp_gR": 16384.0,
        "abs_error_over_ulp_sum_gU_old": 16384.0,
        "argmax_flat_index": 30668,
        "argmax_multi_index_row_major": [
          119,
          204
        ],
        "floor_1e_6_active": false,
        "gR_value": 0.000255584716796875,
        "max_rel_old": 0.0018656715983524919,
        "old_denominator": 0.000255584716796875,
        "sum_gU_old_value": 0.0002551078796386719,
        "ulp_gR": 2.9103830456733704e-11,
        "ulp_sum_gU_old": 2.9103830456733704e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.738233399657064e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 8.819623763669866e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 0.0001220703125,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 1.33514404296875e-05
        }
      },
      "metrics": {
        "E_L2": 3.738233399657064e-08,
        "E_inf": 8.819623763669866e-08,
        "max_abs": 1.33514404296875e-05,
        "max_rel_old": 0.0011660447761194029,
        "norm_S64_inf": 151.38333320617676,
        "norm_S64_l2": 6315.924785309424,
        "norm_gR64_inf": 151.38333129882812,
        "norm_gR64_l2": 6315.92478377171,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 151.38333129882812,
        "M64": 151.38333320617676,
        "ULP_M32": 1.52587890625e-05
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_Q": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.99716398099758e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.1824908740701836e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.99716398099758e-17,
        "E_inf": 2.1824908740701836e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 7.530437330170496e-13,
        "norm_S64_inf": 65.11300864548319,
        "norm_S64_l2": 3315.650651311441,
        "norm_gR64_inf": 65.11300864548319,
        "norm_gR64_l2": 3315.650651311441,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 1.4901161193847656e-07,
        "abs_error_over_ulp_gR": 10240.0,
        "abs_error_over_ulp_sum_gU_old": 10240.0,
        "argmax_flat_index": 44885,
        "argmax_multi_index_row_major": [
          175,
          85
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.00024080276489257812,
        "max_rel_old": 0.0006184292142279446,
        "old_denominator": 0.0002409517765045166,
        "sum_gU_old_value": -0.0002409517765045166,
        "ulp_gR": 1.4551915228366852e-11,
        "ulp_sum_gU_old": 1.4551915228366852e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.429421005508923e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.8585808876687416e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 6.103515625e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.814697265625e-06
        }
      },
      "metrics": {
        "E_L2": 3.429421005508923e-08,
        "E_inf": 5.8585808876687416e-08,
        "max_abs": 3.814697265625e-06,
        "max_rel_old": 0.0006184291898577613,
        "norm_S64_inf": 65.11299133300781,
        "norm_S64_l2": 3315.650559158578,
        "norm_gR64_inf": 65.11299133300781,
        "norm_gR64_l2": 3315.650560072383,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 65.11299133300781,
        "M64": 65.11299133300781,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_V": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.996027174721141e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.0815413895241348e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.996027174721141e-17,
        "E_inf": 2.0815413895241348e-16,
        "max_abs": 2.842170943040401e-14,
        "max_rel_old": 5.999331468445374e-12,
        "norm_S64_inf": 136.5416492482119,
        "norm_S64_l2": 6441.887269621523,
        "norm_gR64_inf": 136.54164924821188,
        "norm_gR64_l2": 6441.887269621522,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 2.9802322387695312e-08,
        "abs_error_over_ulp_gR": 4096.0,
        "abs_error_over_ulp_sum_gU_old": 4096.0,
        "argmax_flat_index": 18276,
        "argmax_multi_index_row_major": [
          71,
          100
        ],
        "floor_1e_6_active": false,
        "gR_value": 0.00011491775512695312,
        "max_rel_old": 0.0002593361132312566,
        "old_denominator": 0.00011491775512695312,
        "sum_gU_old_value": 0.00011488795280456543,
        "ulp_gR": 7.275957614183426e-12,
        "ulp_sum_gU_old": 7.275957614183426e-12
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.733403544617331e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 6.984494213311622e-08
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
        "E_L2": 3.733403544617331e-08,
        "E_inf": 6.984494213311622e-08,
        "max_abs": 9.5367431640625e-06,
        "max_rel_old": 0.0009848335631278313,
        "norm_S64_inf": 136.5416431427002,
        "norm_S64_l2": 6441.887326096472,
        "norm_gR64_inf": 136.54164123535156,
        "norm_gR64_l2": 6441.887326476637,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 136.54164123535156,
        "M64": 136.5416431427002,
        "ULP_M32": 1.52587890625e-05
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_down": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.863653452780742e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.1869892588526417e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.863653452780742e-17,
        "E_inf": 2.1869892588526417e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 9.50737153006248e-12,
        "norm_S64_inf": 64.97907869313191,
        "norm_S64_l2": 4193.9751651318165,
        "norm_gR64_inf": 64.9790786931319,
        "norm_gR64_l2": 4193.975165131816,
        "shape": [
          256,
          1024
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 5.960464477539063e-08,
        "abs_error_over_ulp_gR": 8192.0,
        "abs_error_over_ulp_sum_gU_old": 8192.0,
        "argmax_flat_index": 59724,
        "argmax_multi_index_row_major": [
          58,
          332
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.00011610984802246094,
        "max_rel_old": 0.0005133470403961837,
        "old_denominator": 0.00011610984802246094,
        "sum_gU_old_value": -0.00011605024337768555,
        "ulp_gR": 7.275957614183426e-12,
        "ulp_sum_gU_old": 7.275957614183426e-12
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.654293541494326e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 7.338318066892116e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 6.103515625e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 4.76837158203125e-06
        }
      },
      "metrics": {
        "E_L2": 3.654293541494326e-08,
        "E_inf": 7.338318066892116e-08,
        "max_abs": 4.76837158203125e-06,
        "max_rel_old": 0.0011621900826446281,
        "norm_S64_inf": 64.979079246521,
        "norm_S64_l2": 4193.975242637554,
        "norm_gR64_inf": 64.97908020019531,
        "norm_gR64_l2": 4193.975241586208,
        "shape": [
          256,
          1024
        ]
      },
      "scale_context": {
        "M32": 64.97908020019531,
        "M64": 64.97908020019531,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_gate": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.908494457179228e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.4143088303249235e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.908494457179228e-17,
        "E_inf": 2.4143088303249235e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 1.118987725093862e-12,
        "norm_S64_inf": 58.8609648306239,
        "norm_S64_l2": 4181.6653115098015,
        "norm_gR64_inf": 58.860964830623885,
        "norm_gR64_l2": 4181.6653115098015,
        "shape": [
          1024,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 2.384185791015625e-07,
        "abs_error_over_ulp_gR": 32768.0,
        "abs_error_over_ulp_sum_gU_old": 32768.0,
        "argmax_flat_index": 69707,
        "argmax_multi_index_row_major": [
          272,
          75
        ],
        "floor_1e_6_active": false,
        "gR_value": 7.009506225585938e-05,
        "max_rel_old": 0.003401360474526882,
        "old_denominator": 7.009506225585938e-05,
        "sum_gU_old_value": 6.985664367675781e-05,
        "ulp_gR": 7.275957614183426e-12,
        "ulp_sum_gU_old": 7.275957614183426e-12
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.6566800145852484e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.6707532166618525e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 3.0517578125e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.337860107421875e-06
        }
      },
      "metrics": {
        "E_L2": 3.6566800145852484e-08,
        "E_inf": 5.6707532166618525e-08,
        "max_abs": 3.337860107421875e-06,
        "max_rel_old": 0.001488095238095238,
        "norm_S64_inf": 58.86096477508545,
        "norm_S64_l2": 4181.6653943639585,
        "norm_gR64_inf": 58.860965728759766,
        "norm_gR64_l2": 4181.665395474566,
        "shape": [
          1024,
          256
        ]
      },
      "scale_context": {
        "M32": 58.860965728759766,
        "M64": 58.860965728759766,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_up": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.978993922430235e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.990504495398211e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.978993922430235e-17,
        "E_inf": 1.990504495398211e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 3.4862317534358667e-12,
        "norm_S64_inf": 71.39323095253319,
        "norm_S64_l2": 4172.57586942892,
        "norm_gR64_inf": 71.3932309525332,
        "norm_gR64_l2": 4172.575869428921,
        "shape": [
          1024,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 1.4901161193847656e-08,
        "abs_error_over_ulp_gR": 16384.0,
        "abs_error_over_ulp_sum_gU_old": 16384.0,
        "argmax_flat_index": 54149,
        "argmax_multi_index_row_major": [
          211,
          133
        ],
        "floor_1e_6_active": false,
        "gR_value": -1.0564923286437988e-05,
        "max_rel_old": 0.0014104372821748257,
        "old_denominator": 1.0564923286437988e-05,
        "sum_gU_old_value": -1.055002212524414e-05,
        "ulp_gR": 9.094947017729282e-13,
        "ulp_sum_gU_old": 9.094947017729282e-13
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.6452970986929974e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 6.679024464892589e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 6.103515625e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 4.76837158203125e-06
        }
      },
      "metrics": {
        "E_L2": 3.6452970986929974e-08,
        "E_inf": 6.679024464892589e-08,
        "max_abs": 4.76837158203125e-06,
        "max_rel_old": 0.0028129395218002813,
        "norm_S64_inf": 71.3932294845581,
        "norm_S64_l2": 4172.575939600078,
        "norm_gR64_inf": 71.39323425292969,
        "norm_gR64_l2": 4172.575939068506,
        "shape": [
          1024,
          256
        ]
      },
      "scale_context": {
        "M32": 71.39323425292969,
        "M64": 71.39323425292969,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  }
}
```

### Persisted gradient bundle

```json
{
  "bundle_path": "seed_20261002_raw_gradients.pt",
  "bundle_sha256": "a7a2c05c26451ad3f214169741aed700d2c814daf855a3235283a895cc9ea6e3",
  "bundle_size_bytes": 105685741,
  "tensor_metadata": {
    "seed_20261002/cpu_fp64_oracle/W_K/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "0b8708f08754afacbc26ec9ea3496647eac35c022bbea1bf45ff0cd86b0f7442",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_K/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "f63074a0f1a6335ada7a725c8cec3b2ee05ff9d6ff18882999c0595505e2245e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_K/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "8bf454029c0d6602ec1b061910204dacbbb900d86482db5eda96a64225f3f150",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_K/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "c206c417358871237fac54ee670d64d7af757a5c5dbd60f794a57d1bff5d27b4",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_K/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "fc885a86238110e6c15cfa10c1d010ed886b36803f46cf1079b75d452a7a3d04",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_K/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "3c3716b748acf2d2eb1ca0436ae625ef7153ca1dd6364b7b6b99179460d7d408",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_O/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "5560f310187c193d9df9bd8035c93d4a9f1097d7fbbe2911a39e7ad52baa7e11",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_O/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "9a29218cdaf602e38852b4c5170f5cb5097a78a05c3c3350f8aca1507b7513b6",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_O/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "16e547f7adb31e3d4b131c106b1fa3703a6b3e7797425e1899d014addc66bfe1",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_O/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "6e54d57c51098ad02b6c79b8d5bc6dce2d96badf47bacde2d8d1cabd6d657d9e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_O/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "8c69eaed91c8f81eb56f2b272fd92a8497885b7a63a45fa423eb5e39449fb126",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_O/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "899fe987c82a53e960d22ed951e30c97c63545d979fa35b22a46034b98e89a2e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_Q/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "1870d678392a130c9e24802cb0b337115512c8da96f00316ab1477b6eaa49f1a",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_Q/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "95bd44236e627f4ef7d1ad7dae02a4f41ab6924afd5ecf3807aeec14c0070994",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_Q/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "11fa5f5d79889936defe580de48ffbecd47b98d3a1e51a0a5d6806f385116495",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_Q/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "8ebf5544720bba9d4099fbf0ba39df35710526b68e65f54b98b196cbcd5b5237",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_Q/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "b9464137bfc6596f7257a6ffa729ec1f2a1f3edbda72030bdee54eaa0f53a445",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_Q/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "6d7c90ed910a643591ba51e3bb276d08446696290c3de73501c2236d32d4fa68",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_V/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "622ff8bad459a4602de6220f0a2b52e92b15013e2e535978a744887595a37792",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_V/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "340b830d84c2dabe4874ec0f68f75952865fae2ede19f38e18d484ec9b009b7c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_V/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "03e5032bf89734b499134d233e2f28ab129c5c6ccd58959ecdbe76b2b3a04606",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_V/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "1811c85f6301a97447cbcd36dc19dcdbffc8900bd30ef5b7eaeff7e43a01091e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_V/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "67e1cda45b91c6a6f1cf4c2e5612101c038351e1b59f03451c228b0c0a7fe0fc",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_V/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "4cb9ad23cb563c7d30675a549cdbc8c273cc540eee27ffd0676746fe94e240bb",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_down/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "69bd7d0f53ac5fba0af02a0b1ce48eff285e4a0e7faa1651784181d54251731e",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_down/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "eacbf871ef9bf2164da2792cf3af822908db08b33e685129bf3c6abb17951508",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_down/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "943728806baab95a073d74ee83402dd8383172749408e6593569625dd482185d",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_down/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "23ab21916fa7f46a17755b9782cae1431f878e1788bedd62a00a105652d91536",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_down/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "51e903687078961ad61f1725ab924392092293ddfe8f9bcc6057ff064a0f594e",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_down/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "fcfbbed7b4418aa9759d745a1e3b7d30449d6032b9e5f8c17ffe305da499d608",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_gate/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "381ec65c78893388e13ba47a3ee5d410da8c3335051a10cf308937021333e789",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_gate/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "21b89c210e9b441de198b7aeaa45a32b713b902a618e97a6147c70f920d1c750",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_gate/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "7d7326cd3dc899f504db8b2c450b10e5c130e6eb8f0513200ae98eefa0f2245c",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_gate/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "2717466949cd9d506bb2d21da49a7fc38bfdd410dd1b5e3b1d52efada0ab97c0",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_gate/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "e313d5e24b2444ef27383dbed9a2828aec17f12814916e2dcc574305b4d5845a",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_gate/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "0cc36a0355e1f3032f5a51029d61f469fda4c91602e2e4d1d21516c40b0341da",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_up/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "82524230984f3223bb50bc988ebb15c939ec8fff28144dee06bef98b5cdc1ce8",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_up/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "ad672ff02cc4c5631244739e1eb1a2f2dd6ad8275151dacb599f3ca9c0b2a63e",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_up/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "1ab02644d274733b0d38270bd3fba3898e6a5ac174a6d62c28f905aa13be9f92",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_up/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "487b26c263bb6577e58192c0bb3b710f03f79c2b2fa0fecc65d0531f98cbc046",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_up/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "f8ebb7eaaecba3d951eb64190bd6fddf194e6d481d73eccefd6cfc94f8a872f4",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cpu_fp64_oracle/W_up/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "0fc44893d59247e419df994e690593d20a2d1a167533ccba752fbfe66736fd09",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_K/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "0589951c06e7e11720724b986d8c0c2e162a43a7ba90d6c7313018ec93ff6321",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_K/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "4fd91ba174c9d2f3a976d2d04ffa250c70ebbee583417d645fb4fcf18a13d696",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_K/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "62fadbd178af61c9344c34f8a0ab885d7a7e8835101ab0733fe96b2030e4b0af",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_K/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "bf613833a0fe951e44d015390b9f23ccedb6132dfcd2c8ae7d219c90b6c6d603",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_K/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "3dbf1a2de9a9a31f02051670ccc1e92f9f53844220c0bd00a1c3a1173a96642e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_K/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "2e43e52c02886b5bf99411e4cf7753386a33e293cec51d27b5097068c26ea0bf",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_K/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "cbe94cc33cc9cc7c4bf6554d1ab1e83d6b4cd145b481bb40851bddacd76234b0",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_K/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "19cf7d1f5911dcb3323338d912a32cf3246abf0980a423123656ee413cc129c3",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_K/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "b511768326643ac8b4c4a91ab796eba70dfe39ae38ec0e2bfc24eff1a2aef7e3",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_K/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "fab9f80f5ccf1393f6b0d261e16c054c7f507e60d9840a8bb3da4ad928b36e21",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_K/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "a67315509aeaf356b2faad651bdb226fc99faba7399f9549c9b90ebc0a314194",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_O/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "2e14ff5d5978a0756b9a97b6ef71bff0c1b128ebb1bbd9f056a82f381871287e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_O/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "60392ea9082f7b13968581c0314f9146408127d9f4f29ddca251bb94ad6149b3",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_O/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "5e8d0df730d603d6992c18dfdb96f7fd18ce97ba1d56b0d5fdce0d290ee49007",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_O/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "4289d4959284d74a413ed1d4a6913c740d93a3883d06f6bf61616bc9c40c80ef",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_O/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "9b9c4fdc9c67c594e3e6d22c692a81571c88bbe38beeebdd4d728d5f334c9dc9",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_O/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "e9e129db9719a12373eacbe70590c25746a343b1d052d3a0ed4aba357b672aa2",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_O/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "85c91d05e1f3b32b848849300f76f4ca74370ab5dc4d81e233092eddaa823d52",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_O/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "9289e83b60c23e257a275b55a6549528b7c4b175b8c289211f8e911a038a667a",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_O/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "e8c4d3990b3c3b49a18dea416fd8ebe38142a206a3fe93a3d47cc40fda78c080",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_O/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "22280b221fdd458b9143428163e0eaa73c2e75f4c65491dfb28f597c6d297b90",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_O/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "0da1bd5e6a7e9bd92ae541dab9767f8b1a4e5a59d1ba389a7319ac81b3d91adb",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_Q/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "ba832bd745b277a7bc57f8896e880aac3523a98c26b63c8a4efa23dc448384ea",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_Q/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "f7568d5b1f5e05f6ef9ef3fe01f615c621b8626100be1889298e83ea243678a6",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_Q/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "882f566e897cdc897181187442da5e87e5f6f91fd75c17af4ef0541e890f88f9",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_Q/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "c77e97757c926897643f707b31dd92c03914a68024365877e2249ee74725d9b6",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_Q/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "0c453f15924a435db1bb7411d8408c4c6c837ee9b704ee75db0ed0aea64ae071",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_Q/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "847ff12a1dba4b3c9556a198e747d7ed25f688653e2869acc7e8a3e7a14072d2",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_Q/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "82e496aa3b49c53a67b7571e8552024835ecc39ad3ff3e9f8467eae30bf18250",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_Q/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "e755a228aceca8f3ba8af83228b10a0b10441d002322b80eb10446224565a547",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_Q/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "c95722d246c8195d1033939054074c0c24f3e01b19ac12307a2855c0f51419eb",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_Q/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "7a992851bc7877e4d6b20a4050996c58c9ee897e2a39a79fbeee8ff77e1d8628",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_Q/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "7d604b66131b37f23a6e4ee5b34345f19f55089d09f4bbdc87f70da528410196",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_V/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "ba804a018f52e0a7838bc4b3128671c4026c5863098b01c3e4c57ceba03ed0a4",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_V/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "0c794ac60b0bbd336fc106f6f9e1f2389bbce1be07ca4619755bdaeedaba3e09",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_V/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "fe6a35424e8ef71dc6a7db508d15343eac1c426cbc025c6c4030ac270dc33fda",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_V/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "684fd89e00f075b9947f1b1ed13fa0a4a9ff77defe9521a3efceaa2144e5aa09",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_V/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "696ee2dee81bba0cb15fefdcee02a3fc8949a705298c5d25206b7a522a99b02c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_V/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "5a89d019fb2faebfb48eb8982e936885c4113ab2db569019b828130399e84889",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_V/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "ae994e56233ff984fb96663f160181be2e86db79e82ff0476cbbe11be2c12b2d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_V/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "d20248b267c8b35737c3daa876d93475b171164254f6710e474cbf44e8a6c23a",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_V/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "6cf141d1112d5cb997167f64f16a1441e6d9b712ba3f62bf89fb045e7164356f",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_V/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "27beda8b2720193b2198e98091f2349f20a5b31ee1827f13155ec6b67e50c67a",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_V/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "5a760e8183ab8a9e4602485859885e6a4f233fdc097c01dc0f48e2d03b97607b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_down/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "fcf369e74b2341d76546e531e56f1545e7216d922138439aeb2b0aefb04e2544",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cuda_fp32/W_down/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "aedb18a5f2551c8a4142206542cd4969f8bfdedd9fe6a7252081dbfcf90afbc1",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cuda_fp32/W_down/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "f422f0dad353aee67ec43bd91f0a903e16c47e2fd1d14c88c1a129b7e250c35c",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cuda_fp32/W_down/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "2fc3b96bd8da53dcf194c0a4e0a1ada4e744e02dcf9ec9fd8dc918644b3c908e",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cuda_fp32/W_down/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "7426ce6cf3bd8afa935aec82ca1dbf75ed2cfd11af2264e696792ee1cacfc07d",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cuda_fp32/W_down/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "600fd051bc0b8f800090910f38620a6c75cb82ee8d359010106711f13d9cec28",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cuda_fp32/W_down/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "9518c95187a3f05a2ea5c7ae87ce52606adc2e97cdcc8f5f86748eca46fb2dd7",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cuda_fp32/W_down/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "db71f8eb3cf4c3d12e13637f16ab69e6fd09ed358e0e4bf246fc5065f94b9700",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cuda_fp32/W_down/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "49bf2b10a6e2cccfdcbf75f1d66a6fe6bed71af043027f846bc11dd0662147da",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cuda_fp32/W_down/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "823feebc1a24de5a49a1d2244608a9c4bbeea4e60d042506ec32ed6c28557639",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cuda_fp32/W_down/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "a0c9bc50027f8fac49405cd16f80cde777b6bfdca01b48504d565dfa64a3abeb",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261002/cuda_fp32/W_gate/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "c9be73df36e0c3ce459f7a2d0bbd5469868f9e6626f08e26d25aad6fe47014a0",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_gate/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "401f41d44417fac2c67d34cb40d05a4ad65bc8390eecb81f6eabb55cec391577",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_gate/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "a592ba08df6ccbaa312fc6eea1409233cc3a9f2f2d0fb7ba700c5270ae03659f",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_gate/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "d7ea2cea61cb2cbc606faf23d357e8bdf252033d4af02387e6d8bae1e002e6e0",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_gate/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "3a4912c7b8c7cc4057a9f1b845d3157f2051502e58a3047c3e13559c1e7feed0",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_gate/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "c22af5bdcb5baa58920cd4456433e9fa884afa5b058d0e709749580928cf48c6",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_gate/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "9368799ac70c77155d39124c1658c97db2f443f5d22e649eea8e184ad84675e8",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_gate/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "7697bbb0b1f9c6e6f19e558ca01beebbaf0b29dc0bc4763c33e07f9c77e1e327",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_gate/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "c9f05ae9cec0adee654981f2d5d30ffe7f454dfd91ae642adff72385ed108037",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_gate/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "0b71eda8f54372ba546699f799606987c15fa84128921c282e623f0a7ccf5a6e",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_gate/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "153da8e0a2a6ce84c5b278fdf74cfbedfad2b170022020c388caa599ded26b22",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_up/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "97e3041ddfd197858f54ef6b418dbf08d35edb4318381ece6567b32cbc77b1aa",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_up/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "6e36158861e81ed229f289f07e9277b952bd591d3e59a25d5c381269cb4fd7ed",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_up/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "e831dfa2b602606c1b9300deb3abf9a05b95faa5ee7d7d3c1c04393db7e08e0e",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_up/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "8b0fa128596ee86adb1c02de3565364be4697e745a844697574e25ba677ceecf",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_up/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "1fbb0ad125c6b025062d8878d695061c6513ad742d9dbd3df8cf50edefedf039",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_up/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "5fc20b8528c9e92a1a95003f05694def09eda4ebce95195dda92610287d8204e",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_up/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "9073002c6df92416f37f17ef4e2b66331d0557027292ba65f839de09e3f25305",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_up/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "d1e906e565ffa0d1f889ad999e51f411b80bdce8af06ac292c366f9ff00c1185",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_up/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "463a4c837e386e485d09832a34e58cd29054a98f4de87395a3f7e450e22cf817",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_up/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "f547653c8fbdfea9cf267d96d2598ec7217f7debaf24c2856e6a42995d1bf0b4",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/cuda_fp32/W_up/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "fdd443a711b4fca4d44782a5cf76e836eabc8fafc4b6077a475dcbc9de232a50",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261002/inputs/w_fp32_cpu": {
      "dtype": "torch.float32",
      "raw_sha256": "78d6fcb2c9e6c2ec2702bb564fef1f10582a781b4e99f5de72945e31fa054d93",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261002/inputs/x_fp32_cpu": {
      "dtype": "torch.float32",
      "raw_sha256": "43704e2fbe9de98db4d387c329b9dbeb70217b9d36587d4d189e20ff9e870deb",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261002/structural_traces/R4/round_1": {
      "dtype": "torch.float32",
      "raw_sha256": "c1ba00c0a913a38f41c71e4bdbf9d4d2e237a5dd742a334ee141d1e1785d7088",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261002/structural_traces/R4/round_2": {
      "dtype": "torch.float32",
      "raw_sha256": "9fedaec07e694b58327f32d8ee37bc8575d8055771a7eaff7301f308fc7eb10e",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261002/structural_traces/R4/round_3": {
      "dtype": "torch.float32",
      "raw_sha256": "df4119594acd6936ff2b77e818e839625931ceae5b37292e16490b6a79b18594",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261002/structural_traces/R4/round_4": {
      "dtype": "torch.float32",
      "raw_sha256": "a47ec90c646a5ae4e5fa811d5687b9de1d4deb218356a48cd7f9831827b947ee",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261002/structural_traces/R4_final": {
      "dtype": "torch.float32",
      "raw_sha256": "dd718e85d2109325e647e93e60577cff904c513deed0f8507f605bd2f47a3a06",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261002/structural_traces/U4/round_1": {
      "dtype": "torch.float32",
      "raw_sha256": "67fc20d66c006610af42b8de733d8621991b49e5731d72526d4eb76ac28aa87a",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261002/structural_traces/U4/round_2": {
      "dtype": "torch.float32",
      "raw_sha256": "83231987786ad7b8d398bcee7b5261744ad2b911fbaaa85cb4f87ea43237b32b",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261002/structural_traces/U4/round_3": {
      "dtype": "torch.float32",
      "raw_sha256": "2eba6e540c450146166d3b765a2a07b647eef2f57848d4140b19b204e9c4369d",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261002/structural_traces/U4/round_4": {
      "dtype": "torch.float32",
      "raw_sha256": "99e9e6b5b7017e8cb69d304d6b0463f666c0c34c53d7aba704cbdb2eeb81a97a",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261002/structural_traces/U4_final": {
      "dtype": "torch.float32",
      "raw_sha256": "aa216c8d7d619672677ed67ef0ff1183158619adf464ce6b1c69af80be0e4927",
      "shape": [
        8,
        8,
        256
      ]
    }
  },
  "tensor_raw_sha256": {
    "seed_20261002/cpu_fp64_oracle/W_K/S64_cpu": "0b8708f08754afacbc26ec9ea3496647eac35c022bbea1bf45ff0cd86b0f7442",
    "seed_20261002/cpu_fp64_oracle/W_K/gR64_cpu": "f63074a0f1a6335ada7a725c8cec3b2ee05ff9d6ff18882999c0595505e2245e",
    "seed_20261002/cpu_fp64_oracle/W_K/gU0_64_cpu": "8bf454029c0d6602ec1b061910204dacbbb900d86482db5eda96a64225f3f150",
    "seed_20261002/cpu_fp64_oracle/W_K/gU1_64_cpu": "c206c417358871237fac54ee670d64d7af757a5c5dbd60f794a57d1bff5d27b4",
    "seed_20261002/cpu_fp64_oracle/W_K/gU2_64_cpu": "fc885a86238110e6c15cfa10c1d010ed886b36803f46cf1079b75d452a7a3d04",
    "seed_20261002/cpu_fp64_oracle/W_K/gU3_64_cpu": "3c3716b748acf2d2eb1ca0436ae625ef7153ca1dd6364b7b6b99179460d7d408",
    "seed_20261002/cpu_fp64_oracle/W_O/S64_cpu": "5560f310187c193d9df9bd8035c93d4a9f1097d7fbbe2911a39e7ad52baa7e11",
    "seed_20261002/cpu_fp64_oracle/W_O/gR64_cpu": "9a29218cdaf602e38852b4c5170f5cb5097a78a05c3c3350f8aca1507b7513b6",
    "seed_20261002/cpu_fp64_oracle/W_O/gU0_64_cpu": "16e547f7adb31e3d4b131c106b1fa3703a6b3e7797425e1899d014addc66bfe1",
    "seed_20261002/cpu_fp64_oracle/W_O/gU1_64_cpu": "6e54d57c51098ad02b6c79b8d5bc6dce2d96badf47bacde2d8d1cabd6d657d9e",
    "seed_20261002/cpu_fp64_oracle/W_O/gU2_64_cpu": "8c69eaed91c8f81eb56f2b272fd92a8497885b7a63a45fa423eb5e39449fb126",
    "seed_20261002/cpu_fp64_oracle/W_O/gU3_64_cpu": "899fe987c82a53e960d22ed951e30c97c63545d979fa35b22a46034b98e89a2e",
    "seed_20261002/cpu_fp64_oracle/W_Q/S64_cpu": "1870d678392a130c9e24802cb0b337115512c8da96f00316ab1477b6eaa49f1a",
    "seed_20261002/cpu_fp64_oracle/W_Q/gR64_cpu": "95bd44236e627f4ef7d1ad7dae02a4f41ab6924afd5ecf3807aeec14c0070994",
    "seed_20261002/cpu_fp64_oracle/W_Q/gU0_64_cpu": "11fa5f5d79889936defe580de48ffbecd47b98d3a1e51a0a5d6806f385116495",
    "seed_20261002/cpu_fp64_oracle/W_Q/gU1_64_cpu": "8ebf5544720bba9d4099fbf0ba39df35710526b68e65f54b98b196cbcd5b5237",
    "seed_20261002/cpu_fp64_oracle/W_Q/gU2_64_cpu": "b9464137bfc6596f7257a6ffa729ec1f2a1f3edbda72030bdee54eaa0f53a445",
    "seed_20261002/cpu_fp64_oracle/W_Q/gU3_64_cpu": "6d7c90ed910a643591ba51e3bb276d08446696290c3de73501c2236d32d4fa68",
    "seed_20261002/cpu_fp64_oracle/W_V/S64_cpu": "622ff8bad459a4602de6220f0a2b52e92b15013e2e535978a744887595a37792",
    "seed_20261002/cpu_fp64_oracle/W_V/gR64_cpu": "340b830d84c2dabe4874ec0f68f75952865fae2ede19f38e18d484ec9b009b7c",
    "seed_20261002/cpu_fp64_oracle/W_V/gU0_64_cpu": "03e5032bf89734b499134d233e2f28ab129c5c6ccd58959ecdbe76b2b3a04606",
    "seed_20261002/cpu_fp64_oracle/W_V/gU1_64_cpu": "1811c85f6301a97447cbcd36dc19dcdbffc8900bd30ef5b7eaeff7e43a01091e",
    "seed_20261002/cpu_fp64_oracle/W_V/gU2_64_cpu": "67e1cda45b91c6a6f1cf4c2e5612101c038351e1b59f03451c228b0c0a7fe0fc",
    "seed_20261002/cpu_fp64_oracle/W_V/gU3_64_cpu": "4cb9ad23cb563c7d30675a549cdbc8c273cc540eee27ffd0676746fe94e240bb",
    "seed_20261002/cpu_fp64_oracle/W_down/S64_cpu": "69bd7d0f53ac5fba0af02a0b1ce48eff285e4a0e7faa1651784181d54251731e",
    "seed_20261002/cpu_fp64_oracle/W_down/gR64_cpu": "eacbf871ef9bf2164da2792cf3af822908db08b33e685129bf3c6abb17951508",
    "seed_20261002/cpu_fp64_oracle/W_down/gU0_64_cpu": "943728806baab95a073d74ee83402dd8383172749408e6593569625dd482185d",
    "seed_20261002/cpu_fp64_oracle/W_down/gU1_64_cpu": "23ab21916fa7f46a17755b9782cae1431f878e1788bedd62a00a105652d91536",
    "seed_20261002/cpu_fp64_oracle/W_down/gU2_64_cpu": "51e903687078961ad61f1725ab924392092293ddfe8f9bcc6057ff064a0f594e",
    "seed_20261002/cpu_fp64_oracle/W_down/gU3_64_cpu": "fcfbbed7b4418aa9759d745a1e3b7d30449d6032b9e5f8c17ffe305da499d608",
    "seed_20261002/cpu_fp64_oracle/W_gate/S64_cpu": "381ec65c78893388e13ba47a3ee5d410da8c3335051a10cf308937021333e789",
    "seed_20261002/cpu_fp64_oracle/W_gate/gR64_cpu": "21b89c210e9b441de198b7aeaa45a32b713b902a618e97a6147c70f920d1c750",
    "seed_20261002/cpu_fp64_oracle/W_gate/gU0_64_cpu": "7d7326cd3dc899f504db8b2c450b10e5c130e6eb8f0513200ae98eefa0f2245c",
    "seed_20261002/cpu_fp64_oracle/W_gate/gU1_64_cpu": "2717466949cd9d506bb2d21da49a7fc38bfdd410dd1b5e3b1d52efada0ab97c0",
    "seed_20261002/cpu_fp64_oracle/W_gate/gU2_64_cpu": "e313d5e24b2444ef27383dbed9a2828aec17f12814916e2dcc574305b4d5845a",
    "seed_20261002/cpu_fp64_oracle/W_gate/gU3_64_cpu": "0cc36a0355e1f3032f5a51029d61f469fda4c91602e2e4d1d21516c40b0341da",
    "seed_20261002/cpu_fp64_oracle/W_up/S64_cpu": "82524230984f3223bb50bc988ebb15c939ec8fff28144dee06bef98b5cdc1ce8",
    "seed_20261002/cpu_fp64_oracle/W_up/gR64_cpu": "ad672ff02cc4c5631244739e1eb1a2f2dd6ad8275151dacb599f3ca9c0b2a63e",
    "seed_20261002/cpu_fp64_oracle/W_up/gU0_64_cpu": "1ab02644d274733b0d38270bd3fba3898e6a5ac174a6d62c28f905aa13be9f92",
    "seed_20261002/cpu_fp64_oracle/W_up/gU1_64_cpu": "487b26c263bb6577e58192c0bb3b710f03f79c2b2fa0fecc65d0531f98cbc046",
    "seed_20261002/cpu_fp64_oracle/W_up/gU2_64_cpu": "f8ebb7eaaecba3d951eb64190bd6fddf194e6d481d73eccefd6cfc94f8a872f4",
    "seed_20261002/cpu_fp64_oracle/W_up/gU3_64_cpu": "0fc44893d59247e419df994e690593d20a2d1a167533ccba752fbfe66736fd09",
    "seed_20261002/cuda_fp32/W_K/S64_primary_cuda": "0589951c06e7e11720724b986d8c0c2e162a43a7ba90d6c7313018ec93ff6321",
    "seed_20261002/cuda_fp32/W_K/S_forward_cuda": "4fd91ba174c9d2f3a976d2d04ffa250c70ebbee583417d645fb4fcf18a13d696",
    "seed_20261002/cuda_fp32/W_K/S_pairwise_cuda": "62fadbd178af61c9344c34f8a0ab885d7a7e8835101ab0733fe96b2030e4b0af",
    "seed_20261002/cuda_fp32/W_K/S_reverse_cuda": "bf613833a0fe951e44d015390b9f23ccedb6132dfcd2c8ae7d219c90b6c6d603",
    "seed_20261002/cuda_fp32/W_K/S_stack_cuda": "3dbf1a2de9a9a31f02051670ccc1e92f9f53844220c0bd00a1c3a1173a96642e",
    "seed_20261002/cuda_fp32/W_K/gR64_primary_cuda": "2e43e52c02886b5bf99411e4cf7753386a33e293cec51d27b5097068c26ea0bf",
    "seed_20261002/cuda_fp32/W_K/gR_cuda_fp32": "cbe94cc33cc9cc7c4bf6554d1ab1e83d6b4cd145b481bb40851bddacd76234b0",
    "seed_20261002/cuda_fp32/W_K/gU0_cuda_fp32": "19cf7d1f5911dcb3323338d912a32cf3246abf0980a423123656ee413cc129c3",
    "seed_20261002/cuda_fp32/W_K/gU1_cuda_fp32": "b511768326643ac8b4c4a91ab796eba70dfe39ae38ec0e2bfc24eff1a2aef7e3",
    "seed_20261002/cuda_fp32/W_K/gU2_cuda_fp32": "fab9f80f5ccf1393f6b0d261e16c054c7f507e60d9840a8bb3da4ad928b36e21",
    "seed_20261002/cuda_fp32/W_K/gU3_cuda_fp32": "a67315509aeaf356b2faad651bdb226fc99faba7399f9549c9b90ebc0a314194",
    "seed_20261002/cuda_fp32/W_O/S64_primary_cuda": "2e14ff5d5978a0756b9a97b6ef71bff0c1b128ebb1bbd9f056a82f381871287e",
    "seed_20261002/cuda_fp32/W_O/S_forward_cuda": "60392ea9082f7b13968581c0314f9146408127d9f4f29ddca251bb94ad6149b3",
    "seed_20261002/cuda_fp32/W_O/S_pairwise_cuda": "5e8d0df730d603d6992c18dfdb96f7fd18ce97ba1d56b0d5fdce0d290ee49007",
    "seed_20261002/cuda_fp32/W_O/S_reverse_cuda": "4289d4959284d74a413ed1d4a6913c740d93a3883d06f6bf61616bc9c40c80ef",
    "seed_20261002/cuda_fp32/W_O/S_stack_cuda": "9b9c4fdc9c67c594e3e6d22c692a81571c88bbe38beeebdd4d728d5f334c9dc9",
    "seed_20261002/cuda_fp32/W_O/gR64_primary_cuda": "e9e129db9719a12373eacbe70590c25746a343b1d052d3a0ed4aba357b672aa2",
    "seed_20261002/cuda_fp32/W_O/gR_cuda_fp32": "85c91d05e1f3b32b848849300f76f4ca74370ab5dc4d81e233092eddaa823d52",
    "seed_20261002/cuda_fp32/W_O/gU0_cuda_fp32": "9289e83b60c23e257a275b55a6549528b7c4b175b8c289211f8e911a038a667a",
    "seed_20261002/cuda_fp32/W_O/gU1_cuda_fp32": "e8c4d3990b3c3b49a18dea416fd8ebe38142a206a3fe93a3d47cc40fda78c080",
    "seed_20261002/cuda_fp32/W_O/gU2_cuda_fp32": "22280b221fdd458b9143428163e0eaa73c2e75f4c65491dfb28f597c6d297b90",
    "seed_20261002/cuda_fp32/W_O/gU3_cuda_fp32": "0da1bd5e6a7e9bd92ae541dab9767f8b1a4e5a59d1ba389a7319ac81b3d91adb",
    "seed_20261002/cuda_fp32/W_Q/S64_primary_cuda": "ba832bd745b277a7bc57f8896e880aac3523a98c26b63c8a4efa23dc448384ea",
    "seed_20261002/cuda_fp32/W_Q/S_forward_cuda": "f7568d5b1f5e05f6ef9ef3fe01f615c621b8626100be1889298e83ea243678a6",
    "seed_20261002/cuda_fp32/W_Q/S_pairwise_cuda": "882f566e897cdc897181187442da5e87e5f6f91fd75c17af4ef0541e890f88f9",
    "seed_20261002/cuda_fp32/W_Q/S_reverse_cuda": "c77e97757c926897643f707b31dd92c03914a68024365877e2249ee74725d9b6",
    "seed_20261002/cuda_fp32/W_Q/S_stack_cuda": "0c453f15924a435db1bb7411d8408c4c6c837ee9b704ee75db0ed0aea64ae071",
    "seed_20261002/cuda_fp32/W_Q/gR64_primary_cuda": "847ff12a1dba4b3c9556a198e747d7ed25f688653e2869acc7e8a3e7a14072d2",
    "seed_20261002/cuda_fp32/W_Q/gR_cuda_fp32": "82e496aa3b49c53a67b7571e8552024835ecc39ad3ff3e9f8467eae30bf18250",
    "seed_20261002/cuda_fp32/W_Q/gU0_cuda_fp32": "e755a228aceca8f3ba8af83228b10a0b10441d002322b80eb10446224565a547",
    "seed_20261002/cuda_fp32/W_Q/gU1_cuda_fp32": "c95722d246c8195d1033939054074c0c24f3e01b19ac12307a2855c0f51419eb",
    "seed_20261002/cuda_fp32/W_Q/gU2_cuda_fp32": "7a992851bc7877e4d6b20a4050996c58c9ee897e2a39a79fbeee8ff77e1d8628",
    "seed_20261002/cuda_fp32/W_Q/gU3_cuda_fp32": "7d604b66131b37f23a6e4ee5b34345f19f55089d09f4bbdc87f70da528410196",
    "seed_20261002/cuda_fp32/W_V/S64_primary_cuda": "ba804a018f52e0a7838bc4b3128671c4026c5863098b01c3e4c57ceba03ed0a4",
    "seed_20261002/cuda_fp32/W_V/S_forward_cuda": "0c794ac60b0bbd336fc106f6f9e1f2389bbce1be07ca4619755bdaeedaba3e09",
    "seed_20261002/cuda_fp32/W_V/S_pairwise_cuda": "fe6a35424e8ef71dc6a7db508d15343eac1c426cbc025c6c4030ac270dc33fda",
    "seed_20261002/cuda_fp32/W_V/S_reverse_cuda": "684fd89e00f075b9947f1b1ed13fa0a4a9ff77defe9521a3efceaa2144e5aa09",
    "seed_20261002/cuda_fp32/W_V/S_stack_cuda": "696ee2dee81bba0cb15fefdcee02a3fc8949a705298c5d25206b7a522a99b02c",
    "seed_20261002/cuda_fp32/W_V/gR64_primary_cuda": "5a89d019fb2faebfb48eb8982e936885c4113ab2db569019b828130399e84889",
    "seed_20261002/cuda_fp32/W_V/gR_cuda_fp32": "ae994e56233ff984fb96663f160181be2e86db79e82ff0476cbbe11be2c12b2d",
    "seed_20261002/cuda_fp32/W_V/gU0_cuda_fp32": "d20248b267c8b35737c3daa876d93475b171164254f6710e474cbf44e8a6c23a",
    "seed_20261002/cuda_fp32/W_V/gU1_cuda_fp32": "6cf141d1112d5cb997167f64f16a1441e6d9b712ba3f62bf89fb045e7164356f",
    "seed_20261002/cuda_fp32/W_V/gU2_cuda_fp32": "27beda8b2720193b2198e98091f2349f20a5b31ee1827f13155ec6b67e50c67a",
    "seed_20261002/cuda_fp32/W_V/gU3_cuda_fp32": "5a760e8183ab8a9e4602485859885e6a4f233fdc097c01dc0f48e2d03b97607b",
    "seed_20261002/cuda_fp32/W_down/S64_primary_cuda": "fcf369e74b2341d76546e531e56f1545e7216d922138439aeb2b0aefb04e2544",
    "seed_20261002/cuda_fp32/W_down/S_forward_cuda": "aedb18a5f2551c8a4142206542cd4969f8bfdedd9fe6a7252081dbfcf90afbc1",
    "seed_20261002/cuda_fp32/W_down/S_pairwise_cuda": "f422f0dad353aee67ec43bd91f0a903e16c47e2fd1d14c88c1a129b7e250c35c",
    "seed_20261002/cuda_fp32/W_down/S_reverse_cuda": "2fc3b96bd8da53dcf194c0a4e0a1ada4e744e02dcf9ec9fd8dc918644b3c908e",
    "seed_20261002/cuda_fp32/W_down/S_stack_cuda": "7426ce6cf3bd8afa935aec82ca1dbf75ed2cfd11af2264e696792ee1cacfc07d",
    "seed_20261002/cuda_fp32/W_down/gR64_primary_cuda": "600fd051bc0b8f800090910f38620a6c75cb82ee8d359010106711f13d9cec28",
    "seed_20261002/cuda_fp32/W_down/gR_cuda_fp32": "9518c95187a3f05a2ea5c7ae87ce52606adc2e97cdcc8f5f86748eca46fb2dd7",
    "seed_20261002/cuda_fp32/W_down/gU0_cuda_fp32": "db71f8eb3cf4c3d12e13637f16ab69e6fd09ed358e0e4bf246fc5065f94b9700",
    "seed_20261002/cuda_fp32/W_down/gU1_cuda_fp32": "49bf2b10a6e2cccfdcbf75f1d66a6fe6bed71af043027f846bc11dd0662147da",
    "seed_20261002/cuda_fp32/W_down/gU2_cuda_fp32": "823feebc1a24de5a49a1d2244608a9c4bbeea4e60d042506ec32ed6c28557639",
    "seed_20261002/cuda_fp32/W_down/gU3_cuda_fp32": "a0c9bc50027f8fac49405cd16f80cde777b6bfdca01b48504d565dfa64a3abeb",
    "seed_20261002/cuda_fp32/W_gate/S64_primary_cuda": "c9be73df36e0c3ce459f7a2d0bbd5469868f9e6626f08e26d25aad6fe47014a0",
    "seed_20261002/cuda_fp32/W_gate/S_forward_cuda": "401f41d44417fac2c67d34cb40d05a4ad65bc8390eecb81f6eabb55cec391577",
    "seed_20261002/cuda_fp32/W_gate/S_pairwise_cuda": "a592ba08df6ccbaa312fc6eea1409233cc3a9f2f2d0fb7ba700c5270ae03659f",
    "seed_20261002/cuda_fp32/W_gate/S_reverse_cuda": "d7ea2cea61cb2cbc606faf23d357e8bdf252033d4af02387e6d8bae1e002e6e0",
    "seed_20261002/cuda_fp32/W_gate/S_stack_cuda": "3a4912c7b8c7cc4057a9f1b845d3157f2051502e58a3047c3e13559c1e7feed0",
    "seed_20261002/cuda_fp32/W_gate/gR64_primary_cuda": "c22af5bdcb5baa58920cd4456433e9fa884afa5b058d0e709749580928cf48c6",
    "seed_20261002/cuda_fp32/W_gate/gR_cuda_fp32": "9368799ac70c77155d39124c1658c97db2f443f5d22e649eea8e184ad84675e8",
    "seed_20261002/cuda_fp32/W_gate/gU0_cuda_fp32": "7697bbb0b1f9c6e6f19e558ca01beebbaf0b29dc0bc4763c33e07f9c77e1e327",
    "seed_20261002/cuda_fp32/W_gate/gU1_cuda_fp32": "c9f05ae9cec0adee654981f2d5d30ffe7f454dfd91ae642adff72385ed108037",
    "seed_20261002/cuda_fp32/W_gate/gU2_cuda_fp32": "0b71eda8f54372ba546699f799606987c15fa84128921c282e623f0a7ccf5a6e",
    "seed_20261002/cuda_fp32/W_gate/gU3_cuda_fp32": "153da8e0a2a6ce84c5b278fdf74cfbedfad2b170022020c388caa599ded26b22",
    "seed_20261002/cuda_fp32/W_up/S64_primary_cuda": "97e3041ddfd197858f54ef6b418dbf08d35edb4318381ece6567b32cbc77b1aa",
    "seed_20261002/cuda_fp32/W_up/S_forward_cuda": "6e36158861e81ed229f289f07e9277b952bd591d3e59a25d5c381269cb4fd7ed",
    "seed_20261002/cuda_fp32/W_up/S_pairwise_cuda": "e831dfa2b602606c1b9300deb3abf9a05b95faa5ee7d7d3c1c04393db7e08e0e",
    "seed_20261002/cuda_fp32/W_up/S_reverse_cuda": "8b0fa128596ee86adb1c02de3565364be4697e745a844697574e25ba677ceecf",
    "seed_20261002/cuda_fp32/W_up/S_stack_cuda": "1fbb0ad125c6b025062d8878d695061c6513ad742d9dbd3df8cf50edefedf039",
    "seed_20261002/cuda_fp32/W_up/gR64_primary_cuda": "5fc20b8528c9e92a1a95003f05694def09eda4ebce95195dda92610287d8204e",
    "seed_20261002/cuda_fp32/W_up/gR_cuda_fp32": "9073002c6df92416f37f17ef4e2b66331d0557027292ba65f839de09e3f25305",
    "seed_20261002/cuda_fp32/W_up/gU0_cuda_fp32": "d1e906e565ffa0d1f889ad999e51f411b80bdce8af06ac292c366f9ff00c1185",
    "seed_20261002/cuda_fp32/W_up/gU1_cuda_fp32": "463a4c837e386e485d09832a34e58cd29054a98f4de87395a3f7e450e22cf817",
    "seed_20261002/cuda_fp32/W_up/gU2_cuda_fp32": "f547653c8fbdfea9cf267d96d2598ec7217f7debaf24c2856e6a42995d1bf0b4",
    "seed_20261002/cuda_fp32/W_up/gU3_cuda_fp32": "fdd443a711b4fca4d44782a5cf76e836eabc8fafc4b6077a475dcbc9de232a50",
    "seed_20261002/inputs/w_fp32_cpu": "78d6fcb2c9e6c2ec2702bb564fef1f10582a781b4e99f5de72945e31fa054d93",
    "seed_20261002/inputs/x_fp32_cpu": "43704e2fbe9de98db4d387c329b9dbeb70217b9d36587d4d189e20ff9e870deb",
    "seed_20261002/structural_traces/R4/round_1": "c1ba00c0a913a38f41c71e4bdbf9d4d2e237a5dd742a334ee141d1e1785d7088",
    "seed_20261002/structural_traces/R4/round_2": "9fedaec07e694b58327f32d8ee37bc8575d8055771a7eaff7301f308fc7eb10e",
    "seed_20261002/structural_traces/R4/round_3": "df4119594acd6936ff2b77e818e839625931ceae5b37292e16490b6a79b18594",
    "seed_20261002/structural_traces/R4/round_4": "a47ec90c646a5ae4e5fa811d5687b9de1d4deb218356a48cd7f9831827b947ee",
    "seed_20261002/structural_traces/R4_final": "dd718e85d2109325e647e93e60577cff904c513deed0f8507f605bd2f47a3a06",
    "seed_20261002/structural_traces/U4/round_1": "67fc20d66c006610af42b8de733d8621991b49e5731d72526d4eb76ac28aa87a",
    "seed_20261002/structural_traces/U4/round_2": "83231987786ad7b8d398bcee7b5261744ad2b911fbaaa85cb4f87ea43237b32b",
    "seed_20261002/structural_traces/U4/round_3": "2eba6e540c450146166d3b765a2a07b647eef2f57848d4140b19b204e9c4369d",
    "seed_20261002/structural_traces/U4/round_4": "99e9e6b5b7017e8cb69d304d6b0463f666c0c34c53d7aba704cbdb2eeb81a97a",
    "seed_20261002/structural_traces/U4_final": "aa216c8d7d619672677ed67ef0ff1183158619adf464ce6b1c69af80be0e4927"
  }
}
```

## Seed seed_20261003

### Structural trace checks

```json
{
  "final_output_equal": true,
  "initial_clone_report": {
    "initial_values_bitwise_equal": true,
    "r4_reuses_one_block_object": true,
    "u4_block_storages_pairwise_disjoint": true,
    "u4_has_four_block_objects": true,
    "u4_storage_disjoint_from_r4": true
  },
  "input_copy_bitwise_equal": true,
  "loss_weight_copy_bitwise_equal": true,
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
  ],
  "weight_copy_report": {
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
    ]
  }
}
```

### Per-family scientific and oracle decisions

```json
{
  "W_K": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 1.0032592486319907e-16
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.5034306959106114e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 1.0032592486319907e-16,
        "E_inf": 2.5034306959106114e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 2.2826112547328076e-12,
        "norm_S64_inf": 56.765520764827365,
        "norm_S64_l2": 2846.7275980323516,
        "norm_gR64_inf": 56.76552076482737,
        "norm_gR64_l2": 2846.7275980323516,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 5.960464477539062e-07,
        "abs_error_over_ulp_gR": 20480.0,
        "abs_error_over_ulp_sum_gU_old": 20480.0,
        "argmax_flat_index": 31749,
        "argmax_multi_index_row_major": [
          124,
          5
        ],
        "floor_1e_6_active": false,
        "gR_value": 0.0003914833068847656,
        "max_rel_old": 0.0015225334791466594,
        "old_denominator": 0.0003914833068847656,
        "sum_gU_old_value": 0.0003908872604370117,
        "ulp_gR": 2.9103830456733704e-11,
        "ulp_sum_gU_old": 2.9103830456733704e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.522867294154064e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.880082339969023e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 3.0517578125e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.337860107421875e-06
        }
      },
      "metrics": {
        "E_L2": 3.522867294154064e-08,
        "E_inf": 5.880082339969023e-08,
        "max_abs": 3.337860107421875e-06,
        "max_rel_old": 0.0009135200974421437,
        "norm_S64_inf": 56.76553201675415,
        "norm_S64_l2": 2846.7275584024933,
        "norm_gR64_inf": 56.765533447265625,
        "norm_gR64_l2": 2846.7275590258982,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 56.765533447265625,
        "M64": 56.765533447265625,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_O": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 1.0070727029128616e-16
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.2508176379977552e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 1.0070727029128616e-16,
        "E_inf": 2.2508176379977552e-16,
        "max_abs": 2.842170943040401e-14,
        "max_rel_old": 9.938845022395866e-13,
        "norm_S64_inf": 126.27282170974507,
        "norm_S64_l2": 6039.725965821807,
        "norm_gR64_inf": 126.27282170974507,
        "norm_gR64_l2": 6039.725965821808,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 2.384185791015625e-07,
        "abs_error_over_ulp_gR": 2048.0,
        "abs_error_over_ulp_sum_gU_old": 2048.0,
        "argmax_flat_index": 16132,
        "argmax_multi_index_row_major": [
          63,
          4
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.0012345314025878906,
        "max_rel_old": 0.00019312475342303514,
        "old_denominator": 0.0012345314025878906,
        "sum_gU_old_value": -0.001234292984008789,
        "ulp_gR": 1.1641532182693481e-10,
        "ulp_sum_gU_old": 1.1641532182693481e-10
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.7427512883128104e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 6.041993197063075e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 6.103515625e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 7.62939453125e-06
        }
      },
      "metrics": {
        "E_L2": 3.7427512883128104e-08,
        "E_inf": 6.041993197063075e-08,
        "max_abs": 7.62939453125e-06,
        "max_rel_old": 0.00019312475859405175,
        "norm_S64_inf": 126.27280902862549,
        "norm_S64_l2": 6039.726097669936,
        "norm_gR64_inf": 126.2728042602539,
        "norm_gR64_l2": 6039.726099074056,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 126.27281188964844,
        "M64": 126.27280902862549,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_Q": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 1.0104021198818964e-16
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.215062223760889e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 1.0104021198818964e-16,
        "E_inf": 2.215062223760889e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 1.2029523112423958e-12,
        "norm_S64_inf": 64.1555553734008,
        "norm_S64_l2": 2875.479983690207,
        "norm_gR64_inf": 64.1555553734008,
        "norm_gR64_l2": 2875.479983690207,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 5.960464477539063e-08,
        "abs_error_over_ulp_gR": 8192.0,
        "abs_error_over_ulp_sum_gU_old": 8192.0,
        "argmax_flat_index": 33402,
        "argmax_multi_index_row_major": [
          130,
          122
        ],
        "floor_1e_6_active": false,
        "gR_value": 9.500980377197266e-05,
        "max_rel_old": 0.000627352565061301,
        "old_denominator": 9.500980377197266e-05,
        "sum_gU_old_value": 9.495019912719727e-05,
        "ulp_gR": 7.275957614183426e-12,
        "ulp_sum_gU_old": 7.275957614183426e-12
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.515051693773517e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.2027611529119525e-08
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
        "E_L2": 3.515051693773517e-08,
        "E_inf": 5.2027611529119525e-08,
        "max_abs": 3.337860107421875e-06,
        "max_rel_old": 0.0004231013327691982,
        "norm_S64_inf": 64.15555143356323,
        "norm_S64_l2": 2875.4799624595553,
        "norm_gR64_inf": 64.15554809570312,
        "norm_gR64_l2": 2875.479961939989,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 64.15554809570312,
        "M64": 64.15555143356323,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_V": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.96773343197353e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.234759694461516e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.96773343197353e-17,
        "E_inf": 2.234759694461516e-16,
        "max_abs": 2.842170943040401e-14,
        "max_rel_old": 2.5520986103726668e-12,
        "norm_S64_inf": 127.18015946342032,
        "norm_S64_l2": 5904.170424590143,
        "norm_gR64_inf": 127.18015946342034,
        "norm_gR64_l2": 5904.170424590143,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 6.854534149169922e-07,
        "abs_error_over_ulp_gR": 23552.0,
        "abs_error_over_ulp_sum_gU_old": 23552.0,
        "argmax_flat_index": 18360,
        "argmax_multi_index_row_major": [
          71,
          184
        ],
        "floor_1e_6_active": false,
        "gR_value": 0.0004329681396484375,
        "max_rel_old": 0.001580647425726056,
        "old_denominator": 0.0004336535930633545,
        "sum_gU_old_value": 0.0004336535930633545,
        "ulp_gR": 2.9103830456733704e-11,
        "ulp_sum_gU_old": 2.9103830456733704e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.757102251081795e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 7.498610057639716e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 6.103515625e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 9.5367431640625e-06
        }
      },
      "metrics": {
        "E_L2": 3.757102251081795e-08,
        "E_inf": 7.498610057639716e-08,
        "max_abs": 9.5367431640625e-06,
        "max_rel_old": 0.0025940337224383916,
        "norm_S64_inf": 127.18014335632324,
        "norm_S64_l2": 5904.170600830369,
        "norm_gR64_inf": 127.18014526367188,
        "norm_gR64_l2": 5904.1706020209185,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 127.18014526367188,
        "M64": 127.18014526367188,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_down": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.911584532779591e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.0959337175428843e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.911584532779591e-17,
        "E_inf": 2.0959337175428843e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 6.553323615508616e-13,
        "norm_S64_inf": 67.80202349080841,
        "norm_S64_l2": 4112.758538090043,
        "norm_gR64_inf": 67.80202349080841,
        "norm_gR64_l2": 4112.758538090043,
        "shape": [
          256,
          1024
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 4.76837158203125e-07,
        "abs_error_over_ulp_gR": 16384.0,
        "abs_error_over_ulp_sum_gU_old": 16384.0,
        "argmax_flat_index": 185693,
        "argmax_multi_index_row_major": [
          181,
          349
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.0004448890686035156,
        "max_rel_old": 0.0010706637986004353,
        "old_denominator": 0.00044536590576171875,
        "sum_gU_old_value": -0.00044536590576171875,
        "ulp_gR": 2.9103830456733704e-11,
        "ulp_sum_gU_old": 2.9103830456733704e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.6797254646768494e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.626229753168299e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 6.103515625e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.814697265625e-06
        }
      },
      "metrics": {
        "E_L2": 3.6797254646768494e-08,
        "E_inf": 5.626229753168299e-08,
        "max_abs": 3.814697265625e-06,
        "max_rel_old": 0.0005356186395286556,
        "norm_S64_inf": 67.8020133972168,
        "norm_S64_l2": 4112.758639390706,
        "norm_gR64_inf": 67.80201721191406,
        "norm_gR64_l2": 4112.7586400799955,
        "shape": [
          256,
          1024
        ]
      },
      "scale_context": {
        "M32": 67.80201721191406,
        "M64": 67.80201721191406,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_gate": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.93226458811873e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.677720214643296e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.93226458811873e-17,
        "E_inf": 2.677720214643296e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 2.992845211220427e-11,
        "norm_S64_inf": 53.07072276442092,
        "norm_S64_l2": 4074.4043173105365,
        "norm_gR64_inf": 53.07072276442092,
        "norm_gR64_l2": 4074.404317310536,
        "shape": [
          1024,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 1.1920928955078125e-07,
        "abs_error_over_ulp_gR": 16384.0,
        "abs_error_over_ulp_sum_gU_old": 16384.0,
        "argmax_flat_index": 81651,
        "argmax_multi_index_row_major": [
          318,
          243
        ],
        "floor_1e_6_active": false,
        "gR_value": -8.273124694824219e-05,
        "max_rel_old": 0.0014409221475943923,
        "old_denominator": 8.273124694824219e-05,
        "sum_gU_old_value": -8.26120376586914e-05,
        "ulp_gR": 7.275957614183426e-12,
        "ulp_sum_gU_old": 7.275957614183426e-12
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.6690476254525254e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 8.984939085706974e-08
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
        "E_L2": 3.6690476254525254e-08,
        "E_inf": 8.984939085706974e-08,
        "max_abs": 4.76837158203125e-06,
        "max_rel_old": 0.004934210526315789,
        "norm_S64_inf": 53.07071614265442,
        "norm_S64_l2": 4074.4044408030786,
        "norm_gR64_inf": 53.070716857910156,
        "norm_gR64_l2": 4074.404441263053,
        "shape": [
          1024,
          256
        ]
      },
      "scale_context": {
        "M32": 53.070716857910156,
        "M64": 53.070716857910156,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_up": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.948636147624094e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.290787307574054e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.948636147624094e-17,
        "E_inf": 2.290787307574054e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 1.4017744175210424e-12,
        "norm_S64_inf": 62.03480641007789,
        "norm_S64_l2": 4026.970021140756,
        "norm_gR64_inf": 62.0348064100779,
        "norm_gR64_l2": 4026.970021140756,
        "shape": [
          1024,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 1.1920928955078125e-07,
        "abs_error_over_ulp_gR": 16384.0,
        "abs_error_over_ulp_sum_gU_old": 16384.0,
        "argmax_flat_index": 195972,
        "argmax_multi_index_row_major": [
          765,
          132
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.00010663270950317383,
        "max_rel_old": 0.0011179429711773992,
        "old_denominator": 0.00010663270950317383,
        "sum_gU_old_value": -0.00010651350021362305,
        "ulp_gR": 7.275957614183426e-12,
        "ulp_sum_gU_old": 7.275957614183426e-12
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.656364092133642e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 7.686605886316697e-08
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
        "E_L2": 3.656364092133642e-08,
        "E_inf": 7.686605886316697e-08,
        "max_abs": 4.76837158203125e-06,
        "max_rel_old": 0.0007552870090634441,
        "norm_S64_inf": 62.03480911254883,
        "norm_S64_l2": 4026.970108267898,
        "norm_gR64_inf": 62.034812927246094,
        "norm_gR64_l2": 4026.9701080123496,
        "shape": [
          1024,
          256
        ]
      },
      "scale_context": {
        "M32": 62.034812927246094,
        "M64": 62.034812927246094,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  }
}
```

### Persisted gradient bundle

```json
{
  "bundle_path": "seed_20261003_raw_gradients.pt",
  "bundle_sha256": "b5794d5cf47e03766cd01a78745dcb02d1af44431c8b2b88622cdfb5dd174c01",
  "bundle_size_bytes": 105685741,
  "tensor_metadata": {
    "seed_20261003/cpu_fp64_oracle/W_K/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "c70e9f75561a1db0fe144b9e25cf794433d6b860abdebe5fbf6cc5053dc381c4",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_K/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "6a5dcbbc3affed34bda601b69b486bbefd603e3806e5dfebd1eaf4dfa5e5f6b0",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_K/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "9e511a4c2825092946df6a1666a126cb1f5ff190f4d5ab904df805a0d16d2645",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_K/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "8c32060c88298c57c615b1c053df5d8d5a627da98489c517911961a16dd63e53",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_K/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "157d19ecfbb81c3f2788af810ab39750097ccbffa4e1eaacbff591f7abf33097",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_K/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "164bafc75c8db2f14c9c8db47fb3bbce6da4a7f4b84781337b4e330a618ea589",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_O/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "71d9ba8c54e95d7f4a474fce33fc268f2500ba4bf24ec9c4033f68212cbe962a",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_O/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "6d6ec6e630daeb2598bd75a1a6e2f912b376b7825d5afcad8433f93f5a6ddfa6",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_O/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "676afe9b54e2314fad5fdbbcf971c564322a11f8d0b68ef80ad89d835bb5f7f4",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_O/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "ca861c5840054204698125bfd644547fd5c40b079c967b61f4b62fd7b5d20d32",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_O/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "9d903c5309445d146c86e0aed1b0b093bcb6ce6cf0dfe3b8be80f2dd10b60b8c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_O/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "99f099a71427cd860f098408be31cceb81fefb80a7c96625dbf2bcc57c62c7d9",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_Q/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "55ad3037907618f88c06fad04668d84304aa3469df330e9a4c84d4e08658d61b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_Q/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "c5732071c47897b267e10063360eab4ac145485936fec924c309847a895c3687",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_Q/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "b1c1cc533207250d037e98ed8ac67c718b01e21f4b35361330cb154e01a0207e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_Q/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "df606116f446c6112e0204cf03065b782930ed703eb3d1108d3bbfd86085e059",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_Q/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "1adf998635118ed5d5f6f587ff420cd2f744ac86711995913a9be232c2b037b1",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_Q/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "270570567858060628b0a6f6152becd4224b4c543a7d7af202645bb0215c0bd9",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_V/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "a205a6332c94020eed69889f4a74e5179c884c7234edf0525eba4253b91dfc92",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_V/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "831fccf9183f478b754532d4d744e306fa2f7eb7dab19feef947bb8c413c2ae0",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_V/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "f5f5fdf1f871132036f57cbcbecdd91d3f6056fb65585d9225c548ff8b476a49",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_V/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "29b5ed08d2904f4e0e4b1118cac722d8e02a40d6e9e399919e0704d90ca1fae7",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_V/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "7462cb45e526be1a86950f7df9e41939973dd4d0b1471d9302dc3c40d92f3e40",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_V/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "f8de95ba288d3cc90f44f28c716668bb349b087ae9bfd2d8caae51cfe2c4b2eb",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_down/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "642e229c57f1681ee9753363f265aee3d5b6f543f9aca5af9f06c5205bd7999f",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_down/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "a8c14f7c8dc95ab59adce195960c61c2d40763c60a3193d277dcaac017c64bfc",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_down/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "1201c0eac93d95329ddeacfe959fd2f67d520208c4426a032e0108121a91e637",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_down/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "67c4199e96c6b37cee2fb83cbe9ba8f7addefe48d6da29abff6d4ef6afea1ba0",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_down/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "624b4710a2b1f157a26cf426bfff3042aabdc521af69ebad9fba069199123123",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_down/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "fd923a9ef842760fe8fac73eec6de95a50ca0ed17cb58d3cc52df5cea9271017",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_gate/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "3a10dab1838fec65b091c06ef90a3b19cc5eb18979d76f15a22e51cb8ee8cce0",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_gate/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "8e63d368269be81cbf515635eafe018fbdac7910f224f50f6d289e932563d8a6",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_gate/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "4debc3f08e740a1e48e74464525a42b8f0a18d9ff98788bbf1072a2932f5f378",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_gate/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "40e25aada9fae696ea39a55f23dc98bbd95ca5f28df209fb590bcc84a4ebbfc4",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_gate/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "0b73f5e12a5dc9e01c930991b3c8987f9647b02ca59b04af4e781f67265bb083",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_gate/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "5696c03a0ef6f321132039603e0487099d30777c01bb4356bfc66f387bdc8ddb",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_up/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "01e37355d0a2f2cfa6bf649f517c5876b05d2f1a12a105306a9270d3dab88899",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_up/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "4c5489c066d9f530244fc654f5a8801ac57baeddc08ea1b60710ad5a17f90a9d",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_up/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "1ca20f4fe245bd1764d97f23c15ff272a41df7b1e9bbfd447a54c954678f99bd",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_up/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "4d1eda9d2e3b9ab0f842c2e451df8b3a0b51cbc6322ea371bb1641dd011a3dc2",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_up/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "5394335719f3705aa6ff2f6d6ab98f4389e756b5e4a24af34d07e2c11cf15eab",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cpu_fp64_oracle/W_up/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "4a48d89c50ed7c7eb971cf782309a00d2a4c3d193726d9ee96d723702588bd87",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_K/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "ad95a2217f25e7a08b1ef335aedbf7380a1bbdcf87eceb34f640b3db6c09dec3",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_K/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "1f1e3be3676e6aeaa556a011843fe4ba29b1d707c6f96fc2bb148debe08eef85",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_K/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "100f84e7b6094cfcc29eedb59f8f025e9d243154040565ff5d7d63eda2c7d829",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_K/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "183c53a7fc9350a690e4eb6f07f36cd23153df998a37d00e8c08341fc52342cd",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_K/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "56d6a8b73ab54e2a33ac6c925d14fc23d77851db3d478a39de716e1f23a61db2",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_K/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "599126cba92f118723411325540ea2bfce5b2c89a5fc1e2a32f75247f59f3be2",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_K/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "8622fdbcbc8fd44144d047add0f8bdc18aa3a81e4cdf643f2c508274dafccf4e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_K/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "55e5454e84e30cf465610872b042e53fbe8bdbfbfc9a42bc86d37ca5711f8ee2",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_K/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "29d4a0a7c7555b6f598e8acf55f332a4078d131afc165fc23d0a44f475817b54",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_K/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "f0a50851d9bb4f95fbf9aed1bbe4fa8b0d27ebaca988c1db5b3aaa9a9c4ad4cb",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_K/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "a9ac9efd42033603ca4a6f301f0fdbd2178d18dbf0ebbd27e493eb7a0b668ae6",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_O/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "58f82edc974e60d124eb0c4cdc93341bd602b03fbc1ea1448fbbef560b25de5f",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_O/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "800f4ed2a5d9181c76fae2770d7ac336a63be7b8917b56659093a2fb54bd87d3",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_O/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "e19d7ad36a3cfc056bffeab0518212855a907293ab3adbdcd440947aa34596fd",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_O/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "1a1121944e896f775450b60ae33b497b97c9b76400a6c5dde5b7ce7d4c01ead7",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_O/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "c63ed988167acd231bde25079371a8bd4a3d73f4480d29d7c0cdcd8c65ab7497",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_O/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "55e118e1642a7e86d356cad2906e3f4d4bc75785681f847ef351c4049b8753ca",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_O/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "bb2cb6845245ab0cd05a995bab8c40b262b3af5a2342e9e028bd618a15eaaa10",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_O/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "4dd9887d5d73874a9a130a9a3df6b80d41b8472382997b099898417b8c4388d5",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_O/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "2a1e836802abc9b774b5e9ed955727a35f4d66f38e8df394ff40dab0f4f44002",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_O/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "d0600817c7fd6b68680606e95147af6777d5bacdd887702a08e2a7fb1aa1b5c4",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_O/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "88c76f4ea03bb3b3fd302ac71b4d1b9792c10b70397fda01c032c2d98161b678",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_Q/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "18843a38679f80b0771862ed2a24efa7608e85a04f52b7137f2a5382e2fd5bcd",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_Q/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "9017299ecc3218a2cfbf08456d83708e200b0db47ecb7aa140dabbfef411215c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_Q/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "e33b029049f880e63cd3f27489299f73409dad7af867618c0dffc4d7344aee75",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_Q/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "f5e5a0d2502de0b35b74c83b08b84931a2157b84c62c8e21aaf2be075dc0e679",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_Q/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "5a07e879f03cfba10c5188a45d22c2649026ff5519369191365a6a723bda11da",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_Q/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "cc5cc10bfb2712d01e0daacd574ae49bde11d1bce73f98240c840ac6b4ecf23d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_Q/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "6ec0ff876697090f515d13ca00f329cffa34a633058f99a20eca8a155f6acdda",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_Q/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "bd4b1cc699597ebbf6be13879506407950289ced1b0f0aec411697b2c83a2428",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_Q/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "69f05c4852d18fd4df1e46682998c991895f30845e7b9ff8692a2cd6b9687e0f",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_Q/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "007e489a3413d32ea3e973b2370063d97a0a677b94d7a4df5daade2f670cf23f",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_Q/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "c57fffe18000c1019153aae94824e3f1a49040a384ffcee23846a1af0253f948",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_V/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "38aa24876bf1562efb76d9a696dd5f43262c94625b85494bfbee4a744f24d05e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_V/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "e26960274e4bde26a527ff340e69bd906883e6a680b139a604c636cfc500d63c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_V/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "add2199b3b98ad76b352bd40d39122c0b813ac4c4562b168bfcff5e06211faf2",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_V/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "a7085e23c00fa70da3f567a5b7153cbf65c3fd51b62aa5123d60d47b52af6c5b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_V/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "168b0d4d0e764ee256b4a68fb896c1cff68df96c22a0992b3ba917ccd13b119e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_V/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "cb448cd413d3971a02f2d50c015e11f74e25f3b530a13197a28d29ef175e1db3",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_V/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "80e3cfa23d9e723ad8817196b89d385de7fb31f0d15122e0b5ff8b13538f1ff0",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_V/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "e38f25f0f9303860f8b8453ea6517208cd2ec0edaa987b859aa60284004ccd10",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_V/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "51c456f3c5e8a465444949733625ee7c0467818146e3bf8760f6da2d37628861",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_V/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "9ee5a74e49c805456b49d29790a57bab6ed4902ee1de899186fdc8405e5195a1",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_V/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "71c8b373d257c37aa0e61d6ebd856b2659aa4711c6327031092c67858cf7003a",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_down/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "d7905e73ad5a9935a76f33e3bb62a5e7573a1dc021a76e7ec07d22f6cc0bbcaf",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cuda_fp32/W_down/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "e1bab9424a264ecb013d5e984c65956ece05b097dfcfd536a0913b5a87a99d86",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cuda_fp32/W_down/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "746b9eb625898252a707717f702070935cc07ba80df43b79cabd367e28905540",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cuda_fp32/W_down/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "60f4b43a439767f75f7941f40ca5f35fa6457f6935f9a3d37803604c77a3c03a",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cuda_fp32/W_down/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "21e61dcf498f49042bfc0f662ff1536c96518a018195b58a8bf3ed18cfb8b207",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cuda_fp32/W_down/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "7d5abc519649918111bbab718888c8edc4e17938c5b6ce8705ef23902902259c",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cuda_fp32/W_down/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "21d4905a25b8cca944ba6958f0751737545791f0371410234a8a85e8d5399a87",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cuda_fp32/W_down/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "bca06fc73c04e682caec3e6f139d66fa63f24429377fd241f9dfa4c1b06dba80",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cuda_fp32/W_down/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "a81cef1834b255fd42c24274d88a5de9c1156a6a0cf2d8a03af71928a00b7c8e",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cuda_fp32/W_down/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "de63d0da2c8ed4bd4f0420cff837b4a940315dc27fcef197f028626197522063",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cuda_fp32/W_down/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "a951e697424770e7913f6dfbf9c4da5cbb1b24625489e3cfbd1eb394d32ff942",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261003/cuda_fp32/W_gate/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "5d5c6df58e769073c02a0ab0637413d39fb4fd0a18f91a690014960af375f658",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_gate/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "e2e8bbe8b47f33e98009d24ec25b14124f52c046b6bd41be6bcd631bce5e24f7",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_gate/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "49a64062a02f1a6cba89e64b3ad25fed8e78a20ff8107ddf775f7d1eabede121",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_gate/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "2ca6873697c7a462319b5e77090df8258070b40c9931d216993dd85a665a27d7",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_gate/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "a5cc58137bc7127496567cdfef93f5bc055148ef396fdb2de47bf0f0ba31904c",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_gate/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "77c22a139bc5f4ed3f9559bfb558c9d49bf160ae95a1a01fc8088fd2a7760a32",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_gate/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "3cf76f048068de65f6dc967042454a831d9fa86b5461091b918e0f32cb4d3cdd",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_gate/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "9dd7ffaf5cde6ae34df65c9419070ac60fc0dcab838007c1b114cb811abbf746",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_gate/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "7081dcfec17e847aa30e0ef587e1589b98e7d15400b900ef489acdc65fbb39e6",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_gate/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "ef56d401230a90b973d719960c63607848d52f682a9e243530e9fa29fb0cdb98",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_gate/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "cd01b6d3564053eb5e3104bb59fc2844f8a12d29c7ddd500a49d80cf9ff52e89",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_up/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "d4dec8c7276a207a37fdee2c18bed1a931d3a70f006622b1914b54f20c6b8387",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_up/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "89bdf25338db0290f1a4bf7192808d9b0a040300bc626dbd2574d47105c1d994",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_up/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "4ca4909c92204fbfbfa34ef638e3c84766942a8ec686de14224deb79465c1cc9",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_up/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "7e95e2e2fe2749b115d8360cd68fe3fc15f2b483acd8a938545d565ae091db24",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_up/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "babf14aeff22948e66be374281484cbc0d8d03df869f2d0fa99c5aad45f3c7b8",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_up/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "d2f4cbe4354e68cd6dc1a883d979d24b0cd4307ca419cbd0603befa25924bb48",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_up/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "1ae656c67b7c10dbeb0f744fe1d73625b964700188be19534084ae8c8e0fcc85",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_up/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "73a89ea88da2cc41eb00dee8ded32d03a1d079027e11e5a80487cae8eb536c95",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_up/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "5dcf11983138466a74a3520b967dbf6d443dd4c30c3ee3dbd5742e0cb65d9a3a",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_up/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "b9c101e43eab7ea63cae9035c8edbcaf27bde16da0bcafc86ec74bd3e62e1087",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/cuda_fp32/W_up/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "e8233acfb81c14abaec82c56d8163a1983943780d439f78e1a56a9c25c57c7a5",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261003/inputs/w_fp32_cpu": {
      "dtype": "torch.float32",
      "raw_sha256": "7cf1f870d7e90692b6207db0c222de69a6fc713c4ba49907db48f1dc6fa6baa6",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261003/inputs/x_fp32_cpu": {
      "dtype": "torch.float32",
      "raw_sha256": "726b11badbb1a0037e27c826b4796b51eecab8fac8b07b5696977360276f0e2a",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261003/structural_traces/R4/round_1": {
      "dtype": "torch.float32",
      "raw_sha256": "b5e2ab917aede3b9965d89ed5712618ec164306ad115a866c09610c48484d007",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261003/structural_traces/R4/round_2": {
      "dtype": "torch.float32",
      "raw_sha256": "9784185754dccb48d970e9ea2f8baa697aec1d5aaa87085fe166274f335bfc40",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261003/structural_traces/R4/round_3": {
      "dtype": "torch.float32",
      "raw_sha256": "05e0344a9cd998e39561cf07967b50b13669fb32417d54aeae287052685b274e",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261003/structural_traces/R4/round_4": {
      "dtype": "torch.float32",
      "raw_sha256": "9d83f259843ef549761c29e99707bcdd27837359db6372300457ebcdec6d4647",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261003/structural_traces/R4_final": {
      "dtype": "torch.float32",
      "raw_sha256": "4b2fd2ada2926bb8b9d3acc44e538ebbdbad35b1ae8e15b8cc9db60b96031e6d",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261003/structural_traces/U4/round_1": {
      "dtype": "torch.float32",
      "raw_sha256": "1ebbfcfb8ba4c32916b8dda4ae5b5e2af2dc669c4abeebd6eb2a0e042bc66193",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261003/structural_traces/U4/round_2": {
      "dtype": "torch.float32",
      "raw_sha256": "25e9fcca7ebf6cec633d98e23138484a378c93aab7e280185141487d20129619",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261003/structural_traces/U4/round_3": {
      "dtype": "torch.float32",
      "raw_sha256": "7869f36f9572833ebbe9b6e1b834220308ffd6a33e73b45d4e947bba5736a8d9",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261003/structural_traces/U4/round_4": {
      "dtype": "torch.float32",
      "raw_sha256": "73f63ebeda218f60b862b6c26b9ef10818bf16e6f9406d6fc22324d19eb7ca8b",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261003/structural_traces/U4_final": {
      "dtype": "torch.float32",
      "raw_sha256": "2004c5e5dc7d048bd77ff7c113e59fbbb2f7a4ba3825a2f1b76cea6a7a1e44b3",
      "shape": [
        8,
        8,
        256
      ]
    }
  },
  "tensor_raw_sha256": {
    "seed_20261003/cpu_fp64_oracle/W_K/S64_cpu": "c70e9f75561a1db0fe144b9e25cf794433d6b860abdebe5fbf6cc5053dc381c4",
    "seed_20261003/cpu_fp64_oracle/W_K/gR64_cpu": "6a5dcbbc3affed34bda601b69b486bbefd603e3806e5dfebd1eaf4dfa5e5f6b0",
    "seed_20261003/cpu_fp64_oracle/W_K/gU0_64_cpu": "9e511a4c2825092946df6a1666a126cb1f5ff190f4d5ab904df805a0d16d2645",
    "seed_20261003/cpu_fp64_oracle/W_K/gU1_64_cpu": "8c32060c88298c57c615b1c053df5d8d5a627da98489c517911961a16dd63e53",
    "seed_20261003/cpu_fp64_oracle/W_K/gU2_64_cpu": "157d19ecfbb81c3f2788af810ab39750097ccbffa4e1eaacbff591f7abf33097",
    "seed_20261003/cpu_fp64_oracle/W_K/gU3_64_cpu": "164bafc75c8db2f14c9c8db47fb3bbce6da4a7f4b84781337b4e330a618ea589",
    "seed_20261003/cpu_fp64_oracle/W_O/S64_cpu": "71d9ba8c54e95d7f4a474fce33fc268f2500ba4bf24ec9c4033f68212cbe962a",
    "seed_20261003/cpu_fp64_oracle/W_O/gR64_cpu": "6d6ec6e630daeb2598bd75a1a6e2f912b376b7825d5afcad8433f93f5a6ddfa6",
    "seed_20261003/cpu_fp64_oracle/W_O/gU0_64_cpu": "676afe9b54e2314fad5fdbbcf971c564322a11f8d0b68ef80ad89d835bb5f7f4",
    "seed_20261003/cpu_fp64_oracle/W_O/gU1_64_cpu": "ca861c5840054204698125bfd644547fd5c40b079c967b61f4b62fd7b5d20d32",
    "seed_20261003/cpu_fp64_oracle/W_O/gU2_64_cpu": "9d903c5309445d146c86e0aed1b0b093bcb6ce6cf0dfe3b8be80f2dd10b60b8c",
    "seed_20261003/cpu_fp64_oracle/W_O/gU3_64_cpu": "99f099a71427cd860f098408be31cceb81fefb80a7c96625dbf2bcc57c62c7d9",
    "seed_20261003/cpu_fp64_oracle/W_Q/S64_cpu": "55ad3037907618f88c06fad04668d84304aa3469df330e9a4c84d4e08658d61b",
    "seed_20261003/cpu_fp64_oracle/W_Q/gR64_cpu": "c5732071c47897b267e10063360eab4ac145485936fec924c309847a895c3687",
    "seed_20261003/cpu_fp64_oracle/W_Q/gU0_64_cpu": "b1c1cc533207250d037e98ed8ac67c718b01e21f4b35361330cb154e01a0207e",
    "seed_20261003/cpu_fp64_oracle/W_Q/gU1_64_cpu": "df606116f446c6112e0204cf03065b782930ed703eb3d1108d3bbfd86085e059",
    "seed_20261003/cpu_fp64_oracle/W_Q/gU2_64_cpu": "1adf998635118ed5d5f6f587ff420cd2f744ac86711995913a9be232c2b037b1",
    "seed_20261003/cpu_fp64_oracle/W_Q/gU3_64_cpu": "270570567858060628b0a6f6152becd4224b4c543a7d7af202645bb0215c0bd9",
    "seed_20261003/cpu_fp64_oracle/W_V/S64_cpu": "a205a6332c94020eed69889f4a74e5179c884c7234edf0525eba4253b91dfc92",
    "seed_20261003/cpu_fp64_oracle/W_V/gR64_cpu": "831fccf9183f478b754532d4d744e306fa2f7eb7dab19feef947bb8c413c2ae0",
    "seed_20261003/cpu_fp64_oracle/W_V/gU0_64_cpu": "f5f5fdf1f871132036f57cbcbecdd91d3f6056fb65585d9225c548ff8b476a49",
    "seed_20261003/cpu_fp64_oracle/W_V/gU1_64_cpu": "29b5ed08d2904f4e0e4b1118cac722d8e02a40d6e9e399919e0704d90ca1fae7",
    "seed_20261003/cpu_fp64_oracle/W_V/gU2_64_cpu": "7462cb45e526be1a86950f7df9e41939973dd4d0b1471d9302dc3c40d92f3e40",
    "seed_20261003/cpu_fp64_oracle/W_V/gU3_64_cpu": "f8de95ba288d3cc90f44f28c716668bb349b087ae9bfd2d8caae51cfe2c4b2eb",
    "seed_20261003/cpu_fp64_oracle/W_down/S64_cpu": "642e229c57f1681ee9753363f265aee3d5b6f543f9aca5af9f06c5205bd7999f",
    "seed_20261003/cpu_fp64_oracle/W_down/gR64_cpu": "a8c14f7c8dc95ab59adce195960c61c2d40763c60a3193d277dcaac017c64bfc",
    "seed_20261003/cpu_fp64_oracle/W_down/gU0_64_cpu": "1201c0eac93d95329ddeacfe959fd2f67d520208c4426a032e0108121a91e637",
    "seed_20261003/cpu_fp64_oracle/W_down/gU1_64_cpu": "67c4199e96c6b37cee2fb83cbe9ba8f7addefe48d6da29abff6d4ef6afea1ba0",
    "seed_20261003/cpu_fp64_oracle/W_down/gU2_64_cpu": "624b4710a2b1f157a26cf426bfff3042aabdc521af69ebad9fba069199123123",
    "seed_20261003/cpu_fp64_oracle/W_down/gU3_64_cpu": "fd923a9ef842760fe8fac73eec6de95a50ca0ed17cb58d3cc52df5cea9271017",
    "seed_20261003/cpu_fp64_oracle/W_gate/S64_cpu": "3a10dab1838fec65b091c06ef90a3b19cc5eb18979d76f15a22e51cb8ee8cce0",
    "seed_20261003/cpu_fp64_oracle/W_gate/gR64_cpu": "8e63d368269be81cbf515635eafe018fbdac7910f224f50f6d289e932563d8a6",
    "seed_20261003/cpu_fp64_oracle/W_gate/gU0_64_cpu": "4debc3f08e740a1e48e74464525a42b8f0a18d9ff98788bbf1072a2932f5f378",
    "seed_20261003/cpu_fp64_oracle/W_gate/gU1_64_cpu": "40e25aada9fae696ea39a55f23dc98bbd95ca5f28df209fb590bcc84a4ebbfc4",
    "seed_20261003/cpu_fp64_oracle/W_gate/gU2_64_cpu": "0b73f5e12a5dc9e01c930991b3c8987f9647b02ca59b04af4e781f67265bb083",
    "seed_20261003/cpu_fp64_oracle/W_gate/gU3_64_cpu": "5696c03a0ef6f321132039603e0487099d30777c01bb4356bfc66f387bdc8ddb",
    "seed_20261003/cpu_fp64_oracle/W_up/S64_cpu": "01e37355d0a2f2cfa6bf649f517c5876b05d2f1a12a105306a9270d3dab88899",
    "seed_20261003/cpu_fp64_oracle/W_up/gR64_cpu": "4c5489c066d9f530244fc654f5a8801ac57baeddc08ea1b60710ad5a17f90a9d",
    "seed_20261003/cpu_fp64_oracle/W_up/gU0_64_cpu": "1ca20f4fe245bd1764d97f23c15ff272a41df7b1e9bbfd447a54c954678f99bd",
    "seed_20261003/cpu_fp64_oracle/W_up/gU1_64_cpu": "4d1eda9d2e3b9ab0f842c2e451df8b3a0b51cbc6322ea371bb1641dd011a3dc2",
    "seed_20261003/cpu_fp64_oracle/W_up/gU2_64_cpu": "5394335719f3705aa6ff2f6d6ab98f4389e756b5e4a24af34d07e2c11cf15eab",
    "seed_20261003/cpu_fp64_oracle/W_up/gU3_64_cpu": "4a48d89c50ed7c7eb971cf782309a00d2a4c3d193726d9ee96d723702588bd87",
    "seed_20261003/cuda_fp32/W_K/S64_primary_cuda": "ad95a2217f25e7a08b1ef335aedbf7380a1bbdcf87eceb34f640b3db6c09dec3",
    "seed_20261003/cuda_fp32/W_K/S_forward_cuda": "1f1e3be3676e6aeaa556a011843fe4ba29b1d707c6f96fc2bb148debe08eef85",
    "seed_20261003/cuda_fp32/W_K/S_pairwise_cuda": "100f84e7b6094cfcc29eedb59f8f025e9d243154040565ff5d7d63eda2c7d829",
    "seed_20261003/cuda_fp32/W_K/S_reverse_cuda": "183c53a7fc9350a690e4eb6f07f36cd23153df998a37d00e8c08341fc52342cd",
    "seed_20261003/cuda_fp32/W_K/S_stack_cuda": "56d6a8b73ab54e2a33ac6c925d14fc23d77851db3d478a39de716e1f23a61db2",
    "seed_20261003/cuda_fp32/W_K/gR64_primary_cuda": "599126cba92f118723411325540ea2bfce5b2c89a5fc1e2a32f75247f59f3be2",
    "seed_20261003/cuda_fp32/W_K/gR_cuda_fp32": "8622fdbcbc8fd44144d047add0f8bdc18aa3a81e4cdf643f2c508274dafccf4e",
    "seed_20261003/cuda_fp32/W_K/gU0_cuda_fp32": "55e5454e84e30cf465610872b042e53fbe8bdbfbfc9a42bc86d37ca5711f8ee2",
    "seed_20261003/cuda_fp32/W_K/gU1_cuda_fp32": "29d4a0a7c7555b6f598e8acf55f332a4078d131afc165fc23d0a44f475817b54",
    "seed_20261003/cuda_fp32/W_K/gU2_cuda_fp32": "f0a50851d9bb4f95fbf9aed1bbe4fa8b0d27ebaca988c1db5b3aaa9a9c4ad4cb",
    "seed_20261003/cuda_fp32/W_K/gU3_cuda_fp32": "a9ac9efd42033603ca4a6f301f0fdbd2178d18dbf0ebbd27e493eb7a0b668ae6",
    "seed_20261003/cuda_fp32/W_O/S64_primary_cuda": "58f82edc974e60d124eb0c4cdc93341bd602b03fbc1ea1448fbbef560b25de5f",
    "seed_20261003/cuda_fp32/W_O/S_forward_cuda": "800f4ed2a5d9181c76fae2770d7ac336a63be7b8917b56659093a2fb54bd87d3",
    "seed_20261003/cuda_fp32/W_O/S_pairwise_cuda": "e19d7ad36a3cfc056bffeab0518212855a907293ab3adbdcd440947aa34596fd",
    "seed_20261003/cuda_fp32/W_O/S_reverse_cuda": "1a1121944e896f775450b60ae33b497b97c9b76400a6c5dde5b7ce7d4c01ead7",
    "seed_20261003/cuda_fp32/W_O/S_stack_cuda": "c63ed988167acd231bde25079371a8bd4a3d73f4480d29d7c0cdcd8c65ab7497",
    "seed_20261003/cuda_fp32/W_O/gR64_primary_cuda": "55e118e1642a7e86d356cad2906e3f4d4bc75785681f847ef351c4049b8753ca",
    "seed_20261003/cuda_fp32/W_O/gR_cuda_fp32": "bb2cb6845245ab0cd05a995bab8c40b262b3af5a2342e9e028bd618a15eaaa10",
    "seed_20261003/cuda_fp32/W_O/gU0_cuda_fp32": "4dd9887d5d73874a9a130a9a3df6b80d41b8472382997b099898417b8c4388d5",
    "seed_20261003/cuda_fp32/W_O/gU1_cuda_fp32": "2a1e836802abc9b774b5e9ed955727a35f4d66f38e8df394ff40dab0f4f44002",
    "seed_20261003/cuda_fp32/W_O/gU2_cuda_fp32": "d0600817c7fd6b68680606e95147af6777d5bacdd887702a08e2a7fb1aa1b5c4",
    "seed_20261003/cuda_fp32/W_O/gU3_cuda_fp32": "88c76f4ea03bb3b3fd302ac71b4d1b9792c10b70397fda01c032c2d98161b678",
    "seed_20261003/cuda_fp32/W_Q/S64_primary_cuda": "18843a38679f80b0771862ed2a24efa7608e85a04f52b7137f2a5382e2fd5bcd",
    "seed_20261003/cuda_fp32/W_Q/S_forward_cuda": "9017299ecc3218a2cfbf08456d83708e200b0db47ecb7aa140dabbfef411215c",
    "seed_20261003/cuda_fp32/W_Q/S_pairwise_cuda": "e33b029049f880e63cd3f27489299f73409dad7af867618c0dffc4d7344aee75",
    "seed_20261003/cuda_fp32/W_Q/S_reverse_cuda": "f5e5a0d2502de0b35b74c83b08b84931a2157b84c62c8e21aaf2be075dc0e679",
    "seed_20261003/cuda_fp32/W_Q/S_stack_cuda": "5a07e879f03cfba10c5188a45d22c2649026ff5519369191365a6a723bda11da",
    "seed_20261003/cuda_fp32/W_Q/gR64_primary_cuda": "cc5cc10bfb2712d01e0daacd574ae49bde11d1bce73f98240c840ac6b4ecf23d",
    "seed_20261003/cuda_fp32/W_Q/gR_cuda_fp32": "6ec0ff876697090f515d13ca00f329cffa34a633058f99a20eca8a155f6acdda",
    "seed_20261003/cuda_fp32/W_Q/gU0_cuda_fp32": "bd4b1cc699597ebbf6be13879506407950289ced1b0f0aec411697b2c83a2428",
    "seed_20261003/cuda_fp32/W_Q/gU1_cuda_fp32": "69f05c4852d18fd4df1e46682998c991895f30845e7b9ff8692a2cd6b9687e0f",
    "seed_20261003/cuda_fp32/W_Q/gU2_cuda_fp32": "007e489a3413d32ea3e973b2370063d97a0a677b94d7a4df5daade2f670cf23f",
    "seed_20261003/cuda_fp32/W_Q/gU3_cuda_fp32": "c57fffe18000c1019153aae94824e3f1a49040a384ffcee23846a1af0253f948",
    "seed_20261003/cuda_fp32/W_V/S64_primary_cuda": "38aa24876bf1562efb76d9a696dd5f43262c94625b85494bfbee4a744f24d05e",
    "seed_20261003/cuda_fp32/W_V/S_forward_cuda": "e26960274e4bde26a527ff340e69bd906883e6a680b139a604c636cfc500d63c",
    "seed_20261003/cuda_fp32/W_V/S_pairwise_cuda": "add2199b3b98ad76b352bd40d39122c0b813ac4c4562b168bfcff5e06211faf2",
    "seed_20261003/cuda_fp32/W_V/S_reverse_cuda": "a7085e23c00fa70da3f567a5b7153cbf65c3fd51b62aa5123d60d47b52af6c5b",
    "seed_20261003/cuda_fp32/W_V/S_stack_cuda": "168b0d4d0e764ee256b4a68fb896c1cff68df96c22a0992b3ba917ccd13b119e",
    "seed_20261003/cuda_fp32/W_V/gR64_primary_cuda": "cb448cd413d3971a02f2d50c015e11f74e25f3b530a13197a28d29ef175e1db3",
    "seed_20261003/cuda_fp32/W_V/gR_cuda_fp32": "80e3cfa23d9e723ad8817196b89d385de7fb31f0d15122e0b5ff8b13538f1ff0",
    "seed_20261003/cuda_fp32/W_V/gU0_cuda_fp32": "e38f25f0f9303860f8b8453ea6517208cd2ec0edaa987b859aa60284004ccd10",
    "seed_20261003/cuda_fp32/W_V/gU1_cuda_fp32": "51c456f3c5e8a465444949733625ee7c0467818146e3bf8760f6da2d37628861",
    "seed_20261003/cuda_fp32/W_V/gU2_cuda_fp32": "9ee5a74e49c805456b49d29790a57bab6ed4902ee1de899186fdc8405e5195a1",
    "seed_20261003/cuda_fp32/W_V/gU3_cuda_fp32": "71c8b373d257c37aa0e61d6ebd856b2659aa4711c6327031092c67858cf7003a",
    "seed_20261003/cuda_fp32/W_down/S64_primary_cuda": "d7905e73ad5a9935a76f33e3bb62a5e7573a1dc021a76e7ec07d22f6cc0bbcaf",
    "seed_20261003/cuda_fp32/W_down/S_forward_cuda": "e1bab9424a264ecb013d5e984c65956ece05b097dfcfd536a0913b5a87a99d86",
    "seed_20261003/cuda_fp32/W_down/S_pairwise_cuda": "746b9eb625898252a707717f702070935cc07ba80df43b79cabd367e28905540",
    "seed_20261003/cuda_fp32/W_down/S_reverse_cuda": "60f4b43a439767f75f7941f40ca5f35fa6457f6935f9a3d37803604c77a3c03a",
    "seed_20261003/cuda_fp32/W_down/S_stack_cuda": "21e61dcf498f49042bfc0f662ff1536c96518a018195b58a8bf3ed18cfb8b207",
    "seed_20261003/cuda_fp32/W_down/gR64_primary_cuda": "7d5abc519649918111bbab718888c8edc4e17938c5b6ce8705ef23902902259c",
    "seed_20261003/cuda_fp32/W_down/gR_cuda_fp32": "21d4905a25b8cca944ba6958f0751737545791f0371410234a8a85e8d5399a87",
    "seed_20261003/cuda_fp32/W_down/gU0_cuda_fp32": "bca06fc73c04e682caec3e6f139d66fa63f24429377fd241f9dfa4c1b06dba80",
    "seed_20261003/cuda_fp32/W_down/gU1_cuda_fp32": "a81cef1834b255fd42c24274d88a5de9c1156a6a0cf2d8a03af71928a00b7c8e",
    "seed_20261003/cuda_fp32/W_down/gU2_cuda_fp32": "de63d0da2c8ed4bd4f0420cff837b4a940315dc27fcef197f028626197522063",
    "seed_20261003/cuda_fp32/W_down/gU3_cuda_fp32": "a951e697424770e7913f6dfbf9c4da5cbb1b24625489e3cfbd1eb394d32ff942",
    "seed_20261003/cuda_fp32/W_gate/S64_primary_cuda": "5d5c6df58e769073c02a0ab0637413d39fb4fd0a18f91a690014960af375f658",
    "seed_20261003/cuda_fp32/W_gate/S_forward_cuda": "e2e8bbe8b47f33e98009d24ec25b14124f52c046b6bd41be6bcd631bce5e24f7",
    "seed_20261003/cuda_fp32/W_gate/S_pairwise_cuda": "49a64062a02f1a6cba89e64b3ad25fed8e78a20ff8107ddf775f7d1eabede121",
    "seed_20261003/cuda_fp32/W_gate/S_reverse_cuda": "2ca6873697c7a462319b5e77090df8258070b40c9931d216993dd85a665a27d7",
    "seed_20261003/cuda_fp32/W_gate/S_stack_cuda": "a5cc58137bc7127496567cdfef93f5bc055148ef396fdb2de47bf0f0ba31904c",
    "seed_20261003/cuda_fp32/W_gate/gR64_primary_cuda": "77c22a139bc5f4ed3f9559bfb558c9d49bf160ae95a1a01fc8088fd2a7760a32",
    "seed_20261003/cuda_fp32/W_gate/gR_cuda_fp32": "3cf76f048068de65f6dc967042454a831d9fa86b5461091b918e0f32cb4d3cdd",
    "seed_20261003/cuda_fp32/W_gate/gU0_cuda_fp32": "9dd7ffaf5cde6ae34df65c9419070ac60fc0dcab838007c1b114cb811abbf746",
    "seed_20261003/cuda_fp32/W_gate/gU1_cuda_fp32": "7081dcfec17e847aa30e0ef587e1589b98e7d15400b900ef489acdc65fbb39e6",
    "seed_20261003/cuda_fp32/W_gate/gU2_cuda_fp32": "ef56d401230a90b973d719960c63607848d52f682a9e243530e9fa29fb0cdb98",
    "seed_20261003/cuda_fp32/W_gate/gU3_cuda_fp32": "cd01b6d3564053eb5e3104bb59fc2844f8a12d29c7ddd500a49d80cf9ff52e89",
    "seed_20261003/cuda_fp32/W_up/S64_primary_cuda": "d4dec8c7276a207a37fdee2c18bed1a931d3a70f006622b1914b54f20c6b8387",
    "seed_20261003/cuda_fp32/W_up/S_forward_cuda": "89bdf25338db0290f1a4bf7192808d9b0a040300bc626dbd2574d47105c1d994",
    "seed_20261003/cuda_fp32/W_up/S_pairwise_cuda": "4ca4909c92204fbfbfa34ef638e3c84766942a8ec686de14224deb79465c1cc9",
    "seed_20261003/cuda_fp32/W_up/S_reverse_cuda": "7e95e2e2fe2749b115d8360cd68fe3fc15f2b483acd8a938545d565ae091db24",
    "seed_20261003/cuda_fp32/W_up/S_stack_cuda": "babf14aeff22948e66be374281484cbc0d8d03df869f2d0fa99c5aad45f3c7b8",
    "seed_20261003/cuda_fp32/W_up/gR64_primary_cuda": "d2f4cbe4354e68cd6dc1a883d979d24b0cd4307ca419cbd0603befa25924bb48",
    "seed_20261003/cuda_fp32/W_up/gR_cuda_fp32": "1ae656c67b7c10dbeb0f744fe1d73625b964700188be19534084ae8c8e0fcc85",
    "seed_20261003/cuda_fp32/W_up/gU0_cuda_fp32": "73a89ea88da2cc41eb00dee8ded32d03a1d079027e11e5a80487cae8eb536c95",
    "seed_20261003/cuda_fp32/W_up/gU1_cuda_fp32": "5dcf11983138466a74a3520b967dbf6d443dd4c30c3ee3dbd5742e0cb65d9a3a",
    "seed_20261003/cuda_fp32/W_up/gU2_cuda_fp32": "b9c101e43eab7ea63cae9035c8edbcaf27bde16da0bcafc86ec74bd3e62e1087",
    "seed_20261003/cuda_fp32/W_up/gU3_cuda_fp32": "e8233acfb81c14abaec82c56d8163a1983943780d439f78e1a56a9c25c57c7a5",
    "seed_20261003/inputs/w_fp32_cpu": "7cf1f870d7e90692b6207db0c222de69a6fc713c4ba49907db48f1dc6fa6baa6",
    "seed_20261003/inputs/x_fp32_cpu": "726b11badbb1a0037e27c826b4796b51eecab8fac8b07b5696977360276f0e2a",
    "seed_20261003/structural_traces/R4/round_1": "b5e2ab917aede3b9965d89ed5712618ec164306ad115a866c09610c48484d007",
    "seed_20261003/structural_traces/R4/round_2": "9784185754dccb48d970e9ea2f8baa697aec1d5aaa87085fe166274f335bfc40",
    "seed_20261003/structural_traces/R4/round_3": "05e0344a9cd998e39561cf07967b50b13669fb32417d54aeae287052685b274e",
    "seed_20261003/structural_traces/R4/round_4": "9d83f259843ef549761c29e99707bcdd27837359db6372300457ebcdec6d4647",
    "seed_20261003/structural_traces/R4_final": "4b2fd2ada2926bb8b9d3acc44e538ebbdbad35b1ae8e15b8cc9db60b96031e6d",
    "seed_20261003/structural_traces/U4/round_1": "1ebbfcfb8ba4c32916b8dda4ae5b5e2af2dc669c4abeebd6eb2a0e042bc66193",
    "seed_20261003/structural_traces/U4/round_2": "25e9fcca7ebf6cec633d98e23138484a378c93aab7e280185141487d20129619",
    "seed_20261003/structural_traces/U4/round_3": "7869f36f9572833ebbe9b6e1b834220308ffd6a33e73b45d4e947bba5736a8d9",
    "seed_20261003/structural_traces/U4/round_4": "73f63ebeda218f60b862b6c26b9ef10818bf16e6f9406d6fc22324d19eb7ca8b",
    "seed_20261003/structural_traces/U4_final": "2004c5e5dc7d048bd77ff7c113e59fbbb2f7a4ba3825a2f1b76cea6a7a1e44b3"
  }
}
```

## Seed seed_20261004

### Structural trace checks

```json
{
  "final_output_equal": true,
  "initial_clone_report": {
    "initial_values_bitwise_equal": true,
    "r4_reuses_one_block_object": true,
    "u4_block_storages_pairwise_disjoint": true,
    "u4_has_four_block_objects": true,
    "u4_storage_disjoint_from_r4": true
  },
  "input_copy_bitwise_equal": true,
  "loss_weight_copy_bitwise_equal": true,
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
  ],
  "weight_copy_report": {
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
    ]
  }
}
```

### Per-family scientific and oracle decisions

```json
{
  "W_K": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 1.0008160273891902e-16
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.66145129733245e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 1.0008160273891902e-16,
        "E_inf": 2.66145129733245e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 4.998181978289033e-13,
        "norm_S64_inf": 53.39513343507515,
        "norm_S64_l2": 3023.364979477414,
        "norm_gR64_inf": 53.39513343507515,
        "norm_gR64_l2": 3023.364979477414,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 5.960464477539063e-08,
        "abs_error_over_ulp_gR": 2048.0,
        "abs_error_over_ulp_sum_gU_old": 2048.0,
        "argmax_flat_index": 65334,
        "argmax_multi_index_row_major": [
          255,
          54
        ],
        "floor_1e_6_active": false,
        "gR_value": 0.0003058910369873047,
        "max_rel_old": 0.00019481783965602517,
        "old_denominator": 0.0003059506416320801,
        "sum_gU_old_value": 0.0003059506416320801,
        "ulp_gR": 2.9103830456733704e-11,
        "ulp_sum_gU_old": 2.9103830456733704e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.4311523798996564e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 6.251245742111872e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 3.0517578125e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.337860107421875e-06
        }
      },
      "metrics": {
        "E_L2": 3.4311523798996564e-08,
        "E_inf": 6.251245742111872e-08,
        "max_abs": 3.337860107421875e-06,
        "max_rel_old": 0.0002442598925256473,
        "norm_S64_inf": 53.395119071006775,
        "norm_S64_l2": 3023.365241323563,
        "norm_gR64_inf": 53.395118713378906,
        "norm_gR64_l2": 3023.3652408686953,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 53.395118713378906,
        "M64": 53.395119071006775,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_O": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.985529613192154e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.091861561673506e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.985529613192154e-17,
        "E_inf": 2.091861561673506e-16,
        "max_abs": 2.842170943040401e-14,
        "max_rel_old": 4.225982874148138e-12,
        "norm_S64_inf": 135.8680227752089,
        "norm_S64_l2": 5831.428513827569,
        "norm_gR64_inf": 135.86802277520894,
        "norm_gR64_l2": 5831.428513827569,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 2.384185791015625e-07,
        "abs_error_over_ulp_gR": 32768.0,
        "abs_error_over_ulp_sum_gU_old": 32768.0,
        "argmax_flat_index": 59447,
        "argmax_multi_index_row_major": [
          232,
          55
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.0001068115234375,
        "max_rel_old": 0.0022321429569274187,
        "old_denominator": 0.0001068115234375,
        "sum_gU_old_value": -0.00010657310485839844,
        "ulp_gR": 7.275957614183426e-12,
        "ulp_sum_gU_old": 7.275957614183426e-12
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.7235545636400424e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 7.019121659684231e-08
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
        "E_L2": 3.7235545636400424e-08,
        "E_inf": 7.019121659684231e-08,
        "max_abs": 9.5367431640625e-06,
        "max_rel_old": 0.004429408482142857,
        "norm_S64_inf": 135.8680362701416,
        "norm_S64_l2": 5831.428692557988,
        "norm_gR64_inf": 135.8680419921875,
        "norm_gR64_l2": 5831.428691691587,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 135.8680419921875,
        "M64": 135.8680419921875,
        "ULP_M32": 1.52587890625e-05
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_Q": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.927410982856908e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.423490168819039e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.927410982856908e-17,
        "E_inf": 2.423490168819039e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 1.2536832369223474e-12,
        "norm_S64_inf": 58.637971377151985,
        "norm_S64_l2": 3094.2757647557314,
        "norm_gR64_inf": 58.637971377151985,
        "norm_gR64_l2": 3094.2757647557314,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 1.1920928955078125e-07,
        "abs_error_over_ulp_gR": 8192.0,
        "abs_error_over_ulp_sum_gU_old": 8192.0,
        "argmax_flat_index": 39641,
        "argmax_multi_index_row_major": [
          154,
          217
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.00017523765563964844,
        "max_rel_old": 0.0006802721181884408,
        "old_denominator": 0.00017523765563964844,
        "sum_gU_old_value": -0.00017511844635009766,
        "ulp_gR": 1.4551915228366852e-11,
        "ulp_sum_gU_old": 1.4551915228366852e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.418851064740523e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.6923175043617994e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 3.0517578125e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.337860107421875e-06
        }
      },
      "metrics": {
        "E_L2": 3.418851064740523e-08,
        "E_inf": 5.6923175043617994e-08,
        "max_abs": 3.337860107421875e-06,
        "max_rel_old": 0.0003401360544217687,
        "norm_S64_inf": 58.63798189163208,
        "norm_S64_l2": 3094.2760391415395,
        "norm_gR64_inf": 58.63798141479492,
        "norm_gR64_l2": 3094.2760392075884,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 58.63798141479492,
        "M64": 58.63798189163208,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_V": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 1.002887486961062e-16
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.251780402544915e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 1.002887486961062e-16,
        "E_inf": 2.251780402544915e-16,
        "max_abs": 2.842170943040401e-14,
        "max_rel_old": 3.1911776099699764e-12,
        "norm_S64_inf": 126.21883287678669,
        "norm_S64_l2": 5824.105903043023,
        "norm_gR64_inf": 126.21883287678668,
        "norm_gR64_l2": 5824.105903043023,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 6.705522537231445e-07,
        "abs_error_over_ulp_gR": 23040.0,
        "abs_error_over_ulp_sum_gU_old": 23040.0,
        "argmax_flat_index": 313,
        "argmax_multi_index_row_major": [
          1,
          57
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.000270843505859375,
        "max_rel_old": 0.0024696779437363148,
        "old_denominator": 0.00027151405811309814,
        "sum_gU_old_value": -0.00027151405811309814,
        "ulp_gR": 2.9103830456733704e-11,
        "ulp_sum_gU_old": 2.9103830456733704e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.7260121554773e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 6.04457730694013e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 6.103515625e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 7.62939453125e-06
        }
      },
      "metrics": {
        "E_L2": 3.7260121554773e-08,
        "E_inf": 6.04457730694013e-08,
        "max_abs": 7.62939453125e-06,
        "max_rel_old": 0.002469677844245651,
        "norm_S64_inf": 126.218825340271,
        "norm_S64_l2": 5824.1060585600235,
        "norm_gR64_inf": 126.21882629394531,
        "norm_gR64_l2": 5824.106058855662,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 126.21882629394531,
        "M64": 126.21882629394531,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_down": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.910715001578708e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.1219486257870467e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.910715001578708e-17,
        "E_inf": 2.1219486257870467e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 4.350813571951249e-12,
        "norm_S64_inf": 66.97077649526548,
        "norm_S64_l2": 4093.5518515508056,
        "norm_gR64_inf": 66.97077649526548,
        "norm_gR64_l2": 4093.5518515508056,
        "shape": [
          256,
          1024
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 3.5762786865234375e-07,
        "abs_error_over_ulp_gR": 24576.0,
        "abs_error_over_ulp_sum_gU_old": 24576.0,
        "argmax_flat_index": 21350,
        "argmax_multi_index_row_major": [
          20,
          870
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.0002052783966064453,
        "max_rel_old": 0.001742160296998918,
        "old_denominator": 0.0002052783966064453,
        "sum_gU_old_value": -0.00020492076873779297,
        "ulp_gR": 1.4551915228366852e-11,
        "ulp_sum_gU_old": 1.4551915228366852e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.65990861338802e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 7.120078283267102e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 6.103515625e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 4.76837158203125e-06
        }
      },
      "metrics": {
        "E_L2": 3.65990861338802e-08,
        "E_inf": 7.120078283267102e-08,
        "max_abs": 4.76837158203125e-06,
        "max_rel_old": 0.0045362903225806455,
        "norm_S64_inf": 66.97077465057373,
        "norm_S64_l2": 4093.5519483926273,
        "norm_gR64_inf": 66.97077178955078,
        "norm_gR64_l2": 4093.5519488154614,
        "shape": [
          256,
          1024
        ]
      },
      "scale_context": {
        "M32": 66.97077178955078,
        "M64": 66.97077465057373,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_gate": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.968304474736914e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.2839754237102415e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.968304474736914e-17,
        "E_inf": 1.2839754237102415e-16,
        "max_abs": 7.105427357601002e-15,
        "max_rel_old": 1.8731529821157946e-12,
        "norm_S64_inf": 55.33927851258082,
        "norm_S64_l2": 4106.525052872383,
        "norm_gR64_inf": 55.33927851258081,
        "norm_gR64_l2": 4106.525052872383,
        "shape": [
          1024,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 4.76837158203125e-07,
        "abs_error_over_ulp_gR": 32768.0,
        "abs_error_over_ulp_sum_gU_old": 32768.0,
        "argmax_flat_index": 117124,
        "argmax_multi_index_row_major": [
          457,
          132
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.0002396106719970703,
        "max_rel_old": 0.001990049844607711,
        "old_denominator": 0.0002396106719970703,
        "sum_gU_old_value": -0.0002391338348388672,
        "ulp_gR": 1.4551915228366852e-11,
        "ulp_sum_gU_old": 1.4551915228366852e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.6804270471099486e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 6.893290484639681e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 3.0517578125e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.814697265625e-06
        }
      },
      "metrics": {
        "E_L2": 3.6804270471099486e-08,
        "E_inf": 6.893290484639681e-08,
        "max_abs": 3.814697265625e-06,
        "max_rel_old": 0.000657030223390276,
        "norm_S64_inf": 55.33927631378174,
        "norm_S64_l2": 4106.52515569321,
        "norm_gR64_inf": 55.33927917480469,
        "norm_gR64_l2": 4106.525155122496,
        "shape": [
          1024,
          256
        ]
      },
      "scale_context": {
        "M32": 55.33927917480469,
        "M64": 55.33927917480469,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_up": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.939747338003729e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.41193364523765e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.939747338003729e-17,
        "E_inf": 1.41193364523765e-16,
        "max_abs": 7.105427357601002e-15,
        "max_rel_old": 2.9813201183370203e-12,
        "norm_S64_inf": 50.32408839868002,
        "norm_S64_l2": 4030.6111858605614,
        "norm_gR64_inf": 50.32408839868001,
        "norm_gR64_l2": 4030.6111858605614,
        "shape": [
          1024,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 1.1920928955078125e-07,
        "abs_error_over_ulp_gR": 8192.0,
        "abs_error_over_ulp_sum_gU_old": 8192.0,
        "argmax_flat_index": 220541,
        "argmax_multi_index_row_major": [
          861,
          125
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.0001342296600341797,
        "max_rel_old": 0.0008873114711605012,
        "old_denominator": 0.00013434886932373047,
        "sum_gU_old_value": -0.00013434886932373047,
        "ulp_gR": 1.4551915228366852e-11,
        "ulp_sum_gU_old": 1.4551915228366852e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.686750541241566e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 7.58026093987049e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 3.0517578125e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.814697265625e-06
        }
      },
      "metrics": {
        "E_L2": 3.686750541241566e-08,
        "E_inf": 7.58026093987049e-08,
        "max_abs": 3.814697265625e-06,
        "max_rel_old": 0.0012175324675324675,
        "norm_S64_inf": 50.32408905029297,
        "norm_S64_l2": 4030.6113001289336,
        "norm_gR64_inf": 50.32408905029297,
        "norm_gR64_l2": 4030.6112997399355,
        "shape": [
          1024,
          256
        ]
      },
      "scale_context": {
        "M32": 50.32408905029297,
        "M64": 50.32408905029297,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  }
}
```

### Persisted gradient bundle

```json
{
  "bundle_path": "seed_20261004_raw_gradients.pt",
  "bundle_sha256": "e0303e84a4df3f74188d29f14b2c305702f675656bc3ab95f3fca26fc77a3cd2",
  "bundle_size_bytes": 105685741,
  "tensor_metadata": {
    "seed_20261004/cpu_fp64_oracle/W_K/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "bd0784d8942b4a501886b0f3abd3d8608d333bde45803e801d06facd6ee45a48",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_K/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "3fa35d57ca556ecd3484009ded35b41af78f87bc8dbeecb9d58606bd958d3e18",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_K/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "cf3682b483d8c135b345950e0354cb4a015b05ec94f671db252f4000ec2e3d9a",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_K/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "ffb9a48621b16cbdf57eecb814e6b95c958130db87dbd1d55d261c3d2df8980b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_K/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "f559c61e1cf9b8425c3f2bf055230c219cc5295f0386f48b689f06da779b177b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_K/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "a52c18b4efcd7886eaa882b5d90a77a812561fc49ce9dda671cef0caa5db48e6",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_O/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "d6b29c5a487cfa18e5f77ab53e6d48ce011dee52e5ac2f0121b6f7c2cd61bcbd",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_O/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "8eb06252c42fd6e6942e36033055f02d516007aebf054d7b81bed483c7afa62d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_O/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "be18e962348a90f5d81215dc0db6e0591faa4545de5a7ae60c6f6744dc494efd",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_O/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "3b25026216e25ff2ee2d402185cfdc2a96abd16314b98e8eb5b8ed6a08495ad4",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_O/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "bbdbe710a33d793623c25ea84b69b17f4a1ecb2500f4b3feda471ca77f44ae94",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_O/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "ed4546340979c8670677be212b6942569e0473ab35d1dbda1dbad0cfd092ff0d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_Q/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "5a494e044e9ecda94c5ffebf0f58080e58a40e6f0f6344b443e623875f00e2f0",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_Q/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "fee98dab86382595fa71ce3effdab68829ddb380ecffbf1d5380b0808db93cce",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_Q/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "44b6567786f170d68246188981752ad91e68fd125e6d12e3fe9e6110c4a5489b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_Q/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "81395cbdeb7f241403e1eda8fb755bf02dc1b3b083c269e4a605ae2b931adb07",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_Q/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "1fb669683e6513691a00c4b12a45afb0b13f47a9eff838740f4d7867d0eec951",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_Q/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "eb016cf512a77f6c1cfb9cd081fd5f9f576631d46ac2f3d9f8b3b84a44c49c56",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_V/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "dd47059169982e7765c4dab89eb2e30337461cf1c08c4c18f40829faa33b9dfe",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_V/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "f8046594149fa291a032ed3c5957efd8ab0c0ffdefcf9726e973e19821c0e5a7",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_V/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "69203da0af9f976fc8482e20b3d77b020404b8a565053b6dae9a4a0470cd336c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_V/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "0d3489d8a91ff2cc632d1b6d07d23553bc9744c05101dbfb79a36daf1e7756f8",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_V/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "d63c1b03d59e74a82af2e1c6f53deaa878d9c7f064b488d61c46a8cc0bbc421d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_V/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "d84fc23c6e7aa743f84d666c6be27aea477a894f0b9f8b03452cebd147d218ca",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_down/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "e6c22a5b135ec4ed7cb6ada99383dfe323faba6f970e42ef11e438490f427d6f",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_down/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "aa6020c6d8df066009d1a85aaa59a45948cc88dda2d0ad9427916e5e0851e51a",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_down/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "2a45e9619b8bb68907de68cc41f48962ce2089bed7be27be10e823250adc119b",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_down/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "c1aff49a3aeb786a952f14a42637d0dcc6f479b0f6a320ba7902397659ac4ae2",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_down/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "ece698ac9b927bdac005e3f73e263ec321708e22b20d7acd811f0e8ff288327a",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_down/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "5f33043419ff01b8787ad70e2d90b645b1eadfef9bf1125cf449a5ec22907e11",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_gate/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "12cac22f0c8c8a7a4149f9bbc55ceedae8077559a86d5e34841920b0d70ed6a0",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_gate/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "7698a02c1484a04697bce295e81eae5e295c768286b7be4d6b09d13ec1b3f7ba",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_gate/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "e4e696bbf771d14827efdc8449693f09b4c527d898a0dbaea8abd5bafb74368f",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_gate/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "527b9139075878833bf717e83957d3d4ff50513094c2932b3c59dd55f0f7a995",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_gate/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "cd00a44c3ce3a3270299bc0e3e3ecf35516492f0ff23ca236e71213ea2c1f4db",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_gate/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "f67ce754c335dc64600bdd4f62fb2dbca499c6a5fa68cb66d79e62175b606065",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_up/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "3b73ba5b66915c6c981a6cda139a28931180e97df2c464f1229b99d205122b4d",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_up/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "b55a5184f0f88e385e825b3de5e7b511dec61f4dd519b4faf7c8b3783e73a00c",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_up/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "0a23efb34661d2af994c769e700346b0a0f8a436d2126bf5d719dd615d838786",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_up/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "3f37f3c750e2cb31bffb2017ef5d7ff4d9d0e8ccff61e3d6a62065eeba9ffef1",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_up/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "1c52e1280173483b49b2be46233b8a3102ad293b376f2e64130893327571a2f0",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cpu_fp64_oracle/W_up/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "a3413aabed90e68e083bfbda7d1133a03b8cf80d3f8f8de2cc4550dcf566401b",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_K/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "06459a1bfd971861f132cebfee8d8b342d351ec7661bd298751079e4117bedab",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_K/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "099b81e4411ba863a7d18283529c988d04d8406b946e30a3781179542ad1afd9",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_K/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "7bffb8bcd0e4a7be35219b99f625a7269eca9aed25521743630607b6da15f8cd",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_K/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "1b9d4b7af54cdfa893f843c594181cf9f0cfadd4af1a2c8c03cd70b509c528b4",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_K/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "fc6a41f1a916784f15f634da3e89efac38de906b76e84042972e4f9320bfc454",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_K/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "2de566b7d9ef86aa7a96833aa30e08809e01faabccd920ebc6628f69abff84dc",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_K/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "b43fdb28f8343cbb846e99381013bf1eec0e33bf50929e89b694437691c9b7ca",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_K/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "7d96c371a45ef8e1d847f279b02d3746a714b598898b1f0a0144c7df6066f31d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_K/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "ab9ad4ba67a6aa627f0f0b672ac239f0e03812715370066e076ca6b5c3f5eaaf",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_K/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "8cf12ae1f669eb68cc68838a9fcde5c7a4553a431b40480ddb7a79f0ecf69d75",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_K/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "b216f7eb32eb134d03b4adac9ff95ac06fc4b477a2b9523b5cdb1b41e8f1bd26",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_O/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "9eb58f1059df4c0861fbfe69cba2c6c5a0dfd084b9f45f15e8c99a7cad082bdf",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_O/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "8db921a4c30204bd111f98978e44e159acf417e38e8ef4edda271222cd25efc3",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_O/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "756d28b5f54550de218b1c3087e83c94f862c97420097970d4fa2aebbeb38022",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_O/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "2d3236213a608e2c501ef9403b91f00ca5745adc8a328605424198e234a2080b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_O/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "87693e534c1f35d2269c2bcdccee9ab72a8cfa2e236b56ffa7854e6a0c32e7ad",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_O/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "0c2aa5b4be66409186a39dd5fc6938f88636f7ab1fb67e7c12bd0dce885b23ef",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_O/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "4964bc753a7eb4e51c68d64722373967a6934e99c0621e6398a2bb4b1501f39c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_O/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "5fe5d3c10b49de4b6cc38cb452a30f755bb25a9637b5c6ac57bd800a49fb4ab4",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_O/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "c84a44b5a470f810a4c8d7f4a7a313e656f799f2af959a08642e4868a6438397",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_O/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "31e5f875f8df77457a276c53ac4835e396892126c4c3c6e6732c5fb922d6691f",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_O/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "4a8ad3efb3ecddd4907cd8f83206376dbc718c4429b1978cfe3b50f7eec6877f",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_Q/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "cf08c495d6b2831fff14ae67f9d42ce20c19342b4b0f65543fcf427db6f0185e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_Q/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "29db71a62e1fd079050718868d8e02bb71c0a385ce9ea6c72e1eb49f10b9896b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_Q/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "1e527704f31c0d256ce61ce1616d6104c8433a4e22ae226b4d0240982e6882ea",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_Q/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "917a55ea9cef8af8b8c89740e1b725baa2f5dc432ae005b3bd2e791004032d02",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_Q/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "cc49fd735ae1af214ef337954e100b24f0eb4ac004bfad8c754519251bb090bb",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_Q/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "cc47867d9be434a8e394d84fd3e6c53c4cddac7134baada17bf856da518df3ae",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_Q/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "fdf381a216031d4ecc3fbe78ff908c983c0935c3f2c6626deb06901306b9e6a1",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_Q/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "4ca8fa50ef0bf074f8e30fd4e10d3d17b10c556a09a770661ca2fb3470b17912",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_Q/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "830d71c3e2b9c3f868361e66b06161d217cccd70f03681c0f1f00353ceda6e90",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_Q/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "f980e13d41a8d680838c5d8f56c50ffb95a71e840a11a5761516224a67052994",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_Q/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "3b448643a26a7f45b1b5f6df32dffb3cd37fc8a530001385584f6a1c667494b2",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_V/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "dedc09d7bf4e1bf1405e4afbfef53dd9ac8d15504b3b0e89e3ea0c86b5bc949c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_V/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "7ef74685916a29c8dbb651aa1a5597915fa2fef9f926259221f32979c2ba8a0b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_V/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "213ce0271213b5c69ec8d529eea4f31d9ec86aae4d16331a8400943779c56292",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_V/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "0f194836811914dd063b8330583c654d24013d6a6163adca714e4bbed5c570ba",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_V/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "7fbb7f3ce4951f524f71cac73a5c5d8fa4c6b858b075ceb100fc2983fbb0e8b6",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_V/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "460cae033711975919c740bf192444f82c51bb854d36f4e998942eda4eb1db89",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_V/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "2ab7547d4fdfecc7dee1a376c286bc1025895126b66331c349ff735a9e22ec60",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_V/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "8bb6988b68b2d4dab91a2c955357bcd9378fa16dfb2535b3eda730c21dd09856",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_V/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "38ea516a223248ee9b4d8fc0b2f1b7e7eac3a3e756ef9344c36e153d3d3b9305",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_V/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "ffc0ec471d7345190acc64a50f1a289895b866e3ad037de01fa17d7560c56a62",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_V/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "02a1f055aaaa2e346812febfaf40bde0b3288e1a04e5faf474981049d28d5253",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_down/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "fc20c3da21211a0babc9d734e08b6d3db29ddd639cc1b73c8b2f7bfe626aab37",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cuda_fp32/W_down/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "4cf5d13da0a2ab836a2636d34265df32808e7e423c966aaea60b6eaf6438c335",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cuda_fp32/W_down/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "81893e124ea9c6e5ff89957b54c4d7cdd0498244a75c6923b890f073661ed8f4",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cuda_fp32/W_down/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "1adde9c0d6b7e6b4b58df8ca19d8067e85ababeb03cd6f26a82facee8c36a8d1",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cuda_fp32/W_down/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "42e7de4f2bc4c07bfeef002bd093132d491489129e9f9d80e127fad5863f866c",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cuda_fp32/W_down/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "aefdd3d8f51122bfe9f54ddec92517d8f02976ca799c9f0172d12e660c747f04",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cuda_fp32/W_down/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "048d64a07f0c82ad8e20525b9c9fc2c68b267a699d9b39ed6bb2b99721a569ea",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cuda_fp32/W_down/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "353d2364058db046b048a47ba1b8248a4860119f1179404113254bda4c16d305",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cuda_fp32/W_down/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "192e3992b1e2c4d78f82e97836f9ed5d8e15349f849dc4ecb6b3ec96880e7b2b",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cuda_fp32/W_down/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "840a7b64cdb9bc3b529c58d8c41bc6f1006abe69a58d37ea7ed097158f4c2a57",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cuda_fp32/W_down/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "6e7f5306697cc8ef30ce1236fdbce865ab280b275727821792a37a59c00bcb8b",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261004/cuda_fp32/W_gate/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "d677a27c9ffe324efb98a1c17afacadf7156661fa1673bee5646e93bbdc2695f",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_gate/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "cd011ee3435dace1fd12ee33ffbd633af6923e18a55744b737c5901a2163967c",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_gate/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "c6ba478d0f94054609d011e3454ab658a515a39996a078ec3aacac3ce5846dc3",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_gate/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "320c80502e9edd497b93e73556f6119c330bcf8aaa1610dd38dd6f09e24932cb",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_gate/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "29e826512fadf51a43340f85614550a9a17a5bbea5fcc91740df552994abf98a",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_gate/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "36a048ce52fbea0ac827f2151aab8a5970047a96c06aef71f94312b570ae8fd2",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_gate/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "15c0518afd239bb8ad3081c88fb504f77f8403e7677438f55b4987de15d88de0",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_gate/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "7f3be45486854fb3901d755d21a9cc70b0727d45ed96ac2494a632a05defd75b",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_gate/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "593889e99ad06effdaf6f6af989cd4671b703213726d453c9839f198cd4071fc",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_gate/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "e458d89d6f7d1a885aec0c1c732bf57bd53c48d2216411c3349d91c0ce892514",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_gate/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "eb82fc957b826705c0c619686f8712c46d919a793313ed1fb23e48fe157c0c0f",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_up/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "e821a7df517d724b3db19878660f3a401522f02a712703a023dafe596726590d",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_up/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "396948cc6f5c52f939340134e67e3b9e7b6431732fcd6993c72fd3d59ea62a57",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_up/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "636adbe75dff34c5a40d4a32cd49634a8abf98d0da311d3d7aa49df72866a0f2",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_up/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "66d08dc2b3b0a45dfbed9a4a8623f47fc34f7b8552c33019b14669e4cd69613d",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_up/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "f670ce7dd3bd53af865fb0b99f0e63864aff6582b6128295aa05a1f7a8930076",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_up/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "d3592369b398d4d67057a1393254f6202ffcb67c35a87557c1b3cbb87e15db3b",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_up/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "5581fc06acb61e96edf668272900edfa7aac611dcf4ebb1e2912d38559c2fc3b",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_up/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "33a12f5bdecf23e76de59c54e064003a50193d71d5206eb25eda23ac71d07187",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_up/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "343215e98d57959a1a4ccdafa1d3704589eb81f7ab1a731ef71f360599b89580",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_up/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "85dd55c091a172a29a2d8fc26f8069e44bdf3e3cbc77fc7206ea119f4e072374",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/cuda_fp32/W_up/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "66e0812532b0ba5d53544f02a38ec1ae6760ef47df8463df1549af18383e64b1",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261004/inputs/w_fp32_cpu": {
      "dtype": "torch.float32",
      "raw_sha256": "b8589347f191bda7ba736a4ea8691a2976730bbd348397faee66dcdd7ab019db",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261004/inputs/x_fp32_cpu": {
      "dtype": "torch.float32",
      "raw_sha256": "e4cf2511a393b3117874497e0552267f1766eaefaf7d240295cf701a8504b665",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261004/structural_traces/R4/round_1": {
      "dtype": "torch.float32",
      "raw_sha256": "f5ec44b117607bb5868813fb9509309310557c494a3b094d801994c07cc08be8",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261004/structural_traces/R4/round_2": {
      "dtype": "torch.float32",
      "raw_sha256": "2620ad74cf69591f90b49242ffb4b72a18bab6c1db7ac44b8ab91756d3b9d94a",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261004/structural_traces/R4/round_3": {
      "dtype": "torch.float32",
      "raw_sha256": "245f99a7a92f0f2213c01ba3f515f0c3ef30d6b20b0bc0ddb1dadd44cbc8395f",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261004/structural_traces/R4/round_4": {
      "dtype": "torch.float32",
      "raw_sha256": "777625660f2e7401c3b3b82988e66c4e41756581358f89dc93c5567a2503fa8e",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261004/structural_traces/R4_final": {
      "dtype": "torch.float32",
      "raw_sha256": "3cc429387bee728f5d2972198da835d736c3aaf336b53c84a092ab831fcffb67",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261004/structural_traces/U4/round_1": {
      "dtype": "torch.float32",
      "raw_sha256": "79d438a57ac335ccc16304c574d01bdba4e0ed6585ce2520a619a6f84d5c8d31",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261004/structural_traces/U4/round_2": {
      "dtype": "torch.float32",
      "raw_sha256": "b3bf1b70fe57b019497d5732afe835981696fd9e3a322ff8f9bfdc78191db1f5",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261004/structural_traces/U4/round_3": {
      "dtype": "torch.float32",
      "raw_sha256": "e56e00619adbb3e48b22e39fd017cedba12c9ad8b17b0569a79a388026854740",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261004/structural_traces/U4/round_4": {
      "dtype": "torch.float32",
      "raw_sha256": "cba27159c6040d83b8bba48d064f3a3bad116c280517ec946683070f78479f36",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261004/structural_traces/U4_final": {
      "dtype": "torch.float32",
      "raw_sha256": "29dec7a2ec1edb11c61e5ba9a0bb90dff3af4cb8010bf35d519f0fef7a866982",
      "shape": [
        8,
        8,
        256
      ]
    }
  },
  "tensor_raw_sha256": {
    "seed_20261004/cpu_fp64_oracle/W_K/S64_cpu": "bd0784d8942b4a501886b0f3abd3d8608d333bde45803e801d06facd6ee45a48",
    "seed_20261004/cpu_fp64_oracle/W_K/gR64_cpu": "3fa35d57ca556ecd3484009ded35b41af78f87bc8dbeecb9d58606bd958d3e18",
    "seed_20261004/cpu_fp64_oracle/W_K/gU0_64_cpu": "cf3682b483d8c135b345950e0354cb4a015b05ec94f671db252f4000ec2e3d9a",
    "seed_20261004/cpu_fp64_oracle/W_K/gU1_64_cpu": "ffb9a48621b16cbdf57eecb814e6b95c958130db87dbd1d55d261c3d2df8980b",
    "seed_20261004/cpu_fp64_oracle/W_K/gU2_64_cpu": "f559c61e1cf9b8425c3f2bf055230c219cc5295f0386f48b689f06da779b177b",
    "seed_20261004/cpu_fp64_oracle/W_K/gU3_64_cpu": "a52c18b4efcd7886eaa882b5d90a77a812561fc49ce9dda671cef0caa5db48e6",
    "seed_20261004/cpu_fp64_oracle/W_O/S64_cpu": "d6b29c5a487cfa18e5f77ab53e6d48ce011dee52e5ac2f0121b6f7c2cd61bcbd",
    "seed_20261004/cpu_fp64_oracle/W_O/gR64_cpu": "8eb06252c42fd6e6942e36033055f02d516007aebf054d7b81bed483c7afa62d",
    "seed_20261004/cpu_fp64_oracle/W_O/gU0_64_cpu": "be18e962348a90f5d81215dc0db6e0591faa4545de5a7ae60c6f6744dc494efd",
    "seed_20261004/cpu_fp64_oracle/W_O/gU1_64_cpu": "3b25026216e25ff2ee2d402185cfdc2a96abd16314b98e8eb5b8ed6a08495ad4",
    "seed_20261004/cpu_fp64_oracle/W_O/gU2_64_cpu": "bbdbe710a33d793623c25ea84b69b17f4a1ecb2500f4b3feda471ca77f44ae94",
    "seed_20261004/cpu_fp64_oracle/W_O/gU3_64_cpu": "ed4546340979c8670677be212b6942569e0473ab35d1dbda1dbad0cfd092ff0d",
    "seed_20261004/cpu_fp64_oracle/W_Q/S64_cpu": "5a494e044e9ecda94c5ffebf0f58080e58a40e6f0f6344b443e623875f00e2f0",
    "seed_20261004/cpu_fp64_oracle/W_Q/gR64_cpu": "fee98dab86382595fa71ce3effdab68829ddb380ecffbf1d5380b0808db93cce",
    "seed_20261004/cpu_fp64_oracle/W_Q/gU0_64_cpu": "44b6567786f170d68246188981752ad91e68fd125e6d12e3fe9e6110c4a5489b",
    "seed_20261004/cpu_fp64_oracle/W_Q/gU1_64_cpu": "81395cbdeb7f241403e1eda8fb755bf02dc1b3b083c269e4a605ae2b931adb07",
    "seed_20261004/cpu_fp64_oracle/W_Q/gU2_64_cpu": "1fb669683e6513691a00c4b12a45afb0b13f47a9eff838740f4d7867d0eec951",
    "seed_20261004/cpu_fp64_oracle/W_Q/gU3_64_cpu": "eb016cf512a77f6c1cfb9cd081fd5f9f576631d46ac2f3d9f8b3b84a44c49c56",
    "seed_20261004/cpu_fp64_oracle/W_V/S64_cpu": "dd47059169982e7765c4dab89eb2e30337461cf1c08c4c18f40829faa33b9dfe",
    "seed_20261004/cpu_fp64_oracle/W_V/gR64_cpu": "f8046594149fa291a032ed3c5957efd8ab0c0ffdefcf9726e973e19821c0e5a7",
    "seed_20261004/cpu_fp64_oracle/W_V/gU0_64_cpu": "69203da0af9f976fc8482e20b3d77b020404b8a565053b6dae9a4a0470cd336c",
    "seed_20261004/cpu_fp64_oracle/W_V/gU1_64_cpu": "0d3489d8a91ff2cc632d1b6d07d23553bc9744c05101dbfb79a36daf1e7756f8",
    "seed_20261004/cpu_fp64_oracle/W_V/gU2_64_cpu": "d63c1b03d59e74a82af2e1c6f53deaa878d9c7f064b488d61c46a8cc0bbc421d",
    "seed_20261004/cpu_fp64_oracle/W_V/gU3_64_cpu": "d84fc23c6e7aa743f84d666c6be27aea477a894f0b9f8b03452cebd147d218ca",
    "seed_20261004/cpu_fp64_oracle/W_down/S64_cpu": "e6c22a5b135ec4ed7cb6ada99383dfe323faba6f970e42ef11e438490f427d6f",
    "seed_20261004/cpu_fp64_oracle/W_down/gR64_cpu": "aa6020c6d8df066009d1a85aaa59a45948cc88dda2d0ad9427916e5e0851e51a",
    "seed_20261004/cpu_fp64_oracle/W_down/gU0_64_cpu": "2a45e9619b8bb68907de68cc41f48962ce2089bed7be27be10e823250adc119b",
    "seed_20261004/cpu_fp64_oracle/W_down/gU1_64_cpu": "c1aff49a3aeb786a952f14a42637d0dcc6f479b0f6a320ba7902397659ac4ae2",
    "seed_20261004/cpu_fp64_oracle/W_down/gU2_64_cpu": "ece698ac9b927bdac005e3f73e263ec321708e22b20d7acd811f0e8ff288327a",
    "seed_20261004/cpu_fp64_oracle/W_down/gU3_64_cpu": "5f33043419ff01b8787ad70e2d90b645b1eadfef9bf1125cf449a5ec22907e11",
    "seed_20261004/cpu_fp64_oracle/W_gate/S64_cpu": "12cac22f0c8c8a7a4149f9bbc55ceedae8077559a86d5e34841920b0d70ed6a0",
    "seed_20261004/cpu_fp64_oracle/W_gate/gR64_cpu": "7698a02c1484a04697bce295e81eae5e295c768286b7be4d6b09d13ec1b3f7ba",
    "seed_20261004/cpu_fp64_oracle/W_gate/gU0_64_cpu": "e4e696bbf771d14827efdc8449693f09b4c527d898a0dbaea8abd5bafb74368f",
    "seed_20261004/cpu_fp64_oracle/W_gate/gU1_64_cpu": "527b9139075878833bf717e83957d3d4ff50513094c2932b3c59dd55f0f7a995",
    "seed_20261004/cpu_fp64_oracle/W_gate/gU2_64_cpu": "cd00a44c3ce3a3270299bc0e3e3ecf35516492f0ff23ca236e71213ea2c1f4db",
    "seed_20261004/cpu_fp64_oracle/W_gate/gU3_64_cpu": "f67ce754c335dc64600bdd4f62fb2dbca499c6a5fa68cb66d79e62175b606065",
    "seed_20261004/cpu_fp64_oracle/W_up/S64_cpu": "3b73ba5b66915c6c981a6cda139a28931180e97df2c464f1229b99d205122b4d",
    "seed_20261004/cpu_fp64_oracle/W_up/gR64_cpu": "b55a5184f0f88e385e825b3de5e7b511dec61f4dd519b4faf7c8b3783e73a00c",
    "seed_20261004/cpu_fp64_oracle/W_up/gU0_64_cpu": "0a23efb34661d2af994c769e700346b0a0f8a436d2126bf5d719dd615d838786",
    "seed_20261004/cpu_fp64_oracle/W_up/gU1_64_cpu": "3f37f3c750e2cb31bffb2017ef5d7ff4d9d0e8ccff61e3d6a62065eeba9ffef1",
    "seed_20261004/cpu_fp64_oracle/W_up/gU2_64_cpu": "1c52e1280173483b49b2be46233b8a3102ad293b376f2e64130893327571a2f0",
    "seed_20261004/cpu_fp64_oracle/W_up/gU3_64_cpu": "a3413aabed90e68e083bfbda7d1133a03b8cf80d3f8f8de2cc4550dcf566401b",
    "seed_20261004/cuda_fp32/W_K/S64_primary_cuda": "06459a1bfd971861f132cebfee8d8b342d351ec7661bd298751079e4117bedab",
    "seed_20261004/cuda_fp32/W_K/S_forward_cuda": "099b81e4411ba863a7d18283529c988d04d8406b946e30a3781179542ad1afd9",
    "seed_20261004/cuda_fp32/W_K/S_pairwise_cuda": "7bffb8bcd0e4a7be35219b99f625a7269eca9aed25521743630607b6da15f8cd",
    "seed_20261004/cuda_fp32/W_K/S_reverse_cuda": "1b9d4b7af54cdfa893f843c594181cf9f0cfadd4af1a2c8c03cd70b509c528b4",
    "seed_20261004/cuda_fp32/W_K/S_stack_cuda": "fc6a41f1a916784f15f634da3e89efac38de906b76e84042972e4f9320bfc454",
    "seed_20261004/cuda_fp32/W_K/gR64_primary_cuda": "2de566b7d9ef86aa7a96833aa30e08809e01faabccd920ebc6628f69abff84dc",
    "seed_20261004/cuda_fp32/W_K/gR_cuda_fp32": "b43fdb28f8343cbb846e99381013bf1eec0e33bf50929e89b694437691c9b7ca",
    "seed_20261004/cuda_fp32/W_K/gU0_cuda_fp32": "7d96c371a45ef8e1d847f279b02d3746a714b598898b1f0a0144c7df6066f31d",
    "seed_20261004/cuda_fp32/W_K/gU1_cuda_fp32": "ab9ad4ba67a6aa627f0f0b672ac239f0e03812715370066e076ca6b5c3f5eaaf",
    "seed_20261004/cuda_fp32/W_K/gU2_cuda_fp32": "8cf12ae1f669eb68cc68838a9fcde5c7a4553a431b40480ddb7a79f0ecf69d75",
    "seed_20261004/cuda_fp32/W_K/gU3_cuda_fp32": "b216f7eb32eb134d03b4adac9ff95ac06fc4b477a2b9523b5cdb1b41e8f1bd26",
    "seed_20261004/cuda_fp32/W_O/S64_primary_cuda": "9eb58f1059df4c0861fbfe69cba2c6c5a0dfd084b9f45f15e8c99a7cad082bdf",
    "seed_20261004/cuda_fp32/W_O/S_forward_cuda": "8db921a4c30204bd111f98978e44e159acf417e38e8ef4edda271222cd25efc3",
    "seed_20261004/cuda_fp32/W_O/S_pairwise_cuda": "756d28b5f54550de218b1c3087e83c94f862c97420097970d4fa2aebbeb38022",
    "seed_20261004/cuda_fp32/W_O/S_reverse_cuda": "2d3236213a608e2c501ef9403b91f00ca5745adc8a328605424198e234a2080b",
    "seed_20261004/cuda_fp32/W_O/S_stack_cuda": "87693e534c1f35d2269c2bcdccee9ab72a8cfa2e236b56ffa7854e6a0c32e7ad",
    "seed_20261004/cuda_fp32/W_O/gR64_primary_cuda": "0c2aa5b4be66409186a39dd5fc6938f88636f7ab1fb67e7c12bd0dce885b23ef",
    "seed_20261004/cuda_fp32/W_O/gR_cuda_fp32": "4964bc753a7eb4e51c68d64722373967a6934e99c0621e6398a2bb4b1501f39c",
    "seed_20261004/cuda_fp32/W_O/gU0_cuda_fp32": "5fe5d3c10b49de4b6cc38cb452a30f755bb25a9637b5c6ac57bd800a49fb4ab4",
    "seed_20261004/cuda_fp32/W_O/gU1_cuda_fp32": "c84a44b5a470f810a4c8d7f4a7a313e656f799f2af959a08642e4868a6438397",
    "seed_20261004/cuda_fp32/W_O/gU2_cuda_fp32": "31e5f875f8df77457a276c53ac4835e396892126c4c3c6e6732c5fb922d6691f",
    "seed_20261004/cuda_fp32/W_O/gU3_cuda_fp32": "4a8ad3efb3ecddd4907cd8f83206376dbc718c4429b1978cfe3b50f7eec6877f",
    "seed_20261004/cuda_fp32/W_Q/S64_primary_cuda": "cf08c495d6b2831fff14ae67f9d42ce20c19342b4b0f65543fcf427db6f0185e",
    "seed_20261004/cuda_fp32/W_Q/S_forward_cuda": "29db71a62e1fd079050718868d8e02bb71c0a385ce9ea6c72e1eb49f10b9896b",
    "seed_20261004/cuda_fp32/W_Q/S_pairwise_cuda": "1e527704f31c0d256ce61ce1616d6104c8433a4e22ae226b4d0240982e6882ea",
    "seed_20261004/cuda_fp32/W_Q/S_reverse_cuda": "917a55ea9cef8af8b8c89740e1b725baa2f5dc432ae005b3bd2e791004032d02",
    "seed_20261004/cuda_fp32/W_Q/S_stack_cuda": "cc49fd735ae1af214ef337954e100b24f0eb4ac004bfad8c754519251bb090bb",
    "seed_20261004/cuda_fp32/W_Q/gR64_primary_cuda": "cc47867d9be434a8e394d84fd3e6c53c4cddac7134baada17bf856da518df3ae",
    "seed_20261004/cuda_fp32/W_Q/gR_cuda_fp32": "fdf381a216031d4ecc3fbe78ff908c983c0935c3f2c6626deb06901306b9e6a1",
    "seed_20261004/cuda_fp32/W_Q/gU0_cuda_fp32": "4ca8fa50ef0bf074f8e30fd4e10d3d17b10c556a09a770661ca2fb3470b17912",
    "seed_20261004/cuda_fp32/W_Q/gU1_cuda_fp32": "830d71c3e2b9c3f868361e66b06161d217cccd70f03681c0f1f00353ceda6e90",
    "seed_20261004/cuda_fp32/W_Q/gU2_cuda_fp32": "f980e13d41a8d680838c5d8f56c50ffb95a71e840a11a5761516224a67052994",
    "seed_20261004/cuda_fp32/W_Q/gU3_cuda_fp32": "3b448643a26a7f45b1b5f6df32dffb3cd37fc8a530001385584f6a1c667494b2",
    "seed_20261004/cuda_fp32/W_V/S64_primary_cuda": "dedc09d7bf4e1bf1405e4afbfef53dd9ac8d15504b3b0e89e3ea0c86b5bc949c",
    "seed_20261004/cuda_fp32/W_V/S_forward_cuda": "7ef74685916a29c8dbb651aa1a5597915fa2fef9f926259221f32979c2ba8a0b",
    "seed_20261004/cuda_fp32/W_V/S_pairwise_cuda": "213ce0271213b5c69ec8d529eea4f31d9ec86aae4d16331a8400943779c56292",
    "seed_20261004/cuda_fp32/W_V/S_reverse_cuda": "0f194836811914dd063b8330583c654d24013d6a6163adca714e4bbed5c570ba",
    "seed_20261004/cuda_fp32/W_V/S_stack_cuda": "7fbb7f3ce4951f524f71cac73a5c5d8fa4c6b858b075ceb100fc2983fbb0e8b6",
    "seed_20261004/cuda_fp32/W_V/gR64_primary_cuda": "460cae033711975919c740bf192444f82c51bb854d36f4e998942eda4eb1db89",
    "seed_20261004/cuda_fp32/W_V/gR_cuda_fp32": "2ab7547d4fdfecc7dee1a376c286bc1025895126b66331c349ff735a9e22ec60",
    "seed_20261004/cuda_fp32/W_V/gU0_cuda_fp32": "8bb6988b68b2d4dab91a2c955357bcd9378fa16dfb2535b3eda730c21dd09856",
    "seed_20261004/cuda_fp32/W_V/gU1_cuda_fp32": "38ea516a223248ee9b4d8fc0b2f1b7e7eac3a3e756ef9344c36e153d3d3b9305",
    "seed_20261004/cuda_fp32/W_V/gU2_cuda_fp32": "ffc0ec471d7345190acc64a50f1a289895b866e3ad037de01fa17d7560c56a62",
    "seed_20261004/cuda_fp32/W_V/gU3_cuda_fp32": "02a1f055aaaa2e346812febfaf40bde0b3288e1a04e5faf474981049d28d5253",
    "seed_20261004/cuda_fp32/W_down/S64_primary_cuda": "fc20c3da21211a0babc9d734e08b6d3db29ddd639cc1b73c8b2f7bfe626aab37",
    "seed_20261004/cuda_fp32/W_down/S_forward_cuda": "4cf5d13da0a2ab836a2636d34265df32808e7e423c966aaea60b6eaf6438c335",
    "seed_20261004/cuda_fp32/W_down/S_pairwise_cuda": "81893e124ea9c6e5ff89957b54c4d7cdd0498244a75c6923b890f073661ed8f4",
    "seed_20261004/cuda_fp32/W_down/S_reverse_cuda": "1adde9c0d6b7e6b4b58df8ca19d8067e85ababeb03cd6f26a82facee8c36a8d1",
    "seed_20261004/cuda_fp32/W_down/S_stack_cuda": "42e7de4f2bc4c07bfeef002bd093132d491489129e9f9d80e127fad5863f866c",
    "seed_20261004/cuda_fp32/W_down/gR64_primary_cuda": "aefdd3d8f51122bfe9f54ddec92517d8f02976ca799c9f0172d12e660c747f04",
    "seed_20261004/cuda_fp32/W_down/gR_cuda_fp32": "048d64a07f0c82ad8e20525b9c9fc2c68b267a699d9b39ed6bb2b99721a569ea",
    "seed_20261004/cuda_fp32/W_down/gU0_cuda_fp32": "353d2364058db046b048a47ba1b8248a4860119f1179404113254bda4c16d305",
    "seed_20261004/cuda_fp32/W_down/gU1_cuda_fp32": "192e3992b1e2c4d78f82e97836f9ed5d8e15349f849dc4ecb6b3ec96880e7b2b",
    "seed_20261004/cuda_fp32/W_down/gU2_cuda_fp32": "840a7b64cdb9bc3b529c58d8c41bc6f1006abe69a58d37ea7ed097158f4c2a57",
    "seed_20261004/cuda_fp32/W_down/gU3_cuda_fp32": "6e7f5306697cc8ef30ce1236fdbce865ab280b275727821792a37a59c00bcb8b",
    "seed_20261004/cuda_fp32/W_gate/S64_primary_cuda": "d677a27c9ffe324efb98a1c17afacadf7156661fa1673bee5646e93bbdc2695f",
    "seed_20261004/cuda_fp32/W_gate/S_forward_cuda": "cd011ee3435dace1fd12ee33ffbd633af6923e18a55744b737c5901a2163967c",
    "seed_20261004/cuda_fp32/W_gate/S_pairwise_cuda": "c6ba478d0f94054609d011e3454ab658a515a39996a078ec3aacac3ce5846dc3",
    "seed_20261004/cuda_fp32/W_gate/S_reverse_cuda": "320c80502e9edd497b93e73556f6119c330bcf8aaa1610dd38dd6f09e24932cb",
    "seed_20261004/cuda_fp32/W_gate/S_stack_cuda": "29e826512fadf51a43340f85614550a9a17a5bbea5fcc91740df552994abf98a",
    "seed_20261004/cuda_fp32/W_gate/gR64_primary_cuda": "36a048ce52fbea0ac827f2151aab8a5970047a96c06aef71f94312b570ae8fd2",
    "seed_20261004/cuda_fp32/W_gate/gR_cuda_fp32": "15c0518afd239bb8ad3081c88fb504f77f8403e7677438f55b4987de15d88de0",
    "seed_20261004/cuda_fp32/W_gate/gU0_cuda_fp32": "7f3be45486854fb3901d755d21a9cc70b0727d45ed96ac2494a632a05defd75b",
    "seed_20261004/cuda_fp32/W_gate/gU1_cuda_fp32": "593889e99ad06effdaf6f6af989cd4671b703213726d453c9839f198cd4071fc",
    "seed_20261004/cuda_fp32/W_gate/gU2_cuda_fp32": "e458d89d6f7d1a885aec0c1c732bf57bd53c48d2216411c3349d91c0ce892514",
    "seed_20261004/cuda_fp32/W_gate/gU3_cuda_fp32": "eb82fc957b826705c0c619686f8712c46d919a793313ed1fb23e48fe157c0c0f",
    "seed_20261004/cuda_fp32/W_up/S64_primary_cuda": "e821a7df517d724b3db19878660f3a401522f02a712703a023dafe596726590d",
    "seed_20261004/cuda_fp32/W_up/S_forward_cuda": "396948cc6f5c52f939340134e67e3b9e7b6431732fcd6993c72fd3d59ea62a57",
    "seed_20261004/cuda_fp32/W_up/S_pairwise_cuda": "636adbe75dff34c5a40d4a32cd49634a8abf98d0da311d3d7aa49df72866a0f2",
    "seed_20261004/cuda_fp32/W_up/S_reverse_cuda": "66d08dc2b3b0a45dfbed9a4a8623f47fc34f7b8552c33019b14669e4cd69613d",
    "seed_20261004/cuda_fp32/W_up/S_stack_cuda": "f670ce7dd3bd53af865fb0b99f0e63864aff6582b6128295aa05a1f7a8930076",
    "seed_20261004/cuda_fp32/W_up/gR64_primary_cuda": "d3592369b398d4d67057a1393254f6202ffcb67c35a87557c1b3cbb87e15db3b",
    "seed_20261004/cuda_fp32/W_up/gR_cuda_fp32": "5581fc06acb61e96edf668272900edfa7aac611dcf4ebb1e2912d38559c2fc3b",
    "seed_20261004/cuda_fp32/W_up/gU0_cuda_fp32": "33a12f5bdecf23e76de59c54e064003a50193d71d5206eb25eda23ac71d07187",
    "seed_20261004/cuda_fp32/W_up/gU1_cuda_fp32": "343215e98d57959a1a4ccdafa1d3704589eb81f7ab1a731ef71f360599b89580",
    "seed_20261004/cuda_fp32/W_up/gU2_cuda_fp32": "85dd55c091a172a29a2d8fc26f8069e44bdf3e3cbc77fc7206ea119f4e072374",
    "seed_20261004/cuda_fp32/W_up/gU3_cuda_fp32": "66e0812532b0ba5d53544f02a38ec1ae6760ef47df8463df1549af18383e64b1",
    "seed_20261004/inputs/w_fp32_cpu": "b8589347f191bda7ba736a4ea8691a2976730bbd348397faee66dcdd7ab019db",
    "seed_20261004/inputs/x_fp32_cpu": "e4cf2511a393b3117874497e0552267f1766eaefaf7d240295cf701a8504b665",
    "seed_20261004/structural_traces/R4/round_1": "f5ec44b117607bb5868813fb9509309310557c494a3b094d801994c07cc08be8",
    "seed_20261004/structural_traces/R4/round_2": "2620ad74cf69591f90b49242ffb4b72a18bab6c1db7ac44b8ab91756d3b9d94a",
    "seed_20261004/structural_traces/R4/round_3": "245f99a7a92f0f2213c01ba3f515f0c3ef30d6b20b0bc0ddb1dadd44cbc8395f",
    "seed_20261004/structural_traces/R4/round_4": "777625660f2e7401c3b3b82988e66c4e41756581358f89dc93c5567a2503fa8e",
    "seed_20261004/structural_traces/R4_final": "3cc429387bee728f5d2972198da835d736c3aaf336b53c84a092ab831fcffb67",
    "seed_20261004/structural_traces/U4/round_1": "79d438a57ac335ccc16304c574d01bdba4e0ed6585ce2520a619a6f84d5c8d31",
    "seed_20261004/structural_traces/U4/round_2": "b3bf1b70fe57b019497d5732afe835981696fd9e3a322ff8f9bfdc78191db1f5",
    "seed_20261004/structural_traces/U4/round_3": "e56e00619adbb3e48b22e39fd017cedba12c9ad8b17b0569a79a388026854740",
    "seed_20261004/structural_traces/U4/round_4": "cba27159c6040d83b8bba48d064f3a3bad116c280517ec946683070f78479f36",
    "seed_20261004/structural_traces/U4_final": "29dec7a2ec1edb11c61e5ba9a0bb90dff3af4cb8010bf35d519f0fef7a866982"
  }
}
```

## Seed seed_20261005

### Structural trace checks

```json
{
  "final_output_equal": true,
  "initial_clone_report": {
    "initial_values_bitwise_equal": true,
    "r4_reuses_one_block_object": true,
    "u4_block_storages_pairwise_disjoint": true,
    "u4_has_four_block_objects": true,
    "u4_storage_disjoint_from_r4": true
  },
  "input_copy_bitwise_equal": true,
  "loss_weight_copy_bitwise_equal": true,
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
  ],
  "weight_copy_report": {
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
    ]
  }
}
```

### Per-family scientific and oracle decisions

```json
{
  "W_K": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.992490668667513e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.2633055735247655e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.992490668667513e-17,
        "E_inf": 1.2633055735247655e-16,
        "max_abs": 7.105427357601002e-15,
        "max_rel_old": 4.5967568767589464e-11,
        "norm_S64_inf": 56.244724210121674,
        "norm_S64_l2": 2771.2819944835946,
        "norm_gR64_inf": 56.24472421012167,
        "norm_gR64_l2": 2771.2819944835946,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 2.384185791015625e-07,
        "abs_error_over_ulp_gR": 131072.0,
        "abs_error_over_ulp_sum_gU_old": 262144.0,
        "argmax_flat_index": 58442,
        "argmax_multi_index_row_major": [
          228,
          74
        ],
        "floor_1e_6_active": false,
        "gR_value": -1.52587890625e-05,
        "max_rel_old": 0.015625,
        "old_denominator": 1.52587890625e-05,
        "sum_gU_old_value": -1.5020370483398438e-05,
        "ulp_gR": 1.8189894035458565e-12,
        "ulp_sum_gU_old": 9.094947017729282e-13
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.5268869899434996e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.934528920485451e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 3.0517578125e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.337860107421875e-06
        }
      },
      "metrics": {
        "E_L2": 3.5268869899434996e-08,
        "E_inf": 5.934528920485451e-08,
        "max_abs": 3.337860107421875e-06,
        "max_rel_old": 0.0234375,
        "norm_S64_inf": 56.244733810424805,
        "norm_S64_l2": 2771.2821281194133,
        "norm_gR64_inf": 56.24473571777344,
        "norm_gR64_l2": 2771.2821287343177,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 56.24473571777344,
        "M64": 56.24473571777344,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_O": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.916801831123129e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.3657683271587437e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.916801831123129e-17,
        "E_inf": 2.3657683271587437e-16,
        "max_abs": 2.842170943040401e-14,
        "max_rel_old": 4.1231550115599514e-13,
        "norm_S64_inf": 120.13733172486121,
        "norm_S64_l2": 6064.795964450233,
        "norm_gR64_inf": 120.13733172486124,
        "norm_gR64_l2": 6064.795964450233,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 9.5367431640625e-07,
        "abs_error_over_ulp_gR": 1024.0,
        "abs_error_over_ulp_sum_gU_old": 1024.0,
        "argmax_flat_index": 43611,
        "argmax_multi_index_row_major": [
          170,
          91
        ],
        "floor_1e_6_active": false,
        "gR_value": 0.00849151611328125,
        "max_rel_old": 0.00011229646042920649,
        "old_denominator": 0.008492469787597656,
        "sum_gU_old_value": 0.008492469787597656,
        "ulp_gR": 9.313225746154785e-10,
        "ulp_sum_gU_old": 9.313225746154785e-10
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.7404825000913376e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 6.350562446295079e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 6.103515625e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 7.62939453125e-06
        }
      },
      "metrics": {
        "E_L2": 3.7404825000913376e-08,
        "E_inf": 6.350562446295079e-08,
        "max_abs": 7.62939453125e-06,
        "max_rel_old": 0.00011085245538188671,
        "norm_S64_inf": 120.13730430603027,
        "norm_S64_l2": 6064.796101033479,
        "norm_gR64_inf": 120.13729858398438,
        "norm_gR64_l2": 6064.79609977407,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 120.1373062133789,
        "M64": 120.13730430603027,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_Q": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 1.0110998316535487e-16
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.336026163008422e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 1.0110998316535487e-16,
        "E_inf": 1.336026163008422e-16,
        "max_abs": 7.105427357601002e-15,
        "max_rel_old": 8.148579614654066e-13,
        "norm_S64_inf": 53.1832950157295,
        "norm_S64_l2": 2706.623521284866,
        "norm_gR64_inf": 53.1832950157295,
        "norm_gR64_l2": 2706.623521284866,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 3.5762786865234375e-07,
        "abs_error_over_ulp_gR": 1536.0,
        "abs_error_over_ulp_sum_gU_old": 1536.0,
        "argmax_flat_index": 43771,
        "argmax_multi_index_row_major": [
          170,
          251
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.0024535655975341797,
        "max_rel_old": 0.00014573718362953514,
        "old_denominator": 0.002453923225402832,
        "sum_gU_old_value": -0.002453923225402832,
        "ulp_gR": 2.3283064365386963e-10,
        "ulp_sum_gU_old": 2.3283064365386963e-10
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.449233795108921e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.827847744626509e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 3.0517578125e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.0994415283203125e-06
        }
      },
      "metrics": {
        "E_L2": 3.449233795108921e-08,
        "E_inf": 5.827847744626509e-08,
        "max_abs": 3.0994415283203125e-06,
        "max_rel_old": 0.00027827248441674086,
        "norm_S64_inf": 53.18329620361328,
        "norm_S64_l2": 2706.6236414051814,
        "norm_gR64_inf": 53.18329620361328,
        "norm_gR64_l2": 2706.6236410188635,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 53.18329620361328,
        "M64": 53.18329620361328,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_V": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 1.0066052987451897e-16
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.046460090647229e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 1.0066052987451897e-16,
        "E_inf": 2.046460090647229e-16,
        "max_abs": 2.842170943040401e-14,
        "max_rel_old": 3.509586379179977e-12,
        "norm_S64_inf": 138.88230491421479,
        "norm_S64_l2": 5868.523613916143,
        "norm_gR64_inf": 138.88230491421476,
        "norm_gR64_l2": 5868.523613916143,
        "shape": [
          256,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 9.5367431640625e-07,
        "abs_error_over_ulp_gR": 8192.0,
        "abs_error_over_ulp_sum_gU_old": 8192.0,
        "argmax_flat_index": 51269,
        "argmax_multi_index_row_major": [
          200,
          69
        ],
        "floor_1e_6_active": false,
        "gR_value": 0.0016870498657226562,
        "max_rel_old": 0.0005652911495417356,
        "old_denominator": 0.0016870498657226562,
        "sum_gU_old_value": 0.00168609619140625,
        "ulp_gR": 1.1641532182693481e-10,
        "ulp_sum_gU_old": 1.1641532182693481e-10
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.692421240056949e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.493424316159314e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 0.0001220703125,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 7.62939453125e-06
        }
      },
      "metrics": {
        "E_L2": 3.692421240056949e-08,
        "E_inf": 5.493424316159314e-08,
        "max_abs": 7.62939453125e-06,
        "max_rel_old": 0.00018786398647379298,
        "norm_S64_inf": 138.88230800628662,
        "norm_S64_l2": 5868.523739158204,
        "norm_gR64_inf": 138.88230895996094,
        "norm_gR64_l2": 5868.5237394675405,
        "shape": [
          256,
          256
        ]
      },
      "scale_context": {
        "M32": 138.88230895996094,
        "M64": 138.88230895996094,
        "ULP_M32": 1.52587890625e-05
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_down": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.919484239538804e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.0361174497765924e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.919484239538804e-17,
        "E_inf": 1.0361174497765924e-16,
        "max_abs": 7.105427357601002e-15,
        "max_rel_old": 1.7044137767907214e-12,
        "norm_S64_inf": 68.57743163319056,
        "norm_S64_l2": 3992.8451868100715,
        "norm_gR64_inf": 68.57743163319056,
        "norm_gR64_l2": 3992.845186810071,
        "shape": [
          256,
          1024
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 5.960464477539063e-08,
        "abs_error_over_ulp_gR": 32768.0,
        "abs_error_over_ulp_sum_gU_old": 32768.0,
        "argmax_flat_index": 210487,
        "argmax_multi_index_row_major": [
          205,
          567
        ],
        "floor_1e_6_active": false,
        "gR_value": 2.968311309814453e-05,
        "max_rel_old": 0.0020040080416947603,
        "old_denominator": 2.9742717742919922e-05,
        "sum_gU_old_value": 2.9742717742919922e-05,
        "ulp_gR": 1.8189894035458565e-12,
        "ulp_sum_gU_old": 1.8189894035458565e-12
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.675693793257789e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.56261366244945e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 6.103515625e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.814697265625e-06
        }
      },
      "metrics": {
        "E_L2": 3.675693793257789e-08,
        "E_inf": 5.56261366244945e-08,
        "max_abs": 3.814697265625e-06,
        "max_rel_old": 0.00125344697919278,
        "norm_S64_inf": 68.57742595672607,
        "norm_S64_l2": 3992.8452435867403,
        "norm_gR64_inf": 68.57742309570312,
        "norm_gR64_l2": 3992.8452431081123,
        "shape": [
          256,
          1024
        ]
      },
      "scale_context": {
        "M32": 68.57742309570312,
        "M64": 68.57742595672607,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_gate": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.933853962643564e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 2.3185297434899744e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.933853962643564e-17,
        "E_inf": 2.3185297434899744e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 1.5509436354645041e-12,
        "norm_S64_inf": 61.29252710733427,
        "norm_S64_l2": 4018.8590907064245,
        "norm_gR64_inf": 61.29252710733428,
        "norm_gR64_l2": 4018.8590907064245,
        "shape": [
          1024,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 2.384185791015625e-07,
        "abs_error_over_ulp_gR": 8192.0,
        "abs_error_over_ulp_sum_gU_old": 8192.0,
        "argmax_flat_index": 102245,
        "argmax_multi_index_row_major": [
          399,
          101
        ],
        "floor_1e_6_active": false,
        "gR_value": 0.0004239082336425781,
        "max_rel_old": 0.0005621135351248085,
        "old_denominator": 0.0004241466522216797,
        "sum_gU_old_value": 0.0004241466522216797,
        "ulp_gR": 2.9103830456733704e-11,
        "ulp_sum_gU_old": 2.9103830456733704e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.6961247003740165e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 6.223755589632692e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 3.0517578125e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.814697265625e-06
        }
      },
      "metrics": {
        "E_L2": 3.6961247003740165e-08,
        "E_inf": 6.223755589632692e-08,
        "max_abs": 3.814697265625e-06,
        "max_rel_old": 0.0008429334082607474,
        "norm_S64_inf": 61.292529582977295,
        "norm_S64_l2": 4018.8591876710852,
        "norm_gR64_inf": 61.29253005981445,
        "norm_gR64_l2": 4018.8591877673966,
        "shape": [
          1024,
          256
        ]
      },
      "scale_context": {
        "M32": 61.29253005981445,
        "M64": 61.29253005981445,
        "ULP_M32": 3.814697265625e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  },
  "W_up": {
    "cpu_fp64_oracle": {
      "finite": true,
      "gates": {
        "E_L2_FP64": {
          "limit": 1e-15,
          "pass": true,
          "value": 9.92240763236267e-17
        },
        "E_inf_FP64": {
          "limit": 1e-14,
          "pass": true,
          "value": 1.9115705260217235e-16
        }
      },
      "max_abs_FP64_threshold": null,
      "metrics": {
        "E_L2": 9.92240763236267e-17,
        "E_inf": 1.9115705260217235e-16,
        "max_abs": 1.4210854715202004e-14,
        "max_rel_old": 2.3998996166878063e-12,
        "norm_S64_inf": 74.34125250286742,
        "norm_S64_l2": 4017.1489535004603,
        "norm_gR64_inf": 74.3412525028674,
        "norm_gR64_l2": 4017.1489535004603,
        "shape": [
          1024,
          256
        ]
      },
      "pass": true
    },
    "diagnostics_NON_GATE": {
      "S_reverse_equal_gR_NON_GATE": true,
      "fp32_sum_names": [
        "S_forward",
        "S_reverse",
        "S_pairwise",
        "S_stack"
      ],
      "old_max_rel": {
        "S_reverse_equal_gR_NON_GATE": true,
        "abs_error": 4.76837158203125e-07,
        "abs_error_over_ulp_gR": 32768.0,
        "abs_error_over_ulp_sum_gU_old": 32768.0,
        "argmax_flat_index": 91014,
        "argmax_multi_index_row_major": [
          355,
          134
        ],
        "floor_1e_6_active": false,
        "gR_value": -0.0002396106719970703,
        "max_rel_old": 0.001990049844607711,
        "old_denominator": 0.0002396106719970703,
        "sum_gU_old_value": -0.0002391338348388672,
        "ulp_gR": 1.4551915228366852e-11,
        "ulp_sum_gU_old": 1.4551915228366852e-11
      }
    },
    "operational": {
      "all_cpu_oracle_tensors_finite": true,
      "all_cuda_family_tensors_finite": true
    },
    "primary_S64": {
      "finite": true,
      "gates": {
        "A": {
          "limit": 2e-07,
          "name": "normwise_L2",
          "pass": true,
          "value": 3.6898863037850274e-08
        },
        "B": {
          "limit": 5e-07,
          "name": "normwise_Linf",
          "pass": true,
          "value": 5.131333267908834e-08
        },
        "C": {
          "limit_ulps": 8.0,
          "limit_value": 6.103515625e-05,
          "name": "max_abs_scale_aware",
          "pass": true,
          "value": 3.814697265625e-06
        }
      },
      "metrics": {
        "E_L2": 3.6898863037850274e-08,
        "E_inf": 5.131333267908834e-08,
        "max_abs": 3.814697265625e-06,
        "max_rel_old": 0.001610305958132045,
        "norm_S64_inf": 74.34124946594238,
        "norm_S64_l2": 4017.149032379015,
        "norm_gR64_inf": 74.34124755859375,
        "norm_gR64_l2": 4017.149031896814,
        "shape": [
          1024,
          256
        ]
      },
      "scale_context": {
        "M32": 74.34124755859375,
        "M64": 74.34124946594238,
        "ULP_M32": 7.62939453125e-06
      },
      "scientific_gates_pass": true
    },
    "seed_family_pass": true
  }
}
```

### Persisted gradient bundle

```json
{
  "bundle_path": "seed_20261005_raw_gradients.pt",
  "bundle_sha256": "4e17682fc9ff18107d8b33be0661035b932a18003617d71a9187225fe99f993f",
  "bundle_size_bytes": 105685741,
  "tensor_metadata": {
    "seed_20261005/cpu_fp64_oracle/W_K/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "6437c30b0708d0da73c508a0156d3c2283e64ef0601f8f721bea704270018cb6",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_K/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "707cda11bc32ec677f614096edbb1647a56d387fa742ca9d8a72747fe35e0c75",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_K/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "2d2c4b30bcd9614779609d03acfd041dc515fdee5ab71050ededc82345e4c547",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_K/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "c708fc29dc7d1c009f62ea0f09a20f6e4d7d7b8ffddaeb899386005a3b4112ad",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_K/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "acfcf7bd359015821490bf407ce7b780016f794a6e4548c019883f2a4bacc23d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_K/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "9a35ba40521cfd834845ab0fd7107564be37aabd6cab18e64c510697d39b9ec4",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_O/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "d4c4ca5c278cfce28488b09ac8fcc12efd5cebe9e17bfcacaa4fae6fe75588d3",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_O/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "6e8ca38724480c5e77d4271b7b27d5f79c2eee5703d5488e33b30b9abda95ac5",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_O/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "7747fc07d670bd99fa5928c7e0497ffc90626bd9ad697d3682a53cd86a5766e0",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_O/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "51611a583c335cabb70c67f8dca2fef184b8c5d9ddc946b4b21239c00bf65809",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_O/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "2a7bdd4eab79d187401b836a1572db7c6bc64db9caa2eab7aff26794d1ef4fa2",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_O/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "2e4ef2937912d223810d0480901f1f52c0ce347ed7af282473c868bc61ca326d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_Q/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "a6495ef517dd3b67387246200450034bc9cea43ee1f2f116127f22f643444edf",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_Q/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "cb3409c3549473b43e0bf40d2bdd73c112072e92b912b81f155ea0c323eafa77",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_Q/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "a3edac83be9eb92c3628a543b42936f25d9c916aa8b33d34386b4b38e937d289",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_Q/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "e42c1efab47b7d4034890adaed703d5baf9b0a06adffeaca03232c8cc19bda9a",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_Q/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "948909793e4888b45dc40f94028570ecab5f635663980efd803e1a88cdbceb98",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_Q/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "4825e8b58ce33fd3a82a1fa2189484e7d7d39b121c67045eeb0c6cbca16aa6b2",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_V/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "9595b0177e863365730691ffc9819d06494a3a2c891612be52c77dc821996fa8",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_V/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "27bc33d9cd61068fe15733620db7342cd1105e82dcc42c12e51d314ec326b081",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_V/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "f78321b11e79dc520e965fb9d85e7afa0167bf3d3c1add037da8b27f2c27d629",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_V/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "c3a25a707a3820f8ff3e1533a234f68f58b3b40d6101d287c6e060b42072bf3c",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_V/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "337bf08f4bf0719129f95c9c023d6c3883c4306e93b0db6de31fe34e05e1653a",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_V/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "b65e3fd898bc8fd4e7db683dbe73acea3769f3a9243b410745c24aad65ead503",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_down/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "bb2749eb00ff7e946ec350c200c929e1480b838d5c8620b310d7d680ecdb9ce2",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_down/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "c1946533011926574e2107ceef2d8ec8858078bf14362ab2fd9cdd79614d5f62",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_down/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "789ee232c40299422dbd14adf74b9a1f56c1a1456c1dd4ee6a721f34f3661d6e",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_down/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "0d3c1f414f0d0ee600666488c775b490cb8a6e32d9819393fd905385c59e690b",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_down/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "fc17fedb62fa0aad4cdb95d3d9bf46e525112ba440f855273789fd35a070bf31",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_down/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "71c35420037b5d694942fe2d7374030154e484aa4e375cfe94ef796938b5b54b",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_gate/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "a6ba6d93f66799a4846ef8dd6d9724e78af9304d3d6dc44e8c4ec74e81ca8c5d",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_gate/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "13b7e6cc96ad9ef467a99375ee181574868209057625218271f13e70b23e724e",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_gate/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "e2466043a62b170622d05ac0d9caab5f98470923ee4248368831c2fb835d74b5",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_gate/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "1dbc2d688bedfac09f1c90def9921c4ab18ac96f68a9ae2cc22bc905e00b2d74",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_gate/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "d44f3888c6ee470d0ba534989c18694e60ea5f2adc8e6e6de0126b51f6015a6c",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_gate/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "3f31ece0a88d95b4e2da9c84f00ff711110dab51085ddec000e34965a510d317",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_up/S64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "a711cabe8e675e6619fb9d6a0b9f211232123c6c5aa91437d1faa04c1f531c97",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_up/gR64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "20cae543fbd1c5931cbf0d1298239f0c867b623445fe6698955f5d1f0d050aba",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_up/gU0_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "52d8baed4edfecba28d126032418d23a1a2141d65c17eecd5519b4f91ee59340",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_up/gU1_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "dd73967657bf0074095e32fc2a8f61cf21113e095c4f950d0851f815c8e661c6",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_up/gU2_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "ba37c7add11e32d3d2677e99e99feadd60020e799f11c3fcdd039d1408eb05b5",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cpu_fp64_oracle/W_up/gU3_64_cpu": {
      "dtype": "torch.float64",
      "raw_sha256": "75c695af86c53c38e1cb0c4e27de8680ffdb7eb02be1261758a83428db692452",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_K/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "5674ec3c57839e2232c3f37ee2fb40927bc976c1e8b07903cb7485c9c4466d63",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_K/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "6f84d25ffd1f5060ce205148f88282e96ec8d2fb6fc91b16042fa7ffe19aa281",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_K/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "85b219dc93f1312e229f0c7f6e1c590d01e33af37e5b46a2fb74d351705678e0",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_K/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "79cbb9d16df191503dc70f5394b7f43877fc3fe1f1034af0b0baab3bbce5f3c5",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_K/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "9162b7760b82d8540eedd829ac390393ed28523a367634547a92fd0f0fcb390d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_K/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "bb511bf56d29f820b9371ad94aab3915e50b8c06851d265375e46d9b9e61261b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_K/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "8f90f2f40e83444efd6b845dd22023fa5e0f26d56fbe5babceffc38b6100bd2a",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_K/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "61bb47806258866a632d0d015614e62da22dfb05d785ec3c8e7317bb82bb4294",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_K/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "efa9795b40140ab4418633779230783d59c5a86f8b14e7e247d41c00bbf565d8",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_K/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "5b15a30fedbec6e5f8356d44bc6153adcb31a122043ea03e788f334eed0a6fd7",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_K/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "c9795106a380f33525766b9381d78be2873a846171f219542e652101d5ca1c0d",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_O/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "a8be65ab6eb9fe9160a10f40d0886c1821917e23cd9e3aa86d9125bd40f044bc",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_O/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "7fd880911dfd50efbeb78cfa1ca1c952bc5832d1c4897c2c4109695df4b01e82",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_O/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "0baeddea18761270ea9cd345d552100a9db4f0de998b1e186480d5f832613ca8",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_O/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "cfca4772ddcd55d40ab30eee17675509db389aa1529d0852652f8494ddffc5e9",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_O/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "d3d4d4921f18ce738c949cf2babe0bdcc097e56ed0eb32a306c12bf12dcd044e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_O/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "a48efc1a7c8e9e318340df5b6ce4cc5335e490c5b847562c9057fa34fef08911",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_O/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "f4ab203eac64aba40103387852987e4f290d62d25e9aa626320adb5aacec849e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_O/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "ba2933cead9c41a47f0c643a909b38b71f96771282ca4b3c8d1af1ba1bb2b2bb",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_O/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "7dfd16ebf1dc4b43bff20400e5bc24ce5dbf71f0a368f8b8cc67a81e03d41f03",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_O/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "b47188edb26f2b704d22e3553a55d361f815369a9a6b7c6124096004b6b0ce50",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_O/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "a3bf8fe6685444a14021670529c0c7eff099e7a0d7a771c51deda51158242780",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_Q/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "08e0bd87fcf3c11d05ca3c413b33121bfddc56e3a5644ff60b62cdfe9a852e0f",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_Q/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "6f8d0ac80b9102e701840bdf0285942cabe87866d65120beac424d36c851d697",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_Q/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "227f57ae1c2b2f9bd3b4b76f589e82c4228fc0a4502cc53730d18133f3512cfd",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_Q/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "a7f4b61c615e2e0cf84112b94e6e33bdf49377664bf4114a9f2fd9165b1747db",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_Q/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "a94206b1753ee1533a8ffa45eac1b5b952cbe6674178ec70f245836f322c055b",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_Q/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "9c0126b7b4fd274946b3bd66ecabbd22da0be00d2612f19bbdbbf3ebeba1a84a",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_Q/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "897230b45ddbd7563e5f4c61b08223c0534b8ea83ddd1bff7f265504626f30a8",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_Q/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "eb8ef8af82635a418b44ae9ef5fa9860c371867d494ab24795b262989d175091",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_Q/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "64a57439e05440b248aea1afd2a1f1b10cb20de107acb32a8d50bd3bf34c846e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_Q/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "c79145a869a601e4976341c5ec2818a0790beb1eb4268d66706052bc34145a41",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_Q/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "b3e104fe3788dcfaeadb907ae18022be6f47a74b554f24d733a8ae71602ea4a9",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_V/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "01e0e723074ef00088a51498d6872b4df24ca55764ad2fd4ef999d2c18a21156",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_V/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "04a3710f5310971bed481209ba58ce5dfc251a0b95d10a60dc138dcd0a23472f",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_V/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "3bd7bb801e50e91f28ca2c762f4e57db1d31a9d025956516a84425a6dcdf13fd",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_V/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "59426aa6ce2334eb9dbe14a73105e9a4f38ad01e9a2d70c90a9e43f1202d37d2",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_V/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "65e7551e9e4a97b0601620509c2959f82fb2153b5a28d7d8649152f9c44e77cc",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_V/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "af57cda6273f6d1baf1fc8baf3e3e186dcbdf7d5aa35420b8f6d980d75aed91e",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_V/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "e71b9e6c6590fa5ede32ba1a5fcf27a38d8a83ff29bb5d588666c6b60174a19f",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_V/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "e8a5db6cdf22676182c7a0ebd2143728f26a986d912ae804736544812c57bbf8",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_V/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "65557d1fa6179126d6e3ac635a5449368e9972ef592534d05f0aea28e3ca2192",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_V/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "2bbefe6031638daae552622d0de2ab3bb95ca94dbf9085da8402f9b79ed9ff63",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_V/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "7fca13bc1f45384bfe82cda49720a68a4e202c46a847e1b6dffdf3a560a92282",
      "shape": [
        256,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_down/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "840b96091abdc8ea7412ed2f3688203bb6fbfb7f6bcfe9d63798a82861bd62e9",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cuda_fp32/W_down/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "5016bfb737a5fc3324650cf01a125bf8d535805608c31d431257b2d84042c3a0",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cuda_fp32/W_down/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "e47637b6e107596263955f8c23379f7cee92f4353992231019ac70a02890c78b",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cuda_fp32/W_down/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "3c34ffcf0cd41ff8608c332d181b021e711858e175699798e3a45681d4300b8a",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cuda_fp32/W_down/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "593c0dc6bb40edf91f0096038cc4eb7b42073cdc13b83cf1b4133d3e7ec698fa",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cuda_fp32/W_down/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "b33fdcd958dfd04c61f9e7f0b201d0ae7a467e6367a8aa6741def8f02d777fe1",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cuda_fp32/W_down/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "a09541890d72d58b422e7a8b514a4c5f9e1cde8150b422d87f1bfacd770450df",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cuda_fp32/W_down/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "d956309f02c98dfa439e6289d0827841e8552f941ab67b9dd3cfdfbd3bb65ec9",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cuda_fp32/W_down/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "4818d7d3d13cd81d51c4ac14ebad552c73556d6c8e518d09cec558f8c6fa2162",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cuda_fp32/W_down/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "11e365cb862b8afb989d0389f817e6d0160aba671b85a9937a0d1c5408d0b0ed",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cuda_fp32/W_down/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "73e99e5078147eaed9d41aaa75155dafcdc472535a3dd2615f58d8c20b449b35",
      "shape": [
        256,
        1024
      ]
    },
    "seed_20261005/cuda_fp32/W_gate/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "14c600f7068098638de281e75bb1a57119da452d9cfbdd864195d61ef1355825",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_gate/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "8de97aa941cab65a386a120fceaf14cf47a21dfcf16e3860cbff7be6a357809b",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_gate/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "a82737b41f8768756f29e812aea321ef3ccedb14e5f5f70481862ab1037892fe",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_gate/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "54d42a2cf8b00cb28fed3cfb000dba2d3e78be1980c23f852b3cb08de1287fb7",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_gate/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "4ad4dd7a166bcec9c8fcc44cac128f20689c677c446f982f4eaa4611c7465ff9",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_gate/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "319bab65e54bd6d1930001daaae65f2b70ef7735e086ca41a740f78689b415e4",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_gate/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "01ea4836c942a7fbb0cd33d6a29d73fe27019fc1faabe8eb954588fa062f8fac",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_gate/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "ceaeff9b20429bcdd5fb77eae2e143c1aab7030e5b916375793f9bbc171b7dd0",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_gate/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "1a2fa82e055e032434d704ae5c859e3f458d7dd9d3df44c532ef4cb4985998b1",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_gate/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "b24213354675edbaae27d562fb6626b29e52cfebb114d5483386f1b47680757d",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_gate/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "84fb8c4cd9a4843fdebbab9a7860f75e7ea54c8c247852fce0a21080b1c99fc1",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_up/S64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "5535ed2296b66c294c2d656e33ab708702175cd90eb6d50a72ef724951f5f64e",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_up/S_forward_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "96e33e1823892f8de29f65a7572a3387f33ace999721f9db67524e12ba24b0a8",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_up/S_pairwise_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "6455ca5a40fda3bec80e4c5813c1fcf7342910fba53a7add5bcdf4797ec0c5df",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_up/S_reverse_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "55a16ebbfb60e3f2eb251965d6120317a047f3fb046a7389dbc29e922358dc84",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_up/S_stack_cuda": {
      "dtype": "torch.float32",
      "raw_sha256": "aca07827280f237c2075ef5c3f522fae04db0ea8d323aa802c1e9ec0ed24de35",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_up/gR64_primary_cuda": {
      "dtype": "torch.float64",
      "raw_sha256": "b92720b6f78990d4e351f2249dea279020acf1b425b99d204a809f90d2280fbe",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_up/gR_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "667fd6a8d37547773d19d64b18f0b47f4954bde4378a8c2bdeaaeaeee0b56d1c",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_up/gU0_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "0eaf50ff1efc23b68d2b0f3c3f8636518186ddefddd681cbed81fe1940d35527",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_up/gU1_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "318ef6dfdf1f9408d5cbb85366260dd4b01d1cba6b1d4cbe0dde406f04f8e5d2",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_up/gU2_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "884e4bb70e667d3fb98c3760bd8b9dc28579bde5f6c007095268d1ba3c8b9fd9",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/cuda_fp32/W_up/gU3_cuda_fp32": {
      "dtype": "torch.float32",
      "raw_sha256": "c412b1a22f2078a982f8a96d8af8224dfafe36e2d19c32e27d3cf0f5acfbab77",
      "shape": [
        1024,
        256
      ]
    },
    "seed_20261005/inputs/w_fp32_cpu": {
      "dtype": "torch.float32",
      "raw_sha256": "39d940993e326f835614af3a91f140f28fdecbbd80f29a36335b02ad74aed388",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261005/inputs/x_fp32_cpu": {
      "dtype": "torch.float32",
      "raw_sha256": "0a0a2531f280490a82375cbe6db3cc03958fcde1fcfe904e80e2d2451611bf11",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261005/structural_traces/R4/round_1": {
      "dtype": "torch.float32",
      "raw_sha256": "a523c8f1366828fa696789f1baf7cbb577c4d238016308c7d42896b3bb5bc72a",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261005/structural_traces/R4/round_2": {
      "dtype": "torch.float32",
      "raw_sha256": "5eb0f76278d97d870190de2dcbe381cc862902f1d98c5acabdecf04e8466f2d5",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261005/structural_traces/R4/round_3": {
      "dtype": "torch.float32",
      "raw_sha256": "538643dcca79879d634a0b7d9bd645f6fe01f5e2c51a3089b4b893443f8d5032",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261005/structural_traces/R4/round_4": {
      "dtype": "torch.float32",
      "raw_sha256": "02f3c597c1303168567f778681988b8b4cac5ccd922f9d0b24223353dcf1225e",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261005/structural_traces/R4_final": {
      "dtype": "torch.float32",
      "raw_sha256": "69b34fa69680ebbb32fd8c910f115e7b10c3855937844cad1a3d7aabb13b0883",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261005/structural_traces/U4/round_1": {
      "dtype": "torch.float32",
      "raw_sha256": "2ad90b8f3886e71d85b6247734ace26725972af8282c24b2879fd72f37da5815",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261005/structural_traces/U4/round_2": {
      "dtype": "torch.float32",
      "raw_sha256": "a3342356b1ec3e07badde9349f2625e502173315feb609c086610fc109e8954c",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261005/structural_traces/U4/round_3": {
      "dtype": "torch.float32",
      "raw_sha256": "6df7a459d4f2b601956e2efbde652b8725ca8aaa627e829dbea935530c6e18fc",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261005/structural_traces/U4/round_4": {
      "dtype": "torch.float32",
      "raw_sha256": "014accc643c94622cc0670c2859e640dafc3301706269def9444d5262553e216",
      "shape": [
        8,
        8,
        256
      ]
    },
    "seed_20261005/structural_traces/U4_final": {
      "dtype": "torch.float32",
      "raw_sha256": "a9d2805d7771000b2825b12aeb1a1c8d36b8825e2fe37bb7bf2df9c01cc77970",
      "shape": [
        8,
        8,
        256
      ]
    }
  },
  "tensor_raw_sha256": {
    "seed_20261005/cpu_fp64_oracle/W_K/S64_cpu": "6437c30b0708d0da73c508a0156d3c2283e64ef0601f8f721bea704270018cb6",
    "seed_20261005/cpu_fp64_oracle/W_K/gR64_cpu": "707cda11bc32ec677f614096edbb1647a56d387fa742ca9d8a72747fe35e0c75",
    "seed_20261005/cpu_fp64_oracle/W_K/gU0_64_cpu": "2d2c4b30bcd9614779609d03acfd041dc515fdee5ab71050ededc82345e4c547",
    "seed_20261005/cpu_fp64_oracle/W_K/gU1_64_cpu": "c708fc29dc7d1c009f62ea0f09a20f6e4d7d7b8ffddaeb899386005a3b4112ad",
    "seed_20261005/cpu_fp64_oracle/W_K/gU2_64_cpu": "acfcf7bd359015821490bf407ce7b780016f794a6e4548c019883f2a4bacc23d",
    "seed_20261005/cpu_fp64_oracle/W_K/gU3_64_cpu": "9a35ba40521cfd834845ab0fd7107564be37aabd6cab18e64c510697d39b9ec4",
    "seed_20261005/cpu_fp64_oracle/W_O/S64_cpu": "d4c4ca5c278cfce28488b09ac8fcc12efd5cebe9e17bfcacaa4fae6fe75588d3",
    "seed_20261005/cpu_fp64_oracle/W_O/gR64_cpu": "6e8ca38724480c5e77d4271b7b27d5f79c2eee5703d5488e33b30b9abda95ac5",
    "seed_20261005/cpu_fp64_oracle/W_O/gU0_64_cpu": "7747fc07d670bd99fa5928c7e0497ffc90626bd9ad697d3682a53cd86a5766e0",
    "seed_20261005/cpu_fp64_oracle/W_O/gU1_64_cpu": "51611a583c335cabb70c67f8dca2fef184b8c5d9ddc946b4b21239c00bf65809",
    "seed_20261005/cpu_fp64_oracle/W_O/gU2_64_cpu": "2a7bdd4eab79d187401b836a1572db7c6bc64db9caa2eab7aff26794d1ef4fa2",
    "seed_20261005/cpu_fp64_oracle/W_O/gU3_64_cpu": "2e4ef2937912d223810d0480901f1f52c0ce347ed7af282473c868bc61ca326d",
    "seed_20261005/cpu_fp64_oracle/W_Q/S64_cpu": "a6495ef517dd3b67387246200450034bc9cea43ee1f2f116127f22f643444edf",
    "seed_20261005/cpu_fp64_oracle/W_Q/gR64_cpu": "cb3409c3549473b43e0bf40d2bdd73c112072e92b912b81f155ea0c323eafa77",
    "seed_20261005/cpu_fp64_oracle/W_Q/gU0_64_cpu": "a3edac83be9eb92c3628a543b42936f25d9c916aa8b33d34386b4b38e937d289",
    "seed_20261005/cpu_fp64_oracle/W_Q/gU1_64_cpu": "e42c1efab47b7d4034890adaed703d5baf9b0a06adffeaca03232c8cc19bda9a",
    "seed_20261005/cpu_fp64_oracle/W_Q/gU2_64_cpu": "948909793e4888b45dc40f94028570ecab5f635663980efd803e1a88cdbceb98",
    "seed_20261005/cpu_fp64_oracle/W_Q/gU3_64_cpu": "4825e8b58ce33fd3a82a1fa2189484e7d7d39b121c67045eeb0c6cbca16aa6b2",
    "seed_20261005/cpu_fp64_oracle/W_V/S64_cpu": "9595b0177e863365730691ffc9819d06494a3a2c891612be52c77dc821996fa8",
    "seed_20261005/cpu_fp64_oracle/W_V/gR64_cpu": "27bc33d9cd61068fe15733620db7342cd1105e82dcc42c12e51d314ec326b081",
    "seed_20261005/cpu_fp64_oracle/W_V/gU0_64_cpu": "f78321b11e79dc520e965fb9d85e7afa0167bf3d3c1add037da8b27f2c27d629",
    "seed_20261005/cpu_fp64_oracle/W_V/gU1_64_cpu": "c3a25a707a3820f8ff3e1533a234f68f58b3b40d6101d287c6e060b42072bf3c",
    "seed_20261005/cpu_fp64_oracle/W_V/gU2_64_cpu": "337bf08f4bf0719129f95c9c023d6c3883c4306e93b0db6de31fe34e05e1653a",
    "seed_20261005/cpu_fp64_oracle/W_V/gU3_64_cpu": "b65e3fd898bc8fd4e7db683dbe73acea3769f3a9243b410745c24aad65ead503",
    "seed_20261005/cpu_fp64_oracle/W_down/S64_cpu": "bb2749eb00ff7e946ec350c200c929e1480b838d5c8620b310d7d680ecdb9ce2",
    "seed_20261005/cpu_fp64_oracle/W_down/gR64_cpu": "c1946533011926574e2107ceef2d8ec8858078bf14362ab2fd9cdd79614d5f62",
    "seed_20261005/cpu_fp64_oracle/W_down/gU0_64_cpu": "789ee232c40299422dbd14adf74b9a1f56c1a1456c1dd4ee6a721f34f3661d6e",
    "seed_20261005/cpu_fp64_oracle/W_down/gU1_64_cpu": "0d3c1f414f0d0ee600666488c775b490cb8a6e32d9819393fd905385c59e690b",
    "seed_20261005/cpu_fp64_oracle/W_down/gU2_64_cpu": "fc17fedb62fa0aad4cdb95d3d9bf46e525112ba440f855273789fd35a070bf31",
    "seed_20261005/cpu_fp64_oracle/W_down/gU3_64_cpu": "71c35420037b5d694942fe2d7374030154e484aa4e375cfe94ef796938b5b54b",
    "seed_20261005/cpu_fp64_oracle/W_gate/S64_cpu": "a6ba6d93f66799a4846ef8dd6d9724e78af9304d3d6dc44e8c4ec74e81ca8c5d",
    "seed_20261005/cpu_fp64_oracle/W_gate/gR64_cpu": "13b7e6cc96ad9ef467a99375ee181574868209057625218271f13e70b23e724e",
    "seed_20261005/cpu_fp64_oracle/W_gate/gU0_64_cpu": "e2466043a62b170622d05ac0d9caab5f98470923ee4248368831c2fb835d74b5",
    "seed_20261005/cpu_fp64_oracle/W_gate/gU1_64_cpu": "1dbc2d688bedfac09f1c90def9921c4ab18ac96f68a9ae2cc22bc905e00b2d74",
    "seed_20261005/cpu_fp64_oracle/W_gate/gU2_64_cpu": "d44f3888c6ee470d0ba534989c18694e60ea5f2adc8e6e6de0126b51f6015a6c",
    "seed_20261005/cpu_fp64_oracle/W_gate/gU3_64_cpu": "3f31ece0a88d95b4e2da9c84f00ff711110dab51085ddec000e34965a510d317",
    "seed_20261005/cpu_fp64_oracle/W_up/S64_cpu": "a711cabe8e675e6619fb9d6a0b9f211232123c6c5aa91437d1faa04c1f531c97",
    "seed_20261005/cpu_fp64_oracle/W_up/gR64_cpu": "20cae543fbd1c5931cbf0d1298239f0c867b623445fe6698955f5d1f0d050aba",
    "seed_20261005/cpu_fp64_oracle/W_up/gU0_64_cpu": "52d8baed4edfecba28d126032418d23a1a2141d65c17eecd5519b4f91ee59340",
    "seed_20261005/cpu_fp64_oracle/W_up/gU1_64_cpu": "dd73967657bf0074095e32fc2a8f61cf21113e095c4f950d0851f815c8e661c6",
    "seed_20261005/cpu_fp64_oracle/W_up/gU2_64_cpu": "ba37c7add11e32d3d2677e99e99feadd60020e799f11c3fcdd039d1408eb05b5",
    "seed_20261005/cpu_fp64_oracle/W_up/gU3_64_cpu": "75c695af86c53c38e1cb0c4e27de8680ffdb7eb02be1261758a83428db692452",
    "seed_20261005/cuda_fp32/W_K/S64_primary_cuda": "5674ec3c57839e2232c3f37ee2fb40927bc976c1e8b07903cb7485c9c4466d63",
    "seed_20261005/cuda_fp32/W_K/S_forward_cuda": "6f84d25ffd1f5060ce205148f88282e96ec8d2fb6fc91b16042fa7ffe19aa281",
    "seed_20261005/cuda_fp32/W_K/S_pairwise_cuda": "85b219dc93f1312e229f0c7f6e1c590d01e33af37e5b46a2fb74d351705678e0",
    "seed_20261005/cuda_fp32/W_K/S_reverse_cuda": "79cbb9d16df191503dc70f5394b7f43877fc3fe1f1034af0b0baab3bbce5f3c5",
    "seed_20261005/cuda_fp32/W_K/S_stack_cuda": "9162b7760b82d8540eedd829ac390393ed28523a367634547a92fd0f0fcb390d",
    "seed_20261005/cuda_fp32/W_K/gR64_primary_cuda": "bb511bf56d29f820b9371ad94aab3915e50b8c06851d265375e46d9b9e61261b",
    "seed_20261005/cuda_fp32/W_K/gR_cuda_fp32": "8f90f2f40e83444efd6b845dd22023fa5e0f26d56fbe5babceffc38b6100bd2a",
    "seed_20261005/cuda_fp32/W_K/gU0_cuda_fp32": "61bb47806258866a632d0d015614e62da22dfb05d785ec3c8e7317bb82bb4294",
    "seed_20261005/cuda_fp32/W_K/gU1_cuda_fp32": "efa9795b40140ab4418633779230783d59c5a86f8b14e7e247d41c00bbf565d8",
    "seed_20261005/cuda_fp32/W_K/gU2_cuda_fp32": "5b15a30fedbec6e5f8356d44bc6153adcb31a122043ea03e788f334eed0a6fd7",
    "seed_20261005/cuda_fp32/W_K/gU3_cuda_fp32": "c9795106a380f33525766b9381d78be2873a846171f219542e652101d5ca1c0d",
    "seed_20261005/cuda_fp32/W_O/S64_primary_cuda": "a8be65ab6eb9fe9160a10f40d0886c1821917e23cd9e3aa86d9125bd40f044bc",
    "seed_20261005/cuda_fp32/W_O/S_forward_cuda": "7fd880911dfd50efbeb78cfa1ca1c952bc5832d1c4897c2c4109695df4b01e82",
    "seed_20261005/cuda_fp32/W_O/S_pairwise_cuda": "0baeddea18761270ea9cd345d552100a9db4f0de998b1e186480d5f832613ca8",
    "seed_20261005/cuda_fp32/W_O/S_reverse_cuda": "cfca4772ddcd55d40ab30eee17675509db389aa1529d0852652f8494ddffc5e9",
    "seed_20261005/cuda_fp32/W_O/S_stack_cuda": "d3d4d4921f18ce738c949cf2babe0bdcc097e56ed0eb32a306c12bf12dcd044e",
    "seed_20261005/cuda_fp32/W_O/gR64_primary_cuda": "a48efc1a7c8e9e318340df5b6ce4cc5335e490c5b847562c9057fa34fef08911",
    "seed_20261005/cuda_fp32/W_O/gR_cuda_fp32": "f4ab203eac64aba40103387852987e4f290d62d25e9aa626320adb5aacec849e",
    "seed_20261005/cuda_fp32/W_O/gU0_cuda_fp32": "ba2933cead9c41a47f0c643a909b38b71f96771282ca4b3c8d1af1ba1bb2b2bb",
    "seed_20261005/cuda_fp32/W_O/gU1_cuda_fp32": "7dfd16ebf1dc4b43bff20400e5bc24ce5dbf71f0a368f8b8cc67a81e03d41f03",
    "seed_20261005/cuda_fp32/W_O/gU2_cuda_fp32": "b47188edb26f2b704d22e3553a55d361f815369a9a6b7c6124096004b6b0ce50",
    "seed_20261005/cuda_fp32/W_O/gU3_cuda_fp32": "a3bf8fe6685444a14021670529c0c7eff099e7a0d7a771c51deda51158242780",
    "seed_20261005/cuda_fp32/W_Q/S64_primary_cuda": "08e0bd87fcf3c11d05ca3c413b33121bfddc56e3a5644ff60b62cdfe9a852e0f",
    "seed_20261005/cuda_fp32/W_Q/S_forward_cuda": "6f8d0ac80b9102e701840bdf0285942cabe87866d65120beac424d36c851d697",
    "seed_20261005/cuda_fp32/W_Q/S_pairwise_cuda": "227f57ae1c2b2f9bd3b4b76f589e82c4228fc0a4502cc53730d18133f3512cfd",
    "seed_20261005/cuda_fp32/W_Q/S_reverse_cuda": "a7f4b61c615e2e0cf84112b94e6e33bdf49377664bf4114a9f2fd9165b1747db",
    "seed_20261005/cuda_fp32/W_Q/S_stack_cuda": "a94206b1753ee1533a8ffa45eac1b5b952cbe6674178ec70f245836f322c055b",
    "seed_20261005/cuda_fp32/W_Q/gR64_primary_cuda": "9c0126b7b4fd274946b3bd66ecabbd22da0be00d2612f19bbdbbf3ebeba1a84a",
    "seed_20261005/cuda_fp32/W_Q/gR_cuda_fp32": "897230b45ddbd7563e5f4c61b08223c0534b8ea83ddd1bff7f265504626f30a8",
    "seed_20261005/cuda_fp32/W_Q/gU0_cuda_fp32": "eb8ef8af82635a418b44ae9ef5fa9860c371867d494ab24795b262989d175091",
    "seed_20261005/cuda_fp32/W_Q/gU1_cuda_fp32": "64a57439e05440b248aea1afd2a1f1b10cb20de107acb32a8d50bd3bf34c846e",
    "seed_20261005/cuda_fp32/W_Q/gU2_cuda_fp32": "c79145a869a601e4976341c5ec2818a0790beb1eb4268d66706052bc34145a41",
    "seed_20261005/cuda_fp32/W_Q/gU3_cuda_fp32": "b3e104fe3788dcfaeadb907ae18022be6f47a74b554f24d733a8ae71602ea4a9",
    "seed_20261005/cuda_fp32/W_V/S64_primary_cuda": "01e0e723074ef00088a51498d6872b4df24ca55764ad2fd4ef999d2c18a21156",
    "seed_20261005/cuda_fp32/W_V/S_forward_cuda": "04a3710f5310971bed481209ba58ce5dfc251a0b95d10a60dc138dcd0a23472f",
    "seed_20261005/cuda_fp32/W_V/S_pairwise_cuda": "3bd7bb801e50e91f28ca2c762f4e57db1d31a9d025956516a84425a6dcdf13fd",
    "seed_20261005/cuda_fp32/W_V/S_reverse_cuda": "59426aa6ce2334eb9dbe14a73105e9a4f38ad01e9a2d70c90a9e43f1202d37d2",
    "seed_20261005/cuda_fp32/W_V/S_stack_cuda": "65e7551e9e4a97b0601620509c2959f82fb2153b5a28d7d8649152f9c44e77cc",
    "seed_20261005/cuda_fp32/W_V/gR64_primary_cuda": "af57cda6273f6d1baf1fc8baf3e3e186dcbdf7d5aa35420b8f6d980d75aed91e",
    "seed_20261005/cuda_fp32/W_V/gR_cuda_fp32": "e71b9e6c6590fa5ede32ba1a5fcf27a38d8a83ff29bb5d588666c6b60174a19f",
    "seed_20261005/cuda_fp32/W_V/gU0_cuda_fp32": "e8a5db6cdf22676182c7a0ebd2143728f26a986d912ae804736544812c57bbf8",
    "seed_20261005/cuda_fp32/W_V/gU1_cuda_fp32": "65557d1fa6179126d6e3ac635a5449368e9972ef592534d05f0aea28e3ca2192",
    "seed_20261005/cuda_fp32/W_V/gU2_cuda_fp32": "2bbefe6031638daae552622d0de2ab3bb95ca94dbf9085da8402f9b79ed9ff63",
    "seed_20261005/cuda_fp32/W_V/gU3_cuda_fp32": "7fca13bc1f45384bfe82cda49720a68a4e202c46a847e1b6dffdf3a560a92282",
    "seed_20261005/cuda_fp32/W_down/S64_primary_cuda": "840b96091abdc8ea7412ed2f3688203bb6fbfb7f6bcfe9d63798a82861bd62e9",
    "seed_20261005/cuda_fp32/W_down/S_forward_cuda": "5016bfb737a5fc3324650cf01a125bf8d535805608c31d431257b2d84042c3a0",
    "seed_20261005/cuda_fp32/W_down/S_pairwise_cuda": "e47637b6e107596263955f8c23379f7cee92f4353992231019ac70a02890c78b",
    "seed_20261005/cuda_fp32/W_down/S_reverse_cuda": "3c34ffcf0cd41ff8608c332d181b021e711858e175699798e3a45681d4300b8a",
    "seed_20261005/cuda_fp32/W_down/S_stack_cuda": "593c0dc6bb40edf91f0096038cc4eb7b42073cdc13b83cf1b4133d3e7ec698fa",
    "seed_20261005/cuda_fp32/W_down/gR64_primary_cuda": "b33fdcd958dfd04c61f9e7f0b201d0ae7a467e6367a8aa6741def8f02d777fe1",
    "seed_20261005/cuda_fp32/W_down/gR_cuda_fp32": "a09541890d72d58b422e7a8b514a4c5f9e1cde8150b422d87f1bfacd770450df",
    "seed_20261005/cuda_fp32/W_down/gU0_cuda_fp32": "d956309f02c98dfa439e6289d0827841e8552f941ab67b9dd3cfdfbd3bb65ec9",
    "seed_20261005/cuda_fp32/W_down/gU1_cuda_fp32": "4818d7d3d13cd81d51c4ac14ebad552c73556d6c8e518d09cec558f8c6fa2162",
    "seed_20261005/cuda_fp32/W_down/gU2_cuda_fp32": "11e365cb862b8afb989d0389f817e6d0160aba671b85a9937a0d1c5408d0b0ed",
    "seed_20261005/cuda_fp32/W_down/gU3_cuda_fp32": "73e99e5078147eaed9d41aaa75155dafcdc472535a3dd2615f58d8c20b449b35",
    "seed_20261005/cuda_fp32/W_gate/S64_primary_cuda": "14c600f7068098638de281e75bb1a57119da452d9cfbdd864195d61ef1355825",
    "seed_20261005/cuda_fp32/W_gate/S_forward_cuda": "8de97aa941cab65a386a120fceaf14cf47a21dfcf16e3860cbff7be6a357809b",
    "seed_20261005/cuda_fp32/W_gate/S_pairwise_cuda": "a82737b41f8768756f29e812aea321ef3ccedb14e5f5f70481862ab1037892fe",
    "seed_20261005/cuda_fp32/W_gate/S_reverse_cuda": "54d42a2cf8b00cb28fed3cfb000dba2d3e78be1980c23f852b3cb08de1287fb7",
    "seed_20261005/cuda_fp32/W_gate/S_stack_cuda": "4ad4dd7a166bcec9c8fcc44cac128f20689c677c446f982f4eaa4611c7465ff9",
    "seed_20261005/cuda_fp32/W_gate/gR64_primary_cuda": "319bab65e54bd6d1930001daaae65f2b70ef7735e086ca41a740f78689b415e4",
    "seed_20261005/cuda_fp32/W_gate/gR_cuda_fp32": "01ea4836c942a7fbb0cd33d6a29d73fe27019fc1faabe8eb954588fa062f8fac",
    "seed_20261005/cuda_fp32/W_gate/gU0_cuda_fp32": "ceaeff9b20429bcdd5fb77eae2e143c1aab7030e5b916375793f9bbc171b7dd0",
    "seed_20261005/cuda_fp32/W_gate/gU1_cuda_fp32": "1a2fa82e055e032434d704ae5c859e3f458d7dd9d3df44c532ef4cb4985998b1",
    "seed_20261005/cuda_fp32/W_gate/gU2_cuda_fp32": "b24213354675edbaae27d562fb6626b29e52cfebb114d5483386f1b47680757d",
    "seed_20261005/cuda_fp32/W_gate/gU3_cuda_fp32": "84fb8c4cd9a4843fdebbab9a7860f75e7ea54c8c247852fce0a21080b1c99fc1",
    "seed_20261005/cuda_fp32/W_up/S64_primary_cuda": "5535ed2296b66c294c2d656e33ab708702175cd90eb6d50a72ef724951f5f64e",
    "seed_20261005/cuda_fp32/W_up/S_forward_cuda": "96e33e1823892f8de29f65a7572a3387f33ace999721f9db67524e12ba24b0a8",
    "seed_20261005/cuda_fp32/W_up/S_pairwise_cuda": "6455ca5a40fda3bec80e4c5813c1fcf7342910fba53a7add5bcdf4797ec0c5df",
    "seed_20261005/cuda_fp32/W_up/S_reverse_cuda": "55a16ebbfb60e3f2eb251965d6120317a047f3fb046a7389dbc29e922358dc84",
    "seed_20261005/cuda_fp32/W_up/S_stack_cuda": "aca07827280f237c2075ef5c3f522fae04db0ea8d323aa802c1e9ec0ed24de35",
    "seed_20261005/cuda_fp32/W_up/gR64_primary_cuda": "b92720b6f78990d4e351f2249dea279020acf1b425b99d204a809f90d2280fbe",
    "seed_20261005/cuda_fp32/W_up/gR_cuda_fp32": "667fd6a8d37547773d19d64b18f0b47f4954bde4378a8c2bdeaaeaeee0b56d1c",
    "seed_20261005/cuda_fp32/W_up/gU0_cuda_fp32": "0eaf50ff1efc23b68d2b0f3c3f8636518186ddefddd681cbed81fe1940d35527",
    "seed_20261005/cuda_fp32/W_up/gU1_cuda_fp32": "318ef6dfdf1f9408d5cbb85366260dd4b01d1cba6b1d4cbe0dde406f04f8e5d2",
    "seed_20261005/cuda_fp32/W_up/gU2_cuda_fp32": "884e4bb70e667d3fb98c3760bd8b9dc28579bde5f6c007095268d1ba3c8b9fd9",
    "seed_20261005/cuda_fp32/W_up/gU3_cuda_fp32": "c412b1a22f2078a982f8a96d8af8224dfafe36e2d19c32e27d3cf0f5acfbab77",
    "seed_20261005/inputs/w_fp32_cpu": "39d940993e326f835614af3a91f140f28fdecbbd80f29a36335b02ad74aed388",
    "seed_20261005/inputs/x_fp32_cpu": "0a0a2531f280490a82375cbe6db3cc03958fcde1fcfe904e80e2d2451611bf11",
    "seed_20261005/structural_traces/R4/round_1": "a523c8f1366828fa696789f1baf7cbb577c4d238016308c7d42896b3bb5bc72a",
    "seed_20261005/structural_traces/R4/round_2": "5eb0f76278d97d870190de2dcbe381cc862902f1d98c5acabdecf04e8466f2d5",
    "seed_20261005/structural_traces/R4/round_3": "538643dcca79879d634a0b7d9bd645f6fe01f5e2c51a3089b4b893443f8d5032",
    "seed_20261005/structural_traces/R4/round_4": "02f3c597c1303168567f778681988b8b4cac5ccd922f9d0b24223353dcf1225e",
    "seed_20261005/structural_traces/R4_final": "69b34fa69680ebbb32fd8c910f115e7b10c3855937844cad1a3d7aabb13b0883",
    "seed_20261005/structural_traces/U4/round_1": "2ad90b8f3886e71d85b6247734ace26725972af8282c24b2879fd72f37da5815",
    "seed_20261005/structural_traces/U4/round_2": "a3342356b1ec3e07badde9349f2625e502173315feb609c086610fc109e8954c",
    "seed_20261005/structural_traces/U4/round_3": "6df7a459d4f2b601956e2efbde652b8725ca8aaa627e829dbea935530c6e18fc",
    "seed_20261005/structural_traces/U4/round_4": "014accc643c94622cc0670c2859e640dafc3301706269def9444d5262553e216",
    "seed_20261005/structural_traces/U4_final": "a9d2805d7771000b2825b12aeb1a1c8d36b8825e2fe37bb7bf2df9c01cc77970"
  }
}
```

## Aggregation rule

All five held-out seeds and all seven families must pass. No majority, averaging, outlier exclusion, or per-seed retry is used.
S_reverse equality and old max_rel remain NON-GATE diagnostics.

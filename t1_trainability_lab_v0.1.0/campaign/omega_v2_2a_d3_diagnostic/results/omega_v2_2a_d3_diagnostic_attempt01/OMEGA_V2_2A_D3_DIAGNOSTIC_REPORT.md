# OMEGA-V2-2A-D3-DIAGNOSTIC

- classification: `CALIBRATION_DIAGNOSTIC_ONLY`
- may_rescue_V2_2A: `False`
- architectural_verdict: `None`
- V2-2A-r1 terminal: `OMEGA_V2_2A_LOCAL_PREFLIGHT_FAIL`
- V2-2A-r1 modified: `False`
- held-out D3Q seeds touched: `[]`

## CUDA D3 run reproducibility

- run 01/02 bitwise reproducible: `True`

## Run 1 and run 2 family metrics

| Run | Family | Sum variant | E_L2 | E_inf | max_abs | max_rel_old |
|---|---|---|---:|---:|---:|---:|
| run_01 | W_Q | S_forward | 5.37215357e-08 | 1.45208134e-07 | 7.62939453e-06 | 0.00036101084 |
| run_01 | W_Q | S_reverse | 0 | 0 | 0 | 0 |
| run_01 | W_Q | S_pairwise | 4.39704095e-08 | 7.26040739e-08 | 3.81469727e-06 | 0.00036101084 |
| run_01 | W_Q | S_stack | 5.37215357e-08 | 1.45208134e-07 | 7.62939453e-06 | 0.00036101084 |
| run_01 | W_Q | S_fp64 | 3.41579447e-08 | 6.35285612e-08 | 3.33786011e-06 | 0.000110006967 |
| run_01 | W_K | S_forward | 5.37046638e-08 | 1.45851914e-07 | 7.62939453e-06 | 0.00347222225 |
| run_01 | W_K | S_reverse | 0 | 0 | 0 | 0 |
| run_01 | W_K | S_pairwise | 4.41888375e-08 | 7.29259568e-08 | 3.81469727e-06 | 0.000228832956 |
| run_01 | W_K | S_stack | 5.37046638e-08 | 1.45851914e-07 | 7.62939453e-06 | 0.00347222225 |
| run_01 | W_K | S_fp64 | 3.38817531e-08 | 5.92523426e-08 | 3.09944153e-06 | 0.00347222222 |
| run_01 | W_V | S_forward | 5.40485061e-08 | 1.18906584e-07 | 1.52587891e-05 | 0.000232396007 |
| run_01 | W_V | S_reverse | 0 | 0 | 0 | 0 |
| run_01 | W_V | S_pairwise | 4.3861661e-08 | 1.18906584e-07 | 1.52587891e-05 | 8.73973113e-05 |
| run_01 | W_V | S_stack | 5.40485061e-08 | 1.18906584e-07 | 1.52587891e-05 | 0.000232396007 |
| run_01 | W_V | S_fp64 | 3.74718008e-08 | 6.68849554e-08 | 8.58306885e-06 | 0.000116211505 |
| run_01 | W_O | S_forward | 5.36977431e-08 | 1.22206004e-07 | 1.52587891e-05 | 0.000121300342 |
| run_01 | W_O | S_reverse | 0 | 0 | 0 | 0 |
| run_01 | W_O | S_pairwise | 4.32279883e-08 | 6.11029947e-08 | 7.62939453e-06 | 5.58222637e-05 |
| run_01 | W_O | S_stack | 5.36977431e-08 | 1.22206004e-07 | 1.52587891e-05 | 0.000121300342 |
| run_01 | W_O | S_fp64 | 3.72140347e-08 | 6.11029983e-08 | 7.62939453e-06 | 0.00012130034 |
| run_01 | W_gate | S_forward | 5.35648539e-08 | 1.37467708e-07 | 7.62939453e-06 | 0.00598802418 |
| run_01 | W_gate | S_reverse | 0 | 0 | 0 | 0 |
| run_01 | W_gate | S_pairwise | 4.33600427e-08 | 6.87338542e-08 | 3.81469727e-06 | 0.000990099041 |
| run_01 | W_gate | S_stack | 5.35648539e-08 | 1.37467708e-07 | 7.62939453e-06 | 0.00598802418 |
| run_01 | W_gate | S_fp64 | 3.67111147e-08 | 6.8733851e-08 | 3.81469727e-06 | 0.0015037594 |
| run_01 | W_up | S_forward | 5.34688098e-08 | 1.0001942e-07 | 7.62939453e-06 | 0.000426985469 |
| run_01 | W_up | S_reverse | 0 | 0 | 0 | 0 |
| run_01 | W_up | S_pairwise | 4.33507701e-08 | 5.00097102e-08 | 3.81469727e-06 | 0.000426985469 |
| run_01 | W_up | S_stack | 5.34688098e-08 | 1.0001942e-07 | 7.62939453e-06 | 0.000426985469 |
| run_01 | W_up | S_fp64 | 3.6650102e-08 | 5.00097113e-08 | 3.81469727e-06 | 0.000426985482 |
| run_01 | W_down | S_forward | 5.31621005e-08 | 1.25792937e-07 | 7.62939453e-06 | 0.00299401209 |
| run_01 | W_down | S_reverse | 0 | 0 | 0 | 0 |
| run_01 | W_down | S_pairwise | 4.30283897e-08 | 6.28964685e-08 | 3.81469727e-06 | 0.000111346177 |
| run_01 | W_down | S_stack | 5.31621005e-08 | 1.25792937e-07 | 7.62939453e-06 | 0.00299401209 |
| run_01 | W_down | S_fp64 | 3.65633338e-08 | 6.28964699e-08 | 3.81469727e-06 | 0.00287705838 |
| run_02 | W_Q | S_forward | 5.37215357e-08 | 1.45208134e-07 | 7.62939453e-06 | 0.00036101084 |
| run_02 | W_Q | S_reverse | 0 | 0 | 0 | 0 |
| run_02 | W_Q | S_pairwise | 4.39704095e-08 | 7.26040739e-08 | 3.81469727e-06 | 0.00036101084 |
| run_02 | W_Q | S_stack | 5.37215357e-08 | 1.45208134e-07 | 7.62939453e-06 | 0.00036101084 |
| run_02 | W_Q | S_fp64 | 3.41579447e-08 | 6.35285612e-08 | 3.33786011e-06 | 0.000110006967 |
| run_02 | W_K | S_forward | 5.37046638e-08 | 1.45851914e-07 | 7.62939453e-06 | 0.00347222225 |
| run_02 | W_K | S_reverse | 0 | 0 | 0 | 0 |
| run_02 | W_K | S_pairwise | 4.41888375e-08 | 7.29259568e-08 | 3.81469727e-06 | 0.000228832956 |
| run_02 | W_K | S_stack | 5.37046638e-08 | 1.45851914e-07 | 7.62939453e-06 | 0.00347222225 |
| run_02 | W_K | S_fp64 | 3.38817531e-08 | 5.92523426e-08 | 3.09944153e-06 | 0.00347222222 |
| run_02 | W_V | S_forward | 5.40485061e-08 | 1.18906584e-07 | 1.52587891e-05 | 0.000232396007 |
| run_02 | W_V | S_reverse | 0 | 0 | 0 | 0 |
| run_02 | W_V | S_pairwise | 4.3861661e-08 | 1.18906584e-07 | 1.52587891e-05 | 8.73973113e-05 |
| run_02 | W_V | S_stack | 5.40485061e-08 | 1.18906584e-07 | 1.52587891e-05 | 0.000232396007 |
| run_02 | W_V | S_fp64 | 3.74718008e-08 | 6.68849554e-08 | 8.58306885e-06 | 0.000116211505 |
| run_02 | W_O | S_forward | 5.36977431e-08 | 1.22206004e-07 | 1.52587891e-05 | 0.000121300342 |
| run_02 | W_O | S_reverse | 0 | 0 | 0 | 0 |
| run_02 | W_O | S_pairwise | 4.32279883e-08 | 6.11029947e-08 | 7.62939453e-06 | 5.58222637e-05 |
| run_02 | W_O | S_stack | 5.36977431e-08 | 1.22206004e-07 | 1.52587891e-05 | 0.000121300342 |
| run_02 | W_O | S_fp64 | 3.72140347e-08 | 6.11029983e-08 | 7.62939453e-06 | 0.00012130034 |
| run_02 | W_gate | S_forward | 5.35648539e-08 | 1.37467708e-07 | 7.62939453e-06 | 0.00598802418 |
| run_02 | W_gate | S_reverse | 0 | 0 | 0 | 0 |
| run_02 | W_gate | S_pairwise | 4.33600427e-08 | 6.87338542e-08 | 3.81469727e-06 | 0.000990099041 |
| run_02 | W_gate | S_stack | 5.35648539e-08 | 1.37467708e-07 | 7.62939453e-06 | 0.00598802418 |
| run_02 | W_gate | S_fp64 | 3.67111147e-08 | 6.8733851e-08 | 3.81469727e-06 | 0.0015037594 |
| run_02 | W_up | S_forward | 5.34688098e-08 | 1.0001942e-07 | 7.62939453e-06 | 0.000426985469 |
| run_02 | W_up | S_reverse | 0 | 0 | 0 | 0 |
| run_02 | W_up | S_pairwise | 4.33507701e-08 | 5.00097102e-08 | 3.81469727e-06 | 0.000426985469 |
| run_02 | W_up | S_stack | 5.34688098e-08 | 1.0001942e-07 | 7.62939453e-06 | 0.000426985469 |
| run_02 | W_up | S_fp64 | 3.6650102e-08 | 5.00097113e-08 | 3.81469727e-06 | 0.000426985482 |
| run_02 | W_down | S_forward | 5.31621005e-08 | 1.25792937e-07 | 7.62939453e-06 | 0.00299401209 |
| run_02 | W_down | S_reverse | 0 | 0 | 0 | 0 |
| run_02 | W_down | S_pairwise | 4.30283897e-08 | 6.28964685e-08 | 3.81469727e-06 | 0.000111346177 |
| run_02 | W_down | S_stack | 5.31621005e-08 | 1.25792937e-07 | 7.62939453e-06 | 0.00299401209 |
| run_02 | W_down | S_fp64 | 3.65633338e-08 | 6.28964699e-08 | 3.81469727e-06 | 0.00287705838 |

## Reproducibility and persisted tensor hashes

```json
{
  "reproducibility": {
    "all_bitwise_equal": true,
    "rows": [
      {
        "family": "W_Q",
        "tensor": "gR",
        "torch_equal": true
      },
      {
        "family": "W_Q",
        "tensor": "gU0",
        "torch_equal": true
      },
      {
        "family": "W_Q",
        "tensor": "gU1",
        "torch_equal": true
      },
      {
        "family": "W_Q",
        "tensor": "gU2",
        "torch_equal": true
      },
      {
        "family": "W_Q",
        "tensor": "gU3",
        "torch_equal": true
      },
      {
        "family": "W_K",
        "tensor": "gR",
        "torch_equal": true
      },
      {
        "family": "W_K",
        "tensor": "gU0",
        "torch_equal": true
      },
      {
        "family": "W_K",
        "tensor": "gU1",
        "torch_equal": true
      },
      {
        "family": "W_K",
        "tensor": "gU2",
        "torch_equal": true
      },
      {
        "family": "W_K",
        "tensor": "gU3",
        "torch_equal": true
      },
      {
        "family": "W_V",
        "tensor": "gR",
        "torch_equal": true
      },
      {
        "family": "W_V",
        "tensor": "gU0",
        "torch_equal": true
      },
      {
        "family": "W_V",
        "tensor": "gU1",
        "torch_equal": true
      },
      {
        "family": "W_V",
        "tensor": "gU2",
        "torch_equal": true
      },
      {
        "family": "W_V",
        "tensor": "gU3",
        "torch_equal": true
      },
      {
        "family": "W_O",
        "tensor": "gR",
        "torch_equal": true
      },
      {
        "family": "W_O",
        "tensor": "gU0",
        "torch_equal": true
      },
      {
        "family": "W_O",
        "tensor": "gU1",
        "torch_equal": true
      },
      {
        "family": "W_O",
        "tensor": "gU2",
        "torch_equal": true
      },
      {
        "family": "W_O",
        "tensor": "gU3",
        "torch_equal": true
      },
      {
        "family": "W_gate",
        "tensor": "gR",
        "torch_equal": true
      },
      {
        "family": "W_gate",
        "tensor": "gU0",
        "torch_equal": true
      },
      {
        "family": "W_gate",
        "tensor": "gU1",
        "torch_equal": true
      },
      {
        "family": "W_gate",
        "tensor": "gU2",
        "torch_equal": true
      },
      {
        "family": "W_gate",
        "tensor": "gU3",
        "torch_equal": true
      },
      {
        "family": "W_up",
        "tensor": "gR",
        "torch_equal": true
      },
      {
        "family": "W_up",
        "tensor": "gU0",
        "torch_equal": true
      },
      {
        "family": "W_up",
        "tensor": "gU1",
        "torch_equal": true
      },
      {
        "family": "W_up",
        "tensor": "gU2",
        "torch_equal": true
      },
      {
        "family": "W_up",
        "tensor": "gU3",
        "torch_equal": true
      },
      {
        "family": "W_down",
        "tensor": "gR",
        "torch_equal": true
      },
      {
        "family": "W_down",
        "tensor": "gU0",
        "torch_equal": true
      },
      {
        "family": "W_down",
        "tensor": "gU1",
        "torch_equal": true
      },
      {
        "family": "W_down",
        "tensor": "gU2",
        "torch_equal": true
      },
      {
        "family": "W_down",
        "tensor": "gU3",
        "torch_equal": true
      },
      {
        "family": "initial_weights",
        "tensor": "V2_0_state_dict_sha256",
        "torch_equal": true
      },
      {
        "family": "initial_weights",
        "tensor": "R4_U4_initial_clone",
        "torch_equal": true
      },
      {
        "family": "initial_state",
        "tensor": "run_02_compared_to_run_01_before_forward",
        "torch_equal": true
      },
      {
        "family": "fixed_inputs",
        "tensor": "x",
        "torch_equal": true
      },
      {
        "family": "fixed_inputs",
        "tensor": "w",
        "torch_equal": true
      },
      {
        "family": "initial_copy",
        "tensor": "run_01",
        "torch_equal": true
      },
      {
        "family": "initial_copy",
        "tensor": "run_02",
        "torch_equal": true
      }
    ],
    "schema": "omega-v2-2a-d3-reproducibility-v1"
  },
  "tensor_bundles": {
    "cpu_fp64": {
      "file_sha256": "e3131de5eae7d39066333e4ed4dbd25017c2f2815225657c3aa8925c4b464f14",
      "path": "cpu_fp64_oracle.pt",
      "size_bytes": 50607151,
      "tensor_metadata": {
        "cpu_fp64/W_K/S_fp64_cpu": {
          "dtype": "torch.float64",
          "raw_sha256": "1004baaf63653a35071f48df5af2c1b5eb48b3f23380dc9d4cf5678768b8157e",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_K/gR64": {
          "dtype": "torch.float64",
          "raw_sha256": "ede71a75d46c540854daff0ff21915601ec1f47c91819db17bd0ba61e54fd5a1",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_K/gU0_64": {
          "dtype": "torch.float64",
          "raw_sha256": "254360e19304a9349b2b65168b0293a3678516dce41cd5de7f63e1a89f530a44",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_K/gU1_64": {
          "dtype": "torch.float64",
          "raw_sha256": "41014c048547b0e8ab7944c16f93fbbe498480cd441cf11cd2bf4a54ed7f0772",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_K/gU2_64": {
          "dtype": "torch.float64",
          "raw_sha256": "7eb0b5c55a8ae41b9140f206e403154dc27c58a72c79afbbfec44ac45fe41297",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_K/gU3_64": {
          "dtype": "torch.float64",
          "raw_sha256": "bf84db38b4b829e37d7a909876410924c982acfc58c035dd89ccf39d2c1e4a9e",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_O/S_fp64_cpu": {
          "dtype": "torch.float64",
          "raw_sha256": "18969f37f44e04a21054344c32405e39a7114ba4a5b9b0a50ffbfac8ced876bc",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_O/gR64": {
          "dtype": "torch.float64",
          "raw_sha256": "7a00ac8f1e2f7da99fc221632c4f518861dc3f2141b338b8c563265837f25d6d",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_O/gU0_64": {
          "dtype": "torch.float64",
          "raw_sha256": "a7f4bdf785779aa2acb2667bd23f8bac827a7ada130bb9ee41c708ad8428e177",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_O/gU1_64": {
          "dtype": "torch.float64",
          "raw_sha256": "6e6874d1e6cc46d6404ff4967c417a483fdb9793517833fc36345a211dba07ab",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_O/gU2_64": {
          "dtype": "torch.float64",
          "raw_sha256": "71ae46050312b27e0b08be97f1c65aa742d27678b57e4bcb7a73079fd05e0b29",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_O/gU3_64": {
          "dtype": "torch.float64",
          "raw_sha256": "b0aab849bb5cf27945e0716acb7e75a4afb6ade9b832ffcbde45f8f7bf56f96e",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_Q/S_fp64_cpu": {
          "dtype": "torch.float64",
          "raw_sha256": "7576b2c0ad842182852bae9ce248f0363d983a34ee92e5b92e4b6a98a1964384",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_Q/gR64": {
          "dtype": "torch.float64",
          "raw_sha256": "38f5165242346319770ad711eef952404cc31d569da0ba1bbb85b50c2db79a4b",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_Q/gU0_64": {
          "dtype": "torch.float64",
          "raw_sha256": "595ee9f54135a896c704e323576f5cc4edf5ec2a600556ce37229d307fa64bd8",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_Q/gU1_64": {
          "dtype": "torch.float64",
          "raw_sha256": "fc44ff2a8d367abc692014ff6e38d8fd5566ff1bebac4138d730f8b538ec6c7c",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_Q/gU2_64": {
          "dtype": "torch.float64",
          "raw_sha256": "66d9fb489b796cc4005d0f160a0fddea1250913afbadc69f7b7cb70c9c546bda",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_Q/gU3_64": {
          "dtype": "torch.float64",
          "raw_sha256": "ecbf1b08708ecb073a954f4075cfc223215391c6b90e64715f89a469fa05be98",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_V/S_fp64_cpu": {
          "dtype": "torch.float64",
          "raw_sha256": "0b12c710d1c14c40a260ea2a3d0eec9645e9a21142a68a4aaa01d5fb52d9a868",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_V/gR64": {
          "dtype": "torch.float64",
          "raw_sha256": "de07ba7a86c652637274698690854e09623d30869cb2c0a2bd2cb25fc04b2197",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_V/gU0_64": {
          "dtype": "torch.float64",
          "raw_sha256": "d144271deff95cbc5540e2150c62b6a233e3e0a72295e5a07d990a092994938b",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_V/gU1_64": {
          "dtype": "torch.float64",
          "raw_sha256": "c23bd84e4afd0e593251fecddf83b2393679eaf1d13e76c6deacd8aa43cb32d9",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_V/gU2_64": {
          "dtype": "torch.float64",
          "raw_sha256": "3559a1d8d087dd53272f1c91f72a1dc86456f38be0b6391cfe98cddff4a1d05b",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_V/gU3_64": {
          "dtype": "torch.float64",
          "raw_sha256": "c663aa52158fdd43cdd698df00e8683e0cd84b78ce2fc0971c029588d53bf788",
          "shape": [
            256,
            256
          ]
        },
        "cpu_fp64/W_down/S_fp64_cpu": {
          "dtype": "torch.float64",
          "raw_sha256": "f1f3442f6009b3460443f4a3b3ac21e237a778769c9bcc9062af77c6f2eb1add",
          "shape": [
            256,
            1024
          ]
        },
        "cpu_fp64/W_down/gR64": {
          "dtype": "torch.float64",
          "raw_sha256": "6cdfc25107f705fa23bd668edea1d41b27763d93b9b1d1004f542e8dc273d641",
          "shape": [
            256,
            1024
          ]
        },
        "cpu_fp64/W_down/gU0_64": {
          "dtype": "torch.float64",
          "raw_sha256": "1112319df64e6d88db88cf8cf56cbbff9183eb994ff0a1814b19fec9a77d46d9",
          "shape": [
            256,
            1024
          ]
        },
        "cpu_fp64/W_down/gU1_64": {
          "dtype": "torch.float64",
          "raw_sha256": "387f037d4048b1040cc95449cb7f48e1c3a9cfacfc5af1ad2b4e2c1544e4b0ac",
          "shape": [
            256,
            1024
          ]
        },
        "cpu_fp64/W_down/gU2_64": {
          "dtype": "torch.float64",
          "raw_sha256": "d51b47796c98723cee3b770d5d6b48a1b0b97540bde3e5e61425d412220f70c2",
          "shape": [
            256,
            1024
          ]
        },
        "cpu_fp64/W_down/gU3_64": {
          "dtype": "torch.float64",
          "raw_sha256": "db152906197ba14c43c2501a01073bceefd644b034abe27d279d4b2eaab62ce2",
          "shape": [
            256,
            1024
          ]
        },
        "cpu_fp64/W_gate/S_fp64_cpu": {
          "dtype": "torch.float64",
          "raw_sha256": "d6b2fb796c4312f0c6d9120f5b0bdf5c8f517ea09be8cd49270ffd08e3bee216",
          "shape": [
            1024,
            256
          ]
        },
        "cpu_fp64/W_gate/gR64": {
          "dtype": "torch.float64",
          "raw_sha256": "25a7004442ccffa50f9e4e242c7788262cf19d1d43466daf98ce884156d03a04",
          "shape": [
            1024,
            256
          ]
        },
        "cpu_fp64/W_gate/gU0_64": {
          "dtype": "torch.float64",
          "raw_sha256": "7ff521f56ba90c5c9c8b549edee1a6e195d4e1b24e27c0e1c14388c89d0ca4ed",
          "shape": [
            1024,
            256
          ]
        },
        "cpu_fp64/W_gate/gU1_64": {
          "dtype": "torch.float64",
          "raw_sha256": "028a4e1231d3c0bb65cac1e2273fc4b665b4f188a8950c2546560bcda9c9cea4",
          "shape": [
            1024,
            256
          ]
        },
        "cpu_fp64/W_gate/gU2_64": {
          "dtype": "torch.float64",
          "raw_sha256": "0e4c97ebfcb0aa721cbb03edb2fa2e72a8e3382c9a0697471e1d580299b8b1ba",
          "shape": [
            1024,
            256
          ]
        },
        "cpu_fp64/W_gate/gU3_64": {
          "dtype": "torch.float64",
          "raw_sha256": "062147be02adcdbb4825dcf24eef061153a6dbfbe82bdaabfa15342640af19cc",
          "shape": [
            1024,
            256
          ]
        },
        "cpu_fp64/W_up/S_fp64_cpu": {
          "dtype": "torch.float64",
          "raw_sha256": "aa5dfa94c514c3561abc40386f4b633dae2321e6c0d94415d0a9fdeb6ac4faa3",
          "shape": [
            1024,
            256
          ]
        },
        "cpu_fp64/W_up/gR64": {
          "dtype": "torch.float64",
          "raw_sha256": "4999700bb9e984ba16bd40cebba03b01c7d68aa7dbc46a678590c2f21c77724a",
          "shape": [
            1024,
            256
          ]
        },
        "cpu_fp64/W_up/gU0_64": {
          "dtype": "torch.float64",
          "raw_sha256": "3b18b8af71ef68b16ebf62700fd06a58d42778dd39867ada90736877487ab063",
          "shape": [
            1024,
            256
          ]
        },
        "cpu_fp64/W_up/gU1_64": {
          "dtype": "torch.float64",
          "raw_sha256": "e1d2b8e173a6a0a617d3e7f64586baae6868ff25f0eb0a08b5c741fa4d3c869e",
          "shape": [
            1024,
            256
          ]
        },
        "cpu_fp64/W_up/gU2_64": {
          "dtype": "torch.float64",
          "raw_sha256": "0fb354473586eeacd5645f7340428373153e79cfd8acf737abda047c38e1a658",
          "shape": [
            1024,
            256
          ]
        },
        "cpu_fp64/W_up/gU3_64": {
          "dtype": "torch.float64",
          "raw_sha256": "efcd7d7828a9c483cbfdbd624aef0104d6cf61790acdba097b66a7262fce1256",
          "shape": [
            1024,
            256
          ]
        },
        "cpu_fp64/__inputs__/w_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "3985fa09f1433dbfbee4d01890930e0a16f021cc83451b83210d47187bb73ef8",
          "shape": [
            8,
            8,
            256
          ]
        },
        "cpu_fp64/__inputs__/x_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "f20da260041136f6c914b5bd092b5f5718dd230f81277f6a1198fb33581dc81e",
          "shape": [
            8,
            8,
            256
          ]
        }
      },
      "tensor_raw_sha256": {
        "cpu_fp64/W_K/S_fp64_cpu": "1004baaf63653a35071f48df5af2c1b5eb48b3f23380dc9d4cf5678768b8157e",
        "cpu_fp64/W_K/gR64": "ede71a75d46c540854daff0ff21915601ec1f47c91819db17bd0ba61e54fd5a1",
        "cpu_fp64/W_K/gU0_64": "254360e19304a9349b2b65168b0293a3678516dce41cd5de7f63e1a89f530a44",
        "cpu_fp64/W_K/gU1_64": "41014c048547b0e8ab7944c16f93fbbe498480cd441cf11cd2bf4a54ed7f0772",
        "cpu_fp64/W_K/gU2_64": "7eb0b5c55a8ae41b9140f206e403154dc27c58a72c79afbbfec44ac45fe41297",
        "cpu_fp64/W_K/gU3_64": "bf84db38b4b829e37d7a909876410924c982acfc58c035dd89ccf39d2c1e4a9e",
        "cpu_fp64/W_O/S_fp64_cpu": "18969f37f44e04a21054344c32405e39a7114ba4a5b9b0a50ffbfac8ced876bc",
        "cpu_fp64/W_O/gR64": "7a00ac8f1e2f7da99fc221632c4f518861dc3f2141b338b8c563265837f25d6d",
        "cpu_fp64/W_O/gU0_64": "a7f4bdf785779aa2acb2667bd23f8bac827a7ada130bb9ee41c708ad8428e177",
        "cpu_fp64/W_O/gU1_64": "6e6874d1e6cc46d6404ff4967c417a483fdb9793517833fc36345a211dba07ab",
        "cpu_fp64/W_O/gU2_64": "71ae46050312b27e0b08be97f1c65aa742d27678b57e4bcb7a73079fd05e0b29",
        "cpu_fp64/W_O/gU3_64": "b0aab849bb5cf27945e0716acb7e75a4afb6ade9b832ffcbde45f8f7bf56f96e",
        "cpu_fp64/W_Q/S_fp64_cpu": "7576b2c0ad842182852bae9ce248f0363d983a34ee92e5b92e4b6a98a1964384",
        "cpu_fp64/W_Q/gR64": "38f5165242346319770ad711eef952404cc31d569da0ba1bbb85b50c2db79a4b",
        "cpu_fp64/W_Q/gU0_64": "595ee9f54135a896c704e323576f5cc4edf5ec2a600556ce37229d307fa64bd8",
        "cpu_fp64/W_Q/gU1_64": "fc44ff2a8d367abc692014ff6e38d8fd5566ff1bebac4138d730f8b538ec6c7c",
        "cpu_fp64/W_Q/gU2_64": "66d9fb489b796cc4005d0f160a0fddea1250913afbadc69f7b7cb70c9c546bda",
        "cpu_fp64/W_Q/gU3_64": "ecbf1b08708ecb073a954f4075cfc223215391c6b90e64715f89a469fa05be98",
        "cpu_fp64/W_V/S_fp64_cpu": "0b12c710d1c14c40a260ea2a3d0eec9645e9a21142a68a4aaa01d5fb52d9a868",
        "cpu_fp64/W_V/gR64": "de07ba7a86c652637274698690854e09623d30869cb2c0a2bd2cb25fc04b2197",
        "cpu_fp64/W_V/gU0_64": "d144271deff95cbc5540e2150c62b6a233e3e0a72295e5a07d990a092994938b",
        "cpu_fp64/W_V/gU1_64": "c23bd84e4afd0e593251fecddf83b2393679eaf1d13e76c6deacd8aa43cb32d9",
        "cpu_fp64/W_V/gU2_64": "3559a1d8d087dd53272f1c91f72a1dc86456f38be0b6391cfe98cddff4a1d05b",
        "cpu_fp64/W_V/gU3_64": "c663aa52158fdd43cdd698df00e8683e0cd84b78ce2fc0971c029588d53bf788",
        "cpu_fp64/W_down/S_fp64_cpu": "f1f3442f6009b3460443f4a3b3ac21e237a778769c9bcc9062af77c6f2eb1add",
        "cpu_fp64/W_down/gR64": "6cdfc25107f705fa23bd668edea1d41b27763d93b9b1d1004f542e8dc273d641",
        "cpu_fp64/W_down/gU0_64": "1112319df64e6d88db88cf8cf56cbbff9183eb994ff0a1814b19fec9a77d46d9",
        "cpu_fp64/W_down/gU1_64": "387f037d4048b1040cc95449cb7f48e1c3a9cfacfc5af1ad2b4e2c1544e4b0ac",
        "cpu_fp64/W_down/gU2_64": "d51b47796c98723cee3b770d5d6b48a1b0b97540bde3e5e61425d412220f70c2",
        "cpu_fp64/W_down/gU3_64": "db152906197ba14c43c2501a01073bceefd644b034abe27d279d4b2eaab62ce2",
        "cpu_fp64/W_gate/S_fp64_cpu": "d6b2fb796c4312f0c6d9120f5b0bdf5c8f517ea09be8cd49270ffd08e3bee216",
        "cpu_fp64/W_gate/gR64": "25a7004442ccffa50f9e4e242c7788262cf19d1d43466daf98ce884156d03a04",
        "cpu_fp64/W_gate/gU0_64": "7ff521f56ba90c5c9c8b549edee1a6e195d4e1b24e27c0e1c14388c89d0ca4ed",
        "cpu_fp64/W_gate/gU1_64": "028a4e1231d3c0bb65cac1e2273fc4b665b4f188a8950c2546560bcda9c9cea4",
        "cpu_fp64/W_gate/gU2_64": "0e4c97ebfcb0aa721cbb03edb2fa2e72a8e3382c9a0697471e1d580299b8b1ba",
        "cpu_fp64/W_gate/gU3_64": "062147be02adcdbb4825dcf24eef061153a6dbfbe82bdaabfa15342640af19cc",
        "cpu_fp64/W_up/S_fp64_cpu": "aa5dfa94c514c3561abc40386f4b633dae2321e6c0d94415d0a9fdeb6ac4faa3",
        "cpu_fp64/W_up/gR64": "4999700bb9e984ba16bd40cebba03b01c7d68aa7dbc46a678590c2f21c77724a",
        "cpu_fp64/W_up/gU0_64": "3b18b8af71ef68b16ebf62700fd06a58d42778dd39867ada90736877487ab063",
        "cpu_fp64/W_up/gU1_64": "e1d2b8e173a6a0a617d3e7f64586baae6868ff25f0eb0a08b5c741fa4d3c869e",
        "cpu_fp64/W_up/gU2_64": "0fb354473586eeacd5645f7340428373153e79cfd8acf737abda047c38e1a658",
        "cpu_fp64/W_up/gU3_64": "efcd7d7828a9c483cbfdbd624aef0104d6cf61790acdba097b66a7262fce1256",
        "cpu_fp64/__inputs__/w_fp64": "3985fa09f1433dbfbee4d01890930e0a16f021cc83451b83210d47187bb73ef8",
        "cpu_fp64/__inputs__/x_fp64": "f20da260041136f6c914b5bd092b5f5718dd230f81277f6a1198fb33581dc81e"
      }
    },
    "cuda_run_01": {
      "file_sha256": "1d9efe93b6d02ad58768682d6b1c7e4b52c5516d3254df214121310932e43629",
      "path": "cuda_gradients_run_01.pt",
      "size_bytes": 46290447,
      "tensor_metadata": {
        "run_01/W_K/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "9a0deaa1b421353d876adbe96ff901dd8397df84e8ce721e543280fc3a0625fc",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_K/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "db1fd96b53ee4e4785fd83904154bf988eb57eb9f445b59acc83e9c86da19949",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_K/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "f0ea0ee1ebce38c172c4e47813094dec3704e3cba6d89dfc4a7a544df3152160",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_K/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "6de1337e86cbc9735845f6745cc7c2c4e1f6eec4119896ca5305fc4ac8ac361a",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_K/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "b480ee0a89c7ee6c99b7f0f56eb0e895d0291e3b34eefc599d07a543caed0232",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_K/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "60849bfb6fb2c3d3989f0aaad7b98e8a604032403e76cbd611e98239ce22c929",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_K/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "7e26ea1bd016fc411e629d02b696a8c0f27744300e5401f63eac6e81154c8722",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_K/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "a21a354210e14203dbb87f88dbf2882a25e641ffdeb8a58053cb4b8212383b6a",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_K/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "fd6bcd7c290655f92297e4e22a82c9ef842bb543c2c56fe5a51bc6aec6dd4fb5",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_K/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "0479680341ab43f2574a89ccaa925499e8268384a432da8e74f2851f10b2ae74",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_O/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "86c6d3f80be2aed0ddb1329da8f1648887683a9abe3d64cb726ee5bc83590a39",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_O/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "64739500e0cea2a406126d7e90c3cc8665d180168391384fd886f19393fca6c1",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_O/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "c61a3e2a7753513da83b49a38ac2c93807ba18a23b905fdc16754abfa350712e",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_O/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "d26856ed9f5cc1a3df1d536fa5031ddb656b932fdedfd8d1589e55901ba5fa15",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_O/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "278d365b80325f76a89f116c7873c890aca3f2dd118b09a7dd90d2caee57d0c5",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_O/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "a6821553d3d5370479c17189f94f80ac214d176bd0b496ff7a76f71eb3ad4245",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_O/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "1ec1d0494dcfe09146b798122cbabb4ee492ca13a4cd0e243902f73943088b85",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_O/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "106e078440db055b3956bce6b5c71c592d98dc595b7a67c07e5c82cb31dcba11",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_O/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "5767f8e35a8486f39da43a05e142cf574bf8f78c8299672ed7deddaac863abf6",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_O/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "cca32a9fac6c9a2543f31e6459f046fde9f0b966852687bafd567b5d7f31298a",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_Q/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "154863eae062f35e4df1a1f81301e3099b50bc7f22637ada77a54ba004691a83",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_Q/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "a10f199e29d0c8857fa0902526b2753ba712f19ddf5c9cee5c6812a2267829bb",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_Q/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "5f0ab859954d69b4bf645228afa43e193ed52d652efa1368afb578c627f98d2e",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_Q/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "2db2ae6ebfd6866e6de59a1300f0a3c5bf0feba23168c3ebcfe895190eaa4caa",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_Q/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "b1562ccb878ae7f696d632007e4aa3fd476ea52e457b0c4b540dcc91d5775aa0",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_Q/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "40dc54bbaa8cdfb5dc9b256ca8b092641aff7d455c8f31b1e75293a8df460c08",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_Q/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "640641752f3a733c52a79842832de8abfc74cd7d90d369642df570f940f5b65f",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_Q/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "6267a950a6961a0fd9847aa5ec0a0f11b18e9855b6be0b5e8211185b29eca6c5",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_Q/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "7154dd6720d0ed15f77e53e16c6445458a11ff464d424e348ea8b91695167960",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_Q/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "c55e501d0d793cf43dcc63dde10cd9b859fe5c412fae43966ad86526721e40e6",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_V/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "5d17818a01ddf1ab2be743852b66142b33e160503bb9b3a0dc63d5050f6c1d36",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_V/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "65cb97537b1c270b4d2c53439f8be130097fba6759a3931f82e020a3f50cc1d4",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_V/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "6463414d05889c00435f75c48fb68e396009e61e13e67e98bebad937a36fb10d",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_V/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "5817663b1e681d285ca8753007a303698b88a1ee62275f956e53549a796b6fe0",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_V/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "2a5c5cb2d8437e95b25811a58f1217d4b8df7b82ee1dd6bb0e79544b58e053ca",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_V/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "71c274cabb1c2ce017eeb357f775d728c0cfae52d6e1a29acfa91ca60e16de59",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_V/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "5dd80959c263bd4648483ba04a733c2cb8b30bb922d09f92bf9c6655cfb8f1f3",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_V/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "d3e3e9ad1de60579f1b774a00ca88de16fed0397f2eb1001483c5f3b91bcd400",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_V/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "951de8c4cae6620805f00d636c30adde2b859bce035e8e3ad164eb0512727939",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_V/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "d5ef4b9cae4f8fae055afc4ca4708fb09dae3c1062008685c55f6e96f8833ecb",
          "shape": [
            256,
            256
          ]
        },
        "run_01/W_down/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "d3e5981a1def54c0d3106c23009d3b1463be0c66e8aa4d2ea645c63dd4120112",
          "shape": [
            256,
            1024
          ]
        },
        "run_01/W_down/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "537d36ae4e96ab25694ea73a9118bee65f8e3ceb02d705f569e164cf9e187dfc",
          "shape": [
            256,
            1024
          ]
        },
        "run_01/W_down/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "96d748148daef59c07b07fc3b535e878863156fcad8435070f04a11a01d98e34",
          "shape": [
            256,
            1024
          ]
        },
        "run_01/W_down/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "d14410c6d7deea2424142c0a64396c53d969b9ee1065d5ff1d41f7d85e36470d",
          "shape": [
            256,
            1024
          ]
        },
        "run_01/W_down/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "a672a9dd635bc44ccc9a224d8d8b35cc665139fc47ac66eb6c1223d42f4e0e11",
          "shape": [
            256,
            1024
          ]
        },
        "run_01/W_down/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "ddde5bbb6dfac8e20fc44aa5c85e7e93bab3030362f3081b9ee3bca7ab3b4b1a",
          "shape": [
            256,
            1024
          ]
        },
        "run_01/W_down/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "3c1c98933c8ba0d381d1a152a18fbeea5056a1fda644ce5842f681321b89a900",
          "shape": [
            256,
            1024
          ]
        },
        "run_01/W_down/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "1ade3c96c34f267343db6b1e405656c5766b63f9d332fe3b59345d67d5c92667",
          "shape": [
            256,
            1024
          ]
        },
        "run_01/W_down/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "00f0c7e5f613101c2eb70b0a893cd8ff75e5a65f76870bc225678dd04c695006",
          "shape": [
            256,
            1024
          ]
        },
        "run_01/W_down/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "11e61d1ed7742825df10d1dfda0b957e933a88d9ea89aee54627290268aaee1f",
          "shape": [
            256,
            1024
          ]
        },
        "run_01/W_gate/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "8dcfcfc8790dd0b7ba5aeca51b27d7d65c1102a72b3e831c1f5a925e94536b52",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_gate/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "721078d73aec3114761962b2cae926c3c6ec61e168489354b6e183565b8ad2d8",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_gate/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "337305e090464c00872bb33f2b043d1e6e21ff1cdf067e0808daf6993db907b7",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_gate/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "49446af64b65550a186b37b084e3b21534e854d58f92d6dfd9cde9092923b63f",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_gate/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "f79910dadb0d7f435313517beb7d05b5dfcac1a036012c79ff66361e4280ea66",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_gate/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "99f9415a07cf2c125eed247c33ac6932b84960ebf4ef700809285dacfc9d0d2a",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_gate/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "6d8d8b97f80c43bfb895fc275476b373a8b0a11aaf57441f0839bd526a18a0a2",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_gate/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "1a12365f0b53c680a70e6fb60ec2133091b79a6fa622638bb1c5cc7fcb61f82c",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_gate/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "1704d4775f0d42c037211ed1c1421663d762f5c5d9d20865387a509428cbb0f8",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_gate/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "4b177237e591727d0a654ef129a574c2710c6e354523cb056294de73def67b04",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_up/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "08b737f600ce3fb5ffa843bd20b956528bf12e56e36daf380e28e7039253f23b",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_up/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "7755d0d6758932bc25cd8577a8dc2238f24cca3f7c1bb3d1fc1a64ba86cf1dae",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_up/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "054ecb5eee1781f7d241956fe83f27836c8b2731e0336da72fc2b79b3f23af9c",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_up/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "b2671918a702be17b38747e7f0aa61dad58e3dda6f90b35099a966e62a46c3f8",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_up/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "791f3158ec290713d498838f1b0b52940e8ef6d9b5ecfe5369fe6a1af091a920",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_up/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "373944f7ee89555088a06f6241d5c22a7b0d15829b16043dc5729cb60c114286",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_up/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "de631591fd80ab9058eb39a751cba23f756ab62ed596c2e0db8539d20de2f4ee",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_up/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "bcd9bdde5cfee8952be89b353c2e5a5564cad07821f20d5e42480fdf78527338",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_up/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "11e169d9937832f2eec4c0867b968ea6a8394913cf738e30f7eaec0b3effc60d",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/W_up/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "9ca122b25221ddad2f7ff1a4643fe7e800a3721acba49ed058a98ed306ff96d6",
          "shape": [
            1024,
            256
          ]
        },
        "run_01/__inputs__/w_fp32_cpu": {
          "dtype": "torch.float32",
          "raw_sha256": "58da6243a1162044b797f8dbe110db13a499a5d9ccb5bea5761b77d966945783",
          "shape": [
            8,
            8,
            256
          ]
        },
        "run_01/__inputs__/x_fp32_cpu": {
          "dtype": "torch.float32",
          "raw_sha256": "0bcd1d29460cd7df422fa9ae2bc6a02887cb5b488d1dbbd9374c873a2a35f475",
          "shape": [
            8,
            8,
            256
          ]
        }
      },
      "tensor_raw_sha256": {
        "run_01/W_K/S_forward": "9a0deaa1b421353d876adbe96ff901dd8397df84e8ce721e543280fc3a0625fc",
        "run_01/W_K/S_fp64": "db1fd96b53ee4e4785fd83904154bf988eb57eb9f445b59acc83e9c86da19949",
        "run_01/W_K/S_pairwise": "f0ea0ee1ebce38c172c4e47813094dec3704e3cba6d89dfc4a7a544df3152160",
        "run_01/W_K/S_reverse": "6de1337e86cbc9735845f6745cc7c2c4e1f6eec4119896ca5305fc4ac8ac361a",
        "run_01/W_K/S_stack": "b480ee0a89c7ee6c99b7f0f56eb0e895d0291e3b34eefc599d07a543caed0232",
        "run_01/W_K/gR": "60849bfb6fb2c3d3989f0aaad7b98e8a604032403e76cbd611e98239ce22c929",
        "run_01/W_K/gU0": "7e26ea1bd016fc411e629d02b696a8c0f27744300e5401f63eac6e81154c8722",
        "run_01/W_K/gU1": "a21a354210e14203dbb87f88dbf2882a25e641ffdeb8a58053cb4b8212383b6a",
        "run_01/W_K/gU2": "fd6bcd7c290655f92297e4e22a82c9ef842bb543c2c56fe5a51bc6aec6dd4fb5",
        "run_01/W_K/gU3": "0479680341ab43f2574a89ccaa925499e8268384a432da8e74f2851f10b2ae74",
        "run_01/W_O/S_forward": "86c6d3f80be2aed0ddb1329da8f1648887683a9abe3d64cb726ee5bc83590a39",
        "run_01/W_O/S_fp64": "64739500e0cea2a406126d7e90c3cc8665d180168391384fd886f19393fca6c1",
        "run_01/W_O/S_pairwise": "c61a3e2a7753513da83b49a38ac2c93807ba18a23b905fdc16754abfa350712e",
        "run_01/W_O/S_reverse": "d26856ed9f5cc1a3df1d536fa5031ddb656b932fdedfd8d1589e55901ba5fa15",
        "run_01/W_O/S_stack": "278d365b80325f76a89f116c7873c890aca3f2dd118b09a7dd90d2caee57d0c5",
        "run_01/W_O/gR": "a6821553d3d5370479c17189f94f80ac214d176bd0b496ff7a76f71eb3ad4245",
        "run_01/W_O/gU0": "1ec1d0494dcfe09146b798122cbabb4ee492ca13a4cd0e243902f73943088b85",
        "run_01/W_O/gU1": "106e078440db055b3956bce6b5c71c592d98dc595b7a67c07e5c82cb31dcba11",
        "run_01/W_O/gU2": "5767f8e35a8486f39da43a05e142cf574bf8f78c8299672ed7deddaac863abf6",
        "run_01/W_O/gU3": "cca32a9fac6c9a2543f31e6459f046fde9f0b966852687bafd567b5d7f31298a",
        "run_01/W_Q/S_forward": "154863eae062f35e4df1a1f81301e3099b50bc7f22637ada77a54ba004691a83",
        "run_01/W_Q/S_fp64": "a10f199e29d0c8857fa0902526b2753ba712f19ddf5c9cee5c6812a2267829bb",
        "run_01/W_Q/S_pairwise": "5f0ab859954d69b4bf645228afa43e193ed52d652efa1368afb578c627f98d2e",
        "run_01/W_Q/S_reverse": "2db2ae6ebfd6866e6de59a1300f0a3c5bf0feba23168c3ebcfe895190eaa4caa",
        "run_01/W_Q/S_stack": "b1562ccb878ae7f696d632007e4aa3fd476ea52e457b0c4b540dcc91d5775aa0",
        "run_01/W_Q/gR": "40dc54bbaa8cdfb5dc9b256ca8b092641aff7d455c8f31b1e75293a8df460c08",
        "run_01/W_Q/gU0": "640641752f3a733c52a79842832de8abfc74cd7d90d369642df570f940f5b65f",
        "run_01/W_Q/gU1": "6267a950a6961a0fd9847aa5ec0a0f11b18e9855b6be0b5e8211185b29eca6c5",
        "run_01/W_Q/gU2": "7154dd6720d0ed15f77e53e16c6445458a11ff464d424e348ea8b91695167960",
        "run_01/W_Q/gU3": "c55e501d0d793cf43dcc63dde10cd9b859fe5c412fae43966ad86526721e40e6",
        "run_01/W_V/S_forward": "5d17818a01ddf1ab2be743852b66142b33e160503bb9b3a0dc63d5050f6c1d36",
        "run_01/W_V/S_fp64": "65cb97537b1c270b4d2c53439f8be130097fba6759a3931f82e020a3f50cc1d4",
        "run_01/W_V/S_pairwise": "6463414d05889c00435f75c48fb68e396009e61e13e67e98bebad937a36fb10d",
        "run_01/W_V/S_reverse": "5817663b1e681d285ca8753007a303698b88a1ee62275f956e53549a796b6fe0",
        "run_01/W_V/S_stack": "2a5c5cb2d8437e95b25811a58f1217d4b8df7b82ee1dd6bb0e79544b58e053ca",
        "run_01/W_V/gR": "71c274cabb1c2ce017eeb357f775d728c0cfae52d6e1a29acfa91ca60e16de59",
        "run_01/W_V/gU0": "5dd80959c263bd4648483ba04a733c2cb8b30bb922d09f92bf9c6655cfb8f1f3",
        "run_01/W_V/gU1": "d3e3e9ad1de60579f1b774a00ca88de16fed0397f2eb1001483c5f3b91bcd400",
        "run_01/W_V/gU2": "951de8c4cae6620805f00d636c30adde2b859bce035e8e3ad164eb0512727939",
        "run_01/W_V/gU3": "d5ef4b9cae4f8fae055afc4ca4708fb09dae3c1062008685c55f6e96f8833ecb",
        "run_01/W_down/S_forward": "d3e5981a1def54c0d3106c23009d3b1463be0c66e8aa4d2ea645c63dd4120112",
        "run_01/W_down/S_fp64": "537d36ae4e96ab25694ea73a9118bee65f8e3ceb02d705f569e164cf9e187dfc",
        "run_01/W_down/S_pairwise": "96d748148daef59c07b07fc3b535e878863156fcad8435070f04a11a01d98e34",
        "run_01/W_down/S_reverse": "d14410c6d7deea2424142c0a64396c53d969b9ee1065d5ff1d41f7d85e36470d",
        "run_01/W_down/S_stack": "a672a9dd635bc44ccc9a224d8d8b35cc665139fc47ac66eb6c1223d42f4e0e11",
        "run_01/W_down/gR": "ddde5bbb6dfac8e20fc44aa5c85e7e93bab3030362f3081b9ee3bca7ab3b4b1a",
        "run_01/W_down/gU0": "3c1c98933c8ba0d381d1a152a18fbeea5056a1fda644ce5842f681321b89a900",
        "run_01/W_down/gU1": "1ade3c96c34f267343db6b1e405656c5766b63f9d332fe3b59345d67d5c92667",
        "run_01/W_down/gU2": "00f0c7e5f613101c2eb70b0a893cd8ff75e5a65f76870bc225678dd04c695006",
        "run_01/W_down/gU3": "11e61d1ed7742825df10d1dfda0b957e933a88d9ea89aee54627290268aaee1f",
        "run_01/W_gate/S_forward": "8dcfcfc8790dd0b7ba5aeca51b27d7d65c1102a72b3e831c1f5a925e94536b52",
        "run_01/W_gate/S_fp64": "721078d73aec3114761962b2cae926c3c6ec61e168489354b6e183565b8ad2d8",
        "run_01/W_gate/S_pairwise": "337305e090464c00872bb33f2b043d1e6e21ff1cdf067e0808daf6993db907b7",
        "run_01/W_gate/S_reverse": "49446af64b65550a186b37b084e3b21534e854d58f92d6dfd9cde9092923b63f",
        "run_01/W_gate/S_stack": "f79910dadb0d7f435313517beb7d05b5dfcac1a036012c79ff66361e4280ea66",
        "run_01/W_gate/gR": "99f9415a07cf2c125eed247c33ac6932b84960ebf4ef700809285dacfc9d0d2a",
        "run_01/W_gate/gU0": "6d8d8b97f80c43bfb895fc275476b373a8b0a11aaf57441f0839bd526a18a0a2",
        "run_01/W_gate/gU1": "1a12365f0b53c680a70e6fb60ec2133091b79a6fa622638bb1c5cc7fcb61f82c",
        "run_01/W_gate/gU2": "1704d4775f0d42c037211ed1c1421663d762f5c5d9d20865387a509428cbb0f8",
        "run_01/W_gate/gU3": "4b177237e591727d0a654ef129a574c2710c6e354523cb056294de73def67b04",
        "run_01/W_up/S_forward": "08b737f600ce3fb5ffa843bd20b956528bf12e56e36daf380e28e7039253f23b",
        "run_01/W_up/S_fp64": "7755d0d6758932bc25cd8577a8dc2238f24cca3f7c1bb3d1fc1a64ba86cf1dae",
        "run_01/W_up/S_pairwise": "054ecb5eee1781f7d241956fe83f27836c8b2731e0336da72fc2b79b3f23af9c",
        "run_01/W_up/S_reverse": "b2671918a702be17b38747e7f0aa61dad58e3dda6f90b35099a966e62a46c3f8",
        "run_01/W_up/S_stack": "791f3158ec290713d498838f1b0b52940e8ef6d9b5ecfe5369fe6a1af091a920",
        "run_01/W_up/gR": "373944f7ee89555088a06f6241d5c22a7b0d15829b16043dc5729cb60c114286",
        "run_01/W_up/gU0": "de631591fd80ab9058eb39a751cba23f756ab62ed596c2e0db8539d20de2f4ee",
        "run_01/W_up/gU1": "bcd9bdde5cfee8952be89b353c2e5a5564cad07821f20d5e42480fdf78527338",
        "run_01/W_up/gU2": "11e169d9937832f2eec4c0867b968ea6a8394913cf738e30f7eaec0b3effc60d",
        "run_01/W_up/gU3": "9ca122b25221ddad2f7ff1a4643fe7e800a3721acba49ed058a98ed306ff96d6",
        "run_01/__inputs__/w_fp32_cpu": "58da6243a1162044b797f8dbe110db13a499a5d9ccb5bea5761b77d966945783",
        "run_01/__inputs__/x_fp32_cpu": "0bcd1d29460cd7df422fa9ae2bc6a02887cb5b488d1dbbd9374c873a2a35f475"
      }
    },
    "cuda_run_02": {
      "file_sha256": "6dcc7bbd0eb5bad2c66b128cbeb4767e9149a25f4e29249c135938df192e6365",
      "path": "cuda_gradients_run_02.pt",
      "size_bytes": 46290447,
      "tensor_metadata": {
        "run_02/W_K/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "c33e71b9f46428bec3dffa652a95ef83be889c22f9fb534f46bf1a9adb6425a8",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_K/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "fa4204a91e60363aa1d9a639ce21edab0b80a6f3b9de5baa15faf5f30fcace7b",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_K/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "84a86eb38ad68ea152512f6a5a66b1f8fa00a6b152111110275fcc48e07c2ff1",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_K/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "b07b5377b3e94fb5eaaeb9e3d5f75c7d3d8f46fd6c4dee6440d6cf0e67ee4bcf",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_K/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "5767012d7a15e65581171f191b7150169cc64a7236bc4e6c7eea1e04024df77e",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_K/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "9e362c86e7d7a75dc401c4ae49ded1b8771c5a99ad5c972da9ecba8aa0e705e9",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_K/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "aa696b0c9ed7b718133575dabdcc9c28b726694f79cc7cb3d8e43f62b025a533",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_K/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "e3363246da9841f9b0803b5e936589b6dc79a96b8c5bd08490d374aa85caf6f3",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_K/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "5a57b66e1071e20c3c55a5cee53f2bb6c6e7def2a2c43e9d6ec3c3ca6193af2b",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_K/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "39d5d3fadcfdc4cb3ac60882dff97ff2913c32a7ad23f9ee2a753b852f8cd21c",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_O/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "98e3b14e7079967e9127c6b7c105140a144c9a39f26764f4774c07f2ce7d356b",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_O/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "a7d5555098afe30fbcb5509add6dd89634322b716d34dcc16aded8f31d3f5b1d",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_O/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "cbcce52b49bda2eab5923a0d77934d87affd901e6be7abf9a443f78e1802edb0",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_O/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "19ba3578234c53de74b2d336d5db3d2f3cb67dee1413615a96fc8c4298f1fc32",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_O/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "ff8257ebcf83e208d3249d6f4c5cf3db88a4e9b8c333cdb8b8c3b83685dac63f",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_O/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "232a92f48dfeb9c034440fbf7a2536fa757b9c0af81c663e0c7cb4cb651ede8e",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_O/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "7e864b4ddc9b295b0079c24783c73e6c0f299f7d9b0c9677730194a9b6fc6b2f",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_O/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "f6ee22e9245d7f5ba94efa2e3afef207cbbe9d9da85ced7f60b3486924eb0a3d",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_O/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "de434a2472b69ec2e3387507020b64a88c307c44a4f30e7939f150a69eb44e95",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_O/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "b4223d1bcbea17df997f2fc7090c65c31f96ddd6bedf9fd562712493b827e1d8",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_Q/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "9c2a1cca8865df76333b5892460a4f35ad01386996e2c840b5d65e4dfd9cee21",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_Q/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "478951f4c17c62b82501cae8a0a6c97daed02160552bdbb442b8e61a1248c94a",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_Q/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "5f5d55dc3d046605f3fed3ff505bb73ef128de1e8710dd157a4757448cd16ac0",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_Q/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "24849d5e127d05d413cf0fb67a906ec67dcb8cfa3d059a7cc6452ff3570adf1a",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_Q/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "aec9da741e1f6fcbbd1cb813a25a5a46f86841e6a13a39e6081e2c784c8a27b8",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_Q/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "581f724f954fe5ad7d9f9ac6b792319d7e9bb3faad46ab9d89b80d575e771251",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_Q/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "8cd00cb95413666683efb7a5da93933af9994790ebf9733e0d142d1fe0cd1acc",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_Q/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "f9d72eef9612315f1a5fe552fa2dc5d6b5884f22c84e885d581b044e5ff29999",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_Q/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "c4c1d89294b467beeb580c031af66a7ed889fc468b4fe0809f65f1846b738281",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_Q/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "d386dca0f6c9e8234f3535fa08f5be0ebc05fdcf1268c6afa7fe1c77f64f3268",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_V/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "e7194dfd5291bc9aa9c3e2e7ae084265a3f39be8c2896b73cf1009ff8f1cecc7",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_V/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "166a2c959361554abd4a923d5d27fa440ecb915f1bf933895aeea4c745c9deb0",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_V/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "3517c13a4f282e6a41d9eca4d69400ee09977731b68a7fc6788d17cfe359fb0c",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_V/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "5d798506d8a149d98d37f4c8844587b245b4d184fdaf541fb1ea6f2a80a3a06c",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_V/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "f4739059ae07e56d8fad589c5773629dce1c41fbb6067ae28de1e96c64f87189",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_V/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "0131f369ab839c9ce9f2c8afb1aa1c91c63e90821bac81d9210a086bf6f73e30",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_V/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "0958b85cf9234a6a44f032cfe40ce4284e5b98e72b015621195ead835cad6a67",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_V/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "f39e971cd3ca17f73e52ca43c26c73dbe917575faa16a843b045bdc8759f02b2",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_V/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "fdbc286603a28a6c52f04b48070ab882877a6745b609f1cf2a638981b62568fa",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_V/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "1c0835af2b94dd92423a4643e1e774eb39fcc740419c1b09624c0379828ffe95",
          "shape": [
            256,
            256
          ]
        },
        "run_02/W_down/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "834b3c02a7480fc5001f06ffd7f5ec1118a82f252d128800692872abf3b57c36",
          "shape": [
            256,
            1024
          ]
        },
        "run_02/W_down/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "4c5b73e0957ec1730d326cc1d898d5b82e22ae51ee853ab7e7ee6ff5f8b482f7",
          "shape": [
            256,
            1024
          ]
        },
        "run_02/W_down/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "d1f90db630505541365c308f1e58e30b3e2f882d0ab60be1aa3df43292b5e80b",
          "shape": [
            256,
            1024
          ]
        },
        "run_02/W_down/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "f362cc1293252bdb40fb0631e8e362fdd96e66b45980ca2c07ae2d071734e4be",
          "shape": [
            256,
            1024
          ]
        },
        "run_02/W_down/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "293ae5839d0cb09ef9257bfd94aaf4d04924f6cb5595b885836b3e5a22932be7",
          "shape": [
            256,
            1024
          ]
        },
        "run_02/W_down/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "a42870bd0975e31b3a6eb3eef77db94e098ead0815b7ee0a167c4a50b68160c9",
          "shape": [
            256,
            1024
          ]
        },
        "run_02/W_down/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "9fa94cf2dffec39a1601135ce12194432ee00f79fe5a154c019bec5baebc47bc",
          "shape": [
            256,
            1024
          ]
        },
        "run_02/W_down/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "638e46f3452c08233e59e8485ae2bf280382a24a2bedc6ef8a4e9f756374a013",
          "shape": [
            256,
            1024
          ]
        },
        "run_02/W_down/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "5aaa6810b27f73063053c0465f276265fa2a9a1ebece6f860fe41cb598299014",
          "shape": [
            256,
            1024
          ]
        },
        "run_02/W_down/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "5985d3cb648fc620335db2b31fa1a2687b16dfc4b1662811a22044bbbda4cbff",
          "shape": [
            256,
            1024
          ]
        },
        "run_02/W_gate/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "11c480256804223c0259a335af7512f7ec2b8f5218051a57cb81afd72683ff7d",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_gate/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "677f784fdbd75bfdb8c173ed1845a3adfd1d577ccfc978e6f69e5ae1dfb14025",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_gate/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "f3d43f7358db28671751c9617c1149bb7fe2cc3625d38f5b69471ba6aa9cac96",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_gate/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "470be203c16aaec91deefe9e2451764af913d11f64a4296a1eaec2648876a8d9",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_gate/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "d75e5d7e5e97b092aed1889c60a6065c8e2db19106b47df17ae5bceb3b06768e",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_gate/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "1a5cd98172559f3b56ccc370a1eebf98714cb7783934aff290126df632bd1f68",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_gate/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "f8fb757743b8d180e77459084cf979e29ebdc06be023ccfd2b823ea6bef68f3d",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_gate/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "e3da0047d6305a3003a82b21bd7c72be96aff2fbf843b020334a2437874fe8a2",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_gate/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "4e035ded3c9e4b28a0b2ca04d0c3c42d1ec6ab57f48c0ea9475b8193c564975f",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_gate/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "a5ee5ec05b73c72b59b3c8c38447fc955dcfad5e75881e32c306d116344200af",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_up/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "d5144856925f4e13429d12d3e83706bca6c5f7fb083432b83b1a709cf21e0603",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_up/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "58582c4d4a08e40b3cd110d319596c4b863c86db8158637add6858c6527fec8d",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_up/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "0f741ed184537ac48dc4b48fc661d49adc00c95c4b8ea502f33ff99cce5372b4",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_up/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "ae6b6c05e047eca5cfd67f1cd5dabab2f110539ac65992a6abc88015aa7f45f1",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_up/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "3c09779d175a537c58a8bd3227e4f443255830b21ea2ab1474cc8954d552c5de",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_up/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "a2a4c75423727147107167d542057c8656bcd756625ea37f526809f2d85ff291",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_up/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "f7d6469b33475fe1b732a550b482c460594a4e46512ae43ebfac3659f07b47d0",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_up/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "362b445e8db00e4d65d59eba911e7bbd9a483e18a47c2450e36e0b77ee419444",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_up/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "321d58849ae134e0cdd29f81856174bdfbe5ed85f1e9521e1391f8b9e74061eb",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/W_up/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "c9327d0659251762cc39f40312e6e54fc17f5aca4fcc46d0cfb493a43f9c135f",
          "shape": [
            1024,
            256
          ]
        },
        "run_02/__inputs__/w_fp32_cpu": {
          "dtype": "torch.float32",
          "raw_sha256": "e4e28ad8c68c8e1f7001247696daa0d2f9521899340b7128113d89acaf9e9e91",
          "shape": [
            8,
            8,
            256
          ]
        },
        "run_02/__inputs__/x_fp32_cpu": {
          "dtype": "torch.float32",
          "raw_sha256": "31dc3c72911940723ee628bdf5ca91d0f71d49e839c4741c9977d6f6f05bbf27",
          "shape": [
            8,
            8,
            256
          ]
        }
      },
      "tensor_raw_sha256": {
        "run_02/W_K/S_forward": "c33e71b9f46428bec3dffa652a95ef83be889c22f9fb534f46bf1a9adb6425a8",
        "run_02/W_K/S_fp64": "fa4204a91e60363aa1d9a639ce21edab0b80a6f3b9de5baa15faf5f30fcace7b",
        "run_02/W_K/S_pairwise": "84a86eb38ad68ea152512f6a5a66b1f8fa00a6b152111110275fcc48e07c2ff1",
        "run_02/W_K/S_reverse": "b07b5377b3e94fb5eaaeb9e3d5f75c7d3d8f46fd6c4dee6440d6cf0e67ee4bcf",
        "run_02/W_K/S_stack": "5767012d7a15e65581171f191b7150169cc64a7236bc4e6c7eea1e04024df77e",
        "run_02/W_K/gR": "9e362c86e7d7a75dc401c4ae49ded1b8771c5a99ad5c972da9ecba8aa0e705e9",
        "run_02/W_K/gU0": "aa696b0c9ed7b718133575dabdcc9c28b726694f79cc7cb3d8e43f62b025a533",
        "run_02/W_K/gU1": "e3363246da9841f9b0803b5e936589b6dc79a96b8c5bd08490d374aa85caf6f3",
        "run_02/W_K/gU2": "5a57b66e1071e20c3c55a5cee53f2bb6c6e7def2a2c43e9d6ec3c3ca6193af2b",
        "run_02/W_K/gU3": "39d5d3fadcfdc4cb3ac60882dff97ff2913c32a7ad23f9ee2a753b852f8cd21c",
        "run_02/W_O/S_forward": "98e3b14e7079967e9127c6b7c105140a144c9a39f26764f4774c07f2ce7d356b",
        "run_02/W_O/S_fp64": "a7d5555098afe30fbcb5509add6dd89634322b716d34dcc16aded8f31d3f5b1d",
        "run_02/W_O/S_pairwise": "cbcce52b49bda2eab5923a0d77934d87affd901e6be7abf9a443f78e1802edb0",
        "run_02/W_O/S_reverse": "19ba3578234c53de74b2d336d5db3d2f3cb67dee1413615a96fc8c4298f1fc32",
        "run_02/W_O/S_stack": "ff8257ebcf83e208d3249d6f4c5cf3db88a4e9b8c333cdb8b8c3b83685dac63f",
        "run_02/W_O/gR": "232a92f48dfeb9c034440fbf7a2536fa757b9c0af81c663e0c7cb4cb651ede8e",
        "run_02/W_O/gU0": "7e864b4ddc9b295b0079c24783c73e6c0f299f7d9b0c9677730194a9b6fc6b2f",
        "run_02/W_O/gU1": "f6ee22e9245d7f5ba94efa2e3afef207cbbe9d9da85ced7f60b3486924eb0a3d",
        "run_02/W_O/gU2": "de434a2472b69ec2e3387507020b64a88c307c44a4f30e7939f150a69eb44e95",
        "run_02/W_O/gU3": "b4223d1bcbea17df997f2fc7090c65c31f96ddd6bedf9fd562712493b827e1d8",
        "run_02/W_Q/S_forward": "9c2a1cca8865df76333b5892460a4f35ad01386996e2c840b5d65e4dfd9cee21",
        "run_02/W_Q/S_fp64": "478951f4c17c62b82501cae8a0a6c97daed02160552bdbb442b8e61a1248c94a",
        "run_02/W_Q/S_pairwise": "5f5d55dc3d046605f3fed3ff505bb73ef128de1e8710dd157a4757448cd16ac0",
        "run_02/W_Q/S_reverse": "24849d5e127d05d413cf0fb67a906ec67dcb8cfa3d059a7cc6452ff3570adf1a",
        "run_02/W_Q/S_stack": "aec9da741e1f6fcbbd1cb813a25a5a46f86841e6a13a39e6081e2c784c8a27b8",
        "run_02/W_Q/gR": "581f724f954fe5ad7d9f9ac6b792319d7e9bb3faad46ab9d89b80d575e771251",
        "run_02/W_Q/gU0": "8cd00cb95413666683efb7a5da93933af9994790ebf9733e0d142d1fe0cd1acc",
        "run_02/W_Q/gU1": "f9d72eef9612315f1a5fe552fa2dc5d6b5884f22c84e885d581b044e5ff29999",
        "run_02/W_Q/gU2": "c4c1d89294b467beeb580c031af66a7ed889fc468b4fe0809f65f1846b738281",
        "run_02/W_Q/gU3": "d386dca0f6c9e8234f3535fa08f5be0ebc05fdcf1268c6afa7fe1c77f64f3268",
        "run_02/W_V/S_forward": "e7194dfd5291bc9aa9c3e2e7ae084265a3f39be8c2896b73cf1009ff8f1cecc7",
        "run_02/W_V/S_fp64": "166a2c959361554abd4a923d5d27fa440ecb915f1bf933895aeea4c745c9deb0",
        "run_02/W_V/S_pairwise": "3517c13a4f282e6a41d9eca4d69400ee09977731b68a7fc6788d17cfe359fb0c",
        "run_02/W_V/S_reverse": "5d798506d8a149d98d37f4c8844587b245b4d184fdaf541fb1ea6f2a80a3a06c",
        "run_02/W_V/S_stack": "f4739059ae07e56d8fad589c5773629dce1c41fbb6067ae28de1e96c64f87189",
        "run_02/W_V/gR": "0131f369ab839c9ce9f2c8afb1aa1c91c63e90821bac81d9210a086bf6f73e30",
        "run_02/W_V/gU0": "0958b85cf9234a6a44f032cfe40ce4284e5b98e72b015621195ead835cad6a67",
        "run_02/W_V/gU1": "f39e971cd3ca17f73e52ca43c26c73dbe917575faa16a843b045bdc8759f02b2",
        "run_02/W_V/gU2": "fdbc286603a28a6c52f04b48070ab882877a6745b609f1cf2a638981b62568fa",
        "run_02/W_V/gU3": "1c0835af2b94dd92423a4643e1e774eb39fcc740419c1b09624c0379828ffe95",
        "run_02/W_down/S_forward": "834b3c02a7480fc5001f06ffd7f5ec1118a82f252d128800692872abf3b57c36",
        "run_02/W_down/S_fp64": "4c5b73e0957ec1730d326cc1d898d5b82e22ae51ee853ab7e7ee6ff5f8b482f7",
        "run_02/W_down/S_pairwise": "d1f90db630505541365c308f1e58e30b3e2f882d0ab60be1aa3df43292b5e80b",
        "run_02/W_down/S_reverse": "f362cc1293252bdb40fb0631e8e362fdd96e66b45980ca2c07ae2d071734e4be",
        "run_02/W_down/S_stack": "293ae5839d0cb09ef9257bfd94aaf4d04924f6cb5595b885836b3e5a22932be7",
        "run_02/W_down/gR": "a42870bd0975e31b3a6eb3eef77db94e098ead0815b7ee0a167c4a50b68160c9",
        "run_02/W_down/gU0": "9fa94cf2dffec39a1601135ce12194432ee00f79fe5a154c019bec5baebc47bc",
        "run_02/W_down/gU1": "638e46f3452c08233e59e8485ae2bf280382a24a2bedc6ef8a4e9f756374a013",
        "run_02/W_down/gU2": "5aaa6810b27f73063053c0465f276265fa2a9a1ebece6f860fe41cb598299014",
        "run_02/W_down/gU3": "5985d3cb648fc620335db2b31fa1a2687b16dfc4b1662811a22044bbbda4cbff",
        "run_02/W_gate/S_forward": "11c480256804223c0259a335af7512f7ec2b8f5218051a57cb81afd72683ff7d",
        "run_02/W_gate/S_fp64": "677f784fdbd75bfdb8c173ed1845a3adfd1d577ccfc978e6f69e5ae1dfb14025",
        "run_02/W_gate/S_pairwise": "f3d43f7358db28671751c9617c1149bb7fe2cc3625d38f5b69471ba6aa9cac96",
        "run_02/W_gate/S_reverse": "470be203c16aaec91deefe9e2451764af913d11f64a4296a1eaec2648876a8d9",
        "run_02/W_gate/S_stack": "d75e5d7e5e97b092aed1889c60a6065c8e2db19106b47df17ae5bceb3b06768e",
        "run_02/W_gate/gR": "1a5cd98172559f3b56ccc370a1eebf98714cb7783934aff290126df632bd1f68",
        "run_02/W_gate/gU0": "f8fb757743b8d180e77459084cf979e29ebdc06be023ccfd2b823ea6bef68f3d",
        "run_02/W_gate/gU1": "e3da0047d6305a3003a82b21bd7c72be96aff2fbf843b020334a2437874fe8a2",
        "run_02/W_gate/gU2": "4e035ded3c9e4b28a0b2ca04d0c3c42d1ec6ab57f48c0ea9475b8193c564975f",
        "run_02/W_gate/gU3": "a5ee5ec05b73c72b59b3c8c38447fc955dcfad5e75881e32c306d116344200af",
        "run_02/W_up/S_forward": "d5144856925f4e13429d12d3e83706bca6c5f7fb083432b83b1a709cf21e0603",
        "run_02/W_up/S_fp64": "58582c4d4a08e40b3cd110d319596c4b863c86db8158637add6858c6527fec8d",
        "run_02/W_up/S_pairwise": "0f741ed184537ac48dc4b48fc661d49adc00c95c4b8ea502f33ff99cce5372b4",
        "run_02/W_up/S_reverse": "ae6b6c05e047eca5cfd67f1cd5dabab2f110539ac65992a6abc88015aa7f45f1",
        "run_02/W_up/S_stack": "3c09779d175a537c58a8bd3227e4f443255830b21ea2ab1474cc8954d552c5de",
        "run_02/W_up/gR": "a2a4c75423727147107167d542057c8656bcd756625ea37f526809f2d85ff291",
        "run_02/W_up/gU0": "f7d6469b33475fe1b732a550b482c460594a4e46512ae43ebfac3659f07b47d0",
        "run_02/W_up/gU1": "362b445e8db00e4d65d59eba911e7bbd9a483e18a47c2450e36e0b77ee419444",
        "run_02/W_up/gU2": "321d58849ae134e0cdd29f81856174bdfbe5ed85f1e9521e1391f8b9e74061eb",
        "run_02/W_up/gU3": "c9327d0659251762cc39f40312e6e54fc17f5aca4fcc46d0cfb493a43f9c135f",
        "run_02/__inputs__/w_fp32_cpu": "e4e28ad8c68c8e1f7001247696daa0d2f9521899340b7128113d89acaf9e9e91",
        "run_02/__inputs__/x_fp32_cpu": "31dc3c72911940723ee628bdf5ca91d0f71d49e839c4741c9977d6f6f05bbf27"
      }
    }
  }
}
```

## Per-family old-comparator indices and ULP details

```json
{
  "run_01": {
    "W_K": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 7.62939453125e-06,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 18829,
          "floor_1e_6_active": false,
          "gR_value": -38.861732482910156,
          "multi_index_row_major": [
            73,
            141
          ],
          "old_denominator": 38.861732482910156,
          "old_relative_error": 1.9632152259418945e-07,
          "sum_gU_old_value": -38.861724853515625,
          "ulp_gR": 3.814697265625e-06,
          "ulp_sum_gU_old": 3.814697265625e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 1.1920928955078125e-07,
          "error_over_ulp_gR": 32768.0,
          "error_over_ulp_sum_gU_old": 32768.0,
          "flat_index": 44868,
          "floor_1e_6_active": false,
          "gR_value": 3.4332275390625e-05,
          "multi_index_row_major": [
            175,
            68
          ],
          "old_denominator": 3.4332275390625e-05,
          "old_relative_error": 0.0034722222480922937,
          "sum_gU_old_value": 3.421306610107422e-05,
          "ulp_gR": 3.637978807091713e-12,
          "ulp_sum_gU_old": 3.637978807091713e-12
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 2.0,
          "norm_S_stack_inf": 52.309181213378906,
          "norm_gR_inf": 52.309181213378906,
          "norm_gR_l2": 2728.10595703125,
          "ulp_norm_S_stack_inf": 3.814697265625e-06
        }
      },
      "raw_tensor_sha256": {
        "run/W_K/S_forward": "2f3574be712e52be8f7ecc506816381572eb4e696c1953c0b0cf8097a2f39120",
        "run/W_K/S_fp64": "f3e1e3d0c2379e6b8130c997b42a19c73296c2ff77ac031f69f07af58b966f59",
        "run/W_K/S_pairwise": "2fb1444bb5b00272c72d9366e2a84075db00fd9104841dc8f19dbd99c0b91612",
        "run/W_K/S_reverse": "557479982d1f2be88040fe87055876fa52615452bf86cb572ea5f47c5ed1b478",
        "run/W_K/S_stack": "6d4521b9de797a9a0e18d4daacfcf57c32639b1617dc5866f7c0ba5570256cab",
        "run/W_K/gR": "a5513e80e892755cf4293ef84f75557a61a2a985b611a9658ab2763a4998a3ae",
        "run/W_K/gU0": "a5eb2b0257ef9a68de05d837385029b57ac59600e365f076e4e527f1b95bede0",
        "run/W_K/gU1": "cd482400b12c468c70538094c26ba3715aee4918089bb65104604d23839eecfd",
        "run/W_K/gU2": "e0bc54d4e69b428c0a3fd848faae5dadb71553de6dd31b59784f82fddedd892c",
        "run/W_K/gU3": "aa1a10be044f1b3f4cfd7f641b72fa6d6a38352a8c2db6eccd0e8a8bce2f863d"
      },
      "shape": [
        256,
        256
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.370466382714767e-08,
          "E_inf": 1.458519136576797e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 18829,
          "max_rel_old": 0.0034722222480922937,
          "max_rel_old_flat_index": 44868,
          "norm_gR_inf": 52.309181213378906,
          "norm_gR_l2": 2728.10595703125,
          "norm_sum_gU_inf": 52.309181213378906,
          "shape": [
            256,
            256
          ]
        },
        "S_fp64": {
          "E_L2": 3.388175308859245e-08,
          "E_inf": 5.925234263708148e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.0994415283203125e-06,
          "max_abs_flat_index": 52113,
          "max_rel_old": 0.003472222222222222,
          "max_rel_old_flat_index": 44868,
          "norm_gR_inf": 52.309181213378906,
          "norm_gR_l2": 2728.1059185615673,
          "norm_sum_gU_inf": 52.30918025970459,
          "shape": [
            256,
            256
          ]
        },
        "S_pairwise": {
          "E_L2": 4.418883747803193e-08,
          "E_inf": 7.292595682883984e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 1540,
          "max_rel_old": 0.00022883295605424792,
          "max_rel_old_flat_index": 696,
          "norm_gR_inf": 52.309181213378906,
          "norm_gR_l2": 2728.10595703125,
          "norm_sum_gU_inf": 52.309181213378906,
          "shape": [
            256,
            256
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 52.309181213378906,
          "norm_gR_l2": 2728.10595703125,
          "norm_sum_gU_inf": 52.309181213378906,
          "shape": [
            256,
            256
          ]
        },
        "S_stack": {
          "E_L2": 5.370466382714767e-08,
          "E_inf": 1.458519136576797e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 18829,
          "max_rel_old": 0.0034722222480922937,
          "max_rel_old_flat_index": 44868,
          "norm_gR_inf": 52.309181213378906,
          "norm_gR_l2": 2728.10595703125,
          "norm_sum_gU_inf": 52.309181213378906,
          "shape": [
            256,
            256
          ]
        }
      },
      "tensor_metadata": {
        "run/W_K/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "2f3574be712e52be8f7ecc506816381572eb4e696c1953c0b0cf8097a2f39120",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "f3e1e3d0c2379e6b8130c997b42a19c73296c2ff77ac031f69f07af58b966f59",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "2fb1444bb5b00272c72d9366e2a84075db00fd9104841dc8f19dbd99c0b91612",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "557479982d1f2be88040fe87055876fa52615452bf86cb572ea5f47c5ed1b478",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "6d4521b9de797a9a0e18d4daacfcf57c32639b1617dc5866f7c0ba5570256cab",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "a5513e80e892755cf4293ef84f75557a61a2a985b611a9658ab2763a4998a3ae",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "a5eb2b0257ef9a68de05d837385029b57ac59600e365f076e4e527f1b95bede0",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "cd482400b12c468c70538094c26ba3715aee4918089bb65104604d23839eecfd",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "e0bc54d4e69b428c0a3fd848faae5dadb71553de6dd31b59784f82fddedd892c",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "aa1a10be044f1b3f4cfd7f641b72fa6d6a38352a8c2db6eccd0e8a8bce2f863d",
          "shape": [
            256,
            256
          ]
        }
      }
    },
    "W_O": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 1.52587890625e-05,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 36257,
          "floor_1e_6_active": false,
          "gR_value": 79.351806640625,
          "multi_index_row_major": [
            141,
            161
          ],
          "old_denominator": 79.351806640625,
          "old_relative_error": 1.9229290160183155e-07,
          "sum_gU_old_value": 79.35179138183594,
          "ulp_gR": 7.62939453125e-06,
          "ulp_sum_gU_old": 7.62939453125e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 4.76837158203125e-07,
          "error_over_ulp_gR": 1024.0,
          "error_over_ulp_sum_gU_old": 1024.0,
          "flat_index": 17093,
          "floor_1e_6_active": false,
          "gR_value": 0.0039310455322265625,
          "multi_index_row_major": [
            66,
            197
          ],
          "old_denominator": 0.0039310455322265625,
          "old_relative_error": 0.00012130034156143665,
          "sum_gU_old_value": 0.003930568695068359,
          "ulp_gR": 4.656612873077393e-10,
          "ulp_sum_gU_old": 4.656612873077393e-10
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 2.0,
          "norm_S_stack_inf": 124.8612060546875,
          "norm_gR_inf": 124.86121368408203,
          "norm_gR_l2": 6258.7431640625,
          "ulp_norm_S_stack_inf": 7.62939453125e-06
        }
      },
      "raw_tensor_sha256": {
        "run/W_O/S_forward": "3c49753c6b1e6c9ece0906495fe3cd0dfc2f42a1d0ab11910605f29f9b279b6d",
        "run/W_O/S_fp64": "e35e23789ac7b0a028d31d4a42f12945917cda2fc87dc440402768e7418dae9f",
        "run/W_O/S_pairwise": "e4267e78c29f3e4e625255a64c6845910817d6d0ae85386a1e265d9c371d7801",
        "run/W_O/S_reverse": "efe6239eb413fe1147361bcbcda6fd7c06a623a60e2ac4a21d31b4f2b34795ca",
        "run/W_O/S_stack": "28ee8cffd6c5882127b47e9a1999e82cba582f4474a6431faf5af400f7e083ba",
        "run/W_O/gR": "aebdcd338b3b17a4915adc1edc239fd76af92b79764743ea3e502b93e8d7ecf8",
        "run/W_O/gU0": "86c7ac601d6127fae5e98ffbbf1b6e26ec09469c30e5ee03dc91ef2ca2f823d3",
        "run/W_O/gU1": "698ecf7dee3e36ccc54122032a640df82e72ad982cd8870f3f2161e79b5b4734",
        "run/W_O/gU2": "03982d511cf2a8e283afe0638452641b6128fe6c687f9039e6f4051f48e4cf9d",
        "run/W_O/gU3": "70d5bbc92e7fcb16ff366dc76b1f954a76babb7f7415f273176fa136f5700d93"
      },
      "shape": [
        256,
        256
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.3697743140901366e-08,
          "E_inf": 1.2220600353884947e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 1.52587890625e-05,
          "max_abs_flat_index": 36257,
          "max_rel_old": 0.00012130034156143665,
          "max_rel_old_flat_index": 17093,
          "norm_gR_inf": 124.86121368408203,
          "norm_gR_l2": 6258.7431640625,
          "norm_sum_gU_inf": 124.8612060546875,
          "shape": [
            256,
            256
          ]
        },
        "S_fp64": {
          "E_L2": 3.7214034681009397e-08,
          "E_inf": 6.110299833023837e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 11704,
          "max_rel_old": 0.00012130033964095099,
          "max_rel_old_flat_index": 17093,
          "norm_gR_inf": 124.86121368408203,
          "norm_gR_l2": 6258.742908932774,
          "norm_sum_gU_inf": 124.86121368408203,
          "shape": [
            256,
            256
          ]
        },
        "S_pairwise": {
          "E_L2": 4.322798830003194e-08,
          "E_inf": 6.110299466399738e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 324,
          "max_rel_old": 5.582226367550902e-05,
          "max_rel_old_flat_index": 58698,
          "norm_gR_inf": 124.86121368408203,
          "norm_gR_l2": 6258.7431640625,
          "norm_sum_gU_inf": 124.86122131347656,
          "shape": [
            256,
            256
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 124.86121368408203,
          "norm_gR_l2": 6258.7431640625,
          "norm_sum_gU_inf": 124.86121368408203,
          "shape": [
            256,
            256
          ]
        },
        "S_stack": {
          "E_L2": 5.3697743140901366e-08,
          "E_inf": 1.2220600353884947e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 1.52587890625e-05,
          "max_abs_flat_index": 36257,
          "max_rel_old": 0.00012130034156143665,
          "max_rel_old_flat_index": 17093,
          "norm_gR_inf": 124.86121368408203,
          "norm_gR_l2": 6258.7431640625,
          "norm_sum_gU_inf": 124.8612060546875,
          "shape": [
            256,
            256
          ]
        }
      },
      "tensor_metadata": {
        "run/W_O/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "3c49753c6b1e6c9ece0906495fe3cd0dfc2f42a1d0ab11910605f29f9b279b6d",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "e35e23789ac7b0a028d31d4a42f12945917cda2fc87dc440402768e7418dae9f",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "e4267e78c29f3e4e625255a64c6845910817d6d0ae85386a1e265d9c371d7801",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "efe6239eb413fe1147361bcbcda6fd7c06a623a60e2ac4a21d31b4f2b34795ca",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "28ee8cffd6c5882127b47e9a1999e82cba582f4474a6431faf5af400f7e083ba",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "aebdcd338b3b17a4915adc1edc239fd76af92b79764743ea3e502b93e8d7ecf8",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "86c7ac601d6127fae5e98ffbbf1b6e26ec09469c30e5ee03dc91ef2ca2f823d3",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "698ecf7dee3e36ccc54122032a640df82e72ad982cd8870f3f2161e79b5b4734",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "03982d511cf2a8e283afe0638452641b6128fe6c687f9039e6f4051f48e4cf9d",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "70d5bbc92e7fcb16ff366dc76b1f954a76babb7f7415f273176fa136f5700d93",
          "shape": [
            256,
            256
          ]
        }
      }
    },
    "W_Q": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 7.62939453125e-06,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 3378,
          "floor_1e_6_active": false,
          "gR_value": -37.437400817871094,
          "multi_index_row_major": [
            13,
            50
          ],
          "old_denominator": 37.437408447265625,
          "old_relative_error": 2.0379066256737133e-07,
          "sum_gU_old_value": -37.437408447265625,
          "ulp_gR": 3.814697265625e-06,
          "ulp_sum_gU_old": 3.814697265625e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 1.1920928955078125e-07,
          "error_over_ulp_gR": 4096.0,
          "error_over_ulp_sum_gU_old": 4096.0,
          "flat_index": 58051,
          "floor_1e_6_active": false,
          "gR_value": -0.0003300905227661133,
          "multi_index_row_major": [
            226,
            195
          ],
          "old_denominator": 0.00033020973205566406,
          "old_relative_error": 0.00036101083969697356,
          "sum_gU_old_value": -0.00033020973205566406,
          "ulp_gR": 2.9103830456733704e-11,
          "ulp_sum_gU_old": 2.9103830456733704e-11
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 2.0,
          "norm_S_stack_inf": 52.54109573364258,
          "norm_gR_inf": 52.54109191894531,
          "norm_gR_l2": 2769.7294921875,
          "ulp_norm_S_stack_inf": 3.814697265625e-06
        }
      },
      "raw_tensor_sha256": {
        "run/W_Q/S_forward": "a55303d7b27c7dd2d40f6f159f7a19e1f3ff9674ee1cd9276b7fde9ffdf3a359",
        "run/W_Q/S_fp64": "5d6aa547684c262064d9d6aa0444f60ebdfab75a4df43a72505f84b62b1dd4d5",
        "run/W_Q/S_pairwise": "c5b994fc621aefa0cd94134709fd9eb47b8200794bb81b019de4dbb1584448f8",
        "run/W_Q/S_reverse": "d197ab3428f1eb8c5cf5976198f9fdd922a5779ef0c2a76b5983c19f8a5802ec",
        "run/W_Q/S_stack": "d9ecab578cec499bd95f3597413873f4ff614e9f7a3b94a2d7dae17e53a6fcd4",
        "run/W_Q/gR": "76d393a4c29ddaf04e22539b74ec97e477b40e8e7265a5c11d19742378f79bd3",
        "run/W_Q/gU0": "0f7b7a4554cacc24d265ffb7bfced268fcc8bd895bad17ea7e83f91f4e7978ea",
        "run/W_Q/gU1": "841a5c0f140c3082656c65b4ea9de72da40b60dbdb53d36b954ee3804dd086fe",
        "run/W_Q/gU2": "b3b85ce6a3c1c4937883f5f8520d4888c942a30fa0a8ac26a6a332cfe5ced400",
        "run/W_Q/gU3": "98614ef2897ec5a9c2b0b997e4ccebebdb739ff8181bb79371f2398980655300"
      },
      "shape": [
        256,
        256
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.372153566440829e-08,
          "E_inf": 1.452081335173716e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 3378,
          "max_rel_old": 0.00036101083969697356,
          "max_rel_old_flat_index": 58051,
          "norm_gR_inf": 52.54109191894531,
          "norm_gR_l2": 2769.7294921875,
          "norm_sum_gU_inf": 52.54109573364258,
          "shape": [
            256,
            256
          ]
        },
        "S_fp64": {
          "E_L2": 3.415794472797588e-08,
          "E_inf": 6.352856115464177e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.337860107421875e-06,
          "max_abs_flat_index": 4106,
          "max_rel_old": 0.00011000696710791684,
          "max_rel_old_flat_index": 4346,
          "norm_gR_inf": 52.54109191894531,
          "norm_gR_l2": 2769.729576019558,
          "norm_sum_gU_inf": 52.541094064712524,
          "shape": [
            256,
            256
          ]
        },
        "S_pairwise": {
          "E_L2": 4.3970409535631916e-08,
          "E_inf": 7.260407386411316e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 1470,
          "max_rel_old": 0.00036101083969697356,
          "max_rel_old_flat_index": 58051,
          "norm_gR_inf": 52.54109191894531,
          "norm_gR_l2": 2769.7294921875,
          "norm_sum_gU_inf": 52.54109191894531,
          "shape": [
            256,
            256
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 52.54109191894531,
          "norm_gR_l2": 2769.7294921875,
          "norm_sum_gU_inf": 52.54109191894531,
          "shape": [
            256,
            256
          ]
        },
        "S_stack": {
          "E_L2": 5.372153566440829e-08,
          "E_inf": 1.452081335173716e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 3378,
          "max_rel_old": 0.00036101083969697356,
          "max_rel_old_flat_index": 58051,
          "norm_gR_inf": 52.54109191894531,
          "norm_gR_l2": 2769.7294921875,
          "norm_sum_gU_inf": 52.54109573364258,
          "shape": [
            256,
            256
          ]
        }
      },
      "tensor_metadata": {
        "run/W_Q/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "a55303d7b27c7dd2d40f6f159f7a19e1f3ff9674ee1cd9276b7fde9ffdf3a359",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "5d6aa547684c262064d9d6aa0444f60ebdfab75a4df43a72505f84b62b1dd4d5",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "c5b994fc621aefa0cd94134709fd9eb47b8200794bb81b019de4dbb1584448f8",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "d197ab3428f1eb8c5cf5976198f9fdd922a5779ef0c2a76b5983c19f8a5802ec",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "d9ecab578cec499bd95f3597413873f4ff614e9f7a3b94a2d7dae17e53a6fcd4",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "76d393a4c29ddaf04e22539b74ec97e477b40e8e7265a5c11d19742378f79bd3",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "0f7b7a4554cacc24d265ffb7bfced268fcc8bd895bad17ea7e83f91f4e7978ea",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "841a5c0f140c3082656c65b4ea9de72da40b60dbdb53d36b954ee3804dd086fe",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "b3b85ce6a3c1c4937883f5f8520d4888c942a30fa0a8ac26a6a332cfe5ced400",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "98614ef2897ec5a9c2b0b997e4ccebebdb739ff8181bb79371f2398980655300",
          "shape": [
            256,
            256
          ]
        }
      }
    },
    "W_V": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 1.52587890625e-05,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 6109,
          "floor_1e_6_active": false,
          "gR_value": -99.20011901855469,
          "multi_index_row_major": [
            23,
            221
          ],
          "old_denominator": 99.20011901855469,
          "old_relative_error": 1.5381824880478234e-07,
          "sum_gU_old_value": -99.20010375976562,
          "ulp_gR": 7.62939453125e-06,
          "ulp_sum_gU_old": 7.62939453125e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 1.1920928955078125e-07,
          "error_over_ulp_gR": 2048.0,
          "error_over_ulp_sum_gU_old": 2048.0,
          "flat_index": 14342,
          "floor_1e_6_active": false,
          "gR_value": -0.0005128383636474609,
          "multi_index_row_major": [
            56,
            6
          ],
          "old_denominator": 0.0005129575729370117,
          "old_relative_error": 0.00023239600704982877,
          "sum_gU_old_value": -0.0005129575729370117,
          "ulp_gR": 5.820766091346741e-11,
          "ulp_sum_gU_old": 5.820766091346741e-11
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 1.0,
          "norm_S_stack_inf": 128.3258514404297,
          "norm_gR_inf": 128.3258514404297,
          "norm_gR_l2": 6239.6865234375,
          "ulp_norm_S_stack_inf": 1.52587890625e-05
        }
      },
      "raw_tensor_sha256": {
        "run/W_V/S_forward": "16dc942221e0e748e3993a9679bc63816848093d76209a7b5117f0081bb8d2d2",
        "run/W_V/S_fp64": "ca657c6c5514668f7c5152af5e6d99c4e302b4d7b777fbc41b9b15e0f46b694c",
        "run/W_V/S_pairwise": "74a1838bbd1fe7d13b6acd3f2b7b940743e3104ac233bd7897c77fa1b6714e82",
        "run/W_V/S_reverse": "1a33021c7ab2f5a5d90edcd795027908819cef02058f689036633ca27d262839",
        "run/W_V/S_stack": "08e50a098580b820dc46f3d07e3337e6a70ed7da0fab217984253b46fe9602b1",
        "run/W_V/gR": "0b4bebfeeabde03eb09ad1945d9333d339e83dfe06143237c3b719795410ef4a",
        "run/W_V/gU0": "4df24b4da4b0a2caff499d5599ef95b066c708a5e2cd535f8e62ca51bf6de4d2",
        "run/W_V/gU1": "a105de63566b09f13b1e1783a9881e313bcbc7e8d16a46ff164e26d72b897498",
        "run/W_V/gU2": "5eee0635e9a76e1343ac323ac9c62489793fecdf22f3de3ccfdd820042659534",
        "run/W_V/gU3": "8e0df6a8c9576702d52397f68ebf5b8439539ae2b0ab6e9dfb42f8c9a735077d"
      },
      "shape": [
        256,
        256
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.404850611512302e-08,
          "E_inf": 1.1890658413449273e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 1.52587890625e-05,
          "max_abs_flat_index": 6109,
          "max_rel_old": 0.00023239600704982877,
          "max_rel_old_flat_index": 14342,
          "norm_gR_inf": 128.3258514404297,
          "norm_gR_l2": 6239.6865234375,
          "norm_sum_gU_inf": 128.3258514404297,
          "shape": [
            256,
            256
          ]
        },
        "S_fp64": {
          "E_L2": 3.747180079402311e-08,
          "E_inf": 6.688495537970857e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 8.58306884765625e-06,
          "max_abs_flat_index": 16798,
          "max_rel_old": 0.00011621150493898897,
          "max_rel_old_flat_index": 14342,
          "norm_gR_inf": 128.3258514404297,
          "norm_gR_l2": 6239.686415141664,
          "norm_sum_gU_inf": 128.32584810256958,
          "shape": [
            256,
            256
          ]
        },
        "S_pairwise": {
          "E_L2": 4.386166096992383e-08,
          "E_inf": 1.1890658413449273e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 1.52587890625e-05,
          "max_abs_flat_index": 38372,
          "max_rel_old": 8.73973112902604e-05,
          "max_rel_old_flat_index": 50103,
          "norm_gR_inf": 128.3258514404297,
          "norm_gR_l2": 6239.6865234375,
          "norm_sum_gU_inf": 128.3258514404297,
          "shape": [
            256,
            256
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 128.3258514404297,
          "norm_gR_l2": 6239.6865234375,
          "norm_sum_gU_inf": 128.3258514404297,
          "shape": [
            256,
            256
          ]
        },
        "S_stack": {
          "E_L2": 5.404850611512302e-08,
          "E_inf": 1.1890658413449273e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 1.52587890625e-05,
          "max_abs_flat_index": 6109,
          "max_rel_old": 0.00023239600704982877,
          "max_rel_old_flat_index": 14342,
          "norm_gR_inf": 128.3258514404297,
          "norm_gR_l2": 6239.6865234375,
          "norm_sum_gU_inf": 128.3258514404297,
          "shape": [
            256,
            256
          ]
        }
      },
      "tensor_metadata": {
        "run/W_V/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "16dc942221e0e748e3993a9679bc63816848093d76209a7b5117f0081bb8d2d2",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "ca657c6c5514668f7c5152af5e6d99c4e302b4d7b777fbc41b9b15e0f46b694c",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "74a1838bbd1fe7d13b6acd3f2b7b940743e3104ac233bd7897c77fa1b6714e82",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "1a33021c7ab2f5a5d90edcd795027908819cef02058f689036633ca27d262839",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "08e50a098580b820dc46f3d07e3337e6a70ed7da0fab217984253b46fe9602b1",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "0b4bebfeeabde03eb09ad1945d9333d339e83dfe06143237c3b719795410ef4a",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "4df24b4da4b0a2caff499d5599ef95b066c708a5e2cd535f8e62ca51bf6de4d2",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "a105de63566b09f13b1e1783a9881e313bcbc7e8d16a46ff164e26d72b897498",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "5eee0635e9a76e1343ac323ac9c62489793fecdf22f3de3ccfdd820042659534",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "8e0df6a8c9576702d52397f68ebf5b8439539ae2b0ab6e9dfb42f8c9a735077d",
          "shape": [
            256,
            256
          ]
        }
      }
    },
    "W_down": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 7.62939453125e-06,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 140001,
          "floor_1e_6_active": false,
          "gR_value": 42.20118713378906,
          "multi_index_row_major": [
            136,
            737
          ],
          "old_denominator": 42.201194763183594,
          "old_relative_error": 1.8078621621953062e-07,
          "sum_gU_old_value": 42.201194763183594,
          "ulp_gR": 3.814697265625e-06,
          "ulp_sum_gU_old": 3.814697265625e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 2.384185791015625e-07,
          "error_over_ulp_gR": 32768.0,
          "error_over_ulp_sum_gU_old": 32768.0,
          "flat_index": 158001,
          "floor_1e_6_active": false,
          "gR_value": -7.963180541992188e-05,
          "multi_index_row_major": [
            154,
            305
          ],
          "old_denominator": 7.963180541992188e-05,
          "old_relative_error": 0.002994012087583542,
          "sum_gU_old_value": -7.939338684082031e-05,
          "ulp_gR": 7.275957614183426e-12,
          "ulp_sum_gU_old": 7.275957614183426e-12
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 2.0,
          "norm_S_stack_inf": 60.65041732788086,
          "norm_gR_inf": 60.65041732788086,
          "norm_gR_l2": 4074.2841796875,
          "ulp_norm_S_stack_inf": 3.814697265625e-06
        }
      },
      "raw_tensor_sha256": {
        "run/W_down/S_forward": "ace6fb3e36bb9418773f782f5a30cd5b9a2da97b47588807513f03933de7415c",
        "run/W_down/S_fp64": "a139715d974c33898a5e1a48f829c88d0c74f3123ece41a92b8d3af58c154a46",
        "run/W_down/S_pairwise": "5d0ff22439856349849c4f8ed41b6ef27a48f9d8bfed96ca6421d3b1af8eafcc",
        "run/W_down/S_reverse": "fde3f04b4f3a3fb4b3b9cac5e93092edb21d773d4253363bac55e0c851fbc3e2",
        "run/W_down/S_stack": "bc1b60c8fde4100ed5e545c4992cbbba97bc85a79af39cd042f85a539bec1b18",
        "run/W_down/gR": "2ee294919436fcf0ebebe2317358e61d90a70a42515eb90e5b5240605071146d",
        "run/W_down/gU0": "dcbe910c19b5ad888b41dc2415db6f4396690145d334a41fba8aca6f0a607cab",
        "run/W_down/gU1": "fd9f8b9ec16e2803fe8a88d674b5e14f8514d965e1c4ed9629790b9bc8727b21",
        "run/W_down/gU2": "e95e9e4e6bd950e7e855d5f19b4cdc2ccb523fec3233d884a045e34697647ebf",
        "run/W_down/gU3": "7377d94dfd3c64d2347959a2a535c25e71801bd6e524df19365a9636d1b3b4f8"
      },
      "shape": [
        256,
        1024
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.3162100499548615e-08,
          "E_inf": 1.2579293695580418e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 140001,
          "max_rel_old": 0.002994012087583542,
          "max_rel_old_flat_index": 158001,
          "norm_gR_inf": 60.65041732788086,
          "norm_gR_l2": 4074.2841796875,
          "norm_sum_gU_inf": 60.65041732788086,
          "shape": [
            256,
            1024
          ]
        },
        "S_fp64": {
          "E_L2": 3.656333379082823e-08,
          "E_inf": 6.289646986329798e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 140001,
          "max_rel_old": 0.002877058383233533,
          "max_rel_old_flat_index": 158001,
          "norm_gR_inf": 60.65041732788086,
          "norm_gR_l2": 4074.2843031300195,
          "norm_sum_gU_inf": 60.65041923522949,
          "shape": [
            256,
            1024
          ]
        },
        "S_pairwise": {
          "E_L2": 4.302838974012957e-08,
          "E_inf": 6.289646847790209e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 5547,
          "max_rel_old": 0.00011134617670904845,
          "max_rel_old_flat_index": 3474,
          "norm_gR_inf": 60.65041732788086,
          "norm_gR_l2": 4074.2841796875,
          "norm_sum_gU_inf": 60.65041732788086,
          "shape": [
            256,
            1024
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 60.65041732788086,
          "norm_gR_l2": 4074.2841796875,
          "norm_sum_gU_inf": 60.65041732788086,
          "shape": [
            256,
            1024
          ]
        },
        "S_stack": {
          "E_L2": 5.3162100499548615e-08,
          "E_inf": 1.2579293695580418e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 140001,
          "max_rel_old": 0.002994012087583542,
          "max_rel_old_flat_index": 158001,
          "norm_gR_inf": 60.65041732788086,
          "norm_gR_l2": 4074.2841796875,
          "norm_sum_gU_inf": 60.65041732788086,
          "shape": [
            256,
            1024
          ]
        }
      },
      "tensor_metadata": {
        "run/W_down/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "ace6fb3e36bb9418773f782f5a30cd5b9a2da97b47588807513f03933de7415c",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "a139715d974c33898a5e1a48f829c88d0c74f3123ece41a92b8d3af58c154a46",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "5d0ff22439856349849c4f8ed41b6ef27a48f9d8bfed96ca6421d3b1af8eafcc",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "fde3f04b4f3a3fb4b3b9cac5e93092edb21d773d4253363bac55e0c851fbc3e2",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "bc1b60c8fde4100ed5e545c4992cbbba97bc85a79af39cd042f85a539bec1b18",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "2ee294919436fcf0ebebe2317358e61d90a70a42515eb90e5b5240605071146d",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "dcbe910c19b5ad888b41dc2415db6f4396690145d334a41fba8aca6f0a607cab",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "fd9f8b9ec16e2803fe8a88d674b5e14f8514d965e1c4ed9629790b9bc8727b21",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "e95e9e4e6bd950e7e855d5f19b4cdc2ccb523fec3233d884a045e34697647ebf",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "7377d94dfd3c64d2347959a2a535c25e71801bd6e524df19365a9636d1b3b4f8",
          "shape": [
            256,
            1024
          ]
        }
      }
    },
    "W_gate": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 7.62939453125e-06,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 141417,
          "floor_1e_6_active": false,
          "gR_value": -46.208038330078125,
          "multi_index_row_major": [
            552,
            105
          ],
          "old_denominator": 46.208045959472656,
          "old_relative_error": 1.6510965394900268e-07,
          "sum_gU_old_value": -46.208045959472656,
          "ulp_gR": 3.814697265625e-06,
          "ulp_sum_gU_old": 3.814697265625e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 2.384185791015625e-07,
          "error_over_ulp_gR": 65536.0,
          "error_over_ulp_sum_gU_old": 65536.0,
          "flat_index": 11995,
          "floor_1e_6_active": false,
          "gR_value": 3.9577484130859375e-05,
          "multi_index_row_major": [
            46,
            219
          ],
          "old_denominator": 3.981590270996094e-05,
          "old_relative_error": 0.005988024175167084,
          "sum_gU_old_value": 3.981590270996094e-05,
          "ulp_gR": 3.637978807091713e-12,
          "ulp_sum_gU_old": 3.637978807091713e-12
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 2.0,
          "norm_S_stack_inf": 55.499542236328125,
          "norm_gR_inf": 55.499542236328125,
          "norm_gR_l2": 4135.73779296875,
          "ulp_norm_S_stack_inf": 3.814697265625e-06
        }
      },
      "raw_tensor_sha256": {
        "run/W_gate/S_forward": "eee5e4b0122f1c347b9c7e261414a739fb95b2f53c581297017d13fc98de6259",
        "run/W_gate/S_fp64": "0acbd855b2fcab4c8652c649a00b09237f85371a77700682616f01cd0bd30797",
        "run/W_gate/S_pairwise": "29007fc31e68e255615fedb2397822ea45fa51b4ca40bd7a5f78cd1b8580ff8b",
        "run/W_gate/S_reverse": "6b98f1c83849e19de84776f3e97df2ae08c1a68e1f54e5ca5fac6dc6bcca215a",
        "run/W_gate/S_stack": "6bc52b71e030ff476e0ac8fec74cd278bd09f5dff2faf5c7b2aa7ac4488a9930",
        "run/W_gate/gR": "aeecf834df66cac799c8a93cf57885ce284ffddbe1c5b4cb29c9ee918fbf1b44",
        "run/W_gate/gU0": "3459a032b4d48c2d03fc91b928cebac34d921c6572f7e721a949f74128347fc4",
        "run/W_gate/gU1": "66b94defeab4d18ad3fa75814bae7d07731af69c30cdf4b08a217e48808a653c",
        "run/W_gate/gU2": "8f7ef0a0ccd772c09f189e0f69b4ce90253e476c4f496ced16e4ddf62587a0f2",
        "run/W_gate/gU3": "93dc06c20e361ca23753f0f54fe6f77b36f91175d49b2dfae6b95cb8af6aff26"
      },
      "shape": [
        1024,
        256
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.3564853885745833e-08,
          "E_inf": 1.3746770832767652e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 141417,
          "max_rel_old": 0.005988024175167084,
          "max_rel_old_flat_index": 11995,
          "norm_gR_inf": 55.499542236328125,
          "norm_gR_l2": 4135.73779296875,
          "norm_sum_gU_inf": 55.499542236328125,
          "shape": [
            1024,
            256
          ]
        },
        "S_fp64": {
          "E_L2": 3.671111470897815e-08,
          "E_inf": 6.873385098171184e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 36421,
          "max_rel_old": 0.0015037593984962407,
          "max_rel_old_flat_index": 11995,
          "norm_gR_inf": 55.499542236328125,
          "norm_gR_l2": 4135.737847335061,
          "norm_sum_gU_inf": 55.49954032897949,
          "shape": [
            1024,
            256
          ]
        },
        "S_pairwise": {
          "E_L2": 4.3360042667472953e-08,
          "E_inf": 6.873385416383826e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 5129,
          "max_rel_old": 0.0009900990407913923,
          "max_rel_old_flat_index": 170301,
          "norm_gR_inf": 55.499542236328125,
          "norm_gR_l2": 4135.73779296875,
          "norm_sum_gU_inf": 55.499542236328125,
          "shape": [
            1024,
            256
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 55.499542236328125,
          "norm_gR_l2": 4135.73779296875,
          "norm_sum_gU_inf": 55.499542236328125,
          "shape": [
            1024,
            256
          ]
        },
        "S_stack": {
          "E_L2": 5.3564853885745833e-08,
          "E_inf": 1.3746770832767652e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 141417,
          "max_rel_old": 0.005988024175167084,
          "max_rel_old_flat_index": 11995,
          "norm_gR_inf": 55.499542236328125,
          "norm_gR_l2": 4135.73779296875,
          "norm_sum_gU_inf": 55.499542236328125,
          "shape": [
            1024,
            256
          ]
        }
      },
      "tensor_metadata": {
        "run/W_gate/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "eee5e4b0122f1c347b9c7e261414a739fb95b2f53c581297017d13fc98de6259",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "0acbd855b2fcab4c8652c649a00b09237f85371a77700682616f01cd0bd30797",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "29007fc31e68e255615fedb2397822ea45fa51b4ca40bd7a5f78cd1b8580ff8b",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "6b98f1c83849e19de84776f3e97df2ae08c1a68e1f54e5ca5fac6dc6bcca215a",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "6bc52b71e030ff476e0ac8fec74cd278bd09f5dff2faf5c7b2aa7ac4488a9930",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "aeecf834df66cac799c8a93cf57885ce284ffddbe1c5b4cb29c9ee918fbf1b44",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "3459a032b4d48c2d03fc91b928cebac34d921c6572f7e721a949f74128347fc4",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "66b94defeab4d18ad3fa75814bae7d07731af69c30cdf4b08a217e48808a653c",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "8f7ef0a0ccd772c09f189e0f69b4ce90253e476c4f496ced16e4ddf62587a0f2",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "93dc06c20e361ca23753f0f54fe6f77b36f91175d49b2dfae6b95cb8af6aff26",
          "shape": [
            1024,
            256
          ]
        }
      }
    },
    "W_up": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 7.62939453125e-06,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 13612,
          "floor_1e_6_active": false,
          "gR_value": 62.38463592529297,
          "multi_index_row_major": [
            53,
            44
          ],
          "old_denominator": 62.38463592529297,
          "old_relative_error": 1.22296043514325e-07,
          "sum_gU_old_value": 62.38462829589844,
          "ulp_gR": 3.814697265625e-06,
          "ulp_sum_gU_old": 3.814697265625e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 5.960464477539063e-08,
          "error_over_ulp_gR": 4096.0,
          "error_over_ulp_sum_gU_old": 4096.0,
          "flat_index": 114530,
          "floor_1e_6_active": false,
          "gR_value": -0.00013959407806396484,
          "multi_index_row_major": [
            447,
            98
          ],
          "old_denominator": 0.00013959407806396484,
          "old_relative_error": 0.00042698546894825995,
          "sum_gU_old_value": -0.00013953447341918945,
          "ulp_gR": 1.4551915228366852e-11,
          "ulp_sum_gU_old": 1.4551915228366852e-11
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 1.0,
          "norm_S_stack_inf": 76.27912902832031,
          "norm_gR_inf": 76.27912902832031,
          "norm_gR_l2": 4129.11181640625,
          "ulp_norm_S_stack_inf": 7.62939453125e-06
        }
      },
      "raw_tensor_sha256": {
        "run/W_up/S_forward": "e722b653e7a26a7fe4479578fd73ef8add94946080e1cdb80d7b260fc5331b9f",
        "run/W_up/S_fp64": "2048f71a900cfc0b67c47cd68eafd0f1e4bd1fdf106a196270129e2327d63e3f",
        "run/W_up/S_pairwise": "caf732cc851c34c5855b558c297d68352578d3d420bba8df2b7b5a121fb7cd37",
        "run/W_up/S_reverse": "6b933569530c2a7079725e45289c8bc4bdf8bbc34fab2abf86ba58a81a08d0ff",
        "run/W_up/S_stack": "a315db3b5a8ee594ea7d4cb6e960ba4af03d5f19885a97570a99761db9adf423",
        "run/W_up/gR": "1681e16a5173b125c21bf5b34012e94ae7b778337be6c58923a59382c681703a",
        "run/W_up/gU0": "e026a5fa0d468a7db4764274eae1b3a7c7b49fb0f340d243c3512d01cc6edb16",
        "run/W_up/gU1": "b77b053836e7eff47f0dbc5f5e7be15b6d17a3d25e79cba5b4e1153f8df0af05",
        "run/W_up/gU2": "aa8caee01aa05d44c152c53b9c6fc095881572f97ea73b35a8d4ac67180076b1",
        "run/W_up/gU3": "99dd015e7b1ce46512f75f1d5165a0199cb09900b5e1e70b7f4af69c17fbe25e"
      },
      "shape": [
        1024,
        256
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.346880982415314e-08,
          "E_inf": 1.0001942030157807e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 13612,
          "max_rel_old": 0.00042698546894825995,
          "max_rel_old_flat_index": 114530,
          "norm_gR_inf": 76.27912902832031,
          "norm_gR_l2": 4129.11181640625,
          "norm_sum_gU_inf": 76.27912902832031,
          "shape": [
            1024,
            256
          ]
        },
        "S_fp64": {
          "E_L2": 3.6650101956987336e-08,
          "E_inf": 5.000971126080546e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 54564,
          "max_rel_old": 0.0004269854824935952,
          "max_rel_old_flat_index": 114530,
          "norm_gR_inf": 76.27912902832031,
          "norm_gR_l2": 4129.111778587698,
          "norm_sum_gU_inf": 76.27912998199463,
          "shape": [
            1024,
            256
          ]
        },
        "S_pairwise": {
          "E_L2": 4.3350770084771284e-08,
          "E_inf": 5.000971015078903e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 4289,
          "max_rel_old": 0.00042698546894825995,
          "max_rel_old_flat_index": 114530,
          "norm_gR_inf": 76.27912902832031,
          "norm_gR_l2": 4129.11181640625,
          "norm_sum_gU_inf": 76.27912902832031,
          "shape": [
            1024,
            256
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 76.27912902832031,
          "norm_gR_l2": 4129.11181640625,
          "norm_sum_gU_inf": 76.27912902832031,
          "shape": [
            1024,
            256
          ]
        },
        "S_stack": {
          "E_L2": 5.346880982415314e-08,
          "E_inf": 1.0001942030157807e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 13612,
          "max_rel_old": 0.00042698546894825995,
          "max_rel_old_flat_index": 114530,
          "norm_gR_inf": 76.27912902832031,
          "norm_gR_l2": 4129.11181640625,
          "norm_sum_gU_inf": 76.27912902832031,
          "shape": [
            1024,
            256
          ]
        }
      },
      "tensor_metadata": {
        "run/W_up/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "e722b653e7a26a7fe4479578fd73ef8add94946080e1cdb80d7b260fc5331b9f",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "2048f71a900cfc0b67c47cd68eafd0f1e4bd1fdf106a196270129e2327d63e3f",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "caf732cc851c34c5855b558c297d68352578d3d420bba8df2b7b5a121fb7cd37",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "6b933569530c2a7079725e45289c8bc4bdf8bbc34fab2abf86ba58a81a08d0ff",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "a315db3b5a8ee594ea7d4cb6e960ba4af03d5f19885a97570a99761db9adf423",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "1681e16a5173b125c21bf5b34012e94ae7b778337be6c58923a59382c681703a",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "e026a5fa0d468a7db4764274eae1b3a7c7b49fb0f340d243c3512d01cc6edb16",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "b77b053836e7eff47f0dbc5f5e7be15b6d17a3d25e79cba5b4e1153f8df0af05",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "aa8caee01aa05d44c152c53b9c6fc095881572f97ea73b35a8d4ac67180076b1",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "99dd015e7b1ce46512f75f1d5165a0199cb09900b5e1e70b7f4af69c17fbe25e",
          "shape": [
            1024,
            256
          ]
        }
      }
    }
  },
  "run_02": {
    "W_K": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 7.62939453125e-06,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 18829,
          "floor_1e_6_active": false,
          "gR_value": -38.861732482910156,
          "multi_index_row_major": [
            73,
            141
          ],
          "old_denominator": 38.861732482910156,
          "old_relative_error": 1.9632152259418945e-07,
          "sum_gU_old_value": -38.861724853515625,
          "ulp_gR": 3.814697265625e-06,
          "ulp_sum_gU_old": 3.814697265625e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 1.1920928955078125e-07,
          "error_over_ulp_gR": 32768.0,
          "error_over_ulp_sum_gU_old": 32768.0,
          "flat_index": 44868,
          "floor_1e_6_active": false,
          "gR_value": 3.4332275390625e-05,
          "multi_index_row_major": [
            175,
            68
          ],
          "old_denominator": 3.4332275390625e-05,
          "old_relative_error": 0.0034722222480922937,
          "sum_gU_old_value": 3.421306610107422e-05,
          "ulp_gR": 3.637978807091713e-12,
          "ulp_sum_gU_old": 3.637978807091713e-12
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 2.0,
          "norm_S_stack_inf": 52.309181213378906,
          "norm_gR_inf": 52.309181213378906,
          "norm_gR_l2": 2728.10595703125,
          "ulp_norm_S_stack_inf": 3.814697265625e-06
        }
      },
      "raw_tensor_sha256": {
        "run/W_K/S_forward": "2f3574be712e52be8f7ecc506816381572eb4e696c1953c0b0cf8097a2f39120",
        "run/W_K/S_fp64": "f3e1e3d0c2379e6b8130c997b42a19c73296c2ff77ac031f69f07af58b966f59",
        "run/W_K/S_pairwise": "2fb1444bb5b00272c72d9366e2a84075db00fd9104841dc8f19dbd99c0b91612",
        "run/W_K/S_reverse": "557479982d1f2be88040fe87055876fa52615452bf86cb572ea5f47c5ed1b478",
        "run/W_K/S_stack": "6d4521b9de797a9a0e18d4daacfcf57c32639b1617dc5866f7c0ba5570256cab",
        "run/W_K/gR": "a5513e80e892755cf4293ef84f75557a61a2a985b611a9658ab2763a4998a3ae",
        "run/W_K/gU0": "a5eb2b0257ef9a68de05d837385029b57ac59600e365f076e4e527f1b95bede0",
        "run/W_K/gU1": "cd482400b12c468c70538094c26ba3715aee4918089bb65104604d23839eecfd",
        "run/W_K/gU2": "e0bc54d4e69b428c0a3fd848faae5dadb71553de6dd31b59784f82fddedd892c",
        "run/W_K/gU3": "aa1a10be044f1b3f4cfd7f641b72fa6d6a38352a8c2db6eccd0e8a8bce2f863d"
      },
      "shape": [
        256,
        256
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.370466382714767e-08,
          "E_inf": 1.458519136576797e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 18829,
          "max_rel_old": 0.0034722222480922937,
          "max_rel_old_flat_index": 44868,
          "norm_gR_inf": 52.309181213378906,
          "norm_gR_l2": 2728.10595703125,
          "norm_sum_gU_inf": 52.309181213378906,
          "shape": [
            256,
            256
          ]
        },
        "S_fp64": {
          "E_L2": 3.388175308859245e-08,
          "E_inf": 5.925234263708148e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.0994415283203125e-06,
          "max_abs_flat_index": 52113,
          "max_rel_old": 0.003472222222222222,
          "max_rel_old_flat_index": 44868,
          "norm_gR_inf": 52.309181213378906,
          "norm_gR_l2": 2728.1059185615673,
          "norm_sum_gU_inf": 52.30918025970459,
          "shape": [
            256,
            256
          ]
        },
        "S_pairwise": {
          "E_L2": 4.418883747803193e-08,
          "E_inf": 7.292595682883984e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 1540,
          "max_rel_old": 0.00022883295605424792,
          "max_rel_old_flat_index": 696,
          "norm_gR_inf": 52.309181213378906,
          "norm_gR_l2": 2728.10595703125,
          "norm_sum_gU_inf": 52.309181213378906,
          "shape": [
            256,
            256
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 52.309181213378906,
          "norm_gR_l2": 2728.10595703125,
          "norm_sum_gU_inf": 52.309181213378906,
          "shape": [
            256,
            256
          ]
        },
        "S_stack": {
          "E_L2": 5.370466382714767e-08,
          "E_inf": 1.458519136576797e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 18829,
          "max_rel_old": 0.0034722222480922937,
          "max_rel_old_flat_index": 44868,
          "norm_gR_inf": 52.309181213378906,
          "norm_gR_l2": 2728.10595703125,
          "norm_sum_gU_inf": 52.309181213378906,
          "shape": [
            256,
            256
          ]
        }
      },
      "tensor_metadata": {
        "run/W_K/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "2f3574be712e52be8f7ecc506816381572eb4e696c1953c0b0cf8097a2f39120",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "f3e1e3d0c2379e6b8130c997b42a19c73296c2ff77ac031f69f07af58b966f59",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "2fb1444bb5b00272c72d9366e2a84075db00fd9104841dc8f19dbd99c0b91612",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "557479982d1f2be88040fe87055876fa52615452bf86cb572ea5f47c5ed1b478",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "6d4521b9de797a9a0e18d4daacfcf57c32639b1617dc5866f7c0ba5570256cab",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "a5513e80e892755cf4293ef84f75557a61a2a985b611a9658ab2763a4998a3ae",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "a5eb2b0257ef9a68de05d837385029b57ac59600e365f076e4e527f1b95bede0",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "cd482400b12c468c70538094c26ba3715aee4918089bb65104604d23839eecfd",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "e0bc54d4e69b428c0a3fd848faae5dadb71553de6dd31b59784f82fddedd892c",
          "shape": [
            256,
            256
          ]
        },
        "run/W_K/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "aa1a10be044f1b3f4cfd7f641b72fa6d6a38352a8c2db6eccd0e8a8bce2f863d",
          "shape": [
            256,
            256
          ]
        }
      }
    },
    "W_O": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 1.52587890625e-05,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 36257,
          "floor_1e_6_active": false,
          "gR_value": 79.351806640625,
          "multi_index_row_major": [
            141,
            161
          ],
          "old_denominator": 79.351806640625,
          "old_relative_error": 1.9229290160183155e-07,
          "sum_gU_old_value": 79.35179138183594,
          "ulp_gR": 7.62939453125e-06,
          "ulp_sum_gU_old": 7.62939453125e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 4.76837158203125e-07,
          "error_over_ulp_gR": 1024.0,
          "error_over_ulp_sum_gU_old": 1024.0,
          "flat_index": 17093,
          "floor_1e_6_active": false,
          "gR_value": 0.0039310455322265625,
          "multi_index_row_major": [
            66,
            197
          ],
          "old_denominator": 0.0039310455322265625,
          "old_relative_error": 0.00012130034156143665,
          "sum_gU_old_value": 0.003930568695068359,
          "ulp_gR": 4.656612873077393e-10,
          "ulp_sum_gU_old": 4.656612873077393e-10
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 2.0,
          "norm_S_stack_inf": 124.8612060546875,
          "norm_gR_inf": 124.86121368408203,
          "norm_gR_l2": 6258.7431640625,
          "ulp_norm_S_stack_inf": 7.62939453125e-06
        }
      },
      "raw_tensor_sha256": {
        "run/W_O/S_forward": "3c49753c6b1e6c9ece0906495fe3cd0dfc2f42a1d0ab11910605f29f9b279b6d",
        "run/W_O/S_fp64": "e35e23789ac7b0a028d31d4a42f12945917cda2fc87dc440402768e7418dae9f",
        "run/W_O/S_pairwise": "e4267e78c29f3e4e625255a64c6845910817d6d0ae85386a1e265d9c371d7801",
        "run/W_O/S_reverse": "efe6239eb413fe1147361bcbcda6fd7c06a623a60e2ac4a21d31b4f2b34795ca",
        "run/W_O/S_stack": "28ee8cffd6c5882127b47e9a1999e82cba582f4474a6431faf5af400f7e083ba",
        "run/W_O/gR": "aebdcd338b3b17a4915adc1edc239fd76af92b79764743ea3e502b93e8d7ecf8",
        "run/W_O/gU0": "86c7ac601d6127fae5e98ffbbf1b6e26ec09469c30e5ee03dc91ef2ca2f823d3",
        "run/W_O/gU1": "698ecf7dee3e36ccc54122032a640df82e72ad982cd8870f3f2161e79b5b4734",
        "run/W_O/gU2": "03982d511cf2a8e283afe0638452641b6128fe6c687f9039e6f4051f48e4cf9d",
        "run/W_O/gU3": "70d5bbc92e7fcb16ff366dc76b1f954a76babb7f7415f273176fa136f5700d93"
      },
      "shape": [
        256,
        256
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.3697743140901366e-08,
          "E_inf": 1.2220600353884947e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 1.52587890625e-05,
          "max_abs_flat_index": 36257,
          "max_rel_old": 0.00012130034156143665,
          "max_rel_old_flat_index": 17093,
          "norm_gR_inf": 124.86121368408203,
          "norm_gR_l2": 6258.7431640625,
          "norm_sum_gU_inf": 124.8612060546875,
          "shape": [
            256,
            256
          ]
        },
        "S_fp64": {
          "E_L2": 3.7214034681009397e-08,
          "E_inf": 6.110299833023837e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 11704,
          "max_rel_old": 0.00012130033964095099,
          "max_rel_old_flat_index": 17093,
          "norm_gR_inf": 124.86121368408203,
          "norm_gR_l2": 6258.742908932774,
          "norm_sum_gU_inf": 124.86121368408203,
          "shape": [
            256,
            256
          ]
        },
        "S_pairwise": {
          "E_L2": 4.322798830003194e-08,
          "E_inf": 6.110299466399738e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 324,
          "max_rel_old": 5.582226367550902e-05,
          "max_rel_old_flat_index": 58698,
          "norm_gR_inf": 124.86121368408203,
          "norm_gR_l2": 6258.7431640625,
          "norm_sum_gU_inf": 124.86122131347656,
          "shape": [
            256,
            256
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 124.86121368408203,
          "norm_gR_l2": 6258.7431640625,
          "norm_sum_gU_inf": 124.86121368408203,
          "shape": [
            256,
            256
          ]
        },
        "S_stack": {
          "E_L2": 5.3697743140901366e-08,
          "E_inf": 1.2220600353884947e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 1.52587890625e-05,
          "max_abs_flat_index": 36257,
          "max_rel_old": 0.00012130034156143665,
          "max_rel_old_flat_index": 17093,
          "norm_gR_inf": 124.86121368408203,
          "norm_gR_l2": 6258.7431640625,
          "norm_sum_gU_inf": 124.8612060546875,
          "shape": [
            256,
            256
          ]
        }
      },
      "tensor_metadata": {
        "run/W_O/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "3c49753c6b1e6c9ece0906495fe3cd0dfc2f42a1d0ab11910605f29f9b279b6d",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "e35e23789ac7b0a028d31d4a42f12945917cda2fc87dc440402768e7418dae9f",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "e4267e78c29f3e4e625255a64c6845910817d6d0ae85386a1e265d9c371d7801",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "efe6239eb413fe1147361bcbcda6fd7c06a623a60e2ac4a21d31b4f2b34795ca",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "28ee8cffd6c5882127b47e9a1999e82cba582f4474a6431faf5af400f7e083ba",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "aebdcd338b3b17a4915adc1edc239fd76af92b79764743ea3e502b93e8d7ecf8",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "86c7ac601d6127fae5e98ffbbf1b6e26ec09469c30e5ee03dc91ef2ca2f823d3",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "698ecf7dee3e36ccc54122032a640df82e72ad982cd8870f3f2161e79b5b4734",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "03982d511cf2a8e283afe0638452641b6128fe6c687f9039e6f4051f48e4cf9d",
          "shape": [
            256,
            256
          ]
        },
        "run/W_O/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "70d5bbc92e7fcb16ff366dc76b1f954a76babb7f7415f273176fa136f5700d93",
          "shape": [
            256,
            256
          ]
        }
      }
    },
    "W_Q": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 7.62939453125e-06,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 3378,
          "floor_1e_6_active": false,
          "gR_value": -37.437400817871094,
          "multi_index_row_major": [
            13,
            50
          ],
          "old_denominator": 37.437408447265625,
          "old_relative_error": 2.0379066256737133e-07,
          "sum_gU_old_value": -37.437408447265625,
          "ulp_gR": 3.814697265625e-06,
          "ulp_sum_gU_old": 3.814697265625e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 1.1920928955078125e-07,
          "error_over_ulp_gR": 4096.0,
          "error_over_ulp_sum_gU_old": 4096.0,
          "flat_index": 58051,
          "floor_1e_6_active": false,
          "gR_value": -0.0003300905227661133,
          "multi_index_row_major": [
            226,
            195
          ],
          "old_denominator": 0.00033020973205566406,
          "old_relative_error": 0.00036101083969697356,
          "sum_gU_old_value": -0.00033020973205566406,
          "ulp_gR": 2.9103830456733704e-11,
          "ulp_sum_gU_old": 2.9103830456733704e-11
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 2.0,
          "norm_S_stack_inf": 52.54109573364258,
          "norm_gR_inf": 52.54109191894531,
          "norm_gR_l2": 2769.7294921875,
          "ulp_norm_S_stack_inf": 3.814697265625e-06
        }
      },
      "raw_tensor_sha256": {
        "run/W_Q/S_forward": "a55303d7b27c7dd2d40f6f159f7a19e1f3ff9674ee1cd9276b7fde9ffdf3a359",
        "run/W_Q/S_fp64": "5d6aa547684c262064d9d6aa0444f60ebdfab75a4df43a72505f84b62b1dd4d5",
        "run/W_Q/S_pairwise": "c5b994fc621aefa0cd94134709fd9eb47b8200794bb81b019de4dbb1584448f8",
        "run/W_Q/S_reverse": "d197ab3428f1eb8c5cf5976198f9fdd922a5779ef0c2a76b5983c19f8a5802ec",
        "run/W_Q/S_stack": "d9ecab578cec499bd95f3597413873f4ff614e9f7a3b94a2d7dae17e53a6fcd4",
        "run/W_Q/gR": "76d393a4c29ddaf04e22539b74ec97e477b40e8e7265a5c11d19742378f79bd3",
        "run/W_Q/gU0": "0f7b7a4554cacc24d265ffb7bfced268fcc8bd895bad17ea7e83f91f4e7978ea",
        "run/W_Q/gU1": "841a5c0f140c3082656c65b4ea9de72da40b60dbdb53d36b954ee3804dd086fe",
        "run/W_Q/gU2": "b3b85ce6a3c1c4937883f5f8520d4888c942a30fa0a8ac26a6a332cfe5ced400",
        "run/W_Q/gU3": "98614ef2897ec5a9c2b0b997e4ccebebdb739ff8181bb79371f2398980655300"
      },
      "shape": [
        256,
        256
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.372153566440829e-08,
          "E_inf": 1.452081335173716e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 3378,
          "max_rel_old": 0.00036101083969697356,
          "max_rel_old_flat_index": 58051,
          "norm_gR_inf": 52.54109191894531,
          "norm_gR_l2": 2769.7294921875,
          "norm_sum_gU_inf": 52.54109573364258,
          "shape": [
            256,
            256
          ]
        },
        "S_fp64": {
          "E_L2": 3.415794472797588e-08,
          "E_inf": 6.352856115464177e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.337860107421875e-06,
          "max_abs_flat_index": 4106,
          "max_rel_old": 0.00011000696710791684,
          "max_rel_old_flat_index": 4346,
          "norm_gR_inf": 52.54109191894531,
          "norm_gR_l2": 2769.729576019558,
          "norm_sum_gU_inf": 52.541094064712524,
          "shape": [
            256,
            256
          ]
        },
        "S_pairwise": {
          "E_L2": 4.3970409535631916e-08,
          "E_inf": 7.260407386411316e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 1470,
          "max_rel_old": 0.00036101083969697356,
          "max_rel_old_flat_index": 58051,
          "norm_gR_inf": 52.54109191894531,
          "norm_gR_l2": 2769.7294921875,
          "norm_sum_gU_inf": 52.54109191894531,
          "shape": [
            256,
            256
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 52.54109191894531,
          "norm_gR_l2": 2769.7294921875,
          "norm_sum_gU_inf": 52.54109191894531,
          "shape": [
            256,
            256
          ]
        },
        "S_stack": {
          "E_L2": 5.372153566440829e-08,
          "E_inf": 1.452081335173716e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 3378,
          "max_rel_old": 0.00036101083969697356,
          "max_rel_old_flat_index": 58051,
          "norm_gR_inf": 52.54109191894531,
          "norm_gR_l2": 2769.7294921875,
          "norm_sum_gU_inf": 52.54109573364258,
          "shape": [
            256,
            256
          ]
        }
      },
      "tensor_metadata": {
        "run/W_Q/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "a55303d7b27c7dd2d40f6f159f7a19e1f3ff9674ee1cd9276b7fde9ffdf3a359",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "5d6aa547684c262064d9d6aa0444f60ebdfab75a4df43a72505f84b62b1dd4d5",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "c5b994fc621aefa0cd94134709fd9eb47b8200794bb81b019de4dbb1584448f8",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "d197ab3428f1eb8c5cf5976198f9fdd922a5779ef0c2a76b5983c19f8a5802ec",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "d9ecab578cec499bd95f3597413873f4ff614e9f7a3b94a2d7dae17e53a6fcd4",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "76d393a4c29ddaf04e22539b74ec97e477b40e8e7265a5c11d19742378f79bd3",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "0f7b7a4554cacc24d265ffb7bfced268fcc8bd895bad17ea7e83f91f4e7978ea",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "841a5c0f140c3082656c65b4ea9de72da40b60dbdb53d36b954ee3804dd086fe",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "b3b85ce6a3c1c4937883f5f8520d4888c942a30fa0a8ac26a6a332cfe5ced400",
          "shape": [
            256,
            256
          ]
        },
        "run/W_Q/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "98614ef2897ec5a9c2b0b997e4ccebebdb739ff8181bb79371f2398980655300",
          "shape": [
            256,
            256
          ]
        }
      }
    },
    "W_V": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 1.52587890625e-05,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 6109,
          "floor_1e_6_active": false,
          "gR_value": -99.20011901855469,
          "multi_index_row_major": [
            23,
            221
          ],
          "old_denominator": 99.20011901855469,
          "old_relative_error": 1.5381824880478234e-07,
          "sum_gU_old_value": -99.20010375976562,
          "ulp_gR": 7.62939453125e-06,
          "ulp_sum_gU_old": 7.62939453125e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 1.1920928955078125e-07,
          "error_over_ulp_gR": 2048.0,
          "error_over_ulp_sum_gU_old": 2048.0,
          "flat_index": 14342,
          "floor_1e_6_active": false,
          "gR_value": -0.0005128383636474609,
          "multi_index_row_major": [
            56,
            6
          ],
          "old_denominator": 0.0005129575729370117,
          "old_relative_error": 0.00023239600704982877,
          "sum_gU_old_value": -0.0005129575729370117,
          "ulp_gR": 5.820766091346741e-11,
          "ulp_sum_gU_old": 5.820766091346741e-11
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 1.0,
          "norm_S_stack_inf": 128.3258514404297,
          "norm_gR_inf": 128.3258514404297,
          "norm_gR_l2": 6239.6865234375,
          "ulp_norm_S_stack_inf": 1.52587890625e-05
        }
      },
      "raw_tensor_sha256": {
        "run/W_V/S_forward": "16dc942221e0e748e3993a9679bc63816848093d76209a7b5117f0081bb8d2d2",
        "run/W_V/S_fp64": "ca657c6c5514668f7c5152af5e6d99c4e302b4d7b777fbc41b9b15e0f46b694c",
        "run/W_V/S_pairwise": "74a1838bbd1fe7d13b6acd3f2b7b940743e3104ac233bd7897c77fa1b6714e82",
        "run/W_V/S_reverse": "1a33021c7ab2f5a5d90edcd795027908819cef02058f689036633ca27d262839",
        "run/W_V/S_stack": "08e50a098580b820dc46f3d07e3337e6a70ed7da0fab217984253b46fe9602b1",
        "run/W_V/gR": "0b4bebfeeabde03eb09ad1945d9333d339e83dfe06143237c3b719795410ef4a",
        "run/W_V/gU0": "4df24b4da4b0a2caff499d5599ef95b066c708a5e2cd535f8e62ca51bf6de4d2",
        "run/W_V/gU1": "a105de63566b09f13b1e1783a9881e313bcbc7e8d16a46ff164e26d72b897498",
        "run/W_V/gU2": "5eee0635e9a76e1343ac323ac9c62489793fecdf22f3de3ccfdd820042659534",
        "run/W_V/gU3": "8e0df6a8c9576702d52397f68ebf5b8439539ae2b0ab6e9dfb42f8c9a735077d"
      },
      "shape": [
        256,
        256
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.404850611512302e-08,
          "E_inf": 1.1890658413449273e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 1.52587890625e-05,
          "max_abs_flat_index": 6109,
          "max_rel_old": 0.00023239600704982877,
          "max_rel_old_flat_index": 14342,
          "norm_gR_inf": 128.3258514404297,
          "norm_gR_l2": 6239.6865234375,
          "norm_sum_gU_inf": 128.3258514404297,
          "shape": [
            256,
            256
          ]
        },
        "S_fp64": {
          "E_L2": 3.747180079402311e-08,
          "E_inf": 6.688495537970857e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 8.58306884765625e-06,
          "max_abs_flat_index": 16798,
          "max_rel_old": 0.00011621150493898897,
          "max_rel_old_flat_index": 14342,
          "norm_gR_inf": 128.3258514404297,
          "norm_gR_l2": 6239.686415141664,
          "norm_sum_gU_inf": 128.32584810256958,
          "shape": [
            256,
            256
          ]
        },
        "S_pairwise": {
          "E_L2": 4.386166096992383e-08,
          "E_inf": 1.1890658413449273e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 1.52587890625e-05,
          "max_abs_flat_index": 38372,
          "max_rel_old": 8.73973112902604e-05,
          "max_rel_old_flat_index": 50103,
          "norm_gR_inf": 128.3258514404297,
          "norm_gR_l2": 6239.6865234375,
          "norm_sum_gU_inf": 128.3258514404297,
          "shape": [
            256,
            256
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 128.3258514404297,
          "norm_gR_l2": 6239.6865234375,
          "norm_sum_gU_inf": 128.3258514404297,
          "shape": [
            256,
            256
          ]
        },
        "S_stack": {
          "E_L2": 5.404850611512302e-08,
          "E_inf": 1.1890658413449273e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 1.52587890625e-05,
          "max_abs_flat_index": 6109,
          "max_rel_old": 0.00023239600704982877,
          "max_rel_old_flat_index": 14342,
          "norm_gR_inf": 128.3258514404297,
          "norm_gR_l2": 6239.6865234375,
          "norm_sum_gU_inf": 128.3258514404297,
          "shape": [
            256,
            256
          ]
        }
      },
      "tensor_metadata": {
        "run/W_V/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "16dc942221e0e748e3993a9679bc63816848093d76209a7b5117f0081bb8d2d2",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "ca657c6c5514668f7c5152af5e6d99c4e302b4d7b777fbc41b9b15e0f46b694c",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "74a1838bbd1fe7d13b6acd3f2b7b940743e3104ac233bd7897c77fa1b6714e82",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "1a33021c7ab2f5a5d90edcd795027908819cef02058f689036633ca27d262839",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "08e50a098580b820dc46f3d07e3337e6a70ed7da0fab217984253b46fe9602b1",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "0b4bebfeeabde03eb09ad1945d9333d339e83dfe06143237c3b719795410ef4a",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "4df24b4da4b0a2caff499d5599ef95b066c708a5e2cd535f8e62ca51bf6de4d2",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "a105de63566b09f13b1e1783a9881e313bcbc7e8d16a46ff164e26d72b897498",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "5eee0635e9a76e1343ac323ac9c62489793fecdf22f3de3ccfdd820042659534",
          "shape": [
            256,
            256
          ]
        },
        "run/W_V/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "8e0df6a8c9576702d52397f68ebf5b8439539ae2b0ab6e9dfb42f8c9a735077d",
          "shape": [
            256,
            256
          ]
        }
      }
    },
    "W_down": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 7.62939453125e-06,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 140001,
          "floor_1e_6_active": false,
          "gR_value": 42.20118713378906,
          "multi_index_row_major": [
            136,
            737
          ],
          "old_denominator": 42.201194763183594,
          "old_relative_error": 1.8078621621953062e-07,
          "sum_gU_old_value": 42.201194763183594,
          "ulp_gR": 3.814697265625e-06,
          "ulp_sum_gU_old": 3.814697265625e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 2.384185791015625e-07,
          "error_over_ulp_gR": 32768.0,
          "error_over_ulp_sum_gU_old": 32768.0,
          "flat_index": 158001,
          "floor_1e_6_active": false,
          "gR_value": -7.963180541992188e-05,
          "multi_index_row_major": [
            154,
            305
          ],
          "old_denominator": 7.963180541992188e-05,
          "old_relative_error": 0.002994012087583542,
          "sum_gU_old_value": -7.939338684082031e-05,
          "ulp_gR": 7.275957614183426e-12,
          "ulp_sum_gU_old": 7.275957614183426e-12
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 2.0,
          "norm_S_stack_inf": 60.65041732788086,
          "norm_gR_inf": 60.65041732788086,
          "norm_gR_l2": 4074.2841796875,
          "ulp_norm_S_stack_inf": 3.814697265625e-06
        }
      },
      "raw_tensor_sha256": {
        "run/W_down/S_forward": "ace6fb3e36bb9418773f782f5a30cd5b9a2da97b47588807513f03933de7415c",
        "run/W_down/S_fp64": "a139715d974c33898a5e1a48f829c88d0c74f3123ece41a92b8d3af58c154a46",
        "run/W_down/S_pairwise": "5d0ff22439856349849c4f8ed41b6ef27a48f9d8bfed96ca6421d3b1af8eafcc",
        "run/W_down/S_reverse": "fde3f04b4f3a3fb4b3b9cac5e93092edb21d773d4253363bac55e0c851fbc3e2",
        "run/W_down/S_stack": "bc1b60c8fde4100ed5e545c4992cbbba97bc85a79af39cd042f85a539bec1b18",
        "run/W_down/gR": "2ee294919436fcf0ebebe2317358e61d90a70a42515eb90e5b5240605071146d",
        "run/W_down/gU0": "dcbe910c19b5ad888b41dc2415db6f4396690145d334a41fba8aca6f0a607cab",
        "run/W_down/gU1": "fd9f8b9ec16e2803fe8a88d674b5e14f8514d965e1c4ed9629790b9bc8727b21",
        "run/W_down/gU2": "e95e9e4e6bd950e7e855d5f19b4cdc2ccb523fec3233d884a045e34697647ebf",
        "run/W_down/gU3": "7377d94dfd3c64d2347959a2a535c25e71801bd6e524df19365a9636d1b3b4f8"
      },
      "shape": [
        256,
        1024
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.3162100499548615e-08,
          "E_inf": 1.2579293695580418e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 140001,
          "max_rel_old": 0.002994012087583542,
          "max_rel_old_flat_index": 158001,
          "norm_gR_inf": 60.65041732788086,
          "norm_gR_l2": 4074.2841796875,
          "norm_sum_gU_inf": 60.65041732788086,
          "shape": [
            256,
            1024
          ]
        },
        "S_fp64": {
          "E_L2": 3.656333379082823e-08,
          "E_inf": 6.289646986329798e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 140001,
          "max_rel_old": 0.002877058383233533,
          "max_rel_old_flat_index": 158001,
          "norm_gR_inf": 60.65041732788086,
          "norm_gR_l2": 4074.2843031300195,
          "norm_sum_gU_inf": 60.65041923522949,
          "shape": [
            256,
            1024
          ]
        },
        "S_pairwise": {
          "E_L2": 4.302838974012957e-08,
          "E_inf": 6.289646847790209e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 5547,
          "max_rel_old": 0.00011134617670904845,
          "max_rel_old_flat_index": 3474,
          "norm_gR_inf": 60.65041732788086,
          "norm_gR_l2": 4074.2841796875,
          "norm_sum_gU_inf": 60.65041732788086,
          "shape": [
            256,
            1024
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 60.65041732788086,
          "norm_gR_l2": 4074.2841796875,
          "norm_sum_gU_inf": 60.65041732788086,
          "shape": [
            256,
            1024
          ]
        },
        "S_stack": {
          "E_L2": 5.3162100499548615e-08,
          "E_inf": 1.2579293695580418e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 140001,
          "max_rel_old": 0.002994012087583542,
          "max_rel_old_flat_index": 158001,
          "norm_gR_inf": 60.65041732788086,
          "norm_gR_l2": 4074.2841796875,
          "norm_sum_gU_inf": 60.65041732788086,
          "shape": [
            256,
            1024
          ]
        }
      },
      "tensor_metadata": {
        "run/W_down/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "ace6fb3e36bb9418773f782f5a30cd5b9a2da97b47588807513f03933de7415c",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "a139715d974c33898a5e1a48f829c88d0c74f3123ece41a92b8d3af58c154a46",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "5d0ff22439856349849c4f8ed41b6ef27a48f9d8bfed96ca6421d3b1af8eafcc",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "fde3f04b4f3a3fb4b3b9cac5e93092edb21d773d4253363bac55e0c851fbc3e2",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "bc1b60c8fde4100ed5e545c4992cbbba97bc85a79af39cd042f85a539bec1b18",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "2ee294919436fcf0ebebe2317358e61d90a70a42515eb90e5b5240605071146d",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "dcbe910c19b5ad888b41dc2415db6f4396690145d334a41fba8aca6f0a607cab",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "fd9f8b9ec16e2803fe8a88d674b5e14f8514d965e1c4ed9629790b9bc8727b21",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "e95e9e4e6bd950e7e855d5f19b4cdc2ccb523fec3233d884a045e34697647ebf",
          "shape": [
            256,
            1024
          ]
        },
        "run/W_down/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "7377d94dfd3c64d2347959a2a535c25e71801bd6e524df19365a9636d1b3b4f8",
          "shape": [
            256,
            1024
          ]
        }
      }
    },
    "W_gate": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 7.62939453125e-06,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 141417,
          "floor_1e_6_active": false,
          "gR_value": -46.208038330078125,
          "multi_index_row_major": [
            552,
            105
          ],
          "old_denominator": 46.208045959472656,
          "old_relative_error": 1.6510965394900268e-07,
          "sum_gU_old_value": -46.208045959472656,
          "ulp_gR": 3.814697265625e-06,
          "ulp_sum_gU_old": 3.814697265625e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 2.384185791015625e-07,
          "error_over_ulp_gR": 65536.0,
          "error_over_ulp_sum_gU_old": 65536.0,
          "flat_index": 11995,
          "floor_1e_6_active": false,
          "gR_value": 3.9577484130859375e-05,
          "multi_index_row_major": [
            46,
            219
          ],
          "old_denominator": 3.981590270996094e-05,
          "old_relative_error": 0.005988024175167084,
          "sum_gU_old_value": 3.981590270996094e-05,
          "ulp_gR": 3.637978807091713e-12,
          "ulp_sum_gU_old": 3.637978807091713e-12
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 2.0,
          "norm_S_stack_inf": 55.499542236328125,
          "norm_gR_inf": 55.499542236328125,
          "norm_gR_l2": 4135.73779296875,
          "ulp_norm_S_stack_inf": 3.814697265625e-06
        }
      },
      "raw_tensor_sha256": {
        "run/W_gate/S_forward": "eee5e4b0122f1c347b9c7e261414a739fb95b2f53c581297017d13fc98de6259",
        "run/W_gate/S_fp64": "0acbd855b2fcab4c8652c649a00b09237f85371a77700682616f01cd0bd30797",
        "run/W_gate/S_pairwise": "29007fc31e68e255615fedb2397822ea45fa51b4ca40bd7a5f78cd1b8580ff8b",
        "run/W_gate/S_reverse": "6b98f1c83849e19de84776f3e97df2ae08c1a68e1f54e5ca5fac6dc6bcca215a",
        "run/W_gate/S_stack": "6bc52b71e030ff476e0ac8fec74cd278bd09f5dff2faf5c7b2aa7ac4488a9930",
        "run/W_gate/gR": "aeecf834df66cac799c8a93cf57885ce284ffddbe1c5b4cb29c9ee918fbf1b44",
        "run/W_gate/gU0": "3459a032b4d48c2d03fc91b928cebac34d921c6572f7e721a949f74128347fc4",
        "run/W_gate/gU1": "66b94defeab4d18ad3fa75814bae7d07731af69c30cdf4b08a217e48808a653c",
        "run/W_gate/gU2": "8f7ef0a0ccd772c09f189e0f69b4ce90253e476c4f496ced16e4ddf62587a0f2",
        "run/W_gate/gU3": "93dc06c20e361ca23753f0f54fe6f77b36f91175d49b2dfae6b95cb8af6aff26"
      },
      "shape": [
        1024,
        256
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.3564853885745833e-08,
          "E_inf": 1.3746770832767652e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 141417,
          "max_rel_old": 0.005988024175167084,
          "max_rel_old_flat_index": 11995,
          "norm_gR_inf": 55.499542236328125,
          "norm_gR_l2": 4135.73779296875,
          "norm_sum_gU_inf": 55.499542236328125,
          "shape": [
            1024,
            256
          ]
        },
        "S_fp64": {
          "E_L2": 3.671111470897815e-08,
          "E_inf": 6.873385098171184e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 36421,
          "max_rel_old": 0.0015037593984962407,
          "max_rel_old_flat_index": 11995,
          "norm_gR_inf": 55.499542236328125,
          "norm_gR_l2": 4135.737847335061,
          "norm_sum_gU_inf": 55.49954032897949,
          "shape": [
            1024,
            256
          ]
        },
        "S_pairwise": {
          "E_L2": 4.3360042667472953e-08,
          "E_inf": 6.873385416383826e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 5129,
          "max_rel_old": 0.0009900990407913923,
          "max_rel_old_flat_index": 170301,
          "norm_gR_inf": 55.499542236328125,
          "norm_gR_l2": 4135.73779296875,
          "norm_sum_gU_inf": 55.499542236328125,
          "shape": [
            1024,
            256
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 55.499542236328125,
          "norm_gR_l2": 4135.73779296875,
          "norm_sum_gU_inf": 55.499542236328125,
          "shape": [
            1024,
            256
          ]
        },
        "S_stack": {
          "E_L2": 5.3564853885745833e-08,
          "E_inf": 1.3746770832767652e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 141417,
          "max_rel_old": 0.005988024175167084,
          "max_rel_old_flat_index": 11995,
          "norm_gR_inf": 55.499542236328125,
          "norm_gR_l2": 4135.73779296875,
          "norm_sum_gU_inf": 55.499542236328125,
          "shape": [
            1024,
            256
          ]
        }
      },
      "tensor_metadata": {
        "run/W_gate/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "eee5e4b0122f1c347b9c7e261414a739fb95b2f53c581297017d13fc98de6259",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "0acbd855b2fcab4c8652c649a00b09237f85371a77700682616f01cd0bd30797",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "29007fc31e68e255615fedb2397822ea45fa51b4ca40bd7a5f78cd1b8580ff8b",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "6b98f1c83849e19de84776f3e97df2ae08c1a68e1f54e5ca5fac6dc6bcca215a",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "6bc52b71e030ff476e0ac8fec74cd278bd09f5dff2faf5c7b2aa7ac4488a9930",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "aeecf834df66cac799c8a93cf57885ce284ffddbe1c5b4cb29c9ee918fbf1b44",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "3459a032b4d48c2d03fc91b928cebac34d921c6572f7e721a949f74128347fc4",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "66b94defeab4d18ad3fa75814bae7d07731af69c30cdf4b08a217e48808a653c",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "8f7ef0a0ccd772c09f189e0f69b4ce90253e476c4f496ced16e4ddf62587a0f2",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_gate/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "93dc06c20e361ca23753f0f54fe6f77b36f91175d49b2dfae6b95cb8af6aff26",
          "shape": [
            1024,
            256
          ]
        }
      }
    },
    "W_up": {
      "dtype": "torch.float32",
      "old_comparator_argmax_details": {
        "argmax_max_abs": {
          "abs_error": 7.62939453125e-06,
          "error_over_ulp_gR": 2.0,
          "error_over_ulp_sum_gU_old": 2.0,
          "flat_index": 13612,
          "floor_1e_6_active": false,
          "gR_value": 62.38463592529297,
          "multi_index_row_major": [
            53,
            44
          ],
          "old_denominator": 62.38463592529297,
          "old_relative_error": 1.22296043514325e-07,
          "sum_gU_old_value": 62.38462829589844,
          "ulp_gR": 3.814697265625e-06,
          "ulp_sum_gU_old": 3.814697265625e-06
        },
        "argmax_max_rel_old": {
          "abs_error": 5.960464477539063e-08,
          "error_over_ulp_gR": 4096.0,
          "error_over_ulp_sum_gU_old": 4096.0,
          "flat_index": 114530,
          "floor_1e_6_active": false,
          "gR_value": -0.00013959407806396484,
          "multi_index_row_major": [
            447,
            98
          ],
          "old_denominator": 0.00013959407806396484,
          "old_relative_error": 0.00042698546894825995,
          "sum_gU_old_value": -0.00013953447341918945,
          "ulp_gR": 1.4551915228366852e-11,
          "ulp_sum_gU_old": 1.4551915228366852e-11
        },
        "tensor_magnitude_context": {
          "max_abs_over_ulp_norm_S_stack_inf": 1.0,
          "norm_S_stack_inf": 76.27912902832031,
          "norm_gR_inf": 76.27912902832031,
          "norm_gR_l2": 4129.11181640625,
          "ulp_norm_S_stack_inf": 7.62939453125e-06
        }
      },
      "raw_tensor_sha256": {
        "run/W_up/S_forward": "e722b653e7a26a7fe4479578fd73ef8add94946080e1cdb80d7b260fc5331b9f",
        "run/W_up/S_fp64": "2048f71a900cfc0b67c47cd68eafd0f1e4bd1fdf106a196270129e2327d63e3f",
        "run/W_up/S_pairwise": "caf732cc851c34c5855b558c297d68352578d3d420bba8df2b7b5a121fb7cd37",
        "run/W_up/S_reverse": "6b933569530c2a7079725e45289c8bc4bdf8bbc34fab2abf86ba58a81a08d0ff",
        "run/W_up/S_stack": "a315db3b5a8ee594ea7d4cb6e960ba4af03d5f19885a97570a99761db9adf423",
        "run/W_up/gR": "1681e16a5173b125c21bf5b34012e94ae7b778337be6c58923a59382c681703a",
        "run/W_up/gU0": "e026a5fa0d468a7db4764274eae1b3a7c7b49fb0f340d243c3512d01cc6edb16",
        "run/W_up/gU1": "b77b053836e7eff47f0dbc5f5e7be15b6d17a3d25e79cba5b4e1153f8df0af05",
        "run/W_up/gU2": "aa8caee01aa05d44c152c53b9c6fc095881572f97ea73b35a8d4ac67180076b1",
        "run/W_up/gU3": "99dd015e7b1ce46512f75f1d5165a0199cb09900b5e1e70b7f4af69c17fbe25e"
      },
      "shape": [
        1024,
        256
      ],
      "summation_metrics": {
        "S_forward": {
          "E_L2": 5.346880982415314e-08,
          "E_inf": 1.0001942030157807e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 13612,
          "max_rel_old": 0.00042698546894825995,
          "max_rel_old_flat_index": 114530,
          "norm_gR_inf": 76.27912902832031,
          "norm_gR_l2": 4129.11181640625,
          "norm_sum_gU_inf": 76.27912902832031,
          "shape": [
            1024,
            256
          ]
        },
        "S_fp64": {
          "E_L2": 3.6650101956987336e-08,
          "E_inf": 5.000971126080546e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float64",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 54564,
          "max_rel_old": 0.0004269854824935952,
          "max_rel_old_flat_index": 114530,
          "norm_gR_inf": 76.27912902832031,
          "norm_gR_l2": 4129.111778587698,
          "norm_sum_gU_inf": 76.27912998199463,
          "shape": [
            1024,
            256
          ]
        },
        "S_pairwise": {
          "E_L2": 4.3350770084771284e-08,
          "E_inf": 5.000971015078903e-08,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 3.814697265625e-06,
          "max_abs_flat_index": 4289,
          "max_rel_old": 0.00042698546894825995,
          "max_rel_old_flat_index": 114530,
          "norm_gR_inf": 76.27912902832031,
          "norm_gR_l2": 4129.11181640625,
          "norm_sum_gU_inf": 76.27912902832031,
          "shape": [
            1024,
            256
          ]
        },
        "S_reverse": {
          "E_L2": 0.0,
          "E_inf": 0.0,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 0.0,
          "max_abs_flat_index": 0,
          "max_rel_old": 0.0,
          "max_rel_old_flat_index": 0,
          "norm_gR_inf": 76.27912902832031,
          "norm_gR_l2": 4129.11181640625,
          "norm_sum_gU_inf": 76.27912902832031,
          "shape": [
            1024,
            256
          ]
        },
        "S_stack": {
          "E_L2": 5.346880982415314e-08,
          "E_inf": 1.0001942030157807e-07,
          "dtype_gR": "torch.float32",
          "dtype_sum_gU": "torch.float32",
          "floor_1e_6_active_at_max_rel_old": false,
          "max_abs": 7.62939453125e-06,
          "max_abs_flat_index": 13612,
          "max_rel_old": 0.00042698546894825995,
          "max_rel_old_flat_index": 114530,
          "norm_gR_inf": 76.27912902832031,
          "norm_gR_l2": 4129.11181640625,
          "norm_sum_gU_inf": 76.27912902832031,
          "shape": [
            1024,
            256
          ]
        }
      },
      "tensor_metadata": {
        "run/W_up/S_forward": {
          "dtype": "torch.float32",
          "raw_sha256": "e722b653e7a26a7fe4479578fd73ef8add94946080e1cdb80d7b260fc5331b9f",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/S_fp64": {
          "dtype": "torch.float64",
          "raw_sha256": "2048f71a900cfc0b67c47cd68eafd0f1e4bd1fdf106a196270129e2327d63e3f",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/S_pairwise": {
          "dtype": "torch.float32",
          "raw_sha256": "caf732cc851c34c5855b558c297d68352578d3d420bba8df2b7b5a121fb7cd37",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/S_reverse": {
          "dtype": "torch.float32",
          "raw_sha256": "6b933569530c2a7079725e45289c8bc4bdf8bbc34fab2abf86ba58a81a08d0ff",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/S_stack": {
          "dtype": "torch.float32",
          "raw_sha256": "a315db3b5a8ee594ea7d4cb6e960ba4af03d5f19885a97570a99761db9adf423",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/gR": {
          "dtype": "torch.float32",
          "raw_sha256": "1681e16a5173b125c21bf5b34012e94ae7b778337be6c58923a59382c681703a",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/gU0": {
          "dtype": "torch.float32",
          "raw_sha256": "e026a5fa0d468a7db4764274eae1b3a7c7b49fb0f340d243c3512d01cc6edb16",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/gU1": {
          "dtype": "torch.float32",
          "raw_sha256": "b77b053836e7eff47f0dbc5f5e7be15b6d17a3d25e79cba5b4e1153f8df0af05",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/gU2": {
          "dtype": "torch.float32",
          "raw_sha256": "aa8caee01aa05d44c152c53b9c6fc095881572f97ea73b35a8d4ac67180076b1",
          "shape": [
            1024,
            256
          ]
        },
        "run/W_up/gU3": {
          "dtype": "torch.float32",
          "raw_sha256": "99dd015e7b1ce46512f75f1d5165a0199cb09900b5e1e70b7f4af69c17fbe25e",
          "shape": [
            1024,
            256
          ]
        }
      }
    }
  }
}
```

## V2-2A-r1 read-only cross-check

```json
{
  "comparison": "exact Python float equality, informational only; no PASS/FAIL",
  "metrics": [
    "max_abs",
    "max_rel",
    "L2_relative_error"
  ],
  "read_only": true,
  "rows": [
    {
      "L2_equal_exact_float": true,
      "diagnostic_E_L2": 5.372153566440829e-08,
      "diagnostic_max_abs": 7.62939453125e-06,
      "diagnostic_max_rel_old": 0.00036101083969697356,
      "family": "W_Q",
      "max_abs_equal_exact_float": true,
      "max_rel_equal_exact_float": true,
      "r1_E_L2": 5.372153566440829e-08,
      "r1_max_abs": 7.62939453125e-06,
      "r1_max_rel": 0.00036101083969697356
    },
    {
      "L2_equal_exact_float": true,
      "diagnostic_E_L2": 5.370466382714767e-08,
      "diagnostic_max_abs": 7.62939453125e-06,
      "diagnostic_max_rel_old": 0.0034722222480922937,
      "family": "W_K",
      "max_abs_equal_exact_float": true,
      "max_rel_equal_exact_float": true,
      "r1_E_L2": 5.370466382714767e-08,
      "r1_max_abs": 7.62939453125e-06,
      "r1_max_rel": 0.0034722222480922937
    },
    {
      "L2_equal_exact_float": true,
      "diagnostic_E_L2": 5.404850611512302e-08,
      "diagnostic_max_abs": 1.52587890625e-05,
      "diagnostic_max_rel_old": 0.00023239600704982877,
      "family": "W_V",
      "max_abs_equal_exact_float": true,
      "max_rel_equal_exact_float": true,
      "r1_E_L2": 5.404850611512302e-08,
      "r1_max_abs": 1.52587890625e-05,
      "r1_max_rel": 0.00023239600704982877
    },
    {
      "L2_equal_exact_float": true,
      "diagnostic_E_L2": 5.3697743140901366e-08,
      "diagnostic_max_abs": 1.52587890625e-05,
      "diagnostic_max_rel_old": 0.00012130034156143665,
      "family": "W_O",
      "max_abs_equal_exact_float": true,
      "max_rel_equal_exact_float": true,
      "r1_E_L2": 5.3697743140901366e-08,
      "r1_max_abs": 1.52587890625e-05,
      "r1_max_rel": 0.00012130034156143665
    },
    {
      "L2_equal_exact_float": true,
      "diagnostic_E_L2": 5.3564853885745833e-08,
      "diagnostic_max_abs": 7.62939453125e-06,
      "diagnostic_max_rel_old": 0.005988024175167084,
      "family": "W_gate",
      "max_abs_equal_exact_float": true,
      "max_rel_equal_exact_float": true,
      "r1_E_L2": 5.3564853885745833e-08,
      "r1_max_abs": 7.62939453125e-06,
      "r1_max_rel": 0.005988024175167084
    },
    {
      "L2_equal_exact_float": true,
      "diagnostic_E_L2": 5.346880982415314e-08,
      "diagnostic_max_abs": 7.62939453125e-06,
      "diagnostic_max_rel_old": 0.00042698546894825995,
      "family": "W_up",
      "max_abs_equal_exact_float": true,
      "max_rel_equal_exact_float": true,
      "r1_E_L2": 5.346880982415314e-08,
      "r1_max_abs": 7.62939453125e-06,
      "r1_max_rel": 0.00042698546894825995
    },
    {
      "L2_equal_exact_float": true,
      "diagnostic_E_L2": 5.3162100499548615e-08,
      "diagnostic_max_abs": 7.62939453125e-06,
      "diagnostic_max_rel_old": 0.002994012087583542,
      "family": "W_down",
      "max_abs_equal_exact_float": true,
      "max_rel_equal_exact_float": true,
      "r1_E_L2": 5.3162100499548615e-08,
      "r1_max_abs": 7.62939453125e-06,
      "r1_max_rel": 0.002994012087583542
    }
  ],
  "schema": "omega-v2-2a-d3-r1-metric-crosscheck-v1",
  "source_result_sha256": "3e39184eda5989200190d830b1a100bdeef02a18dd06f13fcc4ee6a8ebec9155",
  "terminal_status": "OMEGA_V2_2A_LOCAL_PREFLIGHT_FAIL"
}
```

## CPU FP64 oracle diagnostic

```json
{
  "copy_checks": {
    "fp32_to_fp64_initial_weights_bitwise_value_preserved": true,
    "fp32_to_fp64_w_exact_value_preserved": true,
    "fp32_to_fp64_x_exact_value_preserved": true
  },
  "metrics": {
    "families": [
      {
        "comparisons": [
          {
            "E_L2": 9.969241660858121e-17,
            "E_inf": 2.704712177962087e-16,
            "comparison": "CPU_FP64_gR64_vs_sum_gU64",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.4210854715202004e-14,
            "max_abs_flat_index": 37691,
            "max_rel_old": 6.845613850893133e-13,
            "max_rel_old_flat_index": 58051,
            "norm_gR_inf": 52.54109783285489,
            "norm_gR_l2": 2769.7293884126148,
            "norm_sum_gU_inf": 52.54109783285488,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 4.58005777128316e-07,
            "E_inf": 4.273892202981642e-07,
            "comparison": "CUDA_gR_vs_CPU_FP64_gR64",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.2455498836393417e-05,
            "max_abs_flat_index": 36407,
            "max_rel_old": 0.017359245512681604,
            "max_rel_old_flat_index": 58051,
            "norm_gR_inf": 52.54109191894531,
            "norm_gR_l2": 2769.7295760195893,
            "norm_sum_gU_inf": 52.54109783285489,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 5.334995500379956e-07,
            "E_inf": 5.029208561739734e-07,
            "comparison": "CUDA_gU0_vs_CPU_FP64_gU0",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.670898744965399e-05,
            "max_abs_flat_index": 1766,
            "max_rel_old": 0.0052626595372737015,
            "max_rel_old_flat_index": 21802,
            "norm_gR_inf": 33.22388458251953,
            "norm_gR_l2": 1668.4905557337308,
            "norm_sum_gU_inf": 33.22389048799741,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 6.181595217519282e-07,
            "E_inf": 6.049390875113745e-07,
            "comparison": "CUDA_gU1_vs_CPU_FP64_gU1",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.0412506890045137e-05,
            "max_abs_flat_index": 37437,
            "max_rel_old": 0.27523428127177474,
            "max_rel_old_flat_index": 32076,
            "norm_gR_inf": 17.212488174438477,
            "norm_gR_l2": 899.9433255102468,
            "norm_sum_gU_inf": 17.212486712878054,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 6.271233533033547e-07,
            "E_inf": 5.675163396623173e-07,
            "comparison": "CUDA_gU2_vs_CPU_FP64_gU2",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 6.613918483999726e-06,
            "max_abs_flat_index": 37873,
            "max_rel_old": 0.042095031746536206,
            "max_rel_old_flat_index": 1276,
            "norm_gR_inf": 11.654145240783691,
            "norm_gR_l2": 549.9262207402331,
            "norm_sum_gU_inf": 11.654146359794906,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 6.459794202089368e-07,
            "E_inf": 5.881387307853023e-07,
            "comparison": "CUDA_gU3_vs_CPU_FP64_gU3",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 4.7121496118407435e-06,
            "max_abs_flat_index": 56308,
            "max_rel_old": 0.061284079216350734,
            "max_rel_old_flat_index": 18133,
            "norm_gR_inf": 8.011969566345215,
            "norm_gR_l2": 380.37127024512733,
            "norm_sum_gU_inf": 8.011967425722418,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 4.588515691905929e-07,
            "E_inf": 4.270555868835254e-07,
            "comparison": "CUDA_S_stack_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.2437969370514566e-05,
            "max_abs_flat_index": 54485,
            "max_rel_old": 0.017713989466697732,
            "max_rel_old_flat_index": 58051,
            "norm_gR_inf": 52.54109573364258,
            "norm_gR_l2": 2769.729576644133,
            "norm_sum_gU_inf": 52.54109783285488,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 4.5680471252485656e-07,
            "E_inf": 4.178573311917941e-07,
            "comparison": "CUDA_S_fp64_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.1954682918323698e-05,
            "max_abs_flat_index": 36585,
            "max_rel_old": 0.017359245512008924,
            "max_rel_old_flat_index": 58051,
            "norm_gR_inf": 52.541094064712524,
            "norm_gR_l2": 2769.7295756088406,
            "norm_sum_gU_inf": 52.54109783285488,
            "shape": [
              256,
              256
            ]
          }
        ],
        "family": "W_Q"
      },
      {
        "comparisons": [
          {
            "E_L2": 9.972897413179324e-17,
            "E_inf": 2.7167044179709407e-16,
            "comparison": "CPU_FP64_gR64_vs_sum_gU64",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.4210854715202004e-14,
            "max_abs_flat_index": 30548,
            "max_rel_old": 6.901964367068754e-12,
            "max_rel_old_flat_index": 44868,
            "norm_gR_inf": 52.309167759280356,
            "norm_gR_l2": 2728.1057108046375,
            "norm_sum_gU_inf": 52.30916775928035,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 4.63764790917861e-07,
            "E_inf": 4.639503888064827e-07,
            "comparison": "CUDA_gR_vs_CPU_FP64_gR64",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.4268864962095904e-05,
            "max_abs_flat_index": 45574,
            "max_rel_old": 0.06294533914317274,
            "max_rel_old_flat_index": 44868,
            "norm_gR_inf": 52.309181213378906,
            "norm_gR_l2": 2728.105918561599,
            "norm_sum_gU_inf": 52.309167759280356,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 5.378730247519867e-07,
            "E_inf": 6.258109089621902e-07,
            "comparison": "CUDA_gU0_vs_CPU_FP64_gU0",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.978576045758018e-05,
            "max_abs_flat_index": 21628,
            "max_rel_old": 0.2539415881699985,
            "max_rel_old_flat_index": 59620,
            "norm_gR_inf": 31.616195678710938,
            "norm_gR_l2": 1643.7333720530696,
            "norm_sum_gU_inf": 31.616196161220294,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 6.304750255906574e-07,
            "E_inf": 6.652195016964352e-07,
            "comparison": "CUDA_gU1_vs_CPU_FP64_gU1",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.2795312571922679e-05,
            "max_abs_flat_index": 62559,
            "max_rel_old": 0.04302092726352743,
            "max_rel_old_flat_index": 61370,
            "norm_gR_inf": 19.23472023010254,
            "norm_gR_l2": 884.6348657541256,
            "norm_sum_gU_inf": 19.234722582985345,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 6.298249373626095e-07,
            "E_inf": 7.351138572355164e-07,
            "comparison": "CUDA_gU2_vs_CPU_FP64_gU2",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 7.705320496653734e-06,
            "max_abs_flat_index": 51118,
            "max_rel_old": 0.01129930388656853,
            "max_rel_old_flat_index": 22524,
            "norm_gR_inf": 10.481804847717285,
            "norm_gR_l2": 534.4710508434326,
            "norm_sum_gU_inf": 10.481805533676802,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 6.529456555792437e-07,
            "E_inf": 5.904405099536052e-07,
            "comparison": "CUDA_gU3_vs_CPU_FP64_gU3",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 4.675794359476626e-06,
            "max_abs_flat_index": 37254,
            "max_rel_old": 0.03304339625708553,
            "max_rel_old_flat_index": 50287,
            "norm_gR_inf": 7.919161796569824,
            "norm_gR_l2": 370.79474555205206,
            "norm_sum_gU_inf": 7.919162524678454,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 4.6445173431862134e-07,
            "E_inf": 4.639503888064827e-07,
            "comparison": "CUDA_S_stack_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.4268864962095904e-05,
            "max_abs_flat_index": 45574,
            "max_rel_old": 0.05968034031836621,
            "max_rel_old_flat_index": 44868,
            "norm_gR_inf": 52.309181213378906,
            "norm_gR_l2": 2728.105919806888,
            "norm_sum_gU_inf": 52.30916775928035,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 4.625589922063744e-07,
            "E_inf": 4.7306614245227284e-07,
            "comparison": "CUDA_S_fp64_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.474570212029903e-05,
            "max_abs_flat_index": 45574,
            "max_rel_old": 0.05968034031836621,
            "max_rel_old_flat_index": 44868,
            "norm_gR_inf": 52.30918025970459,
            "norm_gR_l2": 2728.1059190626856,
            "norm_sum_gU_inf": 52.30916775928035,
            "shape": [
              256,
              256
            ]
          }
        ],
        "family": "W_K"
      },
      {
        "comparisons": [
          {
            "E_L2": 9.979673728580613e-17,
            "E_inf": 2.2148078186832346e-16,
            "comparison": "CPU_FP64_gR64_vs_sum_gU64",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.842170943040401e-14,
            "max_abs_flat_index": 736,
            "max_rel_old": 5.304319543434026e-13,
            "max_rel_old_flat_index": 37989,
            "norm_gR_inf": 128.32584927075754,
            "norm_gR_l2": 6239.686243881576,
            "norm_sum_gU_inf": 128.32584927075752,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 2.9390562420264837e-07,
            "E_inf": 2.903421021662342e-07,
            "comparison": "CUDA_gR_vs_CPU_FP64_gR64",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 3.7258397469486226e-05,
            "max_abs_flat_index": 62420,
            "max_rel_old": 0.012077265814273635,
            "max_rel_old_flat_index": 50103,
            "norm_gR_inf": 128.3258514404297,
            "norm_gR_l2": 6239.686415141729,
            "norm_sum_gU_inf": 128.32584927075754,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 3.7366132547156623e-07,
            "E_inf": 3.71885954218367e-07,
            "comparison": "CUDA_gU0_vs_CPU_FP64_gU0",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.352170574226875e-05,
            "max_abs_flat_index": 44629,
            "max_rel_old": 0.02738205887086695,
            "max_rel_old_flat_index": 28897,
            "norm_gR_inf": 63.24978256225586,
            "norm_gR_l2": 2962.8759589250617,
            "norm_sum_gU_inf": 63.24978051282409,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 3.937157654179227e-07,
            "E_inf": 3.7476054804087694e-07,
            "comparison": "CUDA_gU1_vs_CPU_FP64_gU1",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.699333277116466e-05,
            "max_abs_flat_index": 60600,
            "max_rel_old": 0.01719663309939987,
            "max_rel_old_flat_index": 52838,
            "norm_gR_inf": 45.34450149536133,
            "norm_gR_l2": 1957.7765380017574,
            "norm_sum_gU_inf": 45.344508273349824,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 3.5932984825783535e-07,
            "E_inf": 4.302984829284092e-07,
            "comparison": "CUDA_gU2_vs_CPU_FP64_gU2",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.5401522919233912e-05,
            "max_abs_flat_index": 41025,
            "max_rel_old": 0.006522286381398559,
            "max_rel_old_flat_index": 41133,
            "norm_gR_inf": 35.79264831542969,
            "norm_gR_l2": 1605.9697527368999,
            "norm_sum_gU_inf": 35.79264982395101,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 3.116747864496001e-07,
            "E_inf": 3.9662958062489304e-07,
            "comparison": "CUDA_gU3_vs_CPU_FP64_gU3",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.302766369803976e-05,
            "max_abs_flat_index": 24932,
            "max_rel_old": 0.07062701598542327,
            "max_rel_old_flat_index": 601,
            "norm_gR_inf": 32.84592056274414,
            "norm_gR_l2": 1586.2746917369802,
            "norm_sum_gU_inf": 32.845920556145316,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 2.941392600090841e-07,
            "E_inf": 2.903421021800767e-07,
            "comparison": "CUDA_S_stack_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 3.725839747126258e-05,
            "max_abs_flat_index": 62420,
            "max_rel_old": 0.011990908931914743,
            "max_rel_old_flat_index": 50103,
            "norm_gR_inf": 128.3258514404297,
            "norm_gR_l2": 6239.686413046798,
            "norm_sum_gU_inf": 128.32584927075752,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 2.91277801766068e-07,
            "E_inf": 2.9034210708904234e-07,
            "comparison": "CUDA_S_fp64_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 3.725839747126258e-05,
            "max_abs_flat_index": 62420,
            "max_rel_old": 0.012077265814273635,
            "max_rel_old_flat_index": 50103,
            "norm_gR_inf": 128.32584810256958,
            "norm_gR_l2": 6239.686414861766,
            "norm_sum_gU_inf": 128.32584927075752,
            "shape": [
              256,
              256
            ]
          }
        ],
        "family": "W_V"
      },
      {
        "comparisons": [
          {
            "E_L2": 1.0050867002210991e-16,
            "E_inf": 2.276263998333184e-16,
            "comparison": "CPU_FP64_gR64_vs_sum_gU64",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.842170943040401e-14,
            "max_abs_flat_index": 46768,
            "max_rel_old": 2.5434553439458876e-13,
            "max_rel_old_flat_index": 46747,
            "norm_gR_inf": 124.86121755304339,
            "norm_gR_l2": 6258.742743565076,
            "norm_sum_gU_inf": 124.86121755304337,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 2.899951764967232e-07,
            "E_inf": 2.9348024558874586e-07,
            "comparison": "CUDA_gR_vs_CPU_FP64_gR64",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 3.6644300791977e-05,
            "max_abs_flat_index": 27645,
            "max_rel_old": 0.01528088264375043,
            "max_rel_old_flat_index": 46483,
            "norm_gR_inf": 124.86121368408203,
            "norm_gR_l2": 6258.742908932844,
            "norm_sum_gU_inf": 124.86121755304339,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 3.852817392611589e-07,
            "E_inf": 3.7553175222515785e-07,
            "comparison": "CUDA_gU0_vs_CPU_FP64_gU0",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.0894405452054343e-05,
            "max_abs_flat_index": 4841,
            "max_rel_old": 0.05805069966324713,
            "max_rel_old_flat_index": 17722,
            "norm_gR_inf": 55.63949966430664,
            "norm_gR_l2": 2946.686393566459,
            "norm_sum_gU_inf": 55.639517373025406,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 3.877499094256173e-07,
            "E_inf": 3.7215694454237213e-07,
            "comparison": "CUDA_gU1_vs_CPU_FP64_gU1",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.8408561125227152e-05,
            "max_abs_flat_index": 59055,
            "max_rel_old": 0.006593509291799837,
            "max_rel_old_flat_index": 34638,
            "norm_gR_inf": 49.464508056640625,
            "norm_gR_l2": 1985.8006018120795,
            "norm_sum_gU_inf": 49.464510592066176,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 3.423891197996156e-07,
            "E_inf": 3.9083034069289675e-07,
            "comparison": "CUDA_gU2_vs_CPU_FP64_gU2",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.4540442489163752e-05,
            "max_abs_flat_index": 24871,
            "max_rel_old": 0.006471797373294327,
            "max_rel_old_flat_index": 47154,
            "norm_gR_inf": 37.203975677490234,
            "norm_gR_l2": 1622.0886798402812,
            "norm_sum_gU_inf": 37.20396630573313,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 2.9343627065433473e-07,
            "E_inf": 3.530474589868865e-07,
            "comparison": "CUDA_gU3_vs_CPU_FP64_gU3",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.1768256264588217e-05,
            "max_abs_flat_index": 26145,
            "max_rel_old": 0.023119889107308195,
            "max_rel_old_flat_index": 3260,
            "norm_gR_inf": 33.33335494995117,
            "norm_gR_l2": 1584.9046298814726,
            "norm_sum_gU_inf": 33.33335293193873,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 2.9051157098132743e-07,
            "E_inf": 3.3271270427484237e-07,
            "comparison": "CUDA_S_stack_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 4.154291335112248e-05,
            "max_abs_flat_index": 24871,
            "max_rel_old": 0.01528088264375043,
            "max_rel_old_flat_index": 46483,
            "norm_gR_inf": 124.8612060546875,
            "norm_gR_l2": 6258.7429088771905,
            "norm_sum_gU_inf": 124.86121755304337,
            "shape": [
              256,
              256
            ]
          },
          {
            "E_L2": 2.8780115757268044e-07,
            "E_inf": 3.087559946979697e-07,
            "comparison": "CUDA_S_fp64_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 3.85516494247895e-05,
            "max_abs_flat_index": 27645,
            "max_rel_old": 0.01528088264375043,
            "max_rel_old_flat_index": 46483,
            "norm_gR_inf": 124.86121368408203,
            "norm_gR_l2": 6258.742910329801,
            "norm_sum_gU_inf": 124.86121755304337,
            "shape": [
              256,
              256
            ]
          }
        ],
        "family": "W_O"
      },
      {
        "comparisons": [
          {
            "E_L2": 9.951366786540438e-17,
            "E_inf": 2.560535277903607e-16,
            "comparison": "CPU_FP64_gR64_vs_sum_gU64",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.4210854715202004e-14,
            "max_abs_flat_index": 229625,
            "max_rel_old": 1.848225415646351e-12,
            "max_rel_old_flat_index": 233286,
            "norm_gR_inf": 55.49954666837041,
            "norm_gR_l2": 4135.737731846175,
            "norm_sum_gU_inf": 55.49954666837041,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 3.6861255403885543e-07,
            "E_inf": 3.5589768422942403e-07,
            "comparison": "CUDA_gR_vs_CPU_FP64_gR64",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.9752160135055874e-05,
            "max_abs_flat_index": 102868,
            "max_rel_old": 0.08892819727616792,
            "max_rel_old_flat_index": 86990,
            "norm_gR_inf": 55.499542236328125,
            "norm_gR_l2": 4135.737847335167,
            "norm_sum_gU_inf": 55.49954666837041,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 4.7075389641676e-07,
            "E_inf": 4.2187778043670616e-07,
            "comparison": "CUDA_gU0_vs_CPU_FP64_gU0",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.2087469526633754e-05,
            "max_abs_flat_index": 175886,
            "max_rel_old": 0.41608136579712834,
            "max_rel_old_flat_index": 68836,
            "norm_gR_inf": 28.65158462524414,
            "norm_gR_l2": 1889.8539942202922,
            "norm_sum_gU_inf": 28.651590785656992,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 4.591884012126476e-07,
            "E_inf": 5.820893988869136e-07,
            "comparison": "CUDA_gU1_vs_CPU_FP64_gU1",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.0440029782898819e-05,
            "max_abs_flat_index": 102840,
            "max_rel_old": 0.11848890738245742,
            "max_rel_old_flat_index": 195617,
            "norm_gR_inf": 17.93543815612793,
            "norm_gR_l2": 1255.9731394609069,
            "norm_sum_gU_inf": 17.93544050598158,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 4.381327634523016e-07,
            "E_inf": 4.454867221294157e-07,
            "comparison": "CUDA_gU2_vs_CPU_FP64_gU2",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 6.445303670332692e-06,
            "max_abs_flat_index": 193857,
            "max_rel_old": 0.03689214052801661,
            "max_rel_old_flat_index": 185883,
            "norm_gR_inf": 14.468003273010254,
            "norm_gR_l2": 1002.5151157035942,
            "norm_sum_gU_inf": 14.468003983428051,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 4.2955794906267923e-07,
            "E_inf": 5.482787221670956e-07,
            "comparison": "CUDA_gU3_vs_CPU_FP64_gU3",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 8.072029743289022e-06,
            "max_abs_flat_index": 186568,
            "max_rel_old": 0.25144651608992813,
            "max_rel_old_flat_index": 227724,
            "norm_gR_inf": 14.722493171691895,
            "norm_gR_l2": 933.3534574382588,
            "norm_sum_gU_inf": 14.72248817222064,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 3.688439078654758e-07,
            "E_inf": 3.558976841654107e-07,
            "comparison": "CUDA_S_stack_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.975216013150316e-05,
            "max_abs_flat_index": 102868,
            "max_rel_old": 0.08817337390865633,
            "max_rel_old_flat_index": 86990,
            "norm_gR_inf": 55.499542236328125,
            "norm_gR_l2": 4135.737847193417,
            "norm_sum_gU_inf": 55.49954666837041,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 3.667363092649052e-07,
            "E_inf": 3.376402564563819e-07,
            "comparison": "CUDA_S_fp64_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.873888117032152e-05,
            "max_abs_flat_index": 102868,
            "max_rel_old": 0.08817337390865633,
            "max_rel_old_flat_index": 86990,
            "norm_gR_inf": 55.49954032897949,
            "norm_gR_l2": 4135.737848638311,
            "norm_sum_gU_inf": 55.49954666837041,
            "shape": [
              1024,
              256
            ]
          }
        ],
        "family": "W_gate"
      },
      {
        "comparisons": [
          {
            "E_L2": 9.944400049127765e-17,
            "E_inf": 1.863007025398335e-16,
            "comparison": "CPU_FP64_gR64_vs_sum_gU64",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.4210854715202004e-14,
            "max_abs_flat_index": 47561,
            "max_rel_old": 7.991163724414123e-13,
            "max_rel_old_flat_index": 218569,
            "norm_gR_inf": 76.27912574384167,
            "norm_gR_l2": 4129.111650461418,
            "norm_sum_gU_inf": 76.27912574384166,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 3.599903131575725e-07,
            "E_inf": 2.8365038127204595e-07,
            "comparison": "CUDA_gR_vs_CPU_FP64_gR64",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.1636604031982642e-05,
            "max_abs_flat_index": 256232,
            "max_rel_old": 0.01712766361526847,
            "max_rel_old_flat_index": 259830,
            "norm_gR_inf": 76.27912902832031,
            "norm_gR_l2": 4129.111778587797,
            "norm_sum_gU_inf": 76.27912574384167,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 4.679340002923055e-07,
            "E_inf": 4.226098996615657e-07,
            "comparison": "CUDA_gU0_vs_CPU_FP64_gU0",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.577567365984578e-05,
            "max_abs_flat_index": 177725,
            "max_rel_old": 0.11255154986424018,
            "max_rel_old_flat_index": 126735,
            "norm_gR_inf": 37.32916259765625,
            "norm_gR_l2": 1876.965547787289,
            "norm_sum_gU_inf": 37.329155686157286,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 4.4541579201882865e-07,
            "E_inf": 3.7210892909174854e-07,
            "comparison": "CUDA_gU1_vs_CPU_FP64_gU1",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 8.282214400523458e-06,
            "max_abs_flat_index": 164563,
            "max_rel_old": 0.20948539599321278,
            "max_rel_old_flat_index": 113074,
            "norm_gR_inf": 22.25749969482422,
            "norm_gR_l2": 1247.029458065805,
            "norm_sum_gU_inf": 22.25749976153183,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 4.2896317647450614e-07,
            "E_inf": 4.864023455063778e-07,
            "comparison": "CUDA_gU2_vs_CPU_FP64_gU2",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 6.918172890557628e-06,
            "max_abs_flat_index": 152473,
            "max_rel_old": 0.07362559508224856,
            "max_rel_old_flat_index": 181800,
            "norm_gR_inf": 14.22314453125,
            "norm_gR_l2": 985.7973013664464,
            "norm_sum_gU_inf": 14.223148704916996,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 4.1118132476665073e-07,
            "E_inf": 3.6121836485403854e-07,
            "comparison": "CUDA_gU3_vs_CPU_FP64_gU3",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 7.337127803097587e-06,
            "max_abs_flat_index": 74691,
            "max_rel_old": 0.07456984256573905,
            "max_rel_old_flat_index": 40980,
            "norm_gR_inf": 20.312164306640625,
            "norm_gR_l2": 921.3033545291893,
            "norm_sum_gU_inf": 20.312167145937835,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 3.603283643421877e-07,
            "E_inf": 2.855899168165197e-07,
            "comparison": "CUDA_S_stack_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.1784550114034573e-05,
            "max_abs_flat_index": 177725,
            "max_rel_old": 0.017193863678462876,
            "max_rel_old_flat_index": 259830,
            "norm_gR_inf": 76.27912902832031,
            "norm_gR_l2": 4129.11177898706,
            "norm_sum_gU_inf": 76.27912574384166,
            "shape": [
              1024,
              256
            ]
          },
          {
            "E_L2": 3.58113236526364e-07,
            "E_inf": 2.836503777257275e-07,
            "comparison": "CUDA_S_fp64_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.1636604031982642e-05,
            "max_abs_flat_index": 256232,
            "max_rel_old": 0.017193863678462876,
            "max_rel_old_flat_index": 259830,
            "norm_gR_inf": 76.27912998199463,
            "norm_gR_l2": 4129.111778048528,
            "norm_sum_gU_inf": 76.27912574384166,
            "shape": [
              1024,
              256
            ]
          }
        ],
        "family": "W_up"
      },
      {
        "comparisons": [
          {
            "E_L2": 9.846217930112362e-17,
            "E_inf": 2.343076256172163e-16,
            "comparison": "CPU_FP64_gR64_vs_sum_gU64",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.4210854715202004e-14,
            "max_abs_flat_index": 26271,
            "max_rel_old": 5.526953166961818e-12,
            "max_rel_old_flat_index": 158001,
            "norm_gR_inf": 60.6504149310872,
            "norm_gR_l2": 4074.2841877642477,
            "norm_sum_gU_inf": 60.6504149310872,
            "shape": [
              256,
              1024
            ]
          },
          {
            "E_L2": 3.633280118868767e-07,
            "E_inf": 3.2948616857536427e-07,
            "comparison": "CUDA_gR_vs_CPU_FP64_gR64",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.998347362786035e-05,
            "max_abs_flat_index": 193085,
            "max_rel_old": 0.01875053226760147,
            "max_rel_old_flat_index": 74719,
            "norm_gR_inf": 60.65041732788086,
            "norm_gR_l2": 4074.284303130122,
            "norm_sum_gU_inf": 60.6504149310872,
            "shape": [
              256,
              1024
            ]
          },
          {
            "E_L2": 4.6831919282595605e-07,
            "E_inf": 5.670822793070973e-07,
            "comparison": "CUDA_gU0_vs_CPU_FP64_gU0",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.3915352742088771e-05,
            "max_abs_flat_index": 193085,
            "max_rel_old": 0.476446128323371,
            "max_rel_old_flat_index": 21900,
            "norm_gR_inf": 24.53850746154785,
            "norm_gR_l2": 1864.010760365949,
            "norm_sum_gU_inf": 24.538507345830666,
            "shape": [
              256,
              1024
            ]
          },
          {
            "E_L2": 4.49665724507006e-07,
            "E_inf": 4.41748997886479e-07,
            "comparison": "CUDA_gU1_vs_CPU_FP64_gU1",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 8.632519811868633e-06,
            "max_abs_flat_index": 143413,
            "max_rel_old": 0.5022646350505022,
            "max_rel_old_flat_index": 176031,
            "norm_gR_inf": 19.541685104370117,
            "norm_gR_l2": 1236.1066028541795,
            "norm_sum_gU_inf": 19.541676471850305,
            "shape": [
              256,
              1024
            ]
          },
          {
            "E_L2": 4.292885301576827e-07,
            "E_inf": 4.731958708617821e-07,
            "comparison": "CUDA_gU2_vs_CPU_FP64_gU2",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 7.513263362568523e-06,
            "max_abs_flat_index": 185375,
            "max_rel_old": 0.061503804690459817,
            "max_rel_old_flat_index": 120632,
            "norm_gR_inf": 15.877702713012695,
            "norm_gR_l2": 969.8511652091336,
            "norm_sum_gU_inf": 15.877701562815353,
            "shape": [
              256,
              1024
            ]
          },
          {
            "E_L2": 4.2011695266207567e-07,
            "E_inf": 5.399057570627019e-07,
            "comparison": "CUDA_gU3_vs_CPU_FP64_gU3",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 8.458854134474336e-06,
            "max_abs_flat_index": 91167,
            "max_rel_old": 0.22243884105072903,
            "max_rel_old_flat_index": 239008,
            "norm_gR_inf": 15.667277336120605,
            "norm_gR_l2": 900.1026040238173,
            "norm_sum_gU_inf": 15.667279009014102,
            "shape": [
              256,
              1024
            ]
          },
          {
            "E_L2": 3.6367135089900467e-07,
            "E_inf": 3.2948616857536427e-07,
            "comparison": "CUDA_S_stack_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float32",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 1.998347362786035e-05,
            "max_abs_flat_index": 193085,
            "max_rel_old": 0.01875053226760147,
            "max_rel_old_flat_index": 74719,
            "norm_gR_inf": 60.65041732788086,
            "norm_gR_l2": 4074.2843047741057,
            "norm_sum_gU_inf": 60.6504149310872,
            "shape": [
              256,
              1024
            ]
          },
          {
            "E_L2": 3.6140220872028755e-07,
            "E_inf": 3.3454725915306113e-07,
            "comparison": "CUDA_S_fp64_vs_CPU_FP64_sum",
            "dtype_gR": "torch.float64",
            "dtype_sum_gU": "torch.float64",
            "floor_1e_6_active_at_max_rel_old": false,
            "max_abs": 2.0290431521630126e-05,
            "max_abs_flat_index": 117813,
            "max_rel_old": 0.018949434186736416,
            "max_rel_old_flat_index": 74719,
            "norm_gR_inf": 60.65041923522949,
            "norm_gR_l2": 4074.2843041391384,
            "norm_sum_gU_inf": 60.6504149310872,
            "shape": [
              256,
              1024
            ]
          }
        ],
        "family": "W_down"
      }
    ],
    "schema": "omega-v2-2a-d3-cpu-fp64-arithmetic-oracle-v1",
    "thresholds": null
  },
  "tensor_bundle": {
    "file_sha256": "e3131de5eae7d39066333e4ed4dbd25017c2f2815225657c3aa8925c4b464f14",
    "path": "cpu_fp64_oracle.pt",
    "size_bytes": 50607151,
    "tensor_metadata": {
      "cpu_fp64/W_K/S_fp64_cpu": {
        "dtype": "torch.float64",
        "raw_sha256": "1004baaf63653a35071f48df5af2c1b5eb48b3f23380dc9d4cf5678768b8157e",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_K/gR64": {
        "dtype": "torch.float64",
        "raw_sha256": "ede71a75d46c540854daff0ff21915601ec1f47c91819db17bd0ba61e54fd5a1",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_K/gU0_64": {
        "dtype": "torch.float64",
        "raw_sha256": "254360e19304a9349b2b65168b0293a3678516dce41cd5de7f63e1a89f530a44",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_K/gU1_64": {
        "dtype": "torch.float64",
        "raw_sha256": "41014c048547b0e8ab7944c16f93fbbe498480cd441cf11cd2bf4a54ed7f0772",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_K/gU2_64": {
        "dtype": "torch.float64",
        "raw_sha256": "7eb0b5c55a8ae41b9140f206e403154dc27c58a72c79afbbfec44ac45fe41297",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_K/gU3_64": {
        "dtype": "torch.float64",
        "raw_sha256": "bf84db38b4b829e37d7a909876410924c982acfc58c035dd89ccf39d2c1e4a9e",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_O/S_fp64_cpu": {
        "dtype": "torch.float64",
        "raw_sha256": "18969f37f44e04a21054344c32405e39a7114ba4a5b9b0a50ffbfac8ced876bc",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_O/gR64": {
        "dtype": "torch.float64",
        "raw_sha256": "7a00ac8f1e2f7da99fc221632c4f518861dc3f2141b338b8c563265837f25d6d",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_O/gU0_64": {
        "dtype": "torch.float64",
        "raw_sha256": "a7f4bdf785779aa2acb2667bd23f8bac827a7ada130bb9ee41c708ad8428e177",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_O/gU1_64": {
        "dtype": "torch.float64",
        "raw_sha256": "6e6874d1e6cc46d6404ff4967c417a483fdb9793517833fc36345a211dba07ab",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_O/gU2_64": {
        "dtype": "torch.float64",
        "raw_sha256": "71ae46050312b27e0b08be97f1c65aa742d27678b57e4bcb7a73079fd05e0b29",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_O/gU3_64": {
        "dtype": "torch.float64",
        "raw_sha256": "b0aab849bb5cf27945e0716acb7e75a4afb6ade9b832ffcbde45f8f7bf56f96e",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_Q/S_fp64_cpu": {
        "dtype": "torch.float64",
        "raw_sha256": "7576b2c0ad842182852bae9ce248f0363d983a34ee92e5b92e4b6a98a1964384",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_Q/gR64": {
        "dtype": "torch.float64",
        "raw_sha256": "38f5165242346319770ad711eef952404cc31d569da0ba1bbb85b50c2db79a4b",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_Q/gU0_64": {
        "dtype": "torch.float64",
        "raw_sha256": "595ee9f54135a896c704e323576f5cc4edf5ec2a600556ce37229d307fa64bd8",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_Q/gU1_64": {
        "dtype": "torch.float64",
        "raw_sha256": "fc44ff2a8d367abc692014ff6e38d8fd5566ff1bebac4138d730f8b538ec6c7c",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_Q/gU2_64": {
        "dtype": "torch.float64",
        "raw_sha256": "66d9fb489b796cc4005d0f160a0fddea1250913afbadc69f7b7cb70c9c546bda",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_Q/gU3_64": {
        "dtype": "torch.float64",
        "raw_sha256": "ecbf1b08708ecb073a954f4075cfc223215391c6b90e64715f89a469fa05be98",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_V/S_fp64_cpu": {
        "dtype": "torch.float64",
        "raw_sha256": "0b12c710d1c14c40a260ea2a3d0eec9645e9a21142a68a4aaa01d5fb52d9a868",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_V/gR64": {
        "dtype": "torch.float64",
        "raw_sha256": "de07ba7a86c652637274698690854e09623d30869cb2c0a2bd2cb25fc04b2197",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_V/gU0_64": {
        "dtype": "torch.float64",
        "raw_sha256": "d144271deff95cbc5540e2150c62b6a233e3e0a72295e5a07d990a092994938b",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_V/gU1_64": {
        "dtype": "torch.float64",
        "raw_sha256": "c23bd84e4afd0e593251fecddf83b2393679eaf1d13e76c6deacd8aa43cb32d9",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_V/gU2_64": {
        "dtype": "torch.float64",
        "raw_sha256": "3559a1d8d087dd53272f1c91f72a1dc86456f38be0b6391cfe98cddff4a1d05b",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_V/gU3_64": {
        "dtype": "torch.float64",
        "raw_sha256": "c663aa52158fdd43cdd698df00e8683e0cd84b78ce2fc0971c029588d53bf788",
        "shape": [
          256,
          256
        ]
      },
      "cpu_fp64/W_down/S_fp64_cpu": {
        "dtype": "torch.float64",
        "raw_sha256": "f1f3442f6009b3460443f4a3b3ac21e237a778769c9bcc9062af77c6f2eb1add",
        "shape": [
          256,
          1024
        ]
      },
      "cpu_fp64/W_down/gR64": {
        "dtype": "torch.float64",
        "raw_sha256": "6cdfc25107f705fa23bd668edea1d41b27763d93b9b1d1004f542e8dc273d641",
        "shape": [
          256,
          1024
        ]
      },
      "cpu_fp64/W_down/gU0_64": {
        "dtype": "torch.float64",
        "raw_sha256": "1112319df64e6d88db88cf8cf56cbbff9183eb994ff0a1814b19fec9a77d46d9",
        "shape": [
          256,
          1024
        ]
      },
      "cpu_fp64/W_down/gU1_64": {
        "dtype": "torch.float64",
        "raw_sha256": "387f037d4048b1040cc95449cb7f48e1c3a9cfacfc5af1ad2b4e2c1544e4b0ac",
        "shape": [
          256,
          1024
        ]
      },
      "cpu_fp64/W_down/gU2_64": {
        "dtype": "torch.float64",
        "raw_sha256": "d51b47796c98723cee3b770d5d6b48a1b0b97540bde3e5e61425d412220f70c2",
        "shape": [
          256,
          1024
        ]
      },
      "cpu_fp64/W_down/gU3_64": {
        "dtype": "torch.float64",
        "raw_sha256": "db152906197ba14c43c2501a01073bceefd644b034abe27d279d4b2eaab62ce2",
        "shape": [
          256,
          1024
        ]
      },
      "cpu_fp64/W_gate/S_fp64_cpu": {
        "dtype": "torch.float64",
        "raw_sha256": "d6b2fb796c4312f0c6d9120f5b0bdf5c8f517ea09be8cd49270ffd08e3bee216",
        "shape": [
          1024,
          256
        ]
      },
      "cpu_fp64/W_gate/gR64": {
        "dtype": "torch.float64",
        "raw_sha256": "25a7004442ccffa50f9e4e242c7788262cf19d1d43466daf98ce884156d03a04",
        "shape": [
          1024,
          256
        ]
      },
      "cpu_fp64/W_gate/gU0_64": {
        "dtype": "torch.float64",
        "raw_sha256": "7ff521f56ba90c5c9c8b549edee1a6e195d4e1b24e27c0e1c14388c89d0ca4ed",
        "shape": [
          1024,
          256
        ]
      },
      "cpu_fp64/W_gate/gU1_64": {
        "dtype": "torch.float64",
        "raw_sha256": "028a4e1231d3c0bb65cac1e2273fc4b665b4f188a8950c2546560bcda9c9cea4",
        "shape": [
          1024,
          256
        ]
      },
      "cpu_fp64/W_gate/gU2_64": {
        "dtype": "torch.float64",
        "raw_sha256": "0e4c97ebfcb0aa721cbb03edb2fa2e72a8e3382c9a0697471e1d580299b8b1ba",
        "shape": [
          1024,
          256
        ]
      },
      "cpu_fp64/W_gate/gU3_64": {
        "dtype": "torch.float64",
        "raw_sha256": "062147be02adcdbb4825dcf24eef061153a6dbfbe82bdaabfa15342640af19cc",
        "shape": [
          1024,
          256
        ]
      },
      "cpu_fp64/W_up/S_fp64_cpu": {
        "dtype": "torch.float64",
        "raw_sha256": "aa5dfa94c514c3561abc40386f4b633dae2321e6c0d94415d0a9fdeb6ac4faa3",
        "shape": [
          1024,
          256
        ]
      },
      "cpu_fp64/W_up/gR64": {
        "dtype": "torch.float64",
        "raw_sha256": "4999700bb9e984ba16bd40cebba03b01c7d68aa7dbc46a678590c2f21c77724a",
        "shape": [
          1024,
          256
        ]
      },
      "cpu_fp64/W_up/gU0_64": {
        "dtype": "torch.float64",
        "raw_sha256": "3b18b8af71ef68b16ebf62700fd06a58d42778dd39867ada90736877487ab063",
        "shape": [
          1024,
          256
        ]
      },
      "cpu_fp64/W_up/gU1_64": {
        "dtype": "torch.float64",
        "raw_sha256": "e1d2b8e173a6a0a617d3e7f64586baae6868ff25f0eb0a08b5c741fa4d3c869e",
        "shape": [
          1024,
          256
        ]
      },
      "cpu_fp64/W_up/gU2_64": {
        "dtype": "torch.float64",
        "raw_sha256": "0fb354473586eeacd5645f7340428373153e79cfd8acf737abda047c38e1a658",
        "shape": [
          1024,
          256
        ]
      },
      "cpu_fp64/W_up/gU3_64": {
        "dtype": "torch.float64",
        "raw_sha256": "efcd7d7828a9c483cbfdbd624aef0104d6cf61790acdba097b66a7262fce1256",
        "shape": [
          1024,
          256
        ]
      },
      "cpu_fp64/__inputs__/w_fp64": {
        "dtype": "torch.float64",
        "raw_sha256": "3985fa09f1433dbfbee4d01890930e0a16f021cc83451b83210d47187bb73ef8",
        "shape": [
          8,
          8,
          256
        ]
      },
      "cpu_fp64/__inputs__/x_fp64": {
        "dtype": "torch.float64",
        "raw_sha256": "f20da260041136f6c914b5bd092b5f5718dd230f81277f6a1198fb33581dc81e",
        "shape": [
          8,
          8,
          256
        ]
      }
    },
    "tensor_raw_sha256": {
      "cpu_fp64/W_K/S_fp64_cpu": "1004baaf63653a35071f48df5af2c1b5eb48b3f23380dc9d4cf5678768b8157e",
      "cpu_fp64/W_K/gR64": "ede71a75d46c540854daff0ff21915601ec1f47c91819db17bd0ba61e54fd5a1",
      "cpu_fp64/W_K/gU0_64": "254360e19304a9349b2b65168b0293a3678516dce41cd5de7f63e1a89f530a44",
      "cpu_fp64/W_K/gU1_64": "41014c048547b0e8ab7944c16f93fbbe498480cd441cf11cd2bf4a54ed7f0772",
      "cpu_fp64/W_K/gU2_64": "7eb0b5c55a8ae41b9140f206e403154dc27c58a72c79afbbfec44ac45fe41297",
      "cpu_fp64/W_K/gU3_64": "bf84db38b4b829e37d7a909876410924c982acfc58c035dd89ccf39d2c1e4a9e",
      "cpu_fp64/W_O/S_fp64_cpu": "18969f37f44e04a21054344c32405e39a7114ba4a5b9b0a50ffbfac8ced876bc",
      "cpu_fp64/W_O/gR64": "7a00ac8f1e2f7da99fc221632c4f518861dc3f2141b338b8c563265837f25d6d",
      "cpu_fp64/W_O/gU0_64": "a7f4bdf785779aa2acb2667bd23f8bac827a7ada130bb9ee41c708ad8428e177",
      "cpu_fp64/W_O/gU1_64": "6e6874d1e6cc46d6404ff4967c417a483fdb9793517833fc36345a211dba07ab",
      "cpu_fp64/W_O/gU2_64": "71ae46050312b27e0b08be97f1c65aa742d27678b57e4bcb7a73079fd05e0b29",
      "cpu_fp64/W_O/gU3_64": "b0aab849bb5cf27945e0716acb7e75a4afb6ade9b832ffcbde45f8f7bf56f96e",
      "cpu_fp64/W_Q/S_fp64_cpu": "7576b2c0ad842182852bae9ce248f0363d983a34ee92e5b92e4b6a98a1964384",
      "cpu_fp64/W_Q/gR64": "38f5165242346319770ad711eef952404cc31d569da0ba1bbb85b50c2db79a4b",
      "cpu_fp64/W_Q/gU0_64": "595ee9f54135a896c704e323576f5cc4edf5ec2a600556ce37229d307fa64bd8",
      "cpu_fp64/W_Q/gU1_64": "fc44ff2a8d367abc692014ff6e38d8fd5566ff1bebac4138d730f8b538ec6c7c",
      "cpu_fp64/W_Q/gU2_64": "66d9fb489b796cc4005d0f160a0fddea1250913afbadc69f7b7cb70c9c546bda",
      "cpu_fp64/W_Q/gU3_64": "ecbf1b08708ecb073a954f4075cfc223215391c6b90e64715f89a469fa05be98",
      "cpu_fp64/W_V/S_fp64_cpu": "0b12c710d1c14c40a260ea2a3d0eec9645e9a21142a68a4aaa01d5fb52d9a868",
      "cpu_fp64/W_V/gR64": "de07ba7a86c652637274698690854e09623d30869cb2c0a2bd2cb25fc04b2197",
      "cpu_fp64/W_V/gU0_64": "d144271deff95cbc5540e2150c62b6a233e3e0a72295e5a07d990a092994938b",
      "cpu_fp64/W_V/gU1_64": "c23bd84e4afd0e593251fecddf83b2393679eaf1d13e76c6deacd8aa43cb32d9",
      "cpu_fp64/W_V/gU2_64": "3559a1d8d087dd53272f1c91f72a1dc86456f38be0b6391cfe98cddff4a1d05b",
      "cpu_fp64/W_V/gU3_64": "c663aa52158fdd43cdd698df00e8683e0cd84b78ce2fc0971c029588d53bf788",
      "cpu_fp64/W_down/S_fp64_cpu": "f1f3442f6009b3460443f4a3b3ac21e237a778769c9bcc9062af77c6f2eb1add",
      "cpu_fp64/W_down/gR64": "6cdfc25107f705fa23bd668edea1d41b27763d93b9b1d1004f542e8dc273d641",
      "cpu_fp64/W_down/gU0_64": "1112319df64e6d88db88cf8cf56cbbff9183eb994ff0a1814b19fec9a77d46d9",
      "cpu_fp64/W_down/gU1_64": "387f037d4048b1040cc95449cb7f48e1c3a9cfacfc5af1ad2b4e2c1544e4b0ac",
      "cpu_fp64/W_down/gU2_64": "d51b47796c98723cee3b770d5d6b48a1b0b97540bde3e5e61425d412220f70c2",
      "cpu_fp64/W_down/gU3_64": "db152906197ba14c43c2501a01073bceefd644b034abe27d279d4b2eaab62ce2",
      "cpu_fp64/W_gate/S_fp64_cpu": "d6b2fb796c4312f0c6d9120f5b0bdf5c8f517ea09be8cd49270ffd08e3bee216",
      "cpu_fp64/W_gate/gR64": "25a7004442ccffa50f9e4e242c7788262cf19d1d43466daf98ce884156d03a04",
      "cpu_fp64/W_gate/gU0_64": "7ff521f56ba90c5c9c8b549edee1a6e195d4e1b24e27c0e1c14388c89d0ca4ed",
      "cpu_fp64/W_gate/gU1_64": "028a4e1231d3c0bb65cac1e2273fc4b665b4f188a8950c2546560bcda9c9cea4",
      "cpu_fp64/W_gate/gU2_64": "0e4c97ebfcb0aa721cbb03edb2fa2e72a8e3382c9a0697471e1d580299b8b1ba",
      "cpu_fp64/W_gate/gU3_64": "062147be02adcdbb4825dcf24eef061153a6dbfbe82bdaabfa15342640af19cc",
      "cpu_fp64/W_up/S_fp64_cpu": "aa5dfa94c514c3561abc40386f4b633dae2321e6c0d94415d0a9fdeb6ac4faa3",
      "cpu_fp64/W_up/gR64": "4999700bb9e984ba16bd40cebba03b01c7d68aa7dbc46a678590c2f21c77724a",
      "cpu_fp64/W_up/gU0_64": "3b18b8af71ef68b16ebf62700fd06a58d42778dd39867ada90736877487ab063",
      "cpu_fp64/W_up/gU1_64": "e1d2b8e173a6a0a617d3e7f64586baae6868ff25f0eb0a08b5c741fa4d3c869e",
      "cpu_fp64/W_up/gU2_64": "0fb354473586eeacd5645f7340428373153e79cfd8acf737abda047c38e1a658",
      "cpu_fp64/W_up/gU3_64": "efcd7d7828a9c483cbfdbd624aef0104d6cf61790acdba097b66a7262fce1256",
      "cpu_fp64/__inputs__/w_fp64": "3985fa09f1433dbfbee4d01890930e0a16f021cc83451b83210d47187bb73ef8",
      "cpu_fp64/__inputs__/x_fp64": "f20da260041136f6c914b5bd092b5f5718dd230f81277f6a1198fb33581dc81e"
    }
  }
}
```

## Environment and source hashes

```json
{
  "architectural_verdict": null,
  "attempt_00_artifact_manifest_sha256": "869afce3e4abcedd58a00283ca88c57d6b6285d4915c64ba54cf2b4785a97f79",
  "attempt_00_classification": "HARNESS_ABORT_PRE_CUDA_SCIENCE",
  "attempt_00_incident_json_sha256": "337f3be4f53bf2afba0517526e39836c0a221b0e7b0c998acd4e7e71078dc66d",
  "attempt_00_manifest_sidecar_sha256": "869afce3e4abcedd58a00283ca88c57d6b6285d4915c64ba54cf2b4785a97f79",
  "classification": "CALIBRATION_DIAGNOSTIC_ONLY",
  "cuda_kernel_launches_during_source_seal": 0,
  "environment": {
    "allow_tf32_cudnn": false,
    "allow_tf32_matmul": false,
    "amp": false,
    "compute_capability": [
      7,
      5
    ],
    "cublas_workspace_config": ":4096:8",
    "cuda_autocast_enabled": false,
    "cuda_available": true,
    "cudnn_benchmark": false,
    "cudnn_deterministic": true,
    "deterministic_algorithms": true,
    "device_name": "NVIDIA GeForce GTX 1650 SUPER",
    "device_total_memory_bytes": 4294639616,
    "dropout": 0,
    "dtype": "torch.float32",
    "nvidia_smi": {
      "available": true,
      "path": "C:\\WINDOWS\\system32\\nvidia-smi.EXE",
      "query": "NVIDIA GeForce GTX 1650 SUPER, 617.14",
      "query_returncode": 0,
      "stderr": ""
    },
    "platform": "Windows-11-10.0.26100-SP0",
    "python_version": "3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)]",
    "torch_cuda_runtime": "12.8",
    "torch_num_threads": 1,
    "torch_version": "2.11.0+cu128"
  },
  "fixed_seeds": {
    "INPUT_SEED": 20261938,
    "LOSS_W_SEED": 20262938,
    "MASTER_SEED": 20260930,
    "WEIGHT_SEED": 20260930
  },
  "held_out_D3Q_seeds_forbidden": [
    20261001,
    20261002,
    20261003,
    20261004,
    20261005
  ],
  "may_rescue_V2_2A": false,
  "official_diagnostic_started": false,
  "python_source_sha256": {
    ".gitattributes": "a79691a93b46e49ce460c26ef22afcc03d6eca1e63bf2edbc20e96159510f6c9",
    "OMEGA_V2_2A_D3_DIAGNOSTIC_SPEC.md": "e4579c7c72c70647dabe6a795ae128b796a2c897f358cee17ff8bacaaeb06d44",
    "__init__.py": "6ec6cd26c35f404e7e153bb43d2515849bd783e8fa8e91d846352b22086885ab",
    "__main__.py": "ddf91bf0a955c58fbd38231e50eb113689f13eca13ce8679475e995734933d59",
    "core.py": "9439100b05e1d13250deef64320f6a5a19bafcaa44e1805729b50b68ed4d34e1",
    "diagnostic.py": "17b283b33f45dc4c0f26358a392f870d445701008af19e4c2dd5e37eaaa62801",
    "metrics.py": "3c5f996ef515bce1817e116849a140fc29937522c765be035ff4a7ef7e2326e0",
    "runner.py": "2878dbf0f7f0b76b061f84a7761374e0a632b93878914ba5d60e0ab8757aad70",
    "tests/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "tests/test_d3_control_flow_dry_run.py": "5e2a6113c8283effef73af7d611ae8e6ea581240b52df03a468b0a4fb8828953",
    "tests/test_d3_diagnostic.py": "cf7c0c2caca0f675341e71b01352dbbf8511e0f74964c0ef4bbcda17181e1752"
  },
  "schema": "omega-v2-2a-d3-diagnostic-source-seal-v1",
  "source_seal_process_environment": {
    "cuda_device_query_performed": false,
    "cuda_kernel_launches": 0,
    "platform": "Windows-11-10.0.26100-SP0",
    "python_version": "3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)]",
    "torch_cuda_runtime": null,
    "torch_version": "2.11.0+cpu"
  },
  "spec_sha256": "e4579c7c72c70647dabe6a795ae128b796a2c897f358cee17ff8bacaaeb06d44",
  "v2_0_reference_sha256": {
    "D:\\Install\\Dev\\Projects\\IA\\Ejercise\\t1_trainability_lab_v0.1.0\\campaign\\omega_v2_0_conformance\\V2_0_RESULT_SEAL.json": "95facae0865750783faca8f9c271dcd557b7d1338f9fce512b0974cfa4606623",
    "D:\\Install\\Dev\\Projects\\IA\\Ejercise\\t1_trainability_lab_v0.1.0\\campaign\\omega_v2_0_conformance\\omega_v2\\core.py": "a9222f5d839674060f00822556adcdfe8d2db048bcb4cebeecef3ecb80558315",
    "D:\\Install\\Dev\\Projects\\IA\\Ejercise\\t1_trainability_lab_v0.1.0\\campaign\\omega_v2_0_conformance\\omega_v2\\ledger.py": "90ba550ac439e37dcb8dbf60e3dfa416322cdf7f5456e3f774d7e20dde262bc4",
    "D:\\Install\\Dev\\Projects\\IA\\Ejercise\\t1_trainability_lab_v0.1.0\\campaign\\omega_v2_0_conformance\\omega_v2\\variants.py": "1ab8cadfb4b781896f2a15ef16ee3c5e153864101abcf309b9c44fc489460704",
    "D:\\Install\\Dev\\Projects\\IA\\Ejercise\\t1_trainability_lab_v0.1.0\\campaign\\omega_v2_0_conformance\\results\\omega_v2_0_conformance\\omega_v2_flop_ledger.json": "10cfa2b0b3b00cd9d4287c7b4448d1a992f85c71b03875311cf4f72ca88027a5"
  },
  "v2_2a_r1_reference_sha256": {
    "D:\\Install\\Dev\\Projects\\IA\\Ejercise\\t1_trainability_lab_v0.1.0\\campaign\\omega_v2_2a_cuda_preflight\\SOURCE_SEAL_R1.json": "c57e534c184702cb5d1f9bc476b0c7af586b810778e6539736da8eed25c1c490",
    "D:\\Install\\Dev\\Projects\\IA\\Ejercise\\t1_trainability_lab_v0.1.0\\campaign\\omega_v2_2a_cuda_preflight\\core.py": "029c4abc20a7f4f6c637c29e1bddeda50ccad9941dfca98cfdd33564ad0a1eb9",
    "D:\\Install\\Dev\\Projects\\IA\\Ejercise\\t1_trainability_lab_v0.1.0\\campaign\\omega_v2_2a_cuda_preflight\\results\\omega_v2_2a_local_preflight\\artifact_hashes.json": "e34a50bb8b4601d5459755bde98b54aca840f13ef4fdc53a37e3e32d67955fe1",
    "D:\\Install\\Dev\\Projects\\IA\\Ejercise\\t1_trainability_lab_v0.1.0\\campaign\\omega_v2_2a_cuda_preflight\\results\\omega_v2_2a_local_preflight\\preflight_result.json": "3e39184eda5989200190d830b1a100bdeef02a18dd06f13fcc4ee6a8ebec9155",
    "D:\\Install\\Dev\\Projects\\IA\\Ejercise\\t1_trainability_lab_v0.1.0\\campaign\\omega_v2_2a_cuda_preflight\\variants.py": "e8ddafa2fdfed83a560b71f42a7229c2418e5751ac95af86fb75cdc3a4383f3a"
  }
}
```

No thresholds or PASS/FAIL scientific verdicts are applied in this diagnostic.

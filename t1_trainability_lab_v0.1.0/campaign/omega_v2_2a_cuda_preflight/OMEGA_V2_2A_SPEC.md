# OMEGA-V2-2A CUDA CONFORMANCE / TRAINABILITY PREFLIGHT

**Authority:** MD/317 Q9 and MD/318. Scope is local GTX 1650 SUPER CUDA conformance/trainability preflight only.

## Platform and execution configuration

- GPU: GTX 1650 SUPER, 4 GB, compute capability 7.5.
- Framework: PyTorch CUDA `2.11.0+cu128`.
- Dtype FP32; TF32 OFF; AMP OFF; dropout 0.
- Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization.
- Set `torch.use_deterministic_algorithms(True)`, `torch.backends.cuda.matmul.allow_tf32=False`, `torch.backends.cudnn.allow_tf32=False`, `torch.backends.cudnn.benchmark=False`, and `torch.backends.cudnn.deterministic=True`.

## Core, weights, and fixed data

- `d=256`, `m=8`, `B=8`.
- `R4` is one V2-0 `ContractualCoreBlock` shared for four applications; unique core parameter count `1,048,576`.
- `U4` is four independent blocks, each a bitwise deep copy of R4 before any forward; unique core parameter count `4,194,304`.
- Initialize exactly with V2-0 `ContractualCoreBlock(d=256,dtype=torch.float32,seed=20260930)`: seven Xavier-uniform matrices initialized in `MATRIX_FAMILIES` order with `torch.Generator(device="cpu")`. Create weights on CPU once, then copy them to CUDA; do not initialize independently on GPU.
- `MASTER_SEED=20260930`; `WEIGHT_SEED=20260930`; `INPUT_SEED=20261938`; `LOSS_W_SEED=20262938`; `TARGET_SEED=20263938`.
- Generate input `x ~ N(0,1)` on CPU with shape `[8,8,256]` and `INPUT_SEED`, then copy to CUDA. Generate gradient-gate weights `w ~ N(0,1)` on CPU with shape `[8,8,256]` and `LOSS_W_SEED`, then copy to CUDA. Generate smoke target `target ~ N(0,1)` on CPU with shape `[8,8,256]` and `TARGET_SEED`, then copy to CUDA.

## CUDA operation and round trace

The CUDA step preserves V2-0 `ContractualCoreBlock.step` order and equations: RMSNorm; Q/K/V projections; QKᵀ divided by sqrt(d); softmax; AV; O projection and residual; second RMSNorm; gate/up projections; SiLU and Hadamard product; down projection and residual. It adds no parameters. R4 round `r` uses the same block; U4 round `r` applies block `U[r]` once. A round trace is the full state tensor immediately after that round. The K4 final output is recorded separately from the four round traces.

## Required gates

### D1 — CUDA correctness against the V2-0 CPU FP32 reference

Use `ContractualCoreBlock` V2-0 CPU FP32 reference equations with the same CPU-created weights and input as CUDA; run `d=256,m=8,B=8,K={1,4}`. K1 compares its output. K4 compares all four CPU↔CUDA round traces and the final output. Both checks below must pass for every required output/trace.

For each element `i`:

```text
abs(y_cuda[i] - y_cpu[i]) <= 1e-5 + 1e-4 * max(abs(y_cpu[i]), 1e-6)
```

Also require:

```text
E_L2 = ||y_cuda - y_cpu||_2 / max(||y_cpu||_2, 1e-6) <= 1e-5
```

Report `max_abs`, `max_scaled_error = max_i(abs(error[i])/(1e-5 + 1e-4*max(abs(y_cpu[i]),1e-6)))`, and `L2_relative_error` for each required output/trace.

### D2 — R4/U4 initialization and round-trace parity

Before any forward, U4 must contain four bitwise copies of R4. At `d=256,m=8,B=8,K=4`, compare `torch.equal(R_trace[r],U_trace[r])` for all four rounds and `torch.equal(R_final,U_final)`.

If parity is not bitwise, the fallback applies only if each variant reproduces its own outputs bitwise on two identical executions, the first differing operation/round is identified, and each trace and final output satisfies `max_abs<=1e-6`, `max_rel<=1e-5`, `relative_floor=1e-6`. Fallback `max_rel` is elementwise absolute error divided by `max(abs(reference element),1e-6)`.

### D3 — Gradient-sharing identity

At the initial cloned graph, use `d=256,m=8,B=8,K=4`, the same input, weights, and `L=sum_i(y_i*w_i)` for R4 and U4. For each family `W_Q,W_K,W_V,W_O,W_gate,W_up,W_down`, compare R4 gradient `gR` against `sum_gU=Σ(r=1..4)grad(W_U_r)` and require all of:

```text
E_L2grad = ||gR - sum_gU||_2 / max(||gR||_2, ||sum_gU||_2, 1e-6) <= 1e-5
max_abs <= 2e-5
max_rel <= 2e-4
```

`max_rel` uses elementwise denominator `max(abs(gR),abs(sum_gU),1e-6)`. Report all three metrics per family.

### D4 — Parameter and storage identity

Derive unique parameter counts and storage identities by introspection: R4 exactly `1,048,576` parameters and one storage per family; U4 exactly `4,194,304` parameters and four mutually disjoint storages per family. U4 storages are disjoint from R4 storages.

### D5 — Iso-FLOP

Use the V2-0 analytical ledger, `1 MAC = 2 FLOPs`, at `d=256,m=8,B=8,K=4`:

```text
per-round, B=1 = 16,842,752 FLOPs
K4, B=1        = 67,371,008 FLOPs
K4, B=8        = 538,968,064 FLOPs
R4_forward_flops = 538,968,064
U4_forward_flops = 538,968,064
exact integer equality = REQUIRED
```

Require identical R4/U4 `non_gemm_counts_total` for RMSNorm, softmax, SiLU, Hadamard, and residual operations. Do not use wall time for iso-FLOP and do not assign fictitious FLOPs to non-GEMM operations.

### D6 — K-flex

One R4 shared core runs forward `K={1,2,4,8,16}` and backward `K={1,4,8,16}` with a fresh graph per K, the same base input/weights, and `zero_grad(set_to_none=True)`. K-flex backward uses the D3 scalar loss probe `L=sum_i(y_i*w_i)` with the fixed `LOSS_W_SEED` tensor and no optimizer. Outputs, input gradients, and all seven parameter-family gradients must be finite for backward K values.

Before and after each K-flex forward/backward, require unchanged parameter schema, value hash, and parameter count. `SCHEMA_SHA256` is SHA-256 of canonical JSON containing sorted `(name,shape,dtype)` records. `VALUE_SHA256` follows V2-0 `state_dict_sha256`: sorted state-dict entries, name, NUL separator, dtype ASCII, compact JSON shape, then contiguous raw tensor bytes. VALUE hash invariance applies only without optimizer updates.

### D7 — Optimizer smoke

Run R4 and U4 separately; do not compare their losses and do not select checkpoints. For each variant perform exactly 20 updates at `d=256,m=8,B=8,K=4` using AdamW FP32 with `lr=3e-4`, `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0`, no scheduler, and no gradient clipping. Use fixed `x` and `target` and `L=mean((y-target)^2)`. `L0` is evaluated before updates; `L20` after update 20. Require `L20<=0.99*L0` and finiteness at every step for output, loss, all parameter gradients, all parameters, Adam `exp_avg`, and Adam `exp_avg_sq`.

### D8 — VRAM and time

Before each independent gate/configuration cell, call `torch.cuda.empty_cache()` and `torch.cuda.reset_peak_memory_stats()`; never call `empty_cache()` inside forward/backward. Synchronize before recording `peak_memory_allocated` and `peak_memory_reserved`. Require peak allocated `<=3 GiB` for executed cells. Record peak reserved. A configuration exceeding capacity is labeled `OOM_CAPACITY_DIAGNOSTIC`, not an architectural FAIL. Total V2-2A wall time is `<=30 min`.

For this memory accounting, a cell is one complete invocation of a gate/configuration: D1 K1 and K4, D2 K4 parity, D3 K4 gradient identity, each individual D6 forward K and backward K, and each complete D7 variant smoke. CPU-only introspection/ledger work is not a CUDA memory cell.

### D9 — Failures and continuation

Run all preregistered gates when technically possible; a logical FAIL does not automatically cut off other inexpensive gates. Hard-stop only for CUDA/runtime crash, structural corruption, unrecoverable OOM, or a non-finite state that prevents the next gate. Remaining gates are `NOT_RUN_DUE_TO_HARD_STOP`. Any required gate FAIL yields `OMEGA_V2_2A_LOCAL_PREFLIGHT_FAIL`; PASS is `OMEGA_V2_2A_LOCAL_PREFLIGHT_PASS` only when all required gates in MD/317 pass.

## Required order

1. Environment/config seal.
2. Parameter/storage ledger.
3. CPU↔CUDA correctness.
4. R4/U4 init and trace parity.
5. Gradient-sharing identity.
6. Iso-FLOP ledger.
7. K-flex.
8. R4 optimizer smoke.
9. U4 optimizer smoke.
10. VRAM/runtime summary.
11. Final conformance block.

## Source seal and report fields

Before official execution, `SOURCE_SEAL.json` records this spec SHA-256, SHA-256 for every package `.py`, V2-0 source/ledger hashes, V2-0 FLOP ledger SHA-256, PyTorch/CUDA/driver/GPU metadata, deterministic settings, CUBLAS workspace configuration, and fixed seeds. The official report records each ordered gate, all required output/trace error metrics, gradient metrics for seven families, parameter/storage counts, exact FLOP and non-GEMM ledgers, K-flex schema/value hashes before/after every forward/backward, per-variant smoke `L0/L20` and per-step finiteness, per-cell allocated/reserved VRAM, OOM diagnostics, wall time, source/environment seal, and final conformance status.

## Scope exclusions

No d512, FP16/BF16, RunPod, language quality, NLL, `+0.05 nats/token` quality margin, checkpoint selection, or T3. PASS permits only consideration of OMEGA-V2-2B d512 local contractual pilot. Official GPU preflight execution remains pending a separate GO after review of the committed spec/source seal.

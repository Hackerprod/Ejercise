# OMEGA-V2-2A-D3-DIAGNOSTIC

**Classification:** `CALIBRATION_DIAGNOSTIC_ONLY`  
**may_rescue_V2_2A:** `false`  
**architectural_verdict:** `NONE`

This is a separate post-hoc numerical calibration diagnostic. It does not modify, replace, or reinterpret V2-2A-r1. V2-2A-r1 remains `OMEGA_V2_2A_LOCAL_PREFLIGHT_FAIL` at D3 elementwise `max_rel`; no V2-2A-r1 preflight, parameter, source, result, or seal is changed here.

## Frozen execution configuration

- `d=256`, `m=8`, `B=8`, `K=4`.
- `MASTER_SEED=20260930`, `WEIGHT_SEED=20260930`, `INPUT_SEED=20261938`, `LOSS_W_SEED=20262938`.
- Use the V2-0 `ContractualCoreBlock(d=256,dtype=torch.float32,seed=20260930)` initialized on CPU by its seven Xavier-uniform matrices using the CPU generator in `MATRIX_FAMILIES` order.
- For the CPU FP64 oracle, instantiate `ContractualCoreBlock(dtype=torch.float64)` and then `copy_` the exact initial FP32 parameter values converted with `.double()`; do not use FP64 Xavier initialization. V2-0 `core.py` `rms_norm` and `ContractualCoreBlock.step` use dtype-generic tensor operations without FP32 casts or FP32 hard-codes; `step` enforces CPU device and tensor shape only, so CPU FP64 is supported.
- Create R4 on CPU and clone U4 as four independent bitwise copies of R4 before any forward. Generate `x ~ N(0,1)` with shape `[8,8,256]` on CPU using `INPUT_SEED`; generate `w ~ N(0,1)` with shape `[8,8,256]` on CPU using `LOSS_W_SEED`. Copy the exact FP32 values to CUDA.
- FP32, deterministic algorithms, TF32 OFF, AMP OFF, no optimizer. `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before CUDA initialization.
- Do not generate, derive, load, or touch held-out D3Q master seeds `20261001` through `20261005`.
- Loss for both executions is exactly `L=sum(y*w)`; it is not averaged and no other loss is evaluated.

## Repeated D3 diagnostic runs

Execute D3 twice, sequentially, with freshly constructed R4/U4 instances and freshly CPU-generated `x` and `w` from the same frozen seeds on each run; confirm the run-2 initial weights, `x`, and `w` are bitwise equal to run 1 before forward. Keep operation order and CUDA configuration identical. Each execution builds a fresh graph and computes R4 `gR` and U4 `gU0,gU1,gU2,gU3` for K4. Compare all corresponding raw gradients from run 1 and run 2 with `torch.equal`; report bitwise reproducibility for each family and tensor. Reproducibility is diagnostic information, not a PASS/FAIL gate.

For each run and each family `W_Q,W_K,W_V,W_O,W_gate,W_up,W_down`, persist the complete CUDA FP32 tensors `gR`, `gU0`, `gU1`, `gU2`, and `gU3`, plus the five named U4 sum tensors below. Persist each run in a separate `.pt` file after detaching and copying tensors to CPU, preserving tensor dtype, shape, and values. Record the file size and SHA-256 for each tensor bundle.

## U4 summation variants and old-comparator indices

Use the four captured U4 gradients to compute these five diagnostic sums in FP32:

```text
S_forward = ((gU0 + gU1) + gU2) + gU3
S_reverse = ((gU3 + gU2) + gU1) + gU0
S_pairwise = (gU0 + gU1) + (gU2 + gU3)
S_stack = torch.stack([gU0,gU1,gU2,gU3]).sum(0)
S_fp64 = (((gU0.double() + gU1.double()) + gU2.double()) + gU3.double())
```

`sum_gU_old` for the historical max-relative diagnostic is exactly `S_stack`, matching the V2-2A-r1 reduction. For each family, run, and each of the five sums, compare `gR` with the sum and report `E_L2`, `E_inf`, `max_abs`, and `max_rel_old`. These metrics have no PASS/FAIL threshold in this diagnostic.

For run 1 `S_stack`, read `results/omega_v2_2a_local_preflight/preflight_result.json` read-only and compare each of the seven families' `max_abs`, `max_rel`, and `L2_relative_error` against the sealed V2-2A-r1 D3 values using exact floating-value equality. Report the comparisons as informational fields only; do not modify the V2-2A-r1 result file and do not assign PASS/FAIL to this cross-check.

For the old comparator against `sum_gU_old=S_stack`, report both argmax indices:

- `argmax(max_rel_old)` is the first maximum in a contiguous row-major flatten of the family tensor of `abs(gR-S_stack)/max(abs(gR),abs(S_stack),1e-6)`.
- `argmax(max_abs)` is the first maximum in a contiguous row-major flatten of `abs(gR-S_stack)`.
- Convert each linear index to the parameter's row-major multi-index. At both indices persist `gR`, `sum_gU_old`, `abs_error`, `old_denominator=max(abs(gR),abs(sum_gU_old),1e-6)`, `old_relative_error`, and `floor_1e-6_active=(max(abs(gR),abs(sum_gU_old))<=1e-6)`.

## D3 metrics

For R4 tensor `gR` and each selected U4 sum tensor `S`, report:

```text
E_L2 = ||gR-S||_2 / max(||gR||_2,||S||_2,1e-6)
E_inf = ||gR-S||_inf / max(||gR||_inf,||S||_inf,1e-12)
max_abs = max(abs(gR-S))
max_rel_old = max(abs(gR-S)/max(abs(gR),abs(S),1e-6) elementwise)
```

These values are descriptive only. No diagnostic metric is classified PASS or FAIL.

## Local ULP definition

For each finite FP32 value `x`, define local ULP from its magnitude `a=abs(x)` as:

```text
if a == 0:
    ulp(x) = nextafter(float32(0), +infinity) - float32(0)
else if a < largest_finite_FP32:
    ulp(x) = nextafter(a, +infinity) - a
else:
    ulp(x) = a - nextafter(a, 0)
```

`nextafter` is evaluated in FP32. For zero this is the smallest positive FP32 subnormal. For negative inputs, the absolute magnitude is used. For non-finite values, record the value as non-finite and the ULP-derived fields as null.

At both old-comparator argmax indices, report `ulp_gR`, `ulp_sum_gU_old`, `abs_error/ulp_gR`, and `abs_error/ulp_sum_gU_old`. If an ULP is non-finite or zero, its corresponding ratio is null.

Also persist, per family and run, `||gR||_2`, `||gR||_inf`, `||S_stack||_inf`, `ulp(||S_stack||_inf)`, and `max_abs/ulp(||S_stack||_inf)` to express the absolute error in ULPs of the maximum-magnitude tensor value. Compute the local ULP of that FP32 norm with the same `nextafter` rule above.

## CPU FP64 arithmetic oracle

Copy the exact initial FP32 parameter values, `x`, and `w` to CPU FP64 by dtype conversion; do not reinitialize weights or data in FP64. Run the same V2-0 R4/U4 four-round forward/backward and loss in CPU FP64. Record CPU FP64 `gR64` and `gU0_64..gU3_64` in a separate `.pt` bundle and its SHA-256.

Report, per family:

- CPU identity comparison `gR64` vs left-associated FP64 sum of the four U4 FP64 gradients.
- CUDA `gR` vs CPU FP64 `gR64`.
- Each CUDA `gUr` vs its corresponding CPU FP64 `gUr_64`.
- `S_stack` CUDA vs the CPU FP64 U4 sum and `S_fp64` vs that CPU FP64 sum.

For every comparison report `E_L2`, `E_inf`, `max_abs`, and `max_rel_old` where defined. These are diagnostics only; no threshold is applied.

## Execution order

1. Seal environment/configuration and verify the V2-2A-r1/attempt-00/held-out-seed references.
2. Construct the exact CPU FP32 R4/U4 initialization, fixed tensors, and confirm their initial-copy identity; do not execute any held-out seed.
3. Run CUDA D3 diagnostic execution 1 and persist its raw gradient bundle.
4. Run the identical CUDA D3 diagnostic execution 2 and persist its raw gradient bundle.
5. Compare run 1/run 2 raw gradients bitwise and compute all five U4 summation diagnostics, indices, local ULPs, and error metrics.
6. Run the CPU FP64 oracle from the exact FP32-initialized values and persist its gradient bundle.
7. Write the aggregate diagnostic report and hash manifest.

No optimizer, language data, checkpoint selection, held-out seed, official V2-2A rerun, V2-2A-r1 modification, V2-2A rescue, or architectural verdict is part of this unit.

## Persistence and terminal status

Use a new result directory under this diagnostic package. Persist two CUDA gradient `.pt` bundles, the CPU FP64 gradient `.pt` bundle, a metrics JSON, a Markdown report, and an artifact hash manifest containing the name, size, and SHA-256 of every evidence file. For every persisted tensor, record both (a) SHA-256 over its contiguous raw bytes prefixed by its UTF-8 tensor name plus NUL, ASCII dtype, and compact JSON shape, following V2-0 `state_dict_sha256` semantics, and (b) the SHA-256 of its enclosing `.pt` file; the raw-tensor hash does not rely on `torch.save` byte determinism. The report records both execution-level reproducibility summaries, all per-family raw tensor names/shapes/dtypes/hashes, all diagnostic metrics, the read-only V2-2A-r1 cross-check, and environment/source hashes.

The sealed diagnostic package is executed exactly once. An abort before any CUDA numerical cell and before persisting a new scientific result does not consume that run; the boundary is the first CUDA numerical cell or persistence/exposure of a new scientific gate result, whichever occurs first. After that boundary, no rerun under the same diagnostic ID is permitted.

If all diagnostic computations and artifact sealing complete, terminal classification is `DIAGNOSTIC_COMPLETE`; if execution or persistence cannot complete, terminal classification is `EXECUTION_FAILURE`. Numerical magnitudes, ULP counts, and reproducibility do not create scientific PASS/FAIL, do not rescue V2-2A, and do not authorize D3Q or V2-2B.

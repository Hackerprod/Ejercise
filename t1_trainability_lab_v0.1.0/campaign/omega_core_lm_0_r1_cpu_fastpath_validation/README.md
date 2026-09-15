# OMEGA CORE-LM-0 R1 CPU Fastpath Validation

This isolated unit implements **Step 1 only**: complete FP32/eager/CPU numerical equivalence between the corrected technical baseline and a local fast candidate. It does not benchmark speed, relaunch SCOPE-A, build teacher caches, reduce corpus scope, or authorize Step 2.

## Quick Path

```powershell
python -m pytest -q test_equivalence.py
python -m py_compile omega_fast_candidate.py test_equivalence.py
```

`step1_report.json` is written by the pytest session hook. `status` is `passed` only when every assertion passes. `step2_authorized` remains `false`.

## Candidate Correspondence

`omega_fast_candidate.py` starts from one real `OmegaCoreLM0R1Technical` instance through `OmegaCoreLMFast.from_reference(reference)`. It never creates two independent same-seed models.

| Candidate parameter | Reference parameter(s) |
|---|---|
| `embedding.weight` | `embedding.weight` |
| `prelude.*` | `prelude.*` |
| `prelude_norm_weight` | `prelude_norm.weight` |
| `blocks.i.qkv.weight/bias` | Q/K/V rows `[0:D]`, `[D:2D]`, `[2D:3D]` from `slot_mix.query/key/value` |
| `blocks.i.out.*` | `slot_mix.output.*` |
| `blocks.i.fc1.*` | `core.network.0.*` |
| `blocks.i.fc2.*` | `core.network.2.*` |
| `blocks.i.norm_weight` | `rms_norm.weight` |
| `depth_embedding.weight` | `depth_embedding.weight` |
| `gate_logits` | `gate_logits` |
| `readout_norm_weight` | `readout_norm.weight` |
| `output_projection.weight` | `output_projection.weight` |

The suite checks every parameter, mapped gradient, clipped gradient, AdamW parameter, and AdamW `exp_avg`/`exp_avg_sq` moment. It covers shared K1 and shared K4.

## Mask Semantics

For each token, candidate next state is computed first. Candidate/readout state and logits use that state. Input/state mask is applied only afterward to continuation state. Target/loss masks are separate fixtures and are never reused as input masks. A targeted regression fails if masked-position logits come from reverted continuation state.

## Loss Gate

Original CE, KL, and total are compared independently with local logsumexp CE, KL, and total. Teacher negative entropy is retained exactly as `sum(p * log(p))`. Concentrated teacher distributions and low-KL distributions are covered by the loss-path tests. Gate is absolute-only: `atol=1e-5`, `rtol=0`; failures abort rather than relaxing tolerance.

## Dynamic Constants and SDPA

`_round_constants()` recomputes differentiable gate and depth/W1-derived bias values on every forward/window. The suite performs AdamW, changes trainable W1/depth/gate, and verifies updated values are used with gradients connected. Slot SDPA explicitly uses `dropout_p=0.0` and `is_causal=False`.

## Exclusions

- No Step 2 benchmark or speed measurement.
- No GPU, Pod, VPS, GHCR, spend, C++, `torch.compile`, FP16, BF16, TF32, CUDA Graphs, `vmap`, parallel trainers, thread-count experiments, or test split.
- No edits under `omega_core_lm_0_r1_fable_optimization_proposal/`; its `omega_fast.py` is read-only reference material only.
- No claims about invalid Fable benchmark figures.
- No corpus replacement or Step 2 use of technical documents.

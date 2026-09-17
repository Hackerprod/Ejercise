# OMEGA CORE-LM-0 ER32 Design Unit

This is an isolated, synthetic-only design and correctness gate for the ER32
factorization. It does not import campaign runners, load corpus data, read
checkpoints, initialize production models, or launch training.

## Decision

Use one linked factorization for both tied input and tied output:

```text
C: [50257, 32]
U: [32, 128]
E = C @ U: [50257, 128]
e_t = C[token_t] @ U
z = (h @ U.T) @ C.T / sqrt(128)
```

`C` and `U` are the only tied input/output parameters. The explicit view
materializes `E = C @ U`; the efficient view applies the two factors directly.
Neither view owns an independent dense embedding or output matrix.

Vocabulary IDs, ordering, dense logits, and the teacher-logit interface stay
unchanged. The factorization adds no bias, activation, temperature, or
normalization. Temperature appears only in the synthetic CE+KL loss helper,
where it is fixed by default to `T=2.0`.

K1 and K4 use one shared recurrent block, applied for rounds 1 and 4. The
isolated module contains a compact recurrent core suitable for synthetic
fixtures. That core is held constant between explicit and efficient views;
factorization is the sole variable in their comparisons. It is not a claim to
replace the approved production recurrent implementation.

## Synthetic Fixture Initialization

`OmegaCoreLM0ER32.fresh(seed=...)` is a synthetic fixture-only factory. It uses
a deterministic seed inside `torch.random.fork_rng`, so initialization does not
consume the caller's CPU RNG stream. `fork_rng(devices=[])` does not cover other
devices; this determinism claim is CPU-only. Non-factorized fixture components
retain the normal PyTorch reference initialization convention.

`C` uses standard deviation `1.0`, and `U` uses standard deviation
`1/sqrt(rank)`. Therefore each effective embedding component in `E = C @ U`
has marginal variance `1.0`, matching the R1 `nn.Embedding` marginal variance
convention. This is not full matrix equivalence: ER32 remains rank-restricted.

Future R1 integration needs a separate factory. It must copy non-factorized
components (prelude, SlotMix, RMSNorm, and depth modulation) by explicit
correspondence from a freshly initialized same-seed reference, never from A/B/C
checkpoints or by post-hoc factorization.

This design unit does not call fresh initialization for any real campaign
configuration and does not run real initialization or training.

## Equivalence Gate

Tests clone one model `state_dict` into explicit and efficient instances, then
run identical synthetic tokens, masks, teacher logits, and targets. They
compare:

- logits and candidate states;
- CE, KL, and combined loss (`0.5 * CE + 0.5 * KL`);
- gradients for `C`, `U`, and non-factorized parameters;
- clipped gradients;
- one AdamW update, including `exp_avg` and `exp_avg_sq` moments;
- K1 and K4;
- state and mask transfer from window 0 to window 1;
- an integrated nominal two-window AdamW update, including both step counters;
- a negative no-detach check showing window 1 would retain window 0's graph.

The predeclared comparison tolerance is `atol=1e-4, rtol=1e-4`. This allows
FP32 operation-order differences between `C @ U` followed by one projection
and two direct projections. The views are comparable, not bit-identical; exact
equality is required only for fresh CPU RNG preservation and identical fresh
weights.

## Metrics and Future Matrix

No campaign runs are performed here. Future comparison table has four cells:

Synthetic fixture values and equivalence checks are not integrated OMEGA ER32
performance benchmarks and must not be presented as such.

| Variant | K1 | K4 |
|---|---:|---:|
| R1 | no run | no run |
| ER32 | no run | no run |

For a future approved comparison, record:

```text
δ_K = NLL_ER32,K - NLL_R1,K
Δ_ER32 = NLL_ER32,K1 - NLL_ER32,K4
```

The proposed gate `δ_K <= 0.10 NLL` is **PENDING approval by the user/Sol**.
It is not a decided acceptance criterion and must not be replaced by `0.05`.

## Memory Accounting

Counts below cover only tied vocabulary input/output parameters and use FP32.

| Component | Dense tied | ER32 |
|---|---:|---:|
| Parameter storage | 6,432,896 params | 1,612,320 params |
| Parameter bytes | 25,731,584 B (24.54 MiB) | 6,449,280 B (6.15 MiB) |
| Gradients + two AdamW moments | 77,194,752 B (73.62 MiB) | 19,347,840 B (18.45 MiB) |
| Parameters + gradients + two moments | 102,926,336 B (98.16 MiB) | 25,797,120 B (24.60 MiB) |

Exact parameter reduction:

```text
6,432,896 - 1,612,320 = 4,820,576 params
4,820,576 * 4 = 19,282,304 bytes = 18.39 MiB FP32
```

Gradients and both AdamW moments are separate storage from parameters. They
are not silently included in the parameter count. Activations and temporaries
are separate again: dense logits/softmax remain `[B, T, 50257]`. For example,
one FP32 tensor at `B=8, T=256` contains `8 * 256 * 50,257` values and uses
`411,705,344` bytes, approximately `392.7 MiB`. Factorizing `E` does not
remove this dense vocabulary-axis tensor. Measured process peak is a separate
runtime quantity and must be obtained from an instrumented run; this design
unit makes no automatic L2-residency claim.

## Scope Boundary

Only this sibling directory is in scope. A/B/C source files, checkpoints,
results, corpus data, GPU resources, and real training remain untouched.

Verification commands:

```text
python -m pytest t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_er32_design/test_omega_core_lm_0_er32_design.py
python -m py_compile t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_er32_design/omega_core_lm_0_er32_design.py t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_er32_design/test_omega_core_lm_0_er32_design.py
```

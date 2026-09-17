# OMEGA CORE-LM-0 ER32 Integration Contract

This directory implements **Phase 1 only** of `OMEGA-CORE-LM-0-ER32-INTEGRATION-AND-COST-GATE`.
It covers Gates I and II with synthetic CPU FP32 tests. Phase 2 is intentionally absent:
no real 24-update technical benchmark, inference benchmark, runner, schema, results, corpus,
teacher, validation/test data, GPU execution, or training campaign is implemented or run.

## Authoritative Contract

`MD/174.md` is authoritative for all six gate criteria:

1. `INTEGRATION_CORRESPONDENCE`
2. `EXPLICIT_EFFICIENT_EQUIVALENCE`
3. `PERSISTENT_FOOTPRINT`
4. `MEMORY_SAFETY`
5. `TRAINING_COST`
6. `INFERENCE_COST`

This phase tests only criteria 1 and 2. It makes no quality, `delta_K`, or quality-scoping
decision. `delta_K <= 0.10` remains pending.

## Frozen Test Contract

- Device: CPU only.
- Dtype: FP32 only.
- Vocabulary: `V=50257`.
- Rank: `32`.
- Model dimension: `D=128`.
- Integrated slots: `S=8`.
- Integrated rounds: `K=1` and `K=4`.
- Integrated batch/time: `B=2`, `T=8`.
- Loss temperature: `2.0`.
- Loss chunk size: `512`.
- Equivalence tolerance: `ATOL=1e-4`, `RTOL=1e-4`.

Tolerance is frozen. A failed comparison is classified as `NUMERICAL_EQUIVALENCE_FAIL`; tests
must never widen tolerance.

## Implementation Boundary

`omega_fast_er32.py` imports the existing real `OmegaCoreLMFast` from
`omega_core_lm_0_r1_cpu_fastpath_validation/omega_fast_candidate.py` and imports no copied core
implementation. `OmegaCoreLMFastER32` replaces only `self.embedding` with `FactorizedVocabulary`
and overrides only `logits_from_projected`.

`from_f_reference()` accepts an already-created F reference, copies every non-lexical F state
bit-for-bit, and initializes fresh linked `C[V,32]` and `U[32,D]` factors under isolated RNG.
No existing OmegaCoreLMFast, OmegaCoreLM0R1Technical, A/B/C source, checkpoint, or result is
modified.

## Provenance

Source provenance is the current on-disk implementation paths above plus
`run_omega_core_lm_0_r1_training_technical_preflight.py` for test-only fresh R1 reference setup.
Phase 1 creates no result directory, report self-hash, corpus manifest, teacher cache, or git
metadata. Those are Phase 2 concerns and remain unimplemented.

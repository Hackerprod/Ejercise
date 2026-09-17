# OMEGA CORE-LM-0 ER32 Integration Contract

This directory contains **Phase 1 plus Phase 2 code** for
`OMEGA-CORE-LM-0-ER32-INTEGRATION-AND-COST-GATE`. Phase 1 covers Gates I and II with synthetic
CPU FP32 tests. Phase 2 defines the real technical/inference gate, schema, and synthetic unit
tests, but Phase 2 has **not been executed**. Gate execution remains pending user review and
explicit launch authorization.

## Authoritative Contract

`MD/174.md` is authoritative for all six gate criteria:

1. `INTEGRATION_CORRESPONDENCE`
2. `EXPLICIT_EFFICIENT_EQUIVALENCE`
3. `PERSISTENT_FOOTPRINT`
4. `MEMORY_SAFETY`
5. `TRAINING_COST`
6. `INFERENCE_COST`

Phase 1 tests criteria 1 and 2. Phase 2 implementation covers criteria 3 through 6 when
explicitly launched. No quality, `delta_K`, or quality-scoping decision is made;
`delta_K <= 0.10` remains pending.

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

## Phase 2 Implementation Boundary

`run_er32_cost_gate.py` reuses Step2's pinned source/teacher loaders and approved selection
manifest from `run_step2_benchmark.py` and `omega_nominal_microbatch_runner.py`. It keeps the
original CE+KL route, `T=2`, chunk size `512`, CPU FP32 eager execution, and the eight approved
train documents. It does not reopen teacher caches, load validation/test data, traverse 602
documents, alter vocab/chunk/loss, use reduced precision, compile, or use GPU.

Training and inference combinations launch in separate fresh child processes. The default CLI
refuses execution; real execution requires explicit `--execute` and parent-issued child tokens.
Reports use `results/<run-id>/integration_report.json`, `cost_report.json`,
`inference_report.json`, and `ledger.jsonl`. JSON artifacts are self-hashed and reread before
being accepted. No result directory has been created by Phase 2 implementation work.

The six technical training updates intentionally reuse the same two approved source windows
(`source[:, :256]` and `source[:, 256:512]`) for each 0/1 pair. This is a deliberate cost-only
benchmark simplification: NLL is not interpreted as linguistic quality, and the run does not
claim corpus traversal equivalence with A/B/C.

## Provenance

Source provenance is the current on-disk implementation paths above plus
`run_omega_core_lm_0_r1_training_technical_preflight.py` for test-only fresh R1 reference setup.
Phase 1 creates no result directory, report self-hash, corpus manifest, teacher cache, or git
metadata. Phase 2 defines report provenance without executing or creating those artifacts.

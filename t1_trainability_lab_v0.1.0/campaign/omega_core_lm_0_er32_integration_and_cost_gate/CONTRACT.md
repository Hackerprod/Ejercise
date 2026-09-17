# OMEGA CORE-LM-0 ER32 Integration Contract

This directory contains **Phase 1 plus Phase 2 code** for
`OMEGA-CORE-LM-0-ER32-INTEGRATION-AND-COST-GATE`. Phase 1 covers Gates I and II with synthetic
CPU FP32 tests. The authorized Phase 2 run is preserved under
`results/20260917T154901Z-0fb4048b/`. Its raw evidence remains unchanged; later semantic
analysis repairs correct Gate V/VI formulas without rerunning the benchmark.

## Authoritative Contract

`MD/174.md` is authoritative for all six gate criteria:

1. `INTEGRATION_CORRESPONDENCE`
2. `EXPLICIT_EFFICIENT_EQUIVALENCE`
3. `PERSISTENT_FOOTPRINT`
4. `MEMORY_SAFETY`
5. `TRAINING_COST`
6. `INFERENCE_COST`

Phase 1 tests criteria 1 and 2. Phase 2 execution covers criteria 3 through 6. No quality,
`delta_K`, or quality-scoping decision is made;
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
being accepted. `analysis_repair.json` is a separate write-once, self-hashed semantic analysis
artifact; it reads existing cost/inference reports and never overwrites raw evidence.

## Semantic Analysis Repair

The repair path corrects only formula orientation:

- Gate V uses `T_ER32_total / T_F_total`, retaining threshold `1.25` and
  `COST_REGRESSION`/`QUALITY_SCOPING_HOLD` semantics.
- Gate VI uses `ER32_time / F_time`: `ms_per_token` for Mode A and
  `mean_seconds_per_window` for Mode B. RSS anomaly classification is unchanged.
- `benchmark_rerun` is `false` and `raw_measurements_unchanged` is `true`.
- Existing source artifact SHA256 values, old/corrected formulas, corrected metrics, and
  recalculated Gate V/VI classifications are recorded in `analysis_repair.json`.
- The command refuses overwrite and verifies the new artifact by rereading its self-hash:

```text
python run_er32_cost_gate.py --repair-analysis --run-dir results/<run-id>
```

The timing note is explicit: ledger `total_seconds` is captured before append/fsync, while
report training totals are captured after append/fsync. This known definition difference does
not authorize raw mutation or benchmark rerun.

The six technical training updates intentionally reuse the same two approved source windows
(`source[:, :256]` and `source[:, 256:512]`) for each 0/1 pair. This is a deliberate cost-only
benchmark simplification: NLL is not interpreted as linguistic quality, and the run does not
claim corpus traversal equivalence with A/B/C.

## Provenance

Source provenance is the current on-disk implementation paths above plus
`run_omega_core_lm_0_r1_training_technical_preflight.py` for test-only fresh R1 reference setup.
Phase 1 creates no result directory, report self-hash, corpus manifest, teacher cache, or git
metadata. Phase 2 raw reports and ledger are historical evidence; repair provenance is added
only through `analysis_repair.json`.

# ER32 Integration and Cost Gate, Phase 1

Phase 1 proves ER32 structural integration and explicit-vs-efficient equivalence against the
real CPU fastpath. It runs synthetic CPU FP32 tests only. Gates I and II are in scope; Phase 2
real 24-update technical and inference benchmarks are not implemented or run.

## Quick Path

From repository root:

```text
python -m pytest t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_er32_integration_and_cost_gate/test_er32_integration.py
python -m py_compile t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_er32_integration_and_cost_gate/omega_fast_er32.py t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_er32_integration_and_cost_gate/test_er32_integration.py
```

## Phase 1 Gates

| Gate | Coverage |
| --- | --- |
| `INTEGRATION_CORRESPONDENCE` | Real F inheritance, exact non-lexical state copy, factor shapes/sharing, no dense ER32 parameter, K1/K4 shared block structure, parameter reduction. |
| `EXPLICIT_EFFICIENT_EQUIVALENCE` | Level A lexical tensors and loss; Level B real F core forward states/logits/loss, gradients, clipping, AdamW moments, and detached two-window cycle for K1/K4. |

Diagnostics record category, maximum absolute error, and maximum relative error in assertion
messages. Gate outcome remains pass/fail at frozen `ATOL=1e-4`, `RTOL=1e-4`; tolerance is never
increased.

## Explicit Boundary

All six criteria in `MD/174.md` remain authoritative: integration correspondence,
explicit-efficient equivalence, persistent footprint, memory safety, training cost, and
inference cost. Only first two are evaluated here. No quality or `delta_K` decision is made;
`delta_K <= 0.10` remains pending.

The directory intentionally contains no `run_er32_cost_gate.py`, report schema, or results
directory. Phase 2 must be authorized and implemented separately using real approved corpus and
teacher inputs, if Phase 1 passes.

## Provenance and Restrictions

- Real imported F source: `t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_cpu_fastpath_validation/omega_fast_candidate.py`.
- Test-only R1 factory source: `t1_trainability_lab_v0.1.0/scripts/run_omega_core_lm_0_r1_training_technical_preflight.py`.
- Synthetic CPU FP32 tensors only; no corpus, teacher, validation/test split, GPU, compile, or training campaign.
- Existing R1, F, A/B/C sources and results remain untouched.

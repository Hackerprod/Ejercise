# ER32 Integration and Cost Gate, Phases 1-2

Phase 1 proves ER32 structural integration and explicit-vs-efficient equivalence against the
real CPU fastpath. Phase 2 implementation adds the isolated technical/inference runner, report
schema, and synthetic orchestration tests. Phase 2 benchmarks are implemented but **not
executed**; execution remains pending user review and explicit launch authorization.

## Quick Path

From repository root:

```text
python -m pytest t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_er32_integration_and_cost_gate/test_er32_integration.py
python -m pytest t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_er32_integration_and_cost_gate/test_er32_cost_gate.py
python -m py_compile t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_er32_integration_and_cost_gate/run_er32_cost_gate.py t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_er32_integration_and_cost_gate/test_er32_cost_gate.py
```

## Phase 1 Gates

| Gate | Coverage |
| --- | --- |
| `INTEGRATION_CORRESPONDENCE` | Real F inheritance, exact non-lexical state copy, factor shapes/sharing, no dense ER32 parameter, K1/K4 shared block structure, parameter reduction. |
| `EXPLICIT_EFFICIENT_EQUIVALENCE` | Level A lexical tensors and loss; Level B real F core forward states/logits/loss, gradients, clipping, AdamW moments, and detached two-window cycle for K1/K4. |

Diagnostics record category, maximum absolute error, and maximum relative error in assertion
messages. Gate outcome remains pass/fail at frozen `ATOL=1e-4`, `RTOL=1e-4`; tolerance is never
increased.

## Phase 2 Contract

`run_er32_cost_gate.py` imports existing Step2 helpers for the pinned WikiText-2 raw-v1 train
selection, GPT-2 tokenizer, DistilGPT2 teacher, and manifest verification. It uses no teacher
cache and no alternate loss. Four fresh training children execute one combination each, exactly
six updates on `0,1,0,1,0,1`, measuring updates 2-5. Four additional fresh children cover CPU
FP32 eager inference modes A and B. Memory snapshots, exact tensor inventories, timing ledgers,
Gate I-VI classifications, and self-hashed reports follow `MD/174.md`.

The CLI is safe by default:

```text
python run_er32_cost_gate.py
```

prints refusal and launches nothing. The real gate is intentionally not launched in this phase;
`--execute` is an explicit future authorization boundary.

## Explicit Boundary

All six criteria in `MD/174.md` remain authoritative: integration correspondence,
explicit-efficient equivalence, persistent footprint, memory safety, training cost, and
inference cost. Only first two are evaluated here. No quality or `delta_K` decision is made;
`delta_K <= 0.10` remains pending.

The directory contains Phase 2 runner/schema/tests, but no Phase 2 results directory. Real
approved corpus and teacher inputs remain unused until review and explicit launch authorization.

## Provenance and Restrictions

- Real imported F source: `t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_cpu_fastpath_validation/omega_fast_candidate.py`.
- Test-only R1 factory source: `t1_trainability_lab_v0.1.0/scripts/run_omega_core_lm_0_r1_training_technical_preflight.py`.
- Phase 1 uses synthetic CPU FP32 tensors only. Phase 2 code can use only the approved corpus/teacher path when explicitly authorized; it has not loaded corpus or teacher during implementation verification.
- Existing R1, F, A/B/C sources and results remain untouched.

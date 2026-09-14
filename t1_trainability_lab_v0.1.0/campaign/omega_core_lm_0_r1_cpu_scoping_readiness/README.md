# OMEGA CORE LM 0 R1 CPU Scoping Readiness

Sol-authorized local CPU measurement unit. It remains separate from the
efficiency gate and does not alter CUDA, provider, or pilot-readiness artifacts.

## Quick Path

```text
python -m pytest -q t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_cpu_scoping_readiness/test_cpu_scoping_readiness.py
python -m py_compile t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_cpu_scoping_readiness/run_cpu_scoping_readiness.py
python t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_cpu_scoping_readiness/run_cpu_scoping_readiness.py --cache-root C:\Users\danil\.cache\huggingface
```

Stage 3 requires existing local Hugging Face cache and uses
`local_files_only=True` with exact pinned Salesforce/WikiText and
distilgpt2 revisions. Missing or unusable cache fails closed. No download,
fallback teacher, GPU, provider, external communication, or spend occurs.

## Stage Contract

| Order | Scope | Output |
|---|---|---|
| 1 | Random-weight CPU inference, K1/K4, `forward_token` and `forward_window`, persistent state, inference mode | Token/window timing; no language interpretation |
| 2 | Synthetic FP32 training, A/B/C physical batches, effective batch 8, 2,048 valid targets/window | 36 updates, warmup pair 0/1 plus measured pairs 2/3 and 4/5 per config/variant |
| 3 | Real local teacher, exact eight-document manifest, selected physical batch, shared K1/K4 | Four real updates per variant, two complete 0->1 pairs |
| 3-profile | Integrated into stage 3 real workload | Per-update and per-pair `T0-TR-P` phase totals |
| 3-compile | CPU-only second phase after profiling, same source/config/initial FP32 weights | Eager, `torch.compile()` default, and `max-autotune` timing plus equivalence |
| 4 | SCOPE-A arithmetic only | Projection from measured real stage-3 per-update times; no 2,000-update execution |

## T0-TR-P Phase Profile

Each real update reports total update time, finite loss/gradient status, valid
tokens, physical/micro/effective batch, and independently accumulated phase
keys:

```text
teacher_forward_seconds
student_prelude_seconds
student_recurrent_rounds_seconds
readout_seconds
ce_kl_softmax_seconds
student_backward_seconds
optimizer_step_seconds
ledger_instrumentation_seconds
```

Model and runner hooks are optional. With no profile dictionary, existing
forward and runner behavior remains unchanged. Ledger timing covers append,
flush, and fsync only; it is not mislabeled as model work.

## T0-TR-C Compile Comparison

Each variant/mode runs one first execution and two steady updates. Reported
fields separate `compile_time_seconds`, `first_execution_seconds`, and
`steady_state_seconds_per_update`. PyTorch compilation is lazy, so first
execution includes graph compilation when applicable.

Predeclared policy is absolute tolerance `1e-5` for loss, final state,
gradients before optimizer step, and parameter update. Tolerance never changes
after measurement. Within tolerance is `PASS`; any exceedance is
`PERFORMANCE_INTERESTING/SCIENTIFIC_SCOPING_INELIGIBLE`. Compile failures retain
typed failure data, continue other modes, and classify comparison as
`COMPILE_FAILED_CLOSED` rather than relaxing the gate.

## Persistent Outputs

- `cpu_scoping_ledger.jsonl`: append-only stage records and final report pointer.
- `cpu_scoping_report.json`: self-hashed report with command, CPU identity,
  cache/revision provenance, profiling, compile results, gates, and projection.
- `cpu_scoping_report.schema.json`: report contract for T0-TR-P and T0-TR-C.
- `results/runs/*/events.json`: append-only runner event ledgers.

Stage 2 retains exact existing loss, AdamW, clipping, FP32, and causal calendar.
Stage 3 reports operational finite loss and gradients only; it does not
interpret NLL. Stage 4 projection uses measured real-teacher stage-3 times:

```text
hours = 2 seeds x 2,000 updates x (t_shared_K1 + t_shared_K4) / 3,600
```

No BF16, AMP, precision reduction, C++, native work, scientific scope, or
PILOT_EXECUTION_READINESS change is part of this unit.

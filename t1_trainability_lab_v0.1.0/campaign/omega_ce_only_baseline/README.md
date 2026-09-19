# OMEGA CE-Only Baseline

Isolated unit for `OMEGA-CE-ONLY-BASELINE`, Addendum 220, commit reference
`b0e2909`.

## Scientific contract

- Reuses F / `OmegaCoreLMFast`, dimension 128, slots 8, shared K1/K4.
- Uses seeds `20260913` and `20260914`, CPU FP32 eager, 4 intra-op and 1
  inter-op thread, batch 8, window 256, and chunk size 512.
- CE-only loss is full cross entropy. It does not construct, import, or call a
  teacher and performs zero KL calculations.
- The CE path is `recur_states -> project -> chunks -> logits_from_projected ->
  cross_entropy`; no full `[8,256,50257]` logits tensor is materialized.
- Frozen R1 train and validation manifests are read directly. No equivalent
  manifest is rebuilt.
- Initialization gate compares fresh models against R1 `checkpoint_00000.pt`
  for all four seed/K combinations, using `torch.equal` and a state-dict hash.

## Entry points

```text
python run_omega_ce_only_baseline.py --smoke
python run_omega_ce_only_baseline.py --phase0 --confirm-real-execution
python run_omega_ce_only_baseline.py --generation-audit --confirm-real-execution --ce-root <phase-a-output>
python run_omega_ce_only_baseline.py --phase-a --confirm-real-execution --phase0-report <phase0_report.json>
```

`--smoke` emits only a synthetic Phase 0 report. Real Phase A is deliberately
not invoked during implementation/tests. It requires the self-hashed Phase 0
report, and the initialization gate runs before data loading/training.
Phase 0 launches four fresh worker processes, exactly six updates each, and
collects measured timing/RSS rows. Distill workers lazily import the existing R1
teacher/loss path; CE workers never import or construct a teacher. Direct worker
mode also requires explicit authorization unless `--smoke-worker` is supplied.
Generation audit can run CE-only checkpoints through the same eight prompts and
greedy 64-token protocol, then compares 32 CE records with frozen R1 rows from
commit `ca8f4ad`. It remains documentary-only and emits no PASS/FAIL gate.

Phase 0 reports include component timings, RSS, teacher counters, the cost ratio
`q_cost`, even-boundary `U_equal_cost`, and a self-hash. Phase A comparison
helpers compute deltas against existing R1 validation curves, classify
`NONINFERIOR`/`INFERIOR`/`MIXED`, and automatically continue all four CE runs
from checkpoint 2000 when required. Equal-cost results classify
`CE-ONLY-COST-NONINFERIOR`, `DISTILLATION-COST-ADVANTAGE`, or `COST-MIXED`.

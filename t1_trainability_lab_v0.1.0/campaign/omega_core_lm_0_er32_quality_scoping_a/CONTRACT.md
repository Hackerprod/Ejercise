# OMEGA-CORE-LM-0-ER32-QUALITY-SCOPING-A

Phase 1 implements a synthetic-only runner and verification suite. It does not
launch the four real training runs.

## Frozen scope

- Train only ER32 efficient path; explicit factorization remains a technical oracle.
- Seeds: `20260913`, `20260914`.
- Variants: ER32-K1 and ER32-K4.
- Updates: `0..2000`, with checkpoints and validation at `0/500/1000/1500/2000`.
- CPU, FP32 eager, physical/effective batch `8`, one microbatch, `D=128`, `S=8`, rank `32`.
- AdamW: `lr=3e-4`, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0`, clip norm `1.0`.
- Distillation: `0.5 CE + 0.5 T² KL`, `T=2`, chunk size `512`, window `256`.

## Baseline identity

R1 is never trained by this campaign. Each comparison reads the frozen
`validation_curve.json` for the same seed, same K, same update, and same eight
validation documents. R1 deltas are loaded directly from those values; no mean
baseline or copied constants are allowed.

ER32 starts from a fresh R1 reference converted to F, then constructs efficient
ER32 with `experimental_seed=seed`. Non-lexical parameters are copied bitwise.
K1/K4 lexical factors must be bit-identical within each seed. The technical
benchmark seed `20260917` is forbidden.

## Classification at update 2000

- `QUALITY-PROMISING-A`: both seeds satisfy `delta_K1 <= 0.10`,
  `delta_K4 <= 0.10`, and `Delta_ER32 > 0`.
- `QUALITY-NO-GO-A`: both mean quality deltas exceed `0.10` and mean
  `Delta_ER32 <= 0`.
- Otherwise: `QUALITY-MIXED-A`.

This phase never emits `QUALITY_PASS`.

## Safety and recovery

Available memory below `1 GiB` hard-stops before work. Checkpoints are immutable,
ledger records are append-only, and resume is accepted only at a `500`-update
boundary with matching configuration, manifests, and frozen baseline identity.
The test split, A/B/C, F, R1 artifacts, and cost-gate artifacts are outside this
campaign's write set.

# OMEGA-CORE-LM-0-R1 Pilot Training Efficiency Gate

CPU preparation only. This gate proves that physical batch partitioning is
configurable and numerically equivalent before any authorized GPU benchmark.
It produces no scientific result and performs no real training campaign.

## Quick Path

```text
python t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_pilot_training_efficiency_gate/run_efficiency_gate.py --device cpu --dry-run --output-dir <local-output>
python -m pytest -q t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_pilot_training_efficiency_gate
```

CPU mode executes one synthetic update for A/B/C and records no performance
measurement. Its append-only outputs are `efficiency_gate_ledger.jsonl` and
`efficiency_gate_report.jsonl`; each run gets separate runner event ledgers.

## Fixed Contract

| Item | Contract |
|---|---|
| Configurations | A = physical batch 2 x 4 accumulations; B = 4 x 2; C = 8 x 1 |
| Effective update | Batch 8, 2,048 valid targets/window, one optimizer step |
| Numerical recipe | FP32, exact loss, AdamW, gradient clipping, causal window calendar |
| Varying input | Only physical batch partition |
| Held constant | Documents, weights, input/target masks, loss, optimizer, clipping, calendar |
| Precision policy | FP32 only; no BF16, AMP, or other precision reduction |

Each configuration must use identical initial model/optimizer state and the
same deterministic synthetic fixture for CPU equivalence testing. Ledger event
shapes and eight document IDs remain tied to effective batch 8; microbatch
events use the selected physical batch and accurate microbatch count.

## Explicit GPU Boundary

Future CUDA benchmarking requires new authorization and an explicit command:

```text
python .../run_efficiency_gate.py --device cuda --authorize-gpu-benchmark --output-dir <local-output>
```

The command fails if CUDA is unavailable. `--torch-compile` is allowed only in
this FP32 CUDA mode. No GPU command was run for this unit. Rough future spend
estimate is approximately `$1-$2`; this estimate is not authorization.

Cost accounting remains:

```text
C_campaign = t_update × p_GPU × N_updates + evaluation + I/O + overhead
```

This gate measures update efficiency only. It does not choose campaign size,
make 15-GPU campaign decisions, or produce scientific evidence. Full campaign
sizing remains deferred until after an authorized efficiency gate.

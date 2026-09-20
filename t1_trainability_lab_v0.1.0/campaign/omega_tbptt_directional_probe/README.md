# OMEGA TBPTT Directional Probe

Isolated screening unit for truncated backpropagation through time at horizon
16, K4, seed `20260913`, B8, and 256-token windows.

## Contract

- CE-only R1/F model; no teacher, generation audit, secondary validation, K1,
  seed14, or C++ runtime.
- Each 256-token update has 16 chunks. Each chunk calls `backward()` and then
  detaches recurrent state. Parameters accumulate gradients across chunks.
- One gradient clip and one AdamW step happen after all 16 chunks.
- Correctness gate compares incremental backward with a summed-loss reference,
  including loss, gradients, clip norm, parameters, and AdamW state at 1e-5.
- Full-BPTT quality reference is the frozen CE baseline curve; it is never
  retrained.

## Guarded entrypoints

```text
python run_omega_tbptt_directional_probe.py --cost-preflight --confirm-real-execution
python run_omega_tbptt_directional_probe.py --tbptt-training --confirm-real-execution --cost-report results/cost_preflight_report.json
```

Tests and smoke workers do not run real training. Reports are self-hashed.

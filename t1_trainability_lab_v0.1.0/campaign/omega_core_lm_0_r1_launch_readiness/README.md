# OMEGA CORE LM 0 R1 Launch Readiness

**Status: PARTIAL.** Launch remains blocked by image publication. Read-only offer, budget proposal, stop procedure, and execution command are prepared. No Pod, registry, volume, credential, or CUDA action occurred.

## Five Pieces

1. **Image:** Local derived image identity is recorded; published digest is pending because GHCR is blocked.
2. **Offer:** Current live secure L4 evidence is usable for planning at `$0.49/GPU-hour` in `US-MO-2`, with Low stock and no exposed allocatable count.
3. **Budget:** Two-hour proposal is approximately `$0.9951` total, including approximate storage; spend authorization remains pending.
4. **Stop:** `budget_controller.py` is simulation-only. External authorized operator must stop and verify provider state; `q4t3-vol` must never be deleted.
5. **Execution:** R1 CUDA preflight command is prepared but not executed. Architecture, CE/KL, effective batch, and microbatch recipe are unchanged.

## References

- `regional_gpu_quote.json` - fresh 2026-09-14 RunPod read-only quote evidence.
- `budget_proposal.json` - proposed compute and storage costs.
- `stop_and_execution_readiness.md` - stop/terminate authority and prepared preflight command.
- `../omega_core_lm_0_gpu_environment_preparation/budget_controller.py` - unchanged simulation controller.
- `../omega_core_lm_0_gpu_environment_preparation/omega_nominal_microbatch_runner.py` - prepared runner.
- `../omega_core_lm_0_r1_vps_builder_and_image_validation/validation_manifest.json` - prior image validation source.

## Blockers

- Publish derived image to an authorized registry and obtain provider-accepted digest.
- Obtain later explicit spend authorization.
- Confirm allocatable L4 and post-allocation host facts before any runtime action.

# OMEGA CORE LM 0 R1 Launch Readiness

**Status: PARTIAL.** Image is published to the requested GHCR repository with a commit-bound tag, but private visibility and separate-read-credential pull verification remain pending. No Pod, volume mutation, or CUDA action occurred.

## Five Pieces

1. **Image:** Published as `ghcr.io/hackerprod/omega-core-lm-0-r1:a3d7b47`; digest matches the already validated local image. Package privacy and separate-read pull remain unverified.
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

- Confirm package remains private on GitHub and, if available, perform a pull using separate read-only credentials.
- Obtain later explicit spend authorization.
- Confirm allocatable L4 and post-allocation host facts before any runtime action.

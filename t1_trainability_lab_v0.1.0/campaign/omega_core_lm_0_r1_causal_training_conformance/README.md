# OMEGA R1 Causal Training Conformance

This artifact closes authorized R1 causal-calendar, optimizer, source-selection, ledger, and deterministic-backend gaps without running real preflight training.

## Quick Path

1. Review `omega_nominal_microbatch_runner.py` production helpers and `run_preflight` ordering.
2. Run `python -m pytest -q t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_gpu_environment_preparation/test_omega_nominal_microbatch_runner.py`.
3. Run `python -m pytest -q t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_causal_training_conformance/test_causal_training_conformance.py`.

## Fixed Defects

| Area | Conformance rule |
|---|---|
| Calendar | `window_for_update(update)` and `update_window_schedule()` produce `0,1,0,1,...`; window 0 resets state, window 1 records source update `update - 1`. |
| Optimizer | `construct_adamw()` explicitly sets `lr=3e-4`, `betas=(0.9,0.999)`, `eps=1e-8`, and `weight_decay=0.0`. |
| Source | Production calls approved `select_documents()` after reconstruction, compares all eight approved document metadata and exact windows against an embedded v2 pin, validates hashes/shape, then loads teacher. |
| Ledger | Every event preserves prior fields and adds `document_id`, `input_range`, `target_range`, `teacher_context_range`, `window`, `state_reset`, `state_source_update`, and `valid_tokens`. |
| Backend | Each variant receives fixed technical seed and records Torch/CUDA availability/version, deterministic mode, thread counts, and backend flags. |

## Verification

- Existing runner tests: `3 passed`.
- New conformance tests: `9 passed`.
- Source fixture: eight distinct reconstructed document IDs, source shape `(8, 513)`, two windows with `2048` valid target tokens each.
- Negative tests cover old calendar rejection, approved-manifest mismatch, and source hash/length mismatch before model or teacher work.
- Continuity and reset tests use fixed weights and no optimizer step between paired windows.

## Approved Source Pin

- Provenance: `campaign/omega_core_lm_0_r1_training_technical_preflight/training_technical_preflight_v2.json`.
- Artifact SHA-256: `0afa54c708cc746ca6d44503946a6b3c715bec44b7bc97cacbd84ea346d777ac`.
- Embedded pin contains approved indices `0, 1, 2, 3, 4, 5, 12, 13`, exact headers/ranges/token counts/content hashes, and exact causal windows.
- `_preflight_source()` accepts `approved_manifest=` only for explicit synthetic fixtures; production default uses embedded approved pin.

## Safety Boundary

- No RunPod Pod created.
- No money spent.
- No CUDA execution.
- No live model or dataset download.
- No real linguistic training or preflight CLI execution.
- TF32 is explicitly disabled and float32 matmul precision is explicitly set to `highest` by production seed setup.
- Historical files under `campaign/omega_core_lm_0_r1_cuda_preflight/` were not modified.

See `causal_training_conformance_report.json` for machine-readable scope and results.

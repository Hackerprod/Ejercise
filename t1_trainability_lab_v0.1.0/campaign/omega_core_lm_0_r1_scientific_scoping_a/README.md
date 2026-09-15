# OMEGA R1 Scientific Scoping A

Local CPU scoping implementation. Full campaign remains blocked until smoke
output is reviewed and explicitly authorized. No full report is produced by
default. Scientific Scoping A uses candidate F from CPU fastpath validation,
never the Fable proposal module.

## Quick Path

1. Run smoke:

   ```powershell
   python t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_scientific_scoping_a/run_scientific_scoping_a.py --smoke --output-dir t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_scientific_scoping_a/results
   ```

2. Run focused tests:

   ```powershell
   python -m pytest t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_scientific_scoping_a/test_scientific_scoping_a.py -q
   ```

3. Review `results/smoke_report.json`. It must show synthetic data,
   `campaign_started: false`, validation at `0/2/4`, checkpoints at `0/2/4`,
   fresh-process resume, append-only segments, and a classification limited to
   `PROMISING`, `NO_SIGNAL`, or `MIXED`.

## F Integration Smoke

Run bounded real-data integration smoke only; it does not start full campaign:

```powershell
python t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_scientific_scoping_a/run_scientific_scoping_a.py --integration-smoke --output-dir t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_scientific_scoping_a/results
```

This writes `results/integration_smoke/<run-id>/integration_smoke_report.json`.
Both variants run two causal pairs and four updates each. Initial and resume
phases execute in separate child processes. The report records F identity,
candidate source hash, pinned data and teacher revisions, NLL curve points at
`0/2/4`, immutable checkpoints, memory minima, finite-value guards, and
`test_split_loaded: false`.

## Later Full Command

Only after smoke confirmation and explicit authorization:

```powershell
python t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_scientific_scoping_a/run_scientific_scoping_a.py --full --confirm-smoke --output-dir t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_scientific_scoping_a/results
```

This executes four CPU FP32 eager runs: `shared_K1` and `shared_K4` for seeds
`20260913` and `20260914`, 2,000 updates each, 8,000 updates total. Full
validation and immutable checkpoints occur at `0/500/1000/1500/2000` per run.

## Frozen Contract

| Area | Contract |
|---|---|
| Model | F candidate `campaign/omega_core_lm_0_r1_cpu_fastpath_validation/omega_fast_candidate.py`, built with `OmegaCoreLM0R1Technical` then `OmegaCoreLMFast.from_reference`; dimension 128, slots 8, shared core; rounds 1 or 4 |
| Teacher | `distilbert/distilgpt2`, revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`, local cache only, frozen eval mode |
| Dataset | `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`, local cache only |
| Reconstruction | Exact current level-one `= Title =` rule; all eligible reconstructed documents, first 513 GPT-2 tokens, stable first-occurrence dedupe |
| Batch | physical 8, effective 8, one microbatch |
| Windows | `update % 2`; pair `p = update // 2`; eight cyclic documents shared by both windows; window 1 consumes only detached prior window 0 state |
| Loss | Current `distillation_loss`, `T=2`, `0.5*CE + 0.5*KL` |
| Optimizer | AdamW, `3e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0`, clip norm `1.0` |
| Validation | Fixed separate validation manifest, stable order, pure student NLL over target tokens at exact boundaries |
| Classification | Per-seed `K1 final NLL - K4 final NLL`; `PROMISING`, `NO_SIGNAL`, or `MIXED` only |

## Provenance

The runner records SHA-256 values for itself, current R1 dependency, and F
candidate source in every report. Dataset and teacher IDs/revisions are recorded
in manifests, checkpoints, and reports. Synthetic smoke remains separate from
real-data integration smoke. Full manifests are generated only during an
authorized execution.

Hashes are generated at execution time. No freeze, T0, proposal, or aborted-
campaign hash is claimed here.

## Artifacts

- `run_scientific_scoping_a.py`: blocked full driver and synthetic smoke driver.
- `test_scientific_scoping_a.py`: lightweight contract tests.
- `smoke_report.schema.json`: smoke report schema.
- `full_report.schema.json`: full report schema.
- `full_report.template.json`: non-result template; not a campaign report.

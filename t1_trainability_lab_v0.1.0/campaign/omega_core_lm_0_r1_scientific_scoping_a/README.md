# OMEGA R1 Scientific Scoping A

Local CPU scoping implementation. Full campaign remains blocked until smoke
output is reviewed and explicitly authorized. No full report is produced by
default.

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
| Model | Current R1 model imported from `run_omega_core_lm_0_r1_training_technical_preflight.py`; dimension 128, slots 8, shared core; rounds 1 or 4 |
| Teacher | `distilbert/distilgpt2`, revision `2290a62682d06624634c1f46a6ad5be0f47f38aa`, local cache only, frozen eval mode |
| Dataset | `Salesforce/wikitext`, `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`, local cache only |
| Reconstruction | Exact current level-one `= Title =` rule; all eligible reconstructed documents, first 513 GPT-2 tokens, stable first-occurrence dedupe |
| Batch | physical 8, effective 8, one microbatch |
| Windows | `update % 2`; pair `p = update // 2`; eight cyclic documents shared by both windows; window 1 consumes only detached prior window 0 state |
| Loss | Current `distillation_loss`, `T=2`, `0.5*CE + 0.5*KL` |
| Optimizer | AdamW, `3e-4`, betas `(0.9, 0.999)`, epsilon `1e-8`, weight decay `0`, clip norm `1.0` |
| Validation | Fixed separate validation manifest, stable order, student NLL at exact boundaries |
| Classification | Per-seed `K1 final NLL - K4 final NLL`; `PROMISING`, `NO_SIGNAL`, or `MIXED` only |

## Provenance

The runner records SHA-256 values for itself and current R1 source in every
machine-readable smoke/full report. Dataset and teacher IDs/revisions are
recorded in manifests, checkpoints, and full report. Smoke manifests are
synthetic. Full manifests are generated only during an authorized execution.
No freeze hash claim is made here.

Current source/artifact hashes from this smoke implementation:

| Artifact | SHA-256 |
|---|---|
| `run_scientific_scoping_a.py` | `5be14036cb3da4c187469e602d19fa3a705fcb5d05936dc9f6ddb7fb966a01bc` |
| current R1 source | `bb6d86ca2d1185efd1f359d187c0207384aaf301f17fb7290845c31dc091f2d4` |
| smoke train manifest | `24360f081295c45c16a6bbf08d3b57109345912c34734aac069568ccdd96e5a7` |
| smoke validation manifest | `8051090a09aa99a710f2beaf571b9e6f118cf6c7ebc352c37b8ffb67067c07fe` |

## Artifacts

- `run_scientific_scoping_a.py`: blocked full driver and synthetic smoke driver.
- `test_scientific_scoping_a.py`: lightweight contract tests.
- `smoke_report.schema.json`: smoke report schema.
- `full_report.schema.json`: full report schema.
- `full_report.template.json`: non-result template; not a campaign report.

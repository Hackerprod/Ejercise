# OMEGA-CORE-LM-0-R1 Scientific Pilot Freeze

This document is the declarative scientific contract for the OMEGA-CORE-LM-0-R1 pilot. It freezes the recipe before any scientific-pilot result is observed. This directory records no execution.

## Freeze Status

| Item | Frozen value |
|---|---|
| Status | `declarative_only` |
| Scope | Scientific pilot recipe and evidence contract only |
| Execution in this unit | Zero execution |
| Writer model | Single writer |
| Code changes | None |
| Pod, GPU, training, downloads, provider actions | Prohibited |
| Commits | Prohibited |
| Historical CUDA ledgers | Must remain untouched |

No result may be treated as a scientific-pilot result unless it is produced after this freeze by a separately authorized execution unit. This freeze itself creates no Pod, requests no GPU, performs no training, downloads no provider data, and performs no provider action.

## R1 Identity

Architecture R1 is unchanged. The pilot uses the current source at exact code commit `269a4d79b5a1e6df8c230962e2b8c9e237095f18`, which is later than conformance commit `dcf8b04`.

| Reference | Frozen value |
|---|---|
| Model source | `t1_trainability_lab_v0.1.0/t1_trainability/model.py` |
| Corrected production runner | `t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_gpu_environment_preparation/omega_nominal_microbatch_runner.py` |
| Technical preflight reference | `t1_trainability_lab_v0.1.0/scripts/run_omega_core_lm_0_r1_training_technical_preflight.py` |
| Exact code commit | `269a4d79b5a1e6df8c230962e2b8c9e237095f18` |
| Earlier conformance reference | `dcf8b04` |
| Corrected runner SHA-256 | `b4e93b231f8c85e77a4f3a4c1ba726b54c9325cfac145d856d862ed8b71d65cd` |

Variants are exactly:

- `shared_K1`: shared operator, one recurrent round (`K=1`).
- `shared_K4`: shared operator, four recurrent rounds (`K=4`).
- `untied_K4`: untied operators, four recurrent rounds (`K=4`).

No architecture, variant definition, or source behavior may be changed under this freeze.

## Run Budget And Seeds

Each variant run has exactly `20,000` optimizer updates. The campaign has `15` runs: five paired deterministic replicates for each of three variants.

| Field | Frozen value |
|---|---|
| Updates per run | Exactly `20,000` |
| Total updates | `300,000` across 15 runs |
| Effective batch | `8` |
| Targets per update | `256` |
| Physical microbatch | `2` |
| Gradient accumulations | `4` |
| Optimizer steps | Exactly one per update |
| Gradient clipping | Norm `1.0`, exactly once per update, after accumulation and before the optimizer step |
| Master seed | `20260913` |
| Seed formula | `master_seed + replicate_index`, with replicate index `0..4` |

Repeated seed values across variants are intentional and required for paired comparisons. The 15 assignments are explicit:

| Variant | Replicate 0 | Replicate 1 | Replicate 2 | Replicate 3 | Replicate 4 |
|---|---:|---:|---:|---:|---:|
| `shared_K1` | `20260913` | `20260914` | `20260915` | `20260916` | `20260917` |
| `shared_K4` | `20260913` | `20260914` | `20260915` | `20260916` | `20260917` |
| `untied_K4` | `20260913` | `20260914` | `20260915` | `20260916` | `20260917` |

Explicit assignments, one per run:

```text
shared_K1/replicate_0 = 20260913
shared_K1/replicate_1 = 20260914
shared_K1/replicate_2 = 20260915
shared_K1/replicate_3 = 20260916
shared_K1/replicate_4 = 20260917
shared_K4/replicate_0 = 20260913
shared_K4/replicate_1 = 20260914
shared_K4/replicate_2 = 20260915
shared_K4/replicate_3 = 20260916
shared_K4/replicate_4 = 20260917
untied_K4/replicate_0 = 20260913
untied_K4/replicate_1 = 20260914
untied_K4/replicate_2 = 20260915
untied_K4/replicate_3 = 20260916
untied_K4/replicate_4 = 20260917
```

## Data And Teacher

| Item | Frozen value |
|---|---|
| Dataset | Public `Salesforce/wikitext`, config/split `wikitext-2-raw-v1` |
| Dataset revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| Dataset splits | `train`, `validation`, and `test`, untouched |
| Teacher | `distilbert/distilgpt2` |
| Teacher revision | `2290a62682d06624634c1f46a6ad5be0f47f38aa` |
| Teacher state | Frozen, evaluation mode, no gradients |
| Tokenizer | Exact teacher revision, GPT-2 byte-level BPE, expected vocabulary size `50257` |

Document reconstruction and selection MUST reuse the exact `reconstruct_documents()` and `select_documents()` mechanism from `t1_trainability_lab_v0.1.0/scripts/run_omega_core_lm_0_r1_training_technical_preflight.py`. The mechanism is frozen as follows:

1. Reconstruct documents in dataset order from level-one `= Title =` headers. A deeper heading does not start a document.
2. Tokenize each reconstructed document with the exact fixed teacher/tokenizer revision and no special tokens.
3. A document is eligible only when it has at least `513` GPT-2 tokens. Retain its first `513` tokens.
4. Dedupe eligible documents by the pair `(full-text SHA-256, retained-513-token SHA-256)`.
5. Preserve stable dataset order after deduplication.
6. Abort before model work if the train eligible set has fewer than `8` documents.

For train, let `M` be the number of eligible, deduplicated documents. For pair `p = 0..9999`, select eight consecutive eligible documents using:

```text
document_index = ((8 * p + i) mod M), for i = 0..7
```

The selected eight documents are used for window `0` and the immediately following window `1` of that update pair. This is the only declared cyclic traversal. There is no random data sampling and no replacement beyond this declared cyclic traversal.

Validation and test apply the same reconstruction, exact tokenization, eligibility, hash-pair deduplication, and stable-order policy to their own public split. All eligible validation/test documents are evaluated in stable order. Train documents must not leak into validation or test.

## Objective And Optimizer

The loss is exactly:

```text
L = 0.5 * CE + 0.5 * T^2 * KL
T = 2
```

`CE`, masking, reduction, and temperature handling MUST follow `distillation_loss` in the frozen technical preflight script. The KL term is summed over vocabulary and reduced over valid positions as implemented there; no alternative distillation formulation is allowed.

AdamW is exact:

| Parameter | Frozen value |
|---|---|
| Type | `AdamW` |
| Learning rate | `3e-4` constant for all updates |
| Betas | `(0.9, 0.999)` |
| Epsilon | `1e-8` |
| Weight decay | `0.0` |
| Gradient clip norm | `1.0`, exactly once per update |
| Numeric mode | FP32 only |

AMP, quantization, `torch.compile`, external memory, and any unapproved optimization are prohibited.

## Causal BPTT And Ledger

The causal calendar is exact. For zero-based update index `u`, `window = u % 2`:

| Window | State behavior | Source metadata |
|---:|---|---|
| `0` | Reset to initial state before forward pass | `state_reset=true`, `state_source_update=null` |
| `1` | Use state only from immediately preceding window `0` (`u - 1`) | `state_reset=false`, `state_source_update=u - 1` |

The carried state is detached before use in window `1`. Window `1` may never consume state from another window `1`, an earlier pair, or a later update. Window `0` always resets. The declared schedule is `0,1,0,1,...`.

Every append-only ledger event MUST satisfy the metadata contract used by `RunLedger`: `run_id`, `variant`, `update`, `microbatch`, `phase`, `status`, `elapsed_seconds`, `memory`, `document_id`, `input_range`, `target_range`, `teacher_context_range`, `window`, `state_reset`, `state_source_update`, and `valid_tokens`. Events are canonical JSONL, flushed and fsynced; missing fields, invalid phases, schema mismatch, hash mismatch, non-append behavior, or silent reuse/resume is a failure. Each run uses an immutable run identity and preserves all evidence.

## Evaluation And Checkpoints

This schedule is fixed before campaign execution:

- Evaluate validation at update `0`, then every `500` updates through and including update `20,000`.
- Save an immutable checkpoint at update `0`, every `500`-update boundary, and a final checkpoint at update `20,000`.
- Select exactly one checkpoint per run by lowest validation NLL.
- On equal validation NLL, select the earliest update.
- Never use the test set for checkpoint selection, tuning, or any other decision before selection is finalized.
- At the selected checkpoint, evaluate the test set exactly once under the normal causal schedule.
- At the same selected checkpoint, evaluate the anchor-reset ablation exactly once.
- At the same selected checkpoint, record the secondary `state_shuffle` diagnostic exactly once.

Normal evaluation follows the causal schedule above. Anchor-reset evaluates the selected checkpoint on the same evaluation examples while forcing the persistent state to zero before every window `1`, instead of carrying the immediately preceding window `0` state. `state_shuffle` is diagnostic evidence only and never replaces normal evaluation or either required gate.

## Scientific Gates

The gate definitions are exact:

```text
Delta_depth = NLL_shared_K1 - NLL_shared_K4
PASS iff Delta_depth >= 0.05 nats/token
Orientation: K4 better

Delta_ablation = NLL_anchor-reset - NLL_normal
PASS iff Delta_ablation >= 0.01 nats/token
```

For exact textual reporting, the same definitions are named with the frozen symbols:

```text
Δ_depth = NLL_shared_K1 − NLL_shared_K4
PASS iff Δ_depth ≥ 0.05 nats/token
Δ_ablation = NLL_anchor-reset − NLL_normal
PASS iff Δ_ablation ≥ 0.01 nats/token
```

Compute each metric for each paired seed and report all five seed-level values. Campaign aggregation is the arithmetic mean across the five paired seeds for the applicable comparison. Aggregation does not change thresholds. `state_shuffle` remains a secondary diagnostic and can never substitute for a required gate.

## Stop And Failure Rules

Stop the affected run immediately on any of the following:

- NaN or Inf in loss, gradients, or parameters.
- Ledger corruption, schema mismatch, or hash mismatch.
- Recipe drift, including changed source, data, teacher, seed, schedule, objective, optimizer, precision, or evaluation rule.
- Checkpoint corruption or unverifiable checkpoint identity.
- OOM or other resource failure.

Classify an observed scientific-contract violation, corruption, or drift as `FAIL`. Classify an interrupted or resource-limited run whose required evidence cannot establish the contract as `INCONCLUSIVE`. Preserve logs, ledgers, checkpoints, hashes, and partial evidence. Never silently resume, retry under a changed recipe, discard evidence, or alter this freeze. No campaign or gate may be marked `PASS` when required evidence is missing. The test set remains prohibited for checkpoint selection.

## Contract And Platform Separation

The scientific contract above is immutable. Platform, provider, GPU type, image publication, runtime placement, and parallelization are selected later. Those choices may change execution logistics only; they MUST NOT alter the recipe, seed assignments, data traversal, model, objective, optimizer, causal calendar, checkpoints, evaluation, or gates. Platform is not resolved by this freeze.

Actual conformant CUDA measurements provide an estimate only:

| Variant | Measured seconds/update |
|---|---:|
| `shared_K1` | `3.532970` |
| `shared_K4` | `8.927244` |
| `untied_K4` | `8.934740` |

For `300,000` total updates, the estimate is `594.304 GPU-hours`, approximately `$291.209` at `$0.49/GPU-hour`, or `24.763` serial days. This is an estimate, not spend approval, reservation, provider instruction, or execution authorization.

## Environment Identity

The following provenance is part of the frozen identity and must be recorded by any later execution:

| Artifact | SHA-256 |
|---|---|
| `t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_gpu_environment_preparation/requirements.lock` | `38de8910f95cf6894d79e934ebb739a884392c37bc3af5930d59df8b0132d501` |
| Corrected runner | `b4e93b231f8c85e77a4f3a4c1ba726b54c9325cfac145d856d862ed8b71d65cd` |
| `t1_trainability_lab_v0.1.0/scripts/run_omega_core_lm_0_r1_training_technical_preflight.py` | `a36a4a66fb0284c318f9f4228dc17beb64f57faf1106e3dd16ac415cb8c7e961` |
| Design audit synthetic runner (recorded path `scripts/audit_omega_core_lm_0_design.py`) | `326057dcc3de445c836434b87e21d0270d9c1924e393a82a75563e514a81cd44` |
| `t1_trainability_lab_v0.1.0/t1_trainability/model.py` | `bc0250593dd7db03cc185f140c9603a3e63b70afb565bc548ceba21a9eb075a6` |
| `t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_gpu_environment_preparation/Dockerfile` | `6f2ddb87d7485411abc5d77084caa749c0b3a02531724a5cedc4902a4d260432` |
| Dataset revision | `b08601e04326c79dfdd32d625aee71d232d685c3` |
| Teacher revision | `2290a62682d06624634c1f46a6ad5be0f47f38aa` |

Validated backend reference, for identity comparison only:

| Backend field | Reference |
|---|---|
| Torch | `2.11.0+cu128` |
| CUDA | `12.8` |
| GPU driver | L4 driver `580.126.09` |
| Deterministic algorithms | `true` |
| TF32 | `false` |

This backend is a platform reference only. It is not a platform selection or execution request.

## Image Boundary

The old image below predates the corrected runner and is explicitly excluded:

```text
ghcr.io/hackerprod/omega-core-lm-0-r1@sha256:b6834274f18d9dceaa2f5983a9ebfa099d13a2265a6e607303ca1324284ce4e7
```

It MUST NOT be used for the pilot. A future image must be rebuilt and published from the frozen/current code, with its resulting digest recorded in a later authorized execution artifact. This freeze does not resolve, select, publish, pull, or validate any platform image.

## Review Boundary

Review this document and `freeze_manifest.json` for exact recipe identity, explicit seed assignments, evidence requirements, and prohibited actions. Acceptance of this directory is acceptance of a declaration only. It is not evidence that the scientific pilot ran and is not approval to spend, create infrastructure, download data, or observe pilot results.

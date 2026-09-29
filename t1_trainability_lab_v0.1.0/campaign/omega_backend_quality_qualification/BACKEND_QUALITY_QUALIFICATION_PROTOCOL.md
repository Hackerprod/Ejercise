# OMEGA Backend Quality Qualification

**Phase:** preflight authorization granted conditionally for Block A; final identity manifest and real-runner resume contract test are being sealed before any scientific update.  
**Question:** does selected native backend `be376` preserve enough R1 learning quality for later scientific comparisons?  
**Out of scope:** kernel optimization/rewrite, further speed requalification beyond the authorized corrected-loss 8+32 matrix, alternate loss, ER32/ER64, other BPTT horizon, alternate architecture, or erasing `strict_trajectory_proximity=FAIL_RECORDED`.

## 1. Separate identities: R1 architecture and qualified backend configuration

### Architectural reference: historical R1

- Model architecture source reference: R1 freeze commit `269a4d79b5a1e6df8c230962e2b8c9e237095f18`, shared R1 model, `shared_K1` / `shared_K4` only.
- Frozen source reference: `t1_trainability_lab_v0.1.0/t1_trainability/model.py`, SHA `bc0250593dd7db03cc185f140c9603a3e63b70afb565bc548ceba21a9eb075a6`.
- No untied K4, factorization, ER32/ER64, alternate loss, or alternate architecture.
- Original R1 freeze remains unchanged and retains B2×4. This qualification is **not** claimed to conform to that execution batching arrangement.

### Qualified configuration: selected native runtime setup

| Setting | Qualification value |
|---|---|
| Native backend | Final DLL `be37623d7e69b65551ad9c306092f328cc250576a56642fc34a0068df00600cc` |
| `physical_batch` / `effective_batch` | 8 / 8 |
| Gradient accumulation | 1 |
| Native workers | 4 |
| PyTorch threads | intra-op 4 / inter-op 1 |
| R1 rounds | K1 or K4, shared operator only |
| BPTT | 256 tokens/window; window 0 resets; window 1 uses only detached preceding window-0 state |
| Optimizer | AdamW, constant lr `3e-4`, betas `(0.9,0.999)`, eps `1e-8`, weight decay `0` |
| Optimizer steps and clip | One AdamW step per update; one norm-1 clip after backward and before AdamW |
| Numeric mode | CPU FP32, eager, deterministic algorithms, TF32 disabled |
| Native diagnostics/profiling | instrumentation `0`; optional diagnostic pointers null; profile not compiled/enabled |

**Canonical objective:** `R1_MASKED_TOKEN_MEAN_V1`, one shared PyTorch callable applied after either backend's forward. Inputs are `student_logits [B,L,V]`, `teacher_logits [B,L,V]`, `targets [B,L]`, and explicit Boolean `valid_mask [B,L]`. CE is averaged across valid tokens. Teacher distribution is `softmax(teacher_logits/tau)`; student distribution is `log_softmax(student_logits/tau)`; KL direction is `teacher||student`, sum over vocabulary then mean over valid tokens. `tau=2`; apply `tau²` once; total is `0.5*CE + 0.5*KL`. Current full update has 2,048 valid targets; denominator comes from `valid_mask.sum()` and must be positive, never from a hard-coded sequence length. Do not divide the old total by 256, which would also scale CE incorrectly. Canonical implementation is isolated in `r1_masked_token_mean_loss.py` and must pass direct oracle/gradient comparison to the real R1 technical-preflight function before use.

The historical R2 helper `omega_teacher_logit_cache.distillation_loss` remains byte-for-byte preserved and is recorded by SHA `ce025569ea1f3788c25f25d36afc4ae0a053a71698786983b95bb0689074fa1a` as **historical nonconforming for this contract**. It receives `[B,L,V]`, applies `batchmean` to KL without flattening, and therefore weights KL by sequence length relative to R1 token mean. The old R2 runner and reports remain untouched. The new qualified runner calls only the canonical callable for both arms; no monkey-patching, silent import replacement, or reused loss identifier.

## 2. Execution identity chain

The eventual quality-run manifest must pin the actual bytes, not infer identity from a branch name or a seed.

| Role | Exact reference / SHA-256 |
|---|---|
| Selected native DLL | `C:\Users\danil\bpf2a\t1_trainability_lab_v0.1.0\campaign\omega_native_runtime_p2r0\native\build-diagnostics-out-state-candidate-verified\python\omega_recurrent.dll`; `be37623d7e69b65551ad9c306092f328cc250576a56642fc34a0068df00600cc` |
| Native source stage | branch `omega/diagnostics-out-state`; code commit `18e7225b819af61304189eff8537d602f729cf2f`, coverage commit `d993b3c1f1e19e4bffa32d7c4be35687257b28c6` |
| Production FFI bridge | `campaign/omega_native_runtime_p2r0/omega_recurrent_production_bridge.py`; current SHA `7b359d49b093509aab3db6ccc8492c74abc1a3e035b8103e8d27e42813626697` |
| Canonical R2 route source | `campaign/omega_native_runtime_p2r0/run_omega_native_runtime_r2_benchmark.py`; current SHA `e52cccd91ae9aabf13e06fb90eeb91c6a45471a85fccb9882d4ff15e736f07fd` |
| R2 P0 reference helper | `campaign/omega_native_runtime_p0/run_omega_native_runtime_p0.py`; SHA `a7ec530f94d4cf1f98c2650a01da6f3beb609680c5cb6914fbf0b26f1e3cab67` |
| Shared F model / pure-PyTorch reference | `campaign/omega_core_lm_0_r1_cpu_fastpath_validation/omega_fast_candidate.py`; SHA `044f342fce9544df955afe7600017967d639f6f882c7d4d0e25da7dbeb18bf11` |
| Common model factory | `campaign/omega_ce_only_baseline/run_omega_ce_only_baseline.py`; current SHA `9141e3e071cddcb3a078c77b8eab080712b94fe2b26691f664023f788256701c` |
| R1 technical model/preflight | `scripts/run_omega_core_lm_0_r1_training_technical_preflight.py`; current SHA `bb6d86ca2d1185efd1f359d187c0207384aaf301f17fb7290845c31dc091f2d4`; frozen hash in R1 freeze `a36a4a66fb0284c318f9f4228dc17beb64f57faf1106e3dd16ac415cb8c7e961` |
| Canonical qualification loss | `campaign/omega_backend_quality_qualification/r1_masked_token_mean_loss.py`; callable and current source hash are sealed in the new qualification manifest |
| Canonical loss oracle | `scripts/run_omega_core_lm_0_r1_training_technical_preflight.py::distillation_loss`; its current source hash is sealed in the new qualification manifest |
| Historical nonconforming loss helper | `campaign/omega_teacher_logit_cache/run_omega_teacher_logit_cache.py::distillation_loss`; `ce025569ea1f3788c25f25d36afc4ae0a053a71698786983b95bb0689074fa1a`; preserved unchanged and explicitly labeled |
| Microbatch contract reference | `campaign/omega_core_lm_0_gpu_environment_preparation/omega_nominal_microbatch_runner.py`; SHA `e19e000a6f040d26e6413d4137712d227c3ac32f7780e728b2ce5b63bc8e5fdb` |

The difference between current and frozen technical-preflight hashes is optional profile timing hooks; the current model path must call them with `profile=None`. The canonical R2 route and CE helper have local modifications in the D checkout; their exact current hashes above are part of identity. The R2 runner’s changes add measured-interval timestamps; the CE helper changes are generation-audit additions outside `fresh_model`. Never use the stale bpf2a Python runner copy. The new quality runner is separate and records its own hash.

The PyTorch arm uses the pure-PyTorch recurrent route on the same shared F model/state structure used by the native arm. The native arm changes only recurrent execution to `omega_recurrent_production_bridge.apply` with the be376 DLL. Embedding, prelude, readout, tied embeddings, the single canonical loss callable, optimizer, input tensors, and evaluation code are common. Both arms load exactly cloned initialization tensors; no per-update shadow comparison or state restoration is performed.

## 3. Existing controls and valid reuse

- The original `omega_core_lm_0_r1_scientific_pilot_freeze/freeze_manifest.json` is declarative-only; it is not executed training evidence.
- Existing `omega_core_lm_0_r1_scientific_scoping_a/results/full_campaign` contains F/PyTorch-only runs (seeds 20260913–20260915, K1/K4, 5,000 updates, segmented/resumed). Checkpoints contain model, AdamW, RNG, data position, run/source identities; curves and manifests are preserved. They cannot isolate backend, so they are not PyTorch control trajectories. Update-0 checkpoints may be used only as exact initialization oracles after hash/tensor checks; no later optimizer/checkpoint state is imported into a new arm.
- Existing `run_omega_native_runtime_r1_bridge.py` report is a short per-update shadow/equivalence check, not a quality trajectory. Existing R2 stable runs are performance evidence, not long-horizon NLL controls. R2’s stable warmups are real optimizer updates and mutate weights/optimizer; do not reuse them as quality warmups.
- The R1 full-campaign path uses the masked-token-mean R1 helper or its chunked trace equivalent. ER32/ER64 callers were audited individually: trace paths reduce masked token sums; fallback paths flatten `[B,L,V]` to `[B*L,V]` before `batchmean`, which is a token mean for their all-valid masks. The old R2 helper is not the function used in those runs.
- A 60-document validation report already exists at `campaign/omega_expanded_frozen_validation/results/secondary_frozen_validation.json`: 60 documents, WikiText-2 `validation`, all eligible docs, manifest SHA `857800cd725a5648dac54f1a16285e3437be7732f372f179c5b5981480cd3019`. Its generator verifies stable row order, 513-token eligibility, first-occurrence dedupe and excludes train keys. The 8-doc historical scoping manifest is secondary only. The 60-doc list is primary evaluation input for this qualification; test split is not loaded.

## 4. Block A: first bounded experiment

- **Only Block A is authorized conditionally:** seeds `20260913` and `20260914` × K1/K4 × PyTorch/native = 8 independent runs × 2,000 updates = **16,000 total updates** (not two runs of 16,000 each).
- **Block B is not automatic:** seeds `20260915`–`20260917`, same matrix, 24,000 more updates only after separate authorization. Total if approved: 40,000 updates, not 400,000.
- For each `(seed,K)`, create one immutable common update-0 initialization bundle; clone exact model parameters/buffers into independent backend processes and initialize fresh, empty AdamW states. Verify tensor names/order/shapes/hashes, optimizer emptiness, model factory/source and R1 update-0 oracle where available. Sharing a seed alone is insufficient.
- Fixed route ordering, alternated and recorded per pair (P→N, N→P, etc.); fresh process per arm, no concurrency. No optimizer warmups. N starts and advances its own trajectory; P starts and advances its own trajectory. No shadow model, no PyTorch state restoration into native, no switching backend inside a run.
- Use validated train doc identity/order (602 docs) and cyclic formula `doc_index=((8*p+i) mod 602)`, `p=0..999`, `i=0..7`; window 0/1 share the same eight docs. Do not shuffle, add documents, or change the BPTT window. Existing CE-K4 train manifest has exactly 1,000 pairs, 602 docs and 8,000 documents-consumed; it can be reused only after formula/prefix/cache-key verification. The new quality manifest binds its source path/hash and contains the full 1,000-pair schedule plus execution identity chain.
- The hidden cache is `C:\omega_cache\teacher_hidden.fp32`, SHA `b43e9c37f9585e1c5d605df5cc58caba84a10480b4f8f06e747cb8d2f3cd16ff`; cache manifest `campaign/omega_teacher_hidden_cache_probe/results/cache_manifest.json`, SHA `be3e1d2193670be281ef9ef59d6daa52b4569b9ded6c7080081f59e37e94d9f7`. A preflight will compare all 602 doc key pairs × two windows, teacher/tokenizer/context keys, cache file hash, and manifest origin.
- Save immutable checkpoints at updates 0,500,1000,1500,2000; evaluate common PyTorch NLL at those same boundaries. **Primary endpoint is exactly update 2000**, not the best validation checkpoint. Curves are secondary only. Test set is not loaded in this qualification. Selected metrics are evaluated from both saved checkpoints by the same pure-PyTorch R1/F scorer on the verified 60-doc validation set.

### Checkpoint/resume design

Each checkpoint binds model parameters+buffers, complete AdamW state, RNG states, completed/next update, manifest pair/window cursor, recurrent state/boundary, backend ID, native DLL SHA (or PyTorch identity), model/source/harness hashes, data/cache/validation manifest hashes, protocol config, and hash of last canonical ledger event. Ledger is append-only and records pre-clip norm, applied clip coefficient/intervention, one optimizer step/update, and completion status; clipping behavior is unchanged.

No resume is planned for A. If interrupted, only that same backend’s own verified checkpoint may resume in a fresh process with identical identities and parent ledger hash. Updates after last valid 500 boundary remain an attempted noncanonical segment, not confirmed progress. No cross-backend resume, retry with changed seed/config/margin, or silent discard.

### Real-runner checkpoint/resume preflight (separate from Block A)

Before Block A, exercise the actual scientific runner and selected DLL under `R1_MASKED_TOKEN_MEAN_V1` for each PyTorch/native × K1/K4 route: uninterrupted four updates versus three updates, checkpoint at completed update 3, fresh process resume for update 4. Total budget is 32 technical optimizer updates (4 pairs × (4 continuous +3 prefix +1 resumed)); no validation/test evaluation or warmup updates. Checkpoint must restore parameters+buffers, complete AdamW moments and step, RNG, cursor, and detached causal state. Compare continuous/resumed trajectories only within the same backend/K, excluding timestamps and runtime-only metadata; retain exact state/optimizer equality requirements. This validates orchestration, not backend equivalence. Block A starts only after this real-DLL test passes.

### Physical-batch/runtime compatibility

The resolved config is physical batch 8 / runtime workers 4, so current C runtime partition rule (`batch % workers == 0`) is satisfied. PyTorch intra/inter-op remains 4/1. This deliberately differs from the old freeze’s B2×4 accumulation; the original freeze is preserved and not claimed as execution-conformant.

## 5. Analysis and margins fixed before any new result

For each seed `s` after selected update-2000 validation evaluation:

`δ_K,s = NLL_native,K,s − NLL_pytorch,K,s`  
`η_s = (NLL_native,K1,s − NLL_native,K4,s) − (NLL_pytorch,K1,s − NLL_pytorch,K4,s) = δ_K1,s − δ_K4,s`.

Accepted proposed margins: `δ_K1,δ_K4 ∈ ±0.02 nats/token` (~2.02% perplexity ratio) and `η_depth ∈ ±0.01 nats/token` (~1.005% ratio-of-ratios). These are fresh backend-equivalence margins, not inherited from ER32/ER64, the old depth hypothesis, or kernel tolerances.

Run formal analysis **once**, only after all five paired seeds are complete. For each endpoint, report all five seed values and a paired two-sided 90% t interval (TOST α=.05, df=4). QUALIFIED only if all three intervals lie wholly inside their margins; NOT_EQUIVALENT if an interval lies wholly outside a margin; INCONCLUSIVE if an interval overlaps a bound or data is missing. Do not use the Block A two-seed subset for formal equivalence, adjust margins, or automatically start Block B. Report `Δ_depth` descriptively for both backends; do not apply the old 0.05 scientific gate here and do not alter `strict_trajectory_proximity=FAIL_RECORDED`.

## 6. Cost reference

The historical R2-objective estimate is superseded for planning by the authorized canonical-loss 8+32 result at `results/corrected_loss_stable_8p32_20260924T123036/stable_8p32_report.json`. Corrected-loss update-only means: PyTorch K1 `2.46232 s`, native K1 `1.88836 s`, PyTorch K4 `3.62056 s`, native K4 `2.86194 s`. Estimated 2,000-update cost/run: 1.368 h, 1.049 h, 2.011 h, 1.590 h, respectively. Block A is ≈43,332.7 CPU seconds /12.04 CPU-hours; five-seed matrix ≈108,331.7 seconds /30.09 CPU-hours. Excludes validation, checkpoint, startup and preflight overhead. These are cost estimates only, not a quality-equivalence result.

## 7. Preparation status / execution guard

Sol granted conditional GO for Block A after two preflight conditions: (1) this protocol and immutable manifest seal actual DLL/source, canonical loss/source hash, runner/model factory, teacher/cache, train/validation manifests, evaluator, runtime/optimizer config, and analysis plan; (2) the real-runner 32-update checkpoint/resume contract test above passes. Step 1 loss-contract tests and impact audit passed; the original strict smoke remains `FAIL_RECORDED`. A separate corrected-loss common-origin diagnosis passed all local K4/window1 gates and enabled only the 8+32 performance measurement; it did not change the original smoke verdict or `strict_trajectory_proximity`. The corrected-loss 8+32 measurement completed and is bound above. Block A has not started. Once both final sealing and real-runner resume test pass, its authorized 8-run/16,000-update matrix starts automatically; report preflight PASS through DevMCP before launch. Block B still requires separate authorization.

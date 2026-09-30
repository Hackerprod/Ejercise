# OMEGA-V2-1d — Stage A Numerical Calibration (DRAFT)

**Review status:** `DRAFT_FOR_SOL_REVIEW`  
**Classification:** `CALIBRATION_DIAGNOSTIC_ONLY`  
**Architectural verdict:** `NONE`  
**Stage A calibration execution:** `HOLD_PENDING_SOL_SPEC_REVIEW`  
**Stage B held-out qualification:** `HOLD_PENDING_LATER_EXPLICIT_MD`

## 1. Authority, purpose, and boundary

Authority: `MD/329.md` §§6–10 and 12. This is new V2-1d evidence; it does not rescue, repair, or reinterpret the sealed `V2_1C_INVALID_PREFLIGHT` result. The purpose is to compare the frozen candidate_02 full-block Q4 implementation against its scalar-Q4 reference, characterize numerical differences with a well-conditioned descriptive instrument, and separate functional validity checks from numerical diagnostics.

Stage A selects no scientific tolerance, pass/fail threshold, performance claim, residency claim, or architectural verdict. It has no performance timing. After this spec is reviewed and Stage A is separately authorized, run only the calibration cases below, seal and report the results to Sol, then stop. No held-out data or Stage B qualification is part of this unit.

## 2. Frozen candidate and oracle identity

### 2.1 Candidate implementation

Bind the companion harness to the already qualified candidate_02 implementation. Do not edit candidate_02 sources or overwrite/rebuild its sealed KQ executable.

| Item | Identity |
|---|---|
| Candidate | `KQ2_ROW_TILE4_SLOT2_FUSED` (KQ report/native executable identity) |
| Sealed candidate_02 executable SHA-256 | `be5c195e701ccbbf4b606d39382d951ba887b41c1faf96370d2d0ac2a19d084f` |
| `full_block_candidate2.cpp` SHA-256 | `68cfdbca0856ecfcc015f76fdc2613a2581636a6e76234c3897425f47772fc0b` |
| `q4_kernel_candidate2.cpp` SHA-256 | `c15e1618a25888a9b77dacee85ecc373215cdc03c764ca9445de2ade5a90594a` |
| `kq_candidate2.hpp` SHA-256 | `77c6e0305a5eb87408b1838392376955fcbd563d6e8390ee1d6006d808f48bfd` |
| Compiler/build | MSVC 19.44.35229.0; MSBuild 17.14.60; Release x64 |
| Candidate compute options | `/O2 /GL /arch:AVX2 /fp:precise /W4 /EHsc /LTCG` |

The companion may compile and link byte-identical candidate translation units to expose the full-block entry points. Its source manifest, hashes, full compile/link commands, and its own executable hash must be recorded separately. This does not change the frozen candidate executable. If using another build or source snapshot becomes necessary, stop and document exact flags, source hashes, and resulting binary hash before proceeding.

### 2.2 Scalar-Q4 reference

The reference is the frozen `full_block_round_single_thread_candidate2` implementation, which calls `scalar_step` in `omega_v2_1b_candidate_02/src/full_block_candidate2.cpp` (SHA-256 above). Its Q4 projections call `q4_linear_single_thread` in `src/q4_kernel_candidate2.cpp` (SHA-256 above). It uses the same packed Q4 nibbles and FP16 scales as candidate_02; it is not an FP32-weight oracle.

The existing frozen scalar function exposes only the post-round state. A Stage A-local instrumented scalar mirror may expose the per-stage reference checkpoints listed in §4. It must not modify the sealed sources. Before computing or interpreting numerical diagnostics, verify the mirror's post-round states **bitwise** against `full_block_round_single_thread_candidate2` after every round for every calibration cell. A mismatch invalidates the mirror: report `SCALAR_MIRROR_INVALID`, preserve the discrepancy, and make no numerical comparison claim from its checkpoints.

## 3. Calibration inputs and cell inventory

All inputs are calibration-only and deterministic. There are no held-out seeds and no randomly sampled activation inputs.

### 3.1 Weight sources

**d=512 primary source.** Use the exact V2-0 FP32 source tensors used by candidate_02 KQ, then the same Q4 packing path. The source is `omega_v2.core.ContractualCoreBlock` from seed `20260929`; sealed FP32 weight-state SHA-256 is `3a0117ae300ab98286ae840248f417ddd909ca3077c6f828aa0a83bebdc6dc6d`, and the sealed FP32 weight-stream SHA-256 is `65b9179e13d09513765eba3f56f0368480c03237eb1bbbac48c0f383b56ecbb8`. Record the actual packed-Q4-plus-scales hash used by each execution.

**d=640 calibration source (proposal pending Sol).** V2-1c correctness generated this case with `make_seeded_weights(640, 20260929)` in `omega_v2_1_physical/src/q4_layout.cpp` (source SHA-256 `b3e829e3b2a392c4567a65020ee9037b17787e7f43ee877832c80c4abd2226b7`). The generator uses `std::mt19937(20260929)` and `std::uniform_real_distribution<float>` to create FP32 values before Q4 packing. V2-1c's sealed Q4-weight checksum for that generated d640 family is `15453147065333665836` (the checksum is the existing 64-bit FNV-style Q4 checksum, not an FP32-stream SHA-256). No sealed d640 original-FP32 weight stream or SHA-256 was located in V2-1c evidence. Do not invent another seed or describe these as V2-0 d640 weights. Sol must decide whether the deterministic V2-1c generator plus its source hash and sealed Q4 checksum is sufficient, or whether d640 representation-error analysis should remain unavailable pending a sealed FP32 source.

### 3.2 Fixed activation states

For each `(d,m)`, use one fixed state, generated exactly as in V2-1c `initial_state(d,m)`:

```text
N = d*m
for i in [0,N): state[i] = (((i*37 + d + m) mod 127) - 63) / 256.0f
```

There are eight distinct fixed states: `d={512,640}`, `m={1,4,8,16}`. Record the input bytes' SHA-256 and shape in each case record. The only weight seed is `20260929`; activation generation itself is formulaic and uses no RNG. K values replay the same fixed initial state from scratch.

### 3.3 Scientific calibration grid

Run one deterministic case for each cell:

```text
d = {512, 640}
m = {1, 4, 8, 16}
K = {1, 4}
```

This is 16 scientific calibration cells. `m=1` is included as a calibration diagnostic and is not a production or residency claim. The grid is a characterization proposal, not a qualification gate.

### 3.4 Instrumentation smoke only

The existing small recurrence case `d=32, m=4, K={1,4,8}`, with `make_seeded_weights(32,20260929)` and the frozen V2-1b toy-state formula `(((i*19) mod 83)-41)/256.0f`, may be used to validate instrumentation and mirror wiring. Label every artifact `INSTRUMENTATION_SMOKE_NOT_SCIENTIFIC`; exclude it from calibration summaries and all scientific conclusions.

## 4. Output inventory and comparison boundary

### 4.1 Primary outputs

For every scientific cell, compare the row-major FP32 residual state `state_t[m,d]` after **each** recurrent round `t=1..K`, not only the final round. `state_K` is the final full-block output. Comparing each round localizes recurrence accumulation and prevents an earlier error from being hidden by a later state.

### 4.2 Diagnostic checkpoints

Where the frozen candidate or a validated instrumented scalar mirror makes the checkpoint available, record and compare the following at every round:

1. RMS-normalized block input, shape `[m,d]`;
2. Q, K, and V projections, each `[m,d]`;
3. pre-softmax attention logits and post-softmax probabilities, each `[m,m]`;
4. attention context, `[m,d]`;
5. output projection, `[m,d]`;
6. first residual / hidden state, `[m,d]`;
7. MLP RMS-normalized input, `[m,d]`;
8. gate and up projections, each `[m,4d]`;
9. SiLU(gate)×up, `[m,4d]`;
10. down projection, `[m,d]`;
11. final residual / post-round state, `[m,d]`.

The candidate's `Scratch::scores` is overwritten in place from logits to probabilities. If pre-softmax logits are not directly captured by a source-preserving Stage A hook, a deterministic replay derived from the candidate Q/K checkpoints may be reported only as a separately labeled `RECONSTRUCTED_DIAGNOSTIC`; do not present reconstructed logits as a directly emitted candidate output. Any checkpoint unavailable or not bitwise-validated is explicitly `NOT_CAPTURED`/`NO_SEPARABLE`, not silently imputed.

### 4.3 Functional validity vs numerical diagnostics

Keep two independent result sections:

- **Functional/instrument validity:** candidate and reference execute exactly K rounds; dimensions, output inventory, packed-weight identity, and input hashes match the spec; all expected tensors exist with declared shapes; all candidate/reference values are finite; mirror state matches the frozen scalar function bitwise after each round; repeated deterministic execution is bitwise stable. Report each check explicitly. A failure is a functional/instrument failure, not a numeric tolerance failure.
- **Numerical diagnostics:** metric vectors and component comparisons in §§5–6. They are descriptive only. No numerical threshold, tolerance-based PASS/FAIL, or gate is defined in Stage A.

## 5. Numerical metrics (per output, round, and cell)

Flatten each corresponding candidate/reference FP32 tensor in row-major order. Compute differences and reductions in FP64. Let `c` be candidate, `r` scalar-Q4 reference, `delta[i]=c[i]-r[i]`, `N` element count, and `rms(r)=sqrt(sum(r[i]^2)/N)`.

- `E_L2 = ||delta||_2 / ||r||_2` (normwise relative L2).
- `E_inf = ||delta||_inf / ||r||_inf` (normwise relative infinity).
- `max_abs = max_i |delta[i]|` in FP32 output units.
- `scale_aware_max_abs_ratio = max_i |delta[i]| / max(|r[i]|, sqrt(eps32)*rms(r))`, with `eps32=2^-23`. This reports elementwise error relative to a reference magnitude with a reference-RMS-derived floor, rather than magnifying isolated near-zero reference elements. If `rms(r)==0`, report this ratio as `null` plus the zero-scale condition; do not substitute an arbitrary floor.
- `old_max_rel = max_i |delta[i]| / max(|r[i]|,1e-12)`, diagnostic-only compatibility field. Never use it as a primary comparator or gate. Report its worst element's flat index, tensor coordinates, candidate/reference values, signed and absolute difference, and denominator.

For zero reference norm in `E_L2`/`E_inf`, report `null` and a zero-reference-norm flag; absolute errors remain available. Also record reference norm/RMS/maximum, shape, finite counts, metric location(s), and deterministic output checksum. No metric value is labeled pass/fail.

## 6. Error decomposition plan

All decomposition remains `CALIBRATION_DIAGNOSTIC_ONLY` and uses no thresholds.

### 6.1 Same-Q4 implementation difference

The primary candidate-versus-scalar-Q4 comparison uses the exact same packed Q4 values, FP16 scales, initial state, dimensions, and recurrent rounds. Attribute differences only to implementation/operator-order differences, not Q4 representation. Use per-stage checkpoints to identify where differences first appear and how they propagate.

### 6.2 Q4 representation difference

As a separate diagnostic, run the scalar full-block equations with (a) Q4-dequantized weights and (b) the corresponding pre-Q4 FP32 weight values, on identical inputs and operators. This comparison estimates representation error; it is not the candidate-vs-scalar equivalence comparison.

For d512, use the exact sealed V2-0 FP32 source values and hashes in §3.1. For d640, this diagnostic is pending Sol's decision in §3.1 because V2-1c evidence contains a deterministic generator and a Q4 checksum but no sealed pre-Q4 FP32 stream SHA-256. Until approved and properly bound, report d640 Q4-representation attribution as `NOT_AVAILABLE_PENDING_SOURCE_DECISION`.

### 6.3 Operator-level attribution

Use the instrumented checkpoints and controlled scalar/vectorized substitutions where they can be made without modifying the sealed candidate sources:

- Q4 projection / AVX2 reduction order: compare the frozen vectorized Q4 projections with `q4_linear_single_thread` on identical Q4 weights and input tensors; include attention Q·K and attention-value reductions at their matching checkpoints.
- Softmax: both frozen full-block implementations call `std::exp`; candidate_02 does not contain a separate approximate vector softmax. Record logits, probabilities, and context differences. Any standalone attribution to a “softmax approximation” is `NO_SEPARABLE` unless a controlled replay demonstrates a distinct implementation effect.
- SiLU / reciprocal: at identical gate/up checkpoint inputs, compare the frozen candidate's `exp_approx_avx2` and reciprocal-refinement path to the scalar `std::exp`/division path. Report separate effects only if the controlled replay isolates them; otherwise use `NO_SEPARABLE`.
- Recurrence accumulation: report first-residual and final-residual checkpoint differences at each round; identify the first round and stage at which a difference appears.

For every requested component, report the control actually run, checkpoint(s) used, whether attribution was isolated, and any limitation. Use `NO_SEPARABLE` where the available instrumentation cannot isolate an effect; do not force an attribution.

## 7. Prohibitions and stopping rule

- No performance/QPC timing, candidate_02 scaling, residency, cache, bandwidth, or architectural claims in V2-1d Stage A.
- No held-out data, no Stage B execution, and no threshold selection/tuning from these calibration diagnostics.
- Do not modify candidate_02 sources, candidate_02/KQ artifacts, V2-1c artifacts, MD files, or any sealed result.
- The historical V2-1c `max_rel` floor `1e-12` is not reused as a primary comparator. V2-1c remains `INVALID_PREFLIGHT`.
- Sol's review of this complete spec is required before calibration execution. After the one authorized Stage A run, seal outputs and report to Sol, then stop. Stage B requires a later explicit MD and separately preregistered held-out seeds and thresholds.

## 8. Decisions open for Sol (proposals, not ratified)

1. **Grid and inputs — proposed.** Approve the 16-cell scientific grid `d={512,640}`, `m={1,4,8,16}`, `K={1,4}`, using d512's sealed V2-0 seed-20260929 FP32 source, d640's V2-1c seed-20260929 `make_seeded_weights` source, and one deterministic fixed state per `(d,m)`. d32,m4,K={1,4,8} is instrumentation smoke only. Specifically decide whether the unpersisted d640 pre-Q4 FP32 generator output is sufficient for the separate representation-error diagnostic; otherwise keep that diagnostic unavailable for d640.
2. **Outputs and oracle instrumentation — proposed.** Approve per-round residual states as primary outputs and the full checkpoint inventory in §4.2. Approve a Stage A-local instrumented scalar mirror only if it is bitwise cross-checked against the frozen scalar-Q4 function after every round; no sealed source edits. If it does not match, mark it invalid and do not use its checkpoints.
3. **Error decomposition — proposed.** Approve same-Q4 candidate-vs-scalar as implementation/order difference; scalar Q4-dequantized-vs-original-FP32 as a separately labeled representation diagnostic; and controlled checkpoint/substitution attribution for reductions, softmax, SiLU/reciprocal, and recurrence. Unisolated components are `NO_SEPARABLE`; none of these diagnostics creates a numeric gate.

## 9. Required Stage A report fields

The future sealed report must identify spec SHA-256 and reviewed revision; implementation commit; candidate binary/source/shared-dependency hashes; companion build commands/toolchain/hash; weight source, seed, packed-weight hash, input-state formulas and hashes; calibration cell inventory; functional/instrument-validity checks; per-output/per-round metrics and worst-element records; decomposition controls/results including `NO_SEPARABLE`/`NOT_CAPTURED`; and explicit no-timing/no-held-out/no-threshold declarations. Do not generate or populate scientific result fields during spec authoring or harness QA.

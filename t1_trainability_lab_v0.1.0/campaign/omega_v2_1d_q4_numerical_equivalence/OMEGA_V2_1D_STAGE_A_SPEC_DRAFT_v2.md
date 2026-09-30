# OMEGA-V2-1d — Stage A Numerical Calibration (DRAFT_v2)

**Authority:** `MD/329.md` §§6–10, 12 and `MD/330.md` §§1–22.  
**Review status:** `DRAFT_V2_PENDING_SOL_RATIFICATION_OF_CANONICAL_Q4_SHA_SERIALIZATION`  
**Classification:** `CALIBRATION_DIAGNOSTIC_ONLY`  
**Architectural verdict:** `NONE`  
**Stage A scientific execution:** `HOLD_UNTIL_ALL_PRECONDITIONS_IN_§18_PASS`  
**Stage B held-out qualification:** `HOLD`

## 1. Purpose and boundary

Stage A compares the frozen candidate_02 full-block Q4 implementation with the frozen scalar-Q4 implementation, characterizes numerical differences with symmetric and scale-aware descriptive metrics, and reports functional/instrumental validity separately from numerical magnitudes.

Stage A is calibration evidence only. It selects no threshold and makes no numerical PASS/FAIL, speed, performance, residency, or architectural claim. It uses no held-out data and performs no scientific timing. V2-1c remains `V2_1C_INVALID_PREFLIGHT`; V2-1d is new evidence and does not rescue, repair, or reinterpret V2-1c. After the one authorized Stage A execution, seal and report all cells to Sol, then stop. Stage B requires a later explicit MD.

## 2. Frozen candidate, oracle, and source/build identity

### 2.1 Frozen implementations

| Item | Identity |
|---|---|
| Candidate | `KQ2_ROW_TILE4_SLOT2_FUSED` |
| Sealed candidate_02 KQ executable SHA-256 | `be5c195e701ccbbf4b606d39382d951ba887b41c1faf96370d2d0ac2a19d084f` |
| `omega_v2_1b_candidate_02/src/full_block_candidate2.cpp` SHA-256 | `68cfdbca0856ecfcc015f76fdc2613a2581636a6e76234c3897425f47772fc0b` |
| `omega_v2_1b_candidate_02/src/q4_kernel_candidate2.cpp` SHA-256 | `c15e1618a25888a9b77dacee85ecc373215cdc03c764ca9445de2ade5a90594a` |
| `omega_v2_1b_candidate_02/src/kq_candidate2.hpp` SHA-256 | `77c6e0305a5eb87408b1838392376955fcbd563d6e8390ee1d6006d808f48bfd` |
| Shared `omega_v2_1_physical/src/q4_layout.cpp` SHA-256 | `b3e829e3b2a392c4567a65020ee9037b17787e7f43ee877832c80c4abd2226b7` |
| Shared `omega_v2_1_physical/src/v2_1.hpp` SHA-256 | `96c77cfaf3e6b8e1a1661401411d8641e0e96a258e637c02615b232723bbb594` |
| Shared `omega_v2_1_physical/src/worker_pool.cpp` SHA-256 | `3549c67d69765beafbaabd821e49b94dbab47681d33485e2f3f10ecf2f2e6270` |
| Frozen build | MSVC 19.44.35229.0; MSBuild 17.14.60; Release x64 |
| Frozen compute flags | `/O2 /GL /arch:AVX2 /fp:precise /W4 /EHsc /LTCG` |

The scalar-Q4 oracle is frozen `full_block_round_single_thread_candidate2` / `scalar_step` in `full_block_candidate2.cpp`; its scalar Q4 projections call `q4_linear_single_thread` in `q4_kernel_candidate2.cpp`. Candidate and scalar use the same packed Q4 nibbles and FP16 scales. The scalar oracle is not an FP32-weight oracle.

The companion must link candidate source translation units byte-identically to these frozen sources; no sealed candidate source or executable may be modified or overwritten. If recompilation/build configuration is necessary, preserve exact source hashes, compiler version, compile commands, link command, all flags, companion executable SHA-256, and sealed KQ executable SHA-256 in a build manifest.

### 2.2 Companion binding preflight

Before any Stage A scientific cell, reproduce the established d512 KQ cells using the same d512 source weights and historical fixed state:

| Cell | Sealed KQ output |
|---|---|
| `d512,m4,K1` | `candidate2_full_m4_k1.bin` |
| `d512,m16,K1` | `candidate2_full_m16_k1.bin` |
| `d512,m8,K4` | `candidate2_full_m8_k4.bin` |

The companion's frozen-candidate final state must be bitwise identical to each corresponding sealed KQ output. No timing ratio is used. Record individual comparisons and the companion/KQ executable identities.

- All three bitwise comparisons pass: `COMPANION_BINDING_PASS` (mandatory prerequisite).
- Any mismatch: `COMPANION_BINDING_INVALID`; Stage A remains on HOLD.

## 3. Weight provenance and binding

### 3.1 d512 source and representation diagnostic

Use the exact V2-0 FP32 source tensors used by candidate_02 KQ, source `omega_v2.core.ContractualCoreBlock`, seed `20260929`, then the frozen Q4 packing path. The sealed source FP32 weight-state SHA-256 is `3a0117ae300ab98286ae840248f417ddd909ca3077c6f828aa0a83bebdc6dc6d`; the sealed source FP32 weight-stream SHA-256 is `65b9179e13d09513765eba3f56f0368480c03237eb1bbbac48c0f383b56ecbb8`. Bind and record the exact seven-family packed-Q4-plus-FP16-scales bytes consumed by each execution.

The historical V2-0 FP32 bytes are available and are used directly for the d512 scalar Q4-vs-FP32 comparison. Its classification is `DIRECT_CALIBRATION_DIAGNOSTIC`.

### 3.2 d640 regeneration and historical-Q4 binding

Regenerate d640 weights with `make_seeded_weights(640, 20260929)` from the exact frozen `omega_v2_1_physical/src/q4_layout.cpp` implementation, using the same frozen toolchain. The actual matrix-family fields and generator order in that source are:

1. `W_Q`, shape `[640,640]`;
2. `W_K`, shape `[640,640]`;
3. `W_V`, shape `[640,640]`;
4. `W_O`, shape `[640,640]`;
5. `W_gate`, shape `[2560,640]`;
6. `W_up`, shape `[2560,640]`;
7. `W_down`, shape `[640,2560]`.

These are the actual `CoreWeights` members filled in this order by `make_seeded_weights`; the checksum covers packed Q4 and FP16-scale bytes for these seven families.

Before any d640 scientific cell, require:

```text
regenerated_q4_checksum == 15453147065333665836
```

Create and persist a new canonical SHA-256 over the seven named matrix families and their shape, logical packed bytes, and logical FP16-scale bytes. “Logical bytes” are `AlignedBuffer.logical_bytes` only, excluding allocator padding/alignment (`allocated_bytes-logical_bytes`). Scale bytes are the FP16 logical bytes as stored, in little-endian order on the frozen Windows x64 target.

#### Canonical serialization — `PROPOSED_PENDING_SOL_RATIFICATION`

The current MD/330 checksum decision does not prescribe a byte-level serialization for the additional SHA-256. The following exact format is proposed, not frozen pending Sol's ratification:

- Prefix the stream with the fixed ASCII domain tag `OMEGA-V2-1D-D640-Q4-CANONICAL-V1` followed by one NUL byte.
- In the seven-family order above, append for each matrix, with no separators: `uint32_le(name_utf8_byte_length)`, UTF-8 matrix-name bytes, `uint32_le(number_of_dimensions)` (2), each dimension as `uint32_le`, `uint64_le(packed_logical_byte_length)`, packed logical bytes, `uint64_le(scale_logical_byte_length)`, then FP16-scale logical bytes exactly as stored (little-endian on the frozen target).
- SHA-256 the concatenated byte stream. Persist both the exact serialization version and resulting digest.

Do not treat this proposal as ratified or use a digest from another serialization. If Sol does not ratify it, hold d640 binding until Sol specifies the exact serialization. If the historical Q4 checksum does not match, terminal status is `D640_WEIGHT_BINDING_HOLD`; do not execute any d640 scientific cell and report the mismatch.

### 3.3 d640 regenerated FP32 source

The historical pre-Q4 d640 FP32 stream was not sealed. A newly regenerated stream must not be described as the historical original FP32 tensor. Its source identity is:

```text
D640_FP32_SOURCE = REGENERATED_FP32_SOURCE_BOUND_TO_HISTORICAL_Q4
historical_pre_Q4_FP32_bytes_available = false
```

The regenerated source uses the same frozen `q4_layout.cpp`, seed `20260929`, frozen toolchain, and is eligible only for the scalar-Q4-dequantized vs FP32 **calibration representation diagnostic**, after the regenerated packed-plus-FP16 representation matches the historical checksum. Its classification is `RECONSTRUCTED_CALIBRATION_DIAGNOSTIC`. d640 candidate-vs-scalar same-Q4 is `DIRECT_CALIBRATION_DIAGNOSTIC`. No d640 reconstructed representation result proves a claim about unavailable historical FP32 bytes; neither representation diagnostic changes candidate-vs-scalar comparison.

## 4. Calibration states and scientific grid

Native candidate and scalar implementations consume the exact same pre-generated contiguous raw FP32 state bytes. Native C++ must not regenerate `CAL_RANDN` states or use an RNG.

### 4.1 State family A — `HISTORICAL_FORMULA`

For `N=d*m`, row-major flat index `i`:

```text
state[i] = (((i*37 + d + m) mod 127) - 63) / 256.0f
```

This is the V2-1c historical fixed state and is also used in companion binding.

### 4.2 State family B — `CAL_RANDN`

Use V2-0 CPU FP32 random-input semantics:

```python
STATE_SEED(d, m) = 20260930 + d + m
g = torch.Generator(device="cpu").manual_seed(STATE_SEED)
state = torch.randn((m, d), generator=g, dtype=torch.float32)
```

Make the tensor contiguous, persist its exact row-major raw FP32 bytes and SHA-256 **before native execution**, and record Python/PyTorch versions, seed, dtype, device, shape, and byte order. Both native frozen implementations consume these exact bytes.

### 4.3 Frozen Stage A grid

```text
d            = {512,640}
m            = {1,4,8,16}
K            = {1,4}
state_family = {HISTORICAL_FORMULA,CAL_RANDN}
```

This is `2 × 4 × 2 × 2 = 32` scientific calibration cells. `m=1` is calibration/diagnostic only. Do not add K8 to the scientific grid.

### 4.4 d32 instrumentation smoke

Keep the d32 smoke unchanged: `d=32`, `m=4`, `K={1,4,8}`, `make_seeded_weights(32,20260929)`, and the V2-1b toy-state formula `(((i*19) mod 83)-41)/256.0f`. Classification: `INSTRUMENTATION_SMOKE_NOT_SCIENTIFIC`. It must never enter threshold selection or Stage A numerical summaries.

## 5. Primary outputs and frozen comparison

For each of the 32 scientific cells, persist the complete row-major FP32 residual state after every recurrent round. For K1 persist one state; for K4 persist rounds 1, 2, 3, and 4. These frozen-candidate/frozen-scalar residual states are the primary numerical evidence.

The primary comparison always calls the frozen candidate implementation and frozen scalar-Q4 implementation; an instrumented mirror never replaces either. Preserve each frozen output's exact bytes, shape, checksum, cell, state family, and round.

## 6. Diagnostic checkpoint inventory and provenance

At each round, record and compare when available:

1. input RMSNorm output `[m,d]`;
2. Q, K, V projections `[m,d]`;
3. pre-softmax attention logits and post-softmax probabilities `[m,m]`;
4. attention V accumulation/context `[m,d]`;
5. output projection `[m,d]`;
6. first residual/hidden `[m,d]`;
7. post-attention/pre-MLP RMSNorm output `[m,d]`;
8. gate and up Q4 projections `[m,4d]`;
9. SiLU(gate)×up `[m,4d]`;
10. down projection `[m,d]`;
11. final residual/post-round state `[m,d]`.

Every checkpoint field must carry exactly one provenance label:

- `FROZEN_DIRECT`
- `VALIDATED_MIRROR`
- `RECONSTRUCTED_DIAGNOSTIC`
- `NOT_CAPTURED`

The candidate's `Scratch::scores` is overwritten from logits to probabilities. Any replay/reconstruction of candidate logits is `RECONSTRUCTED_DIAGNOSTIC`, never silently presented as a directly emitted frozen output. Do not mix provenance classes without labeling each checkpoint.

## 7. Instrumented mirrors and validity

Two Stage A-local mirrors may be used for diagnostic checkpoints:

- `SCALAR_INSTRUMENTED_MIRROR`
- `CANDIDATE_INSTRUMENTED_MIRROR` (only if candidate checkpoints require a copied/instrumented local implementation)

Neither mirror replaces the frozen functions in the primary comparison. For every scientific cell and every round:

```text
scalar_mirror_post_round_state == frozen_scalar_post_round_state  (bitwise)
candidate_mirror_post_round_state == frozen_candidate_post_round_state  (bitwise; if used)
```

If a mirror mismatches, mark that mirror `INVALID`; it does not erase or invalidate the primary frozen-vs-frozen residual comparison. Checkpoint attribution from that mirror is `NOT_USABLE`; affected decomposition fields become `NOT_CAPTURED` or `NO_SEPARABLE`. No checkpoint-level scientific statement may rely on an invalid mirror.

## 8. Primary normwise metrics

For matching candidate tensor `c` and scalar-Q4 tensor `r`, let `delta=c-r`. All reductions are FP64. These are symmetric primary descriptive metrics:

```text
E_L2  = ||delta||2  / max(||c||2,  ||r||2)
E_inf = ||delta||inf / max(||c||inf, ||r||inf)
```

If both norms in a denominator are exactly zero, define the corresponding E as `0` and persist `both_zero=true`. These values have no threshold and no numerical PASS/FAIL interpretation.

## 9. Primary scale-aware elementwise diagnostic

Let `eps32=2^-23` and:

```text
s = sqrt(eps32) * max(RMS(c), RMS(r))
scale_aware_max_abs_ratio = max_i |c_i-r_i| / max(|c_i|, |r_i|, s)
```

If both tensors are identically zero, define the value as `0` and persist `zero_scale=true`. This is the primary well-conditioned elementwise diagnostic for later Stage B threshold design, but Stage A chooses no threshold. Persist at least `max_abs`, `RMS(c)`, `RMS(r)`, `||c||2`, `||r||2`, `||c||inf`, and `||r||inf` with this diagnostic.

## 10. Historical relative diagnostic

Retain only as `HISTORICAL_COMPATIBILITY_DIAGNOSTIC`:

```text
old_max_rel = max_i |c_i-r_i| / max(|r_i|,1e-12)
```

Never use it as a gate or primary comparator. Persist the worst flat index, tensor coordinates, candidate/reference values, and absolute error. The report may additionally include reference-normalized `E_L2_ref`/`E_inf_ref`, explicitly non-primary.

## 11. Functional and instrumental validity (only PASS/FAIL checks)

The only PASS/FAIL-type checks in Stage A are:

- exact expected round count;
- expected shapes and output inventory;
- all required tensors finite;
- weight identity bound;
- input hashes exact;
- frozen candidate deterministic repeat;
- frozen scalar deterministic repeat;
- required companion binding is `COMPANION_BINDING_PASS`.

Report each check individually. Numerical candidate-vs-scalar magnitudes receive no threshold, no PASS/FAIL, and no numerical terminal verdict.

## 12. Error decomposition

All decomposition is `CALIBRATION_DIAGNOSTIC_ONLY`. Report the executed control, checkpoint provenance, whether the effect was isolated, and limitations. Where not isolated, use `NO_SEPARABLE` or `NOT_CAPTURED`; never force an attribution.

At minimum inventory and report:

1. input RMSNorm reduction/order (candidate AVX2/FMA accumulation plus lane reduction vs scalar accumulation);
2. Q4 projection/reduction;
3. Q·K reduction;
4. softmax;
5. attention V accumulation;
6. output projection;
7. first residual;
8. post-attention/pre-MLP RMSNorm reduction/order (candidate AVX2/FMA plus lane reduction vs scalar accumulation);
9. gate/up Q4 projections;
10. SiLU/reciprocal approximation;
11. down projection;
12. final residual;
13. recurrent accumulation across rounds.

Same-Q4 candidate-vs-scalar differences measure implementation/operator-order differences, not Q4 representation. The separate Q4 representation diagnostic is specified in §3.1 and §3.3.

## 13. Softmax attribution

Both frozen full-block implementations use `std::exp`; the frozen candidate contains no distinct approximate softmax exponential. Probability differences may arise from differing logits caused by reduction order. A controlled replay with identical logits may classify `SOFTMAX_OPERATOR_EQUAL_UNDER_IDENTICAL_INPUT` or reveal another effect. Without such a replay, record:

```text
SOFTMAX_STANDALONE_ATTRIBUTION = NO_SEPARABLE
```

Do not describe softmax as an approximation error analogous to SiLU.

## 14. SiLU attribution

The frozen implementations differ:

- candidate: `exp_approx_avx2`, approximate reciprocal, Newton refinement;
- scalar: `std::exp`, ordinary division.

A controlled replay with identical gate/up inputs is approved. Report approximation effects separately only when the replay isolates them; otherwise report `NO_SEPARABLE`.

## 15. Terminal vocabulary

Use only `STAGE_A_DIAGNOSTIC_COMPLETE` when all required scientific cells and artifacts complete. Operational/instrument failures may use:

- `COMPANION_BINDING_INVALID`
- `D640_WEIGHT_BINDING_HOLD`
- `INSTRUMENTATION_FAILURE`
- `EXECUTION_FAILURE`

`NUMERICAL_PASS` and `NUMERICAL_FAIL` are prohibited Stage A statuses.

## 16. Execution metadata, prohibitions, and stop rule

Use one external launcher for Stage A. Record its exact command, stdout, stderr, UTC start/end, and `time_ns`/process wall time as operational metadata only. No QPC or scientific performance timing.

No candidate_02 timing/scaling, residency, cache/bandwidth, or architectural claim; no held-out data; no Stage B execution; and no post hoc threshold selection/tuning. Preserve V2-1c as `INVALID_PREFLIGHT`. Do not modify candidate_02 sources/executables, V2-1b/KQ artifacts, V2-1c artifacts, MD files, or sealed results.

After the first scientific cell, do not rerun under the same Stage A ID. Preserve and report any pre-scientific operational abort before any retry. Any ambiguity, source drift, binding mismatch, or QA failure means STOP and report to Sol; do not execute scientific cells.

## 17. Required source seal and Stage A report

Before scientific execution, seal the reviewed spec and source/build manifest. The source/build manifest records all source SHA-256 values, exact compiler version, compile commands, link command, all flags, companion executable SHA-256, sealed KQ executable SHA-256, and byte-identity of candidate translation units. Record d512 and d640 weight identity bindings and persisted state bytes/hashes.

The Stage A report identifies spec/source-seal identities; companion binding; d640 checksum and canonical SHA version/digest; all 32 cells; each state's family, seed/formula, source bytes and SHA-256; frozen primary outputs after every round; checkpoint provenance; functional/instrument validity; symmetric metrics and historical diagnostics; decomposition results; launcher metadata; and explicit no-timing/no-held-out/no-threshold declarations. State explicitly `historical_pre_Q4_FP32_bytes_available=false` for d640 reconstructed representation diagnostics.

## 18. Required PASS preconditions before any scientific cell

All must be satisfied and preserved before the single Stage A execution:

1. spec committed;
2. source/build manifest committed;
3. companion binding `COMPANION_BINDING_PASS`;
4. d640 Q4 binding PASS (historical checksum and Sol-ratified canonical SHA serialization);
5. d32 instrumentation smoke PASS;
6. unit/static tests PASS;
7. source seal created;
8. working tree clean.

No scientific cell may start unless every precondition passes exactly. The Stage A execution authorization is conditional on this checklist; this draft does not authorize launching a cell before it is met.

Once every §18 precondition is satisfied exactly, one Stage A calibration execution is GO. No further Sol review is required if the judge verifies that the frozen spec incorporates MD/330 literally and QA/binding are PASS. If any ambiguity, binding mismatch, source drift, or QA failure appears, STOP and report to Sol; do not execute scientific cells.

## 19. Representation diagnostic status and provenance wording

| Dimension / comparison | Classification | Provenance wording |
|---|---|---|
| d512 scalar Q4 vs exact source FP32 | `DIRECT_CALIBRATION_DIAGNOSTIC` | V2-0 seed-20260929 exact source and hashes in §3.1 |
| d640 frozen candidate vs frozen scalar, same Q4 | `DIRECT_CALIBRATION_DIAGNOSTIC` | regenerated Q4 bytes bound to historical checksum in §3.2 |
| d640 scalar Q4-dequantized vs regenerated pre-Q4 FP32 | `RECONSTRUCTED_CALIBRATION_DIAGNOSTIC` | `REGENERATED_FP32_SOURCE_BOUND_TO_HISTORICAL_Q4`; `historical_pre_Q4_FP32_bytes_available=false` |

The d640 reconstructed-source comparison is calibration diagnostics only and may not be cited as evidence about unavailable historical FP32 bytes.

## 20. Stage A completion and subsequent hold

After the one authorized run, seal and report all 32 cells, checkpoint provenance, decomposition results, and the d640 reconstructed-source caveat to Sol; then STOP. Stage B remains HOLD until Sol later freezes its primary metrics, exact thresholds, held-out seeds, whether K8 is added, and final grid. Do not touch Stage B data beforehand.

## 21. Decisions open for Sol

1. **Canonical d640 Q4 SHA-256 serialization — pending ratification.** Ratify or replace the exact proposal in §3.2. No canonical SHA-256 is considered bound until Sol ratifies the byte serialization; d640 scientific execution remains on HOLD until then.

All other listed MD/330 design amendments are incorporated in this DRAFT_v2. Once Sol ratifies the canonical serialization, the revision may be frozen subject to the Stage A process prerequisites in §18.

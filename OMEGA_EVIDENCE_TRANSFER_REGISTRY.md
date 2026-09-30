# OMEGA_EVIDENCE_TRANSFER_REGISTRY

STATUS: v1.0 CANDIDATE (2026-09-29). Documentation only: built by reading git history, sealed reports and result files; no measurement, training or binary was executed. Awaiting Sol ratification; it supersedes draft v0.1 but is not itself a ratified decision.

Authority: Sol MD/314 (systematic transferability audit and table format), MD/315 (hardware re-baseline), MD/316 (V2-1c terminal), MD/317 (registry ordered as unit A, hardware baseline V3 ratified, V2-1d ordered).

Sources: `OMEGA_AUDITORIA_CONFORMIDAD_COMPLETA_2026-09-29.md` (sections 3, 5, 14, Appendix B), `OMEGA_HARDWARE_BASELINE_V3.md`, `MD/314.md` to `MD/317.md`, `T1.5_Spec_MIX_O.md`, V2-0 / V2-1 / V2-1b / V2-1c seals and reports, `Trash/cpu-native-arch/` (CPU-Z report, T0 summaries, provenance manifest), `git log`.

Method: strictest class applied when in doubt. Any value not confirmed in an artifact is written `unknown` with the reason. Commit cells give the commit that created or sealed the artifact (or the range for multi-commit families); Appendix A lists the first and last commit of every directory. Class assignments are never more favourable than the audit; where stricter, `reason` says so.

## 0. Change log v0.1 to v1.0 candidate

- Rows: 30 (v0.1, Spanish) to 64 (this file). The T0, T1/T1.5, T2..T7, d128 and K-curve families were split so each row maps to real directories. All 426 campaign entries (379 directories + 47 loose files) and all 53 `Trash/cpu-native-arch/sweep-output` entries are assigned to exactly one row (Appendix A). Family rows list their member directories in Appendix A.
- `nv` commit cells replaced by real commit hashes from `git log`; truncated be37623d and 2fdeac27 completed to full SHA-256.
- Old T0 CPU model resolved from `Trash/cpu-native-arch/hardware/cpuz_report.txt`: AMD Ryzen AI 5 330 (attributed, not stamped in the run artifacts; see section 1.3).
- V2-1c added (R62, VALID_WITH_SCOPE, terminal V2_1C_INVALID_PREFLIGHT). The v0.1 statement that V2-1 attempt_02 is superseded by "V2-1c in progress" is removed: V2-1c produced no timing and does not supersede it.
- LN row split into R55 (LN_PROTOCOL, reusable methodology) and R56 (CURRENT_L1_BANK, INSTRUMENT_INVALID_L1, Phase B frozen).
- "Physical target Ryzen AI 5 330" replaced by PRIMARY_RESEARCH_PLATFORM i7-13700F + GTX 1650 SUPER and LEGACY_DEPLOYMENT_PROFILE Ryzen AI 5 330 (see `OMEGA_HARDWARE_BASELINE_V3.md`). Column `transferable_to_target_Ryzen` keeps its MD/314 name but means "transferable to the LEGACY_DEPLOYMENT_PROFILE".
- Teacher-cache host corrected from i7 (v0.1) to the legacy laptop by build date (2026-09-19/20, before the 2026-09-27 migration).
- Language unified to English; column names and class names unchanged.

## 1. Reference contract and hardware hierarchy

### 1.1 Contract (V2 target)

- Contractual block: 4d^2 (Q/K/V/O without bias) + 12d^2 (SwiGLU, hidden 4d) = 16d^2. d512 = 4,194,304 recurrent weights; d640 = 6,553,600.
- m workspace slots in {4,8,16}; K flexible rounds with a shared block and no per-round tables; R4 vs U4 clones.
- d128 surrogate: FastWorkspaceUpdateBlock, 12d^2+10d = 197,888 weights, m=8, K=1/4/6, finite tables depth_embedding[K,D] and gate_logits[K,D], tied head 50,257x128. It fulfils neither 16d^2 nor flexible K (CONFORMANCE_HOLD, audit 2026-09-29).

### 1.2 Hardware hierarchy (OMEGA_HARDWARE_BASELINE_V3, MD/315, ratified MD/317)

| Role | Platform / profile |
|---|---|
| PRIMARY_RESEARCH_PLATFORM | i7-13700F + GTX 1650 SUPER (RunPod only after explicit GO) |
| PRIMARY_CPU_PHYSICAL_PROFILE | 8 P-cores / 8 threads |
| THROUGHPUT_PROFILES | 8P/16T and 24T (24T is not a residency gate) |
| LEGACY_DEPLOYMENT_PROFILE | Ryzen AI 5 330, 4 physical cores; device no longer owned, not measurable |
| CPU Q4 native runtime | SECONDARY_ENGINEERING_TRACK |

Universality rule (MD/317): no rho or cache-knee threshold from `Conversacion.md` is universal. Host-specific physical gates must be re-derived before use. Nothing in this registry may be read as an i7 to Ryzen (or Ryzen to i7) transfer of a residency number.

### 1.3 Host attribution legend (column `hardware`)

Per-run `cpu_model` fields are NOT stamped in the pre-migration artifacts. Host codes state how the host was determined:

- `H-LEGACY (inferred, not stamped)`: attributed to the legacy Ryzen AI 5 330 laptop because the run date precedes the 2026-09-27 i7 migration (session memory) and, for T0, because workers are pinned to CPUs 0,2,4,6 and `Trash/cpu-native-arch/hardware/cpuz_report.txt` (commit 4683fb9) reports "AMD Ryzen AI 5 330 w/ Radeon 820M" (4C/8T, L2 1 MiB per core, L3 8 MB). Not verifiable from the run artifacts themselves.
- `H-I7 (inferred by date, not stamped)`: run on or after the migration date.
- `H-I7 (stamped cpu_model)`: the artifact itself prints "13th Gen Intel(R) Core(TM) i7-13700F".

## 2. Evidence classes and transfer rules

| Class | Meaning |
|---|---|
| VALID_DIRECT | The artifact measured or verified exactly the contractual V2 object (16d^2 block, flexible K); its claim is citable without scope reduction inside what it declares. |
| VALID_WITH_SCOPE | Valid and reproducible, but only for the declared scope (scale, host, task, method). Not citable outside it. |
| SURROGATE | Obtained on a substitute (d128, different block, toy). Diagnostic or engineering guidance; not a verdict on the contractual core. |
| NO_TRANSFER | Not citable as evidence for V2 or for the LEGACY_DEPLOYMENT_PROFILE. Methods may be reused, results may not. |
| SUPERSEDED | Replaced by a later measurement or decision; historical value only. |

Rules:

- A rho or cache measurement on the i7-13700F is host-specific: it neither proves nor refutes residency on the Ryzen AI 5 330 (L2, LLC and topology differ), and vice versa. Hence no physical-measurement row has `transferable_to_target_Ryzen = yes`.
- A cache-resizing system would be SURROGATE_DIAGNOSTIC, not a substitute for a real host.
- Every row with `transferable_to_V2 = no` requires re-measurement or re-test on the contractual block before it supports a V2 claim.
- Historical Ryzen measurements cannot be repeated (device not owned); they stay `historical only`.

## 3. Registry

Cell conventions: `unknown` = not verifiable from artifacts, reason given. `MD/N` = `MD/N.md`. Row ids R01..R64 are used by Appendix A. The bracket after each artifact_id states how many directories/files the row covers; the per-directory member list is in Appendix A.

| artifact_id | commit/hash | hardware | d/m/K | block_formula | evidence_class | claim_scope | transferable_to_V2 | transferable_to_target_Ryzen | reusable_components | forbidden_claims | reason | superseded_by |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| R01 T0-M-PHASE3-RERUN (`t0m-phase3-median10-rerun`) [1 Trash dirs] | ec16f83 (2026-09-03); moved to Trash/ in 99121f6 | H-LEGACY (inferred, not stamped); CPU-Z report `hardware/cpuz_report.txt` (4683fb9) names AMD Ryzen AI 5 330; workers pinned to CPUs 0,2,4,6 | D=512 int8; m, K unknown (summary does not state them) | synthetic int8 GEMV microbenchmark, not the 16d^2 block | VALID_WITH_SCOPE | status=PASS_STRONG matrixization (G8/G16 above gate at many points) in the measured d512 int8 scope with 4 workers (audit 3.1 cites 4,240 measured processes; not recounted here) | with scope (compute-bound reference only) | historical only (device not owned; cannot be re-measured) | median-of-10 protocol, STOP-invariant gates, worker pinning, single explicit self-test preflight | Citing it as residency, as speed of the contractual block, or as proof that any V2 kernel reaches a given MAC/s or GB/s | Audit 3.1 rates T0-M matrixization VALID / PASS_STRONG in measured d512 scope; a stricter class is assigned because the block is synthetic int8 and the CPU model is attributed, not stamped | - |
| R02 T0-M-PHASE3-ORIG (`t0m-phase3`) and empty `t0m-phase3-median10` [2 Trash dirs] | 4683fb9 (2026-09-02) for `t0m-phase3`; `t0m-phase3-median10` is an empty untracked directory (no commit) | H-LEGACY (inferred, not stamped) | D=512 int8; m, K unknown | synthetic int8 | SUPERSEDED | status=STOP_INVARIANT_GATE, 448 invocations (summary); first Phase 3 pass | no | no | STOP-invariant gate design | Citing the STOP as a hardware or method failure; citing the empty directory as evidence | Replaced by the median10 rerun (PASS_STRONG). `t0m-phase3-median10` holds zero files (verified by listing) | T0-M-PHASE3-RERUN |
| R03 T0-M-RECURRENCE-D512 (`t0m-recurrence-*`, `t0m-small-static-control-d512`) [9 Trash dirs] | ec16f83 (2026-09-03); moved to Trash/ in 99121f6 | H-LEGACY (inferred, not stamped) | D=512; S=1..16, R=1..16 (25 cells) in performance runs; workers=4 (CPUs 0,2,4,6); m, K not mapped to the contract slots | real int8 GEMV + Norm/Requantize/residual recurrence probe, not 16d^2 | SURROGATE | All summaries status=PASS (harness validation, not a residency verdict). Residency A/B at D=512,S=1,R=16: A_median 2.766e9 MAC/s, B_median 2.941e9 MAC/s, A_over_B=0.9408 (no shared-weight throughput advantage in that run). Descriptive only | no (engineering guidance only) | historical only (device not owned; cannot be re-measured) | recurrence probe design, A/B pair-adjacent ordering, checksum determinism checks | Treating PASS as a residency gate result; deriving rho from it; citing 0.9408 as a property of the contractual block or of any other host | Synthetic probe on the legacy host; no cache-level accounting; the static-control comparisons inside are explicitly "descriptive and non-equivalent" (summary). MD/317: host cache knees are not universal | V2-1 attempt_02 (contractual 16d^2 A/B/C on i7) |
| R04 T0-M-DIAGNOSTICS (barrier, checksum-order, fresh-verification, L1 comparisons, legacy-compare, R1/R16 diagnostics, sharding) [19 Trash dirs] | ec16f83 (2026-09-03) for all but `t0m-sharding-diagnostic` (4683fb9, 2026-09-02) | H-LEGACY (inferred, not stamped) | D=512 and related probe shapes; m, K unknown | synthetic int8 kernel variants | NO_TRANSFER | Kernel-variant diagnostics. Recorded statuses include DIAGNOSTIC_COMPLETE, PASS and FAIL: `t0m-barrier-diagnostic` STOPPED_R1_GATE_FAIL; `t0m-r16-spillfree-compare` FAIL; `t0m-r16-alternating-order-diagnostic-prebuild-stale-exe` FAIL (stale executable, invalid by construction) | no | no | diagnostic patterns (order alternation, self-test gating, stale-executable detection) | Citing any number from these directories as a V2, residency or kernel-quality result | Debugging-level evidence for one legacy kernel; several members are explicit FAIL or stale-binary runs; nothing in them measures the contractual block | - |
| R05 T0-BRIDGE-1-LITERAL-FUSED (`t0m-bridge1-literal-fused-*`) [3 Trash dirs] | ec16f83 (2026-09-03) | H-LEGACY (inferred, not stamped) | D=1472 (and a rectangular variant); S=1, R=16; not d512/d640 | synthetic int8 fused kernel, not 16d^2 | NO_TRANSFER | status=PASS for all three; in `t0m-bridge1-literal-fused-d1472` A_over_Bclone=1.4840 with exact checksum equality over 10 paired runs | no | historical only (device not owned; cannot be re-measured) | byte-identical Bclone construction, checksum-equality proof | Using A_over_Bclone=1.484 as a residency result or as a counterweight to the Bridge-1 FAIL | d1472 is not a contractual size and the Bridge-1 sequence was stopped (see T0-BRIDGE-1); PASS here validates a control construction, not the architecture thesis | - |
| R06 T0-BRIDGE-1 (`t0m-bridge1-residency-d1472`) [1 Trash dirs] | ec16f83 (2026-09-03) | H-LEGACY (inferred, not stamped) | d1472 (not contractual); m, K unknown | synthetic int8, not 16d^2 | NO_TRANSFER | status=FAIL: A_over_Bclone=0.8619, outside the expected T0-R range; summary states Bridge 2/3/4 must not start | no | no (host-specific; LEGACY_DEPLOYMENT_PROFILE device no longer owned, not measurable) | design lesson: the B reference must be a clean non-resident control | Citing it as a refutation or confirmation of residency; starting Bridges 2/3/4 | Audit 3.1: FAIL_RECORDED against a historical reference of insufficient validity; d1472 is not d512/d640 | V2-1 attempt_02 |
| R07 T0-R-INT8-SHARDED (`t0r-int8-sharded`, 240 rows) [1 Trash dirs] | 4683fb9 (2026-09-02); provenance manifest `archive/t0r/t0r-provenance-manifest.md` in the same commit | H-LEGACY (inferred, not stamped); provenance manifest cites D:\ASUS.json topology (4 physical cores, 8 logical) and the CPU-Z report | D unknown; m, K unknown | synthetic int8 sharded A/B/C | NO_TRANSFER | Measured DRAM ceiling 32.9295 GB/s; 240 valid rows; CPUs 0,2,4,6 | no | no (host-specific; LEGACY_DEPLOYMENT_PROFILE device no longer owned, not measurable) | sweep and sharding procedure, corrected-accounting lesson (8x repetition inflation bug found and fixed) | Using B_over_measured_dram as a clean control; using the DRAM ceiling as a property of any host other than the legacy laptop | Audit 3.1: DEGRADED; B has depth blocks while A/C have one, so small depths can stay cache-resident, and B_over_measured_dram is derived from MAC/s, not from a traffic counter. Stricter than the audit for residency verdicts | V2-1 attempt_02 |
| R08 T0-R-BCLONE (`t0r-bclone-*`, 6 dirs) [6 Trash dirs] | ec16f83 (2026-09-03) for `original-frozen` and `control-original`; 91993f7 (2026-09-03) for the four timing dirs | H-LEGACY (inferred, not stamped) | unknown | synthetic int8 | NO_TRANSFER | `t0r-bclone-original-frozen-20260903`: status=STOP_CONTROL_FAILURE, control gate FAIL (frozen A_over_B median 2.2122 vs confirmed reference 2.6876). The four 91993f7 timing dirs (block-loop, defender-excluded, ready, warmup-affinity) have no summary file; their outcomes were not read | no | no (host-specific; LEGACY_DEPLOYMENT_PROFILE device no longer owned, not measurable) | Bclone control idea; Defender-exclusion and warm-up/affinity timing checks | Citing any Bclone ratio as residency evidence | The control gate failed, so the ratios cannot support a residency verdict | V2-1 attempt_02 |
| R09 T0-H0-PROBES (`h0-*`, `h0_sweep.*`) [4 Trash dirs, 7 Trash loose files] | 4683fb9 (2026-09-02); loose files h0_probe.*, h0_sweep.*, manual.*.tmp were committed to Trash/ in 99121f6 (2026-09-10) | H-LEGACY (inferred, not stamped) | unknown (manual.stdout.tmp shows a scalar Q4 probe m=819, K=512 with 8 active logical CPUs) | Q4 AVX2 GEMV microprobe, not 16d^2 | SURROGATE | Microbenchmark sweeps (1 to 8 logical workers). Caveats printed in the summaries: approximate resident working set; SMT, scheduler, boost, thermal and affinity state affect results; physical-core identity not hard-coded in the 8-worker runs | no | historical only (device not owned; cannot be re-measured) | H0 method (later re-implemented for V2-1c) | Citing throughput numbers as V2 kernel performance or as a cache-knee threshold | Host-specific microbenchmark; MD/317: no cache-knee threshold is universal | V2-1c H0 (i7) for host-specific selection |
| R10 TRASH-PREDECESSORS (`Trash/` non-OMEGA work) [sub-results only, no dedicated directory] | moved to `Trash/` in 99121f6 (2026-09-10, "move non-OMEGA predecessor work"); origin commits not surveyed except `cpu-native-arch` (4683fb9..91993f7) | unknown | n/a | n/a | NO_TRANSFER | Directories under `Trash/`: cpu_native_recurrence_lab_v0.4.0 and v0.4.1 (+ .zip), relational-language-engine, mrdl-production-ready-3.0.0, remote-phase-e, `Trash/sweep-output` (t0m-r1-repeat-diagnostic, t0r), the T1..T1.4 spec documents, patches and scripts. Not opened one by one | no | no | historical specs may inform method choices (read them as prior art) | Citing any result from `Trash/` other than through the explicit T0 rows above | Explicitly retired as non-OMEGA predecessor work; classified as a family, not individually | - |
| R11 T1-TOY-CORE (pointer chasing, typed-slot, associative recall, multi-hop, sequential update SU1/2/4/5, variable binding, gate-init, wd-norm, `runs`, P2 replication + campaign-level logs) [20 campaign dirs, 21 loose campaign files] | 4fc5589 (2026-09-04, "full typed-slot campaign checkpoint") | H-LEGACY (inferred, not stamped) | d=64 (config.json: pointer/associative/multihop/sequential/variable-binding); rounds 4 (sequential update 6); slots 1 (pointer) or 4 (gate-init); vocabulary and m otherwise unknown | toy shared core, not 16d^2 | VALID_WITH_SCOPE | Learnability of synthetic memory/binding tasks with a shared recurrent core at d=64 (audit 3.2: T1_SYNTHETIC_TRAINABILITY VALID_WITH_SCOPE) | with scope (mechanism, not scale) | no | task generators, oracles, metrics, replication protocol | Extrapolating to d512/d640, to real language, to flexible K, or to any host property | Audit 3.2: valid in scope; does not establish language nor d512 conformance. Directories `associative_typed_replication`, `multihop_pointer_replication` show no d/m/K keys in the JSON surveyed (unknown) | - |
| R12 T1-W (`t1w_*`) [4 campaign dirs] | 80d6a6a (2026-09-04) | H-LEGACY (inferred, not stamped) | d=64 (config.json); rounds 1/2/4/6 per the draft registry (not re-read) | toy shared core with workspace, not 16d^2 | VALID_WITH_SCOPE | Workspace/identity tasks at d=64 | with scope (mechanism) | no | workspace task protocol | As T1-TOY-CORE | Same audit 3.2 rating | - |
| R13 T1-U0A-ISO (`u0a_iso_clean_seed*_12000`) [5 campaign dirs, 1 loose campaign files] | 852a4cb (2026-09-05, 5 seeds); `u0a_u0b_provenance_archive.json` in 88ecbc9 | H-LEGACY (inferred, not stamped) | d=64 (config.json), 12,000 steps; m, K unknown | toy, not 16d^2 | VALID_WITH_SCOPE | Isolated clean-training controls over five seeds | with scope (mechanism) | no | five-seed protocol | As T1-TOY-CORE | Audit 3.2 (T1 valid in scope) | - |
| R14 T1-U0B-ABLATIONS (loose `u0b_b1..b9`, `u0b_b5_*`) [11 loose campaign files] | per-file commits, all 2026-09-05 (bcc4884, 1a792c7, b1add52, a5f3e06, e442b11, 7b555df, fdf397f, 4e98a6c, b828590, 7b9fde6, 57a1a9e) | H-LEGACY (inferred, not stamped) | unknown (JSON files not opened) | toy | VALID_WITH_SCOPE | U0B ablation result files that exist only as loose JSON at the campaign root; contents were not read for this registry | with scope (mechanism) | no | ablation list (pointer residual, frozen pointer, write-E disabled, raw register, permutations, workspace replace/frozen, no-payload, zero reader payload) | Citing individual ablation numbers without opening the file | Class inherited from the T1 audit rating; individual contents unverified | - |
| R15 T1-U0C-C0-C1 (`u0c_c0_*`, `u0c_c1_joint*`, `u0c_c1_lossnorm*`, `u0c_c1_workspace_audit*`, `u0c_select_integration*`) [16 campaign dirs, 1 loose campaign files] | multiple 2026-09-05 commits (e.g. fb14208, c4a66d8, a1588c4, 285368c, 7100832, d78d2f0, a4c7be5, 9c9ab4c, 71413bb); per-dir in Appendix A. `u0c_correction_capacity_audit.json` in fdf397f | H-LEGACY (inferred, not stamped) | d=64 (config.json); 12,000 steps for the seed sweeps; m, K unknown | toy, not 16d^2 | VALID_WITH_SCOPE | Oracle/reader/joint controls and loss-normalisation variants at d=64 | with scope (mechanism) | no | loss-normalisation and annealing recipes as prior art | Reading loss-norm comparisons as evidence about d512 or about the V2 loss | Audit 3.2 | - |
| R16 T1.5-MIX-O (`u0c_c1_mix_o_*`, `u0c_c1_read_set_validation`) [18 campaign dirs] | 2026-09-06 commits (d8136a2, acdc18a, 51b83fa, c2187f2, 5b54410, ce29d39, d3f6d3d, 0db782e, 1911bf2, 7e223e5); per-dir in Appendix A; addenda in `T1.5_Spec_MIX_O.md` | H-LEGACY (inferred, not stamped) | d unknown (JSON does not expose it under the keys searched); seeds 101..505; m, K unknown | toy | VALID_WITH_SCOPE | Composition, depth, memory and E/R/ALU controls of the mixed task at toy scale | with scope (control design) | no | control design and verdict criteria | Claiming generalisation of the contractual core | Toy scale; task mixture is not language | - |
| R17 T1.5-CTRL-1 (`u0c_ctrl1_*`) [2 campaign dirs] | 2026-09-06 commits: d945111, b2953f3 | H-LEGACY (inferred, not stamped) | parameter_count 4,225 (Linear(64,64)->SiLU->Linear(64,1) probe, config.json) where exposed; otherwise d/m/K unknown | toy control probe | VALID_WITH_SCOPE | Control 1: pilot and preflight (seed101, frozen) | with scope (control design) | no | ablation-control design (CTRL-1..7 in `T1.5_Spec_MIX_O.md`) | Using the control probe as evidence about the contractual core | Toy scale; class inherited from the T1/T1.5 rating | - |
| R18 T1.5-CTRL-2 (`u0c_ctrl2_*`) [26 campaign dirs] | 2026-09-06 commits: c3f0ebe, 6d6eeac, 7b63a95, 0fe27bd, 1954e35, a2427cc, 30c20fa, b5c5950, b7a0bbb, df34e14 | H-LEGACY (inferred, not stamped) | parameter_count 4,225 (Linear(64,64)->SiLU->Linear(64,1) probe, config.json) where exposed; otherwise d/m/K unknown | toy control probe | VALID_WITH_SCOPE | Control 2: G coverage, O canon/consistency diagnostics, ordinal audit, replicas seeds 2202-2205 | with scope (control design) | no | ablation-control design (CTRL-1..7 in `T1.5_Spec_MIX_O.md`) | Using the control probe as evidence about the contractual core | Toy scale; class inherited from the T1/T1.5 rating | - |
| R19 T1.5-CTRL-3 (`u0c_ctrl3_*`) [10 campaign dirs] | 2026-09-06 commits: a6c98be, dc1b62f, 3bf4a52, fb4d91a | H-LEGACY (inferred, not stamped) | parameter_count 4,225 (Linear(64,64)->SiLU->Linear(64,1) probe, config.json) where exposed; otherwise d/m/K unknown | toy control probe | VALID_WITH_SCOPE | Control 3: canonical vs real-R, seeds 2201-2205 | with scope (control design) | no | ablation-control design (CTRL-1..7 in `T1.5_Spec_MIX_O.md`) | Using the control probe as evidence about the contractual core | Toy scale; class inherited from the T1/T1.5 rating | - |
| R20 T1.5-CTRL-4 (`u0c_ctrl4_*`) [4 campaign dirs] | 2026-09-06 commits: 1f2f2c8, b232158, 2cbfc13 | H-LEGACY (inferred, not stamped) | parameter_count 4,225 (Linear(64,64)->SiLU->Linear(64,1) probe, config.json) where exposed; otherwise d/m/K unknown | toy control probe | VALID_WITH_SCOPE | Control 4: evaluation, pilot, interop, preflight | with scope (control design) | no | ablation-control design (CTRL-1..7 in `T1.5_Spec_MIX_O.md`) | Using the control probe as evidence about the contractual core | Toy scale; class inherited from the T1/T1.5 rating | - |
| R21 T1.5-CTRL-5 (`u0c_ctrl5_*`) [3 campaign dirs] | 2026-09-06 commits: f382678, 1240f0c | H-LEGACY (inferred, not stamped) | parameter_count 4,225 (Linear(64,64)->SiLU->Linear(64,1) probe, config.json) where exposed; otherwise d/m/K unknown | toy control probe | VALID_WITH_SCOPE | Control 5: evaluation, pilot, preflight | with scope (control design) | no | ablation-control design (CTRL-1..7 in `T1.5_Spec_MIX_O.md`) | Using the control probe as evidence about the contractual core | Toy scale; class inherited from the T1/T1.5 rating | - |
| R22 T1.5-CTRL-6 (`u0c_ctrl6_*`) [4 campaign dirs] | 2026-09-08 commits: 16a2da7, e6f5a31, 65f23ee | H-LEGACY (inferred, not stamped) | parameter_count 4,225 (Linear(64,64)->SiLU->Linear(64,1) probe, config.json) where exposed; otherwise d/m/K unknown | toy control probe | VALID_WITH_SCOPE | Control 6: clamp/trained evaluation, pilot, preflight | with scope (control design) | no | ablation-control design (CTRL-1..7 in `T1.5_Spec_MIX_O.md`) | Using the control probe as evidence about the contractual core | Toy scale; class inherited from the T1/T1.5 rating | - |
| R23 T1.5-CTRL-7 (`u0c_ctrl7_*`) [4 campaign dirs] | 2026-09-08 commits: aa9ce74, 5635d74, b054450 | H-LEGACY (inferred, not stamped) | parameter_count 4,225 (Linear(64,64)->SiLU->Linear(64,1) probe, config.json) where exposed; otherwise d/m/K unknown | toy control probe | VALID_WITH_SCOPE | Control 7: held-out, pilot, trained evaluation, preflight | with scope (control design) | no | ablation-control design (CTRL-1..7 in `T1.5_Spec_MIX_O.md`) | Using the control probe as evidence about the contractual core | Toy scale; class inherited from the T1/T1.5 rating | - |
| R24 T2-I0 (`t2_i0_*`) [31 campaign dirs] | 2026-09-08..09-09; 30 dirs, per-dir commits in Appendix A (e.g. a6d715b, e42c2dd, 470592c, c26d902, 05c4c3d, c1e57a9, ba6d594, e3a4539, 741ec56, 1be6c8c ...) | H-LEGACY (inferred, not stamped) | unknown; tiny models (~25 KB per earlier survey) | unknown, clearly not 16d^2 | SURROGATE | I0 baselines A/B, R1/R1.1 controls and held-out, R2 controls/held-out across seeds 5701-5705 | no | no | task/oracle protocols, control layout | Citing as evidence for the contractual architecture | Scale is minimal and d/m/K/block are not exposed by the JSON keys searched (or are toy-scale); the strictest class was chosen over VALID_WITH_SCOPE (draft v0.1 decision, kept) | V2-2 (future) |
| R25 T2-I1 (`t2_i1_*`) [3 campaign dirs] | 8acc41e, ac8a18d (2026-09-09) | H-LEGACY (inferred, not stamped) | unknown | unknown, not 16d^2 | SURROGATE | I1 seed 5801 development, controls, held-out | no | no | protocol | As T2-I0 | Scale is minimal and d/m/K/block are not exposed by the JSON keys searched (or are toy-scale); the strictest class was chosen over VALID_WITH_SCOPE (draft v0.1 decision, kept) | V2-2 (future) |
| R26 T2-I2 (`t2_i2_*` + loose `t2_i2_*.json`) [6 campaign dirs, 8 loose campaign files] | 2026-09-10..09-11: 9719c0c, b0ffb1b, 146725e, 812b596, 6e3ab28, b5f3c9c | H-LEGACY (inferred, not stamped) | unknown | unknown, not 16d^2 | SURROGATE | I2 architecture reports, smoke tests, R1/R2/R3 (alg and non-alg) | no | no | architecture-report format | As T2-I0 | Scale is minimal and d/m/K/block are not exposed by the JSON keys searched (or are toy-scale); the strictest class was chosen over VALID_WITH_SCOPE (draft v0.1 decision, kept) | V2-2 (future) |
| R27 T2-I3 (`t2_i3_*` + loose `t2_i3_*.json`) [42 campaign dirs, 3 loose campaign files] | 2026-09-11..09-12; 46 dirs, per-dir commits in Appendix A (e.g. 23760da, 1d729f6, 8070ffd, bc994c2, 3e1b605, 7774140) | H-LEGACY (inferred, not stamped) | parameter_count 1,072 and rounds 2 exposed in one config; K0..K8 depth sweeps; d/m otherwise unknown | tiny model, not 16d^2 | SURROGATE | I3 depth (K0-K8), COMP-0, G1-G4, think diagnostics. `t2_i3_view0_seed6401` is an empty untracked dir (no evidence) | no | no | depth-sweep harness, COMP-0 calibration approach | Citing K0-K8 curves as flexible-K evidence for V2 | Scale is minimal and d/m/K/block are not exposed by the JSON keys searched (or are toy-scale); the strictest class was chosen over VALID_WITH_SCOPE (draft v0.1 decision, kept) | V2-2 (future) |
| R28 T2-XF-CLAUSE (`t2_xf_clause_*`) [17 campaign dirs] | 2026-09-09; commits 78ca247, fdde50a, f753ed4, c934f1c, ad8f8c3, 1598425, 43d2b3c | H-LEGACY (inferred, not stamped) | transformer control with 5,380 parameters (earlier survey); d/m/K of the recurrent arm unknown | not 16d^2 | SURROGATE | Clause task, R1.1 controls and held-out, seeds 6001-6005, plus residual audit | no | no | XF comparison protocol | Claiming advantage over Transformers | Scale is minimal and d/m/K/block are not exposed by the JSON keys searched (or are toy-scale); the strictest class was chosen over VALID_WITH_SCOPE (draft v0.1 decision, kept) | V2-2 (future) |
| R29 T2-XF-SEQ (`t2_xf_seq_*`) [15 campaign dirs] | 2026-09-09; commits a19c2c1, 409cd11, 953fbce, 917626d, 00e30a4, f471fcc | H-LEGACY (inferred, not stamped) | as T2-XF-CLAUSE | not 16d^2 | SURROGATE | Sequence task, R1.1 controls and held-out, seeds 6001-6005 | no | no | XF comparison protocol | As T2-XF-CLAUSE | Scale is minimal and d/m/K/block are not exposed by the JSON keys searched (or are toy-scale); the strictest class was chosen over VALID_WITH_SCOPE (draft v0.1 decision, kept) | V2-2 (future) |
| R30 T2-NOBYPASS-0/1 (`t2_nobypass_*`, `t2_nobypass1_*`, R2-STAB family) [31 campaign dirs] | 2026-09-12; 42 dirs, per-dir commits in Appendix A | H-LEGACY (inferred, not stamped) | unknown; task label rows only | unknown, not 16d^2 | SURROGATE | No-bypass mechanism studies: staged HARDPTR, NB3/NB4/NBF0, R0..R14 relational-key/gate diagnostics | no | no | no-bypass test design | Citing as evidence for the architecture | Scale is minimal and d/m/K/block are not exposed by the JSON keys searched (or are toy-scale); the strictest class was chosen over VALID_WITH_SCOPE (draft v0.1 decision, kept) | V2-2 (future) |
| R31 T2-NOBYPASS-2 / NB5 (`nb5_*`, `t2_nobypass2_*` + loose `nb5_fresh_690x_manifest_check.json`) [36 campaign dirs, 1 loose campaign files] | 2026-09-12; 43 dirs (NB5 failure-alg 36204ac; fresh 690x 3d5eb6a; RCSEP-A 29b0760; RCSEP-B 51272bf, abffc05; secondary/stage fec3a9f; 78f6b77; manifests f52b614; closure 3942431; freeze 13b9eea) | H-LEGACY (inferred, not stamped) | unknown; small models | unknown, not 16d^2 | SURROGATE | NB5 role separation, RCSEP A/B/C, fresh 690x (commit message: "FIRST genuinely fresh PASS_STRONG"), final closure | no | no | fresh-manifest sealing discipline | Citing PASS_STRONG as evidence for the contractual core | Scale is minimal and d/m/K/block are not exposed by the JSON keys searched (or are toy-scale); the strictest class was chosen over VALID_WITH_SCOPE (draft v0.1 decision, kept) | V2-2 (future) |
| R32 T3-NOBYPASS-1 (`t3_nobypass1_noop_distractor` + loose manifest check) [1 campaign dirs, 1 loose campaign files] | 754b28f (2026-09-12); manifests sealed in d90459d | H-LEGACY (inferred, not stamped) | d_model=16 (config.json); m, K unknown | toy, not 16d^2 | SURROGATE | NOOP-distractor variant of the binder task (a legacy "T3" name that is NOT the contractual external-memory T3) | no | no | distractor design | Reading it as external-memory (contractual T3) evidence; contractual T3 remains HOLD | Scale is minimal and d/m/K/block are not exposed by the JSON keys searched (or are toy-scale); the strictest class was chosen over VALID_WITH_SCOPE (draft v0.1 decision, kept). Name collision with the contractual T3 is intentional to flag | - |
| R33 T4 (`t4_*`) [4 campaign dirs] | 2026-09-12..09-13 (6b547ef, 49c47f3, ed1bdef, 384c90d, ccbff5e, 8525a29) | H-LEGACY (inferred, not stamped) | unknown; T4 config names "T4RelKeyEncoder with shared W_v" | unknown, not 16d^2 | SURROGATE | Binder/role families: three active roles, RCSEP-3, fresh sealed manifests, final closure. Several member directories are preparation, design or seal-manifest artifacts with "zero training" (commit messages) and hold no result | no | no | manifest sealing, static audits, final-closure format | Citing as evidence for the architecture or for external memory | Scale is minimal and d/m/K/block are not exposed by the JSON keys searched (or are toy-scale); the strictest class was chosen over VALID_WITH_SCOPE (draft v0.1 decision, kept) | V2-2 (future) |
| R34 T5 (`t5_*`) [6 campaign dirs] | 2026-09-13 (ce1fe26, 2a7e0e0, a7ccb09, b1b2dc4, d6c98a8, 58ccb36) | H-LEGACY (inferred, not stamped) | unknown | unknown, not 16d^2 | SURROGATE | Binder/role families: N=3/N=4 development, fresh, preparation and design audit, final closure. Several member directories are preparation, design or seal-manifest artifacts with "zero training" (commit messages) and hold no result | no | no | manifest sealing, static audits, final-closure format | Citing as evidence for the architecture or for external memory | Scale is minimal and d/m/K/block are not exposed by the JSON keys searched (or are toy-scale); the strictest class was chosen over VALID_WITH_SCOPE (draft v0.1 decision, kept) | V2-2 (future) |
| R35 T6 (`t6_*`) [12 campaign dirs] | 2026-09-13 (bd54f53, d6060e7, 1bb3ecb, e35dab4, 22abe57, 286c838) | H-LEGACY (inferred, not stamped) | unknown | unknown, not 16d^2 | SURROGATE | Binder/role families: active-set midpoint development/fresh/preparation, final closure. Several member directories are preparation, design or seal-manifest artifacts with "zero training" (commit messages) and hold no result | no | no | manifest sealing, static audits, final-closure format | Citing as evidence for the architecture or for external memory | Scale is minimal and d/m/K/block are not exposed by the JSON keys searched (or are toy-scale); the strictest class was chosen over VALID_WITH_SCOPE (draft v0.1 decision, kept) | V2-2 (future) |
| R36 T7 (`t7_*`) [15 campaign dirs] | 2026-09-13 (7c0dece, 1860fc5, 26427d5, f33f525, 60f8d86, 2769776, 96ab55a, 13a123f, 5a77827, df37132, 5a4da75, fea9ea4, 20b15e9, 2ae3121, 8f22eee, b7fb256) | H-LEGACY (inferred, not stamped) | unknown; d_model=16 in one T7 config | unknown, not 16d^2 | SURROGATE | Binder/role families: NOOP-none lexical stages A/B/C, paired development/fresh, final closure. Several member directories are preparation, design or seal-manifest artifacts with "zero training" (commit messages) and hold no result | no | no | manifest sealing, static audits, final-closure format | Citing as evidence for the architecture or for external memory | Scale is minimal and d/m/K/block are not exposed by the JSON keys searched (or are toy-scale); the strictest class was chosen over VALID_WITH_SCOPE (draft v0.1 decision, kept) | V2-2 (future) |
| R37 LM0-DESIGN-R1-INFRA (`omega_core_lm_0_design_audit`, `omega_core_lm_0_r1_*` except scoping and Fable proposal, `omega_core_lm_0_gpu_environment_preparation`) [15 campaign dirs] | 2026-09-13..09-14 (e.g. d151c4a, 1238ac3, a89f7f4, 8d5c34d, e2589be, a226489, bd42b21, a05c130, 7fb92b9, 2901e58, 18240b8, a3d7b47); `omega_core_lm_0_r1_pilot_execution_readiness` and `omega_core_lm_0_gpu_environment_preparation` were last touched in 6b61c04 (2026-09-28); per-dir in Appendix A | H-LEGACY (inferred, not stamped); `..._runtime_recovery_and_gpu_readiness` reports an "AMD Radeon(TM) 820M Graphics" GPU model, consistent with the legacy laptop; VPS builder unit may have run elsewhere (not verified) | d=128, m=8, K=1/4/6 (audit 3.3) | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | SURROGATE | Design audit, preflights (architecture, cost, device, CUDA, CPU fast path, launch and pilot readiness), pilot freeze, VPS builder/image validation for the d128 substitute | with scope (infrastructure only) | no | data pipeline, launch scripts, preflight patterns | Claiming they validate 16d^2 or flexible K | Audit 3.3: d128 is a surrogate; contract is d512 16d^2 without per-round tables | V2-2 |
| R38 LM0-SCOPE-A/B/C (`omega_core_lm_0_r1_scientific_scoping_a`) [1 campaign dirs] | afe69f7 (2026-09-14) created; b19e682 (SCOPE-A complete, "PROMISING"), 287403a (SCOPE-B complete), 3c7de18 (SCOPE-C complete, 2026-09-17) last | H-LEGACY (inferred, not stamped) | d=128, m=8, K=1/4/6 (audit 3.3) | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | SURROGATE | Shared K1/K4 scoping runs (5,000 updates in B/C); results directories include full_campaign, interrupted (memory pressure) and aborted (batch-1 bug) runs whose status was not read here | no | no | comparison protocol, NLL metrics, resume-from-checkpoint logic | Citing NLL or rankings as a verdict on the contractual core; using aborted/interrupted runs as results | d128 substitute (audit 3.3) | V2-2 |
| R39 LM0-ER32 (`omega_core_lm_0_er32_*`) [3 campaign dirs] | 2026-09-17 (ac8c2d5, 0df64b5, f5499b5, 34277cb, fe26223, 3ca4bf4) | H-LEGACY (inferred, not stamped) | d=128, m=8, K=1/4/6 (audit 3.3); the ER32 configuration was not re-read | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | SURROGATE | ER32 design, integration and cost gate, quality scoping A | no | no | cost-gate protocol | As LM0-SCOPE-A/B/C | d128 substitute; verdict files not re-read for this registry | V2-2 |
| R40 LM0-ER64 (`omega_core_lm_0_er64_*`; QUALITY-NO-GO-A) [2 campaign dirs] | 2026-09-18..09-19 (543a1f5, 12d46ea "real 8000-update closure, QUALITY-NO-GO-A", 36029b2, ade11cf) | H-LEGACY (inferred, not stamped) | d=128, m=8, K=1/4/6 (audit 3.3); ER64 configuration not re-read | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | SURROGATE | Bounded negative (QUALITY-NO-GO-A) plus cost/scaling gate at d128 | no | no | cost-gate protocol | Extending the NO-GO to d512 or to flexible K | Valid only on the substitute | V2-2 |
| R41 LM0-TBPTT-DIRECTIONAL (`omega_tbptt_directional_probe`) [1 campaign dirs] | 4b519c9 (implementation, 2026-09-19); ff0adff (real execution, "DIRECTIONAL-STOP", 2026-09-19) | H-LEGACY (inferred, not stamped) | d=128, m=8, K=1/4/6 (audit 3.3) | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | SURROGATE | TBPTT16 directional probe closed with DIRECTIONAL-STOP | no | no | directional stop criterion | Concluding that TBPTT does not work in general | Audit 3.3: DIRECTIONAL-STOP is not the full T2 state-training plan | V2-2 |
| R42 LM0-TEACHER-CACHES-AND-CE-ONLY (`omega_teacher_logit_cache`, `omega_teacher_hidden_cache_probe`, `omega_hidden_cache_production_integration`, `omega_readout_cache_survival_audit`, `omega_ce_only_baseline`) [5 campaign dirs] | 2026-09-18..09-20 (9e4a095, b9774d8, 02e502e, 2c55aca, da6f1e7, 26a59e8, 96e28e0, 1532858, 3d8428c; ce_only touched again in 6b61c04) | H-LEGACY (inferred, not stamped) for the builds (dates); caches were later restored on the i7 workstation via a junction (session memory), so use-host differs from build-host | d=128, m=8, K=1/4/6 (audit 3.3) | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | SURROGATE | Teacher logit/hidden caches, readout-cache survival audit, CE-only baseline for d128 | with scope (data reusable only if tokenizer and vocabulary match V2; not verified) | no | cache format and scripts | Assuming caches are valid for V2 without a tokenizer/vocabulary check | Draft v0.1 listed host as i7 from the migration memory; corrected here because the commits pre-date the 2026-09-27 migration | - |
| R43 LM0-RANK-SVD-AND-EXPANDED-VALIDATION (`omega_rank_svd_diagnostic`, `omega_expanded_frozen_validation`) [2 campaign dirs] | aa9826e (2026-09-18); 6190a62 latest | H-LEGACY (inferred, not stamped) | d=128, m=8, K=1/4/6 (audit 3.3) | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | SURROGATE | Diagnostics on d128 checkpoints (contents not re-read) | no | no | SVD/rank diagnostic code | Citing as V2 evidence | d128 substitute; not opened for this registry | V2-2 |
| R44 LM0-OLD513-FREE-GENERATION (`omega_core_lm_0_autoregressive_generation_audit`) [1 campaign dirs] | 21fdd46 (implementation), ca8f4ad ("AUDIT_COMPLETE, 64/64 real generations verified", 2026-09-18) | H-LEGACY (inferred, not stamped) | d=128, m=8, K=1/4/6 (audit 3.3) | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | VALID_WITH_SCOPE | Old-513 update-2000 free generation produced severe loops (audit 5.6: VALID_STRONG_NEGATIVE_EVIDENCE); not carried over to the current Coverage-C checkpoint | no | no | prompt set and generation-audit method | Citing as a verdict on the contractual core or on the current checkpoint | Audit 5.6; d128 substitute | V2-2 |
| R45 NATIVE-BACKEND-BE376 (`omega_native_runtime_p0`, `omega_native_runtime_p2r0`) [2 campaign dirs] | 97aed42 / 363d1fe (P0, 2026-09-20); 1a4103c / 6b61c04 (P2R0); DLL SHA-256 be37623d7e69b65551ad9c306092f328cc250576a56642fc34a0068df00600cc (from BACKEND_QUALITY_QUALIFICATION_PROTOCOL.md and KERNEL_CLOSEOUT_SCOPE_NOTE.md) | built and benchmarked on the legacy laptop per dates (H-LEGACY, inferred); i7 portability preflight file dated 2026-09-27 | d=128, m=8 | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | NO_TRANSFER | Native CPU kernel for the substitute; Fable CLOSED. Historical performance is valid for the exact executed objective only: the Python loss callable had a CE/KL scaling defect (KERNEL_CLOSEOUT_SCOPE_NOTE.md), so historical percentages must not be quoted as canonical-loss performance | no | no | ABI, harness, tests, golden cases, optimization techniques | Using the DLL for d512/d640 or assuming its speedups; quoting historical percentages as canonical-R1-loss performance | Audit 5.7: BE376_BINARY_AS_V2_BACKEND NOT_TRANSFERABLE (different block and tables) | V2-1b KQ candidate_02 (for the contractual block) |
| R46 FABLE-METHODS (`omega_core_lm_0_r1_fable_optimization_proposal`; `OMEGA_FABLE_CLOSEOUT.md`) [1 campaign dirs] | 6b3a9ad (2026-09-14); closeout note in `omega_native_runtime_p2r0` | n/a (methods) | n/a | n/a | VALID_WITH_SCOPE | Reusable optimization techniques (blocking, fusion, quantization, replay/backward tricks) | with scope (techniques, not numbers) | with scope (techniques, not numbers) | techniques | Citing Fable speedups as V2 results | Audit 5.7: FABLE_OPTIMIZATION_METHODS REUSABLE | - |
| R47 QUALIFIED-AT-2000 (`omega_backend_quality_qualification`) [1 campaign dirs] | 6b61c04 (2026-09-28, code and docs; results/ directory is gitignored, on disk only). Backend DLL be37623d7e69b65551ad9c306092f328cc250576a56642fc34a0068df00600cc ADOPTED (MD/274) | H-LEGACY (inferred, not stamped) for blocks A/B (result dirs dated 2026-09-24..09-26); one i7 portability preflight (2026-09-27) | d=128, m=8, K=1/4/6 (audit 3.3) | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | VALID_WITH_SCOPE | Native vs PyTorch at d128, K1/K4, 2,000 updates: about 20% better under the canonical (masked token mean) loss; strict_trajectory_proximity=FAIL_RECORDED preserved; training_quality_equivalence=NOT_ESTABLISHED (KERNEL_CLOSEOUT_SCOPE_NOTE.md) | no | no | qualification protocol (parity, loss contract, causal diagnostics), harness, ABI | Extending to d512/d640 or to any other kernel; certifying be376 as the V2 backend | Audit 5.1: BACKEND_QUALIFIED_AT_2000 VALID_WITH_SCOPE; TRANSFER_TO_D512_D640 NOT_TRANSFERABLE | - |
| R48 CAUSAL-STATE-USE-D128 (sub-results of `omega_backend_quality_qualification/results`: `r1_state_causal_diagnostics_*`, `gate_init_diagnostic_*`) [sub-results only, no dedicated directory] | 6b61c04 (code); results on disk only (gitignored), dated 2026-09-26 | H-LEGACY (inferred, not stamped) | d=128, m=8, K=1/4/6 (audit 3.3) | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | VALID_WITH_SCOPE | Reset/shuffle/gate-init showed the carried state and small gate adaptation are functionally used at d128 (audit 5.2: STATE_CAUSAL_USE_D128 VALID) | no (STATE_CAUSAL_USE_V2 NOT_YET_ESTABLISHED) | no | causal-diagnostic instrumentation | Extending to d512 or m=8/d512 optimality | Audit 5.2 | - |
| R49 K-CURVE-A-K4-GT-K1 (`omega_r1_k_curve_a_preflight/results/k_curve_primary_20260927_md286`) [sub-results only, no dedicated directory] | 6b61c04 (2026-09-28, code; results gitignored); Sol MD/286 | H-I7 (inferred by date, not stamped) | d=128, K=1/4 (K6 in the saturation row) | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | VALID_WITH_SCOPE | K4 improves on K1 at update 2000, document-balanced, d128 (audit 5.3: K4_GT_K1_AT_2K_D128 VALID) | no | no | K-curve protocol | Presenting as a property of flexible K (tables are finite) | Audit 5.3 | - |
| R50 K-CURVE-A-SATURATION-BEYOND-K4 [sub-results only, no dedicated directory] | as above (K6 sentinel and K2/K6 preflights: `k6_seed_sentinel_20260927_md281`, `k2_*`) | H-I7 (inferred by date, not stamped) | d=128, K=4 vs 6 | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | VALID_WITH_SCOPE | K4 to K6 interval includes zero at update 2000 (audit 5.3: SATURATION_BEYOND_K4_AT_2K_D128 VALID_WITH_SCOPE) | no | no | protocol | Elevating to a law or using it to exclude K>4 in V2 | Audit 5.3; K effect varies with budget (effect modification), which does not invalidate the fixed-budget interval | - |
| R51 K-CURVE-A-K4-OPERATING-DEPTH (K4 as final depth; K2 excluded) [sub-results only, no dedicated directory] | as above (`k2_local_adjudication_*`) | H-I7 (inferred by date, not stamped) | d=128 | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | SUPERSEDED | Selection of K4 as operating depth; K2 excluded for a numerical edge case (session memory) | no | no | none | Fixing K=4 in V2 | Audit 5.3: K4_AS_FINAL_OPERATING_DEPTH SUPERSEDED / NOT_ESTABLISHED; K_EFFECT_STABLE_ACROSS_BUDGET NOT_ESTABLISHED; flexible K is part of the contract | V2 design (flexible K) |
| R52 R1-KCURVE-SUPPORT-DIAGNOSTICS (`i7_topology_probe_20260927_md282`, `shared_grad_cancellation_audit_20260927_md283`, K2/K6 preflights) [sub-results only, no dedicated directory] | 6b61c04 (2026-09-28); results on disk only | H-I7 (inferred by date, not stamped) | d=128, m=8, K=1/4/6 (audit 3.3) | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | SURROGATE | Diagnostics supporting the K-curve unit (topology probe on the i7, gradient cancellation audit, preflights) | no | no | topology-probe idea (P/E-core identification) | Citing as physical gates | Diagnostics on the substitute; host-specific to i7 | CPU workstation diagnostics (MD/317 order C) |
| R53 COVERAGE-A/B/B2/C (`omega_r1_k_curve_a_preflight/results/train_data_coverage_*`, `coverage_*`) [1 campaign dirs] | 6b61c04 (2026-09-28); the directory also holds the K-curve, TTR-A and support rows; results gitignored | H-I7 (inferred by date, not stamped) | data policy; d=128 for the models used | n/a (data) | VALID_WITH_SCOPE | Coverage-C isolated document weighting and showed benefit on VALL token-weighted and document-macro (audit 5.4: DOCUMENT_BALANCED_MULTICHUNK VALID_DATA_POLICY_RESULT) | with scope (data recipe, after tokenizer/manifest equivalence is checked) | with scope (data, not performance) | data recipe, multichunk builder | Reading it as validation of the d128 or d512 architecture | Audit 5.4. This row is also the primary directory row for the shared K-curve directory | - |
| R54 TTR-A (`omega_r1_k_curve_a_preflight/results/test_time_recurrence_scaling_a_20260928_md295`) [sub-results only, no dedicated directory] | 6b61c04 (2026-09-28); results gitignored | H-I7 (inferred by date, not stamped) | d=128, K finite (depth_embedding[K,D], gate_logits[K,D]) | FastWorkspaceUpdateBlock 12d^2+10d = 197,888 weights; finite depth_embedding[K,D] and gate_logits[K,D]; tied head 50,257x128 | VALID_WITH_SCOPE | TAIL-REPEAT degraded K1/K4 and was neutral only for K6 to K8 before degrading (audit 5.5: TAIL_REPEAT_V1 VALID_NEGATIVE_RESULT) | no | no | experiment design | Citing as a refutation of arbitrary K | Audit 5.5: ORIGINAL_ARBITRARY_K_THESIS NOT_TESTED_CLEANLY; V1 tables force an invented policy for new rounds | V2 flexible-K tests |
| R55 LN_PROTOCOL (`omega_minimum_language_competence`; protocol and log in "Conversacion LN.md") [1 campaign dirs] | e4d4322dc082b2c7de6e06fb0d3076234afbeffa (base protocol, from the Phase B report); implementation 3751494 (2026-09-28) and 015d916 (2026-09-29) | H-I7 (inferred by date, not stamped) | n/a (instrument) | n/a | VALID_WITH_SCOPE | Reusable methodology: two-level instrument design, DistilGPT2/KN5 reference calibration, OOV policy handling, human-rater protocol, leakage report, self-hash conventions | with scope (method) | with scope (method) | protocol, harness (`phase_b_utils.py`, `human_pilot_harness.py`), OOV decision procedure | Using a d128 LN result as an architecture-conformance verdict or as a T3 release; using the protocol without the four open definition holes raised by Opus | MD/317 split of the former single LN row; audit 5.6: BASELINE_DIAGNOSTIC_ONLY | - |
| R56 CURRENT_L1_BANK (Phase B `phase_b_20260929_md298*`, INSTRUMENT_INVALID_L1) [sub-results only, no dedicated directory] | 015d916 (2026-09-29) for tracked `phase_b_20260929_md298` (9 files); `..._oov_resume1..4` directories are untracked on disk; final `PHASE_B_REPORT.md` SHA-256 file hash listed in the report | H-I7 (inferred by date, not stamped) | n/a | n/a | NO_TRANSFER | Phase B verdict INSTRUMENT_INVALID_L1: reason DISTILGPT2_PAIR_SUCCESS_FULL_LT_0_60 (pair success full 0.40625; trunc5 0.0). L0 accuracy DistilGPT2 0.9375, KN5 0.5. FP_U_FP_threshold NOT_RUN_L1_INSTRUMENT_INVALID; RATER_PANEL_READY false. Phase B frozen (HOLD) | no | no | none for the current bank; reference-model NLL values in `absolute_lm_calibration.json` (DistilGPT2 3.9899) are recorded but not classified as a verdict here | Using the L1 bank, any pair-success number, or a derived threshold as evidence; releasing T3 on it | MD/317: CURRENT_L1_BANK = INSTRUMENT_INVALID_L1. No d128 LN result has been executed, so the former row "LN result d128" is removed | - |
| R57 V2-0 conformance (`omega_v2_0_conformance`, OMEGA_V2_0_CONFORMANT_PASS) [1 campaign dirs] | execution bc9c1745f70e08f15fd340097d04fff6f56cba43; validated head 5df1cf270837a9cc4f56a3dd31f277af868f61ce; implementation c8de2a3; sealed 367d73a (2026-09-29) | n/a: no performance measured; judge environment python 3.14.3, torch 2.11.0+cpu; host not stamped in the seal (V2_0_RESULT_SEAL.json) | contractual: d512/d640, m in {4,8,16}, K flexible | 4d^2 + 12d^2 = 16d^2 | VALID_DIRECT | Implementation conformance: 16d^2 block, flexible K, bit-identical R4/U4 clones, equal FLOP ledger, Q4 logical only; 10/10 tests; no training | yes (implementation conformance only) | not applicable (residency NOT_MEASURED) | specification, conformance tests, R4/U4 reference | Citing as evidence of residency, speed, quality or training | Only artifact that directly verifies the contractual object; measures no performance | - |
| R58 V2-1 attempt_01 (root-level files of `omega_v2_1_physical/results/omega_v2_1_physical`) [sub-results only, no dedicated directory] | no tracked artifact; cause fixed in 18f98a1 (2026-09-29, "Fix Windows topology and QPC preflight") | H-I7 (attempt_02 preflight prints i7-13700F; attempt_01 not re-read) | d in {512,640} | 16d^2 | SUPERSEDED | MEASUREMENT_INVALID (QPC / topology), per audit and judge notebook | no | no | none | Citing its numbers | Invalid measurement; files are untracked (gitignored results) | V2-1 attempt_02 |
| R59 V2-1 attempt_02 (RESIDENCY_GATE_FAIL, `omega_v2_1_physical/results/.../attempt_02`) [1 campaign dirs] | implementation 9742005 / 18f98a1d83f8dfe021e7bbe9943d44e95dc79859; results a23ec1c (2026-09-29); artifact manifest d17d5b79d43fe72eba644069e72b8539ce6990032579313d953190a7313f5287 | H-I7 (stamped cpu_model): hardware_preflight.json prints "13th Gen Intel(R) Core(TM) i7-13700F" (8P+8E, L2 25,165,824 B, LLC 31,457,280 B); 4 P-cores / 4 threads emulating the legacy profile | d in {512,640}; m in {1,4,8,16}; K in {1,4,8}; 72 cells | 16d^2 | VALID_WITH_SCOPE | rho_resident=0.99853 (gate <=0.50: FAIL), delta_CB=0.0045, G_matrix=2.7448, measured on the i7 emulating the legacy 4-core profile | yes (measurement of the contractual block on that host) | no (host-specific to i7-13700F; no Ryzen transfer may be claimed) | A/B/C protocol, B-pool, eviction, QPC, v_i selection | Concluding the block is not resident on any other host; concluding it is; citing rho as a host-independent property; treating the 0.50 gate as universal | Correct object, specific host: stricter than VALID_DIRECT (MD/314, MD/317 host-specific rule). V2-1c closed INVALID_PREFLIGHT and therefore did NOT supersede it | - (V2-1c did not supersede; residency for the workstation must be re-derived by a future unit) |
| R60 V2-1b KQ candidate_01 (`omega_v2_1b_kernel_qualification`, run_02) [1 campaign dirs] | implementation e285f4be1abfebd22c490f7be64d2e8580e0fb72; candidate-1 result recorded 0e0e8f9; kernel-qualification results 5ef5be6; exe SHA-256 2fdeac276797899b473da0a6dcb17844a95cb915d47e5096d274cee43717f15f | H-I7 (stamped cpu_model) | d512, several m/K | 16d^2 | VALID_WITH_SCOPE | Kernel qualification: S_native=0.5286 (gate >=1.20 not met) | with scope (kernel qualification of the contractual block) | no (host-specific to i7-13700F; no Ryzen transfer may be claimed) | KQ harness; criteria E_Q4>=0.25, E_FULL>=0.60, S_native>=1.20 | Citing as a competitive backend or as a residency verdict | Valid result for a discarded candidate | V2-1b KQ candidate_02 |
| R61 V2-1b KQ candidate_02 (`omega_v2_1b_candidate_02`; run_01, PROJECT_NATIVE_SPEED_GATE_FAIL) [1 campaign dirs] | implementation 2e8ac1207e742e659aab34ef0ab4bf51d43dbe55 (2e8ac12); result 5ef5be6 (2026-09-29); exe SHA-256 be5c195e701ccbbf4b606d39382d951ba887b41c1faf96370d2d0ac2a19d084f | H-I7 (stamped cpu_model) | d512, several m/K | 16d^2 | VALID_WITH_SCOPE | Sealed kernel; S_native=0.7785 (gate failed); attempt_03 not executed. Its numerical equivalence to the scalar oracle was later found NOT to satisfy the frozen KQ criterion in V2-1c | with scope (frozen kernel used in V2-1c preflights) | no (host-specific to i7-13700F; no Ryzen transfer may be claimed) | frozen q4_kernel_candidate2 / full_block_candidate2 / kq_candidate2 | Citing as a competitive backend, as language quality or trainability evidence; timing candidate_02 full block before V2-1d-Q4-NUMERICAL-EQUIVALENCE | V2-1b closed with GATE_FAIL; used only as the frozen kernel of V2-1c | - |
| R62 V2-1c residency-only (`omega_v2_1c_residency_only`, V2_1C_INVALID_PREFLIGHT) [1 campaign dirs] | sealed in 1072b09 (`results/v2_1c_sealed/`); SEAL_MANIFEST.json SHA-256 61d7cefaef64c456c615ee1773ef0b62e6ac79a07295f64bbac757bf42f7b94d (recomputed here); implementation 832c0ff; spec MD/312 05b3bda, MD/313 13bad70; MD/316 recorded 01e2f99; offline-seal verification 78ec56e | H-I7 (stamped cpu_model) (prints i7-13700F); timed CPU-set IDs [266,260,256,264] | d512 and d640, m in {1,4,8,16}, K=1/4/8; no timed cells executed | 16d^2 | VALID_WITH_SCOPE | Terminal V2_1C_INVALID_PREFLIGHT at CORRECTNESS_PREFLIGHT; scientific_timing_executed=false; residency_verdict=NOT_ESTABLISHED; protocol_deviation=false. Binding-preflight, H0 core-selection and v_i seals are valid preflight evidence (MD/317 Q6: SEALS_VALID=YES, OPERATIONAL_PROCESS_INCIDENT=YES, REMEASUREMENT_REQUIRED=NO). Correctness FAIL is binding under the frozen comparator (max_abs<=1e-5 and max_rel<=1e-4, floor 1e-12, exact scalar oracle): e.g. candidate_02 KQ d512 m4 K4 max_abs 1.049e-05 and m16 K4 1.061e-05. OFFLINE_SEAL_RECOVERY_VERIFICATION status PASS, 0 executable invocations | with scope (preflight seals only; no timing, no residency) | no (host-specific to i7-13700F; no Ryzen transfer may be claimed) | binding/H0/v_i seal machinery, CPU-set selection, offline-recovery verification procedure | Citing any residency number; treating candidate_02 as numerically equivalent; timing candidate_02 full block before V2-1d-Q4-NUMERICAL-EQUIVALENCE; treating recovered seals as new measurements | MD/316 (no amendment) and MD/317 Q6-Q7; disclosed offline seal recovery (parser bugs, saved JSON reused, EXE not re-invoked) | V2-1d-Q4-NUMERICAL-EQUIVALENCE (ordered, no artifact yet) |
| R63 V2-GPU-DIAG (`omega_v2_gpu_diag`, local GTX 1650 SUPER memory diagnostic) [1 campaign dirs] | e9f8786 (2026-09-29); `gpu_memory_diag.json` SHA-256 6331390E1811CD63B172766AD22DF74CDF09E3291473D3A70400FF1A575A64F4 (verified) | GTX 1650 SUPER 4 GB (cc 7.5) on the i7-13700F host; .venv_cuda torch 2.11.0+cu128; FP32, <=3 GiB | contractual block sizes | 16d^2 | VALID_WITH_SCOPE | Engineering diagnostic: parity max_abs<=4.3e-6 and peak memory; supports feasibility of local validation (MD/311) | with scope (feasibility, not performance) | no | parity procedure, memory profile | Citing as CPU residency, quality or speed evidence | Engineering diagnostic on different hardware | V2-2A (in progress) |
| R64 HOUSEKEEPING (`.pytest_cache`, empty `b5_failure_alg`; empty `t2_i3_view0_seed6401` is listed under T2-I3) [2 campaign dirs] | untracked (no commit) | n/a | n/a | n/a | NO_TRANSFER | `.pytest_cache` is tooling cache; `b5_failure_alg` and `t2_i3_view0_seed6401` are empty directories (verified); `nb5_failure_alg` (tracked, 36204ac) is the populated sibling of `b5_failure_alg` | no | no | none | Citing as evidence | No content | - |

## 4. Not yet evidence (do not cite)

- V2-1d-Q4-NUMERICAL-EQUIVALENCE (MD/317 order B): ordered, not yet preregistered or run. Prerequisite for any candidate_02 full-block timing.
- CPU non-gate diagnostics (order C: FMA roofline, DRAM bandwidth, cache-size probes, topology, thread scaling 1P/2P/4P/8P, 8P/16T, E-cores, 24T): not run.
- Re-derivation of T0 on the i7 (order D, cache knees, d1024 to d2048): not run.
- V2-2A local GPU preflight (d256, m8, FP32, GTX 1650 SUPER): AUTHORIZED (MD/317 GO, final parameters MD/318). The untracked directory `campaign/omega_v2_2a_cuda_preflight/` existed at session start; no sealed result was read, so nothing from it is evidence yet. V2-2B (d512) only after V2-2A PASS.
- RunPod and contractual T3 (external memory): HOLD. Phase B / current L1 bank: frozen (INSTRUMENT_INVALID_L1).
- Residency verdict for the i7 workstation: NOT_ESTABLISHED (V2-1 attempt_02 measured rho=0.99853 on an emulated 4-core profile; V2-1c produced none).

## 5. Remaining gaps (unknowns not resolvable from artifacts)

1. Per-run CPU model is not stamped in any pre-2026-09-27 artifact; the legacy-laptop attribution is by date and by the CPU-Z report. The old T0 CPU model is documented only in `hardware/cpuz_report.txt`.
2. Most `results/` directories are gitignored (`**/results/`, `*.log`, `*.obj`): the evidence exists on disk only and has no commit. Commit cells for those rows give the commit of the code/docs that produced them, not of the results.
3. Contents of the u0b_b* JSON files, T2/T4..T7 config d/m/K and the ER32/ER64 configurations were not opened; those cells say `unknown` or "not re-read".
4. The four `t0r-bclone-*` timing directories (block-loop, defender-excluded, ready, warmup-affinity; commit 91993f7) have no summary file; their outcomes were not read.
5. Empty untracked directories with no evidence: `b5_failure_alg`, `t2_i3_view0_seed6401`, `Trash/cpu-native-arch/sweep-output/t0m-phase3-median10`. `.pytest_cache` is untracked tooling.
6. Whether the VPS builder / image-validation units of LM0-DESIGN-R1-INFRA ran on the laptop or on the VPS is not verified.
7. Tokenizer/vocabulary compatibility of the teacher caches with V2 is not verified.
8. The "~379" figure in the brief = 378 campaign directories + `.pytest_cache` (379 directory entries) plus 47 loose files. The 4,240 measured processes for T0-M is quoted from the audit and was not recounted.
9. V2-1c OFFLINE_SEAL_RECOVERY_VERIFICATION (PASS) was committed as 78ec56e while this registry was being written; the V2-1c row reflects it.
10. The four open definition holes of the LN protocol (raised by Opus) remain unresolved; they are recorded in "Conversacion LN.md", which was not re-read here.

## 6. Class counts (computed by script over section 3)

| Class | Rows |
|---|---|
| VALID_DIRECT | 1 |
| VALID_WITH_SCOPE | 28 |
| SURROGATE | 23 |
| NO_TRANSFER | 9 |
| SUPERSEDED | 3 |
| Total | 64 |

v0.1 counts were VALID_DIRECT 1, VALID_WITH_SCOPE 16, SURROGATE 6, NO_TRANSFER 4, SUPERSEDED 3 (total 30).

## Appendix A. Per-entry assignment (every entry maps to exactly one row)

Columns: entry, kind, row, tracked files, on-disk files, first commit (date), last commit (date). Campaign root = `t1_trainability_lab_v0.1.0/campaign/`. Trash entries are under `Trash/cpu-native-arch/sweep-output/` and were moved there in 99121f6 (2026-09-10); their origin commits are in the row cells.

| entry | kind | row | tracked_files | on_disk_files | first_commit | last_commit |
|---|---|---|---|---|---|---|
| `.pytest_cache` | dir | R64 | 0 | 4 | UNTRACKED |  |
| `associative_typed_replication` | dir | R11 | 21 | 21 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `associative_typed_seed101` | dir | R11 | 4 | 4 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `b5_failure_alg` | dir | R64 | 0 | 0 | UNTRACKED |  |
| `gate_init_p05` | dir | R11 | 45 | 45 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `multihop_pointer_replication` | dir | R11 | 21 | 21 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `multihop_pointer_seed101` | dir | R11 | 4 | 4 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `nb5_failure_alg` | dir | R31 | 1 | 1 | 36204ac (2026-09-12) | 36204ac (2026-09-12) |
| `nb5_fresh_690x` | dir | R31 | 1 | 1 | 3d5eb6a (2026-09-12) | 3d5eb6a (2026-09-12) |
| `nb5_fresh_690x_6901` | dir | R31 | 2 | 2 | 3d5eb6a (2026-09-12) | 3d5eb6a (2026-09-12) |
| `nb5_fresh_690x_6902` | dir | R31 | 2 | 2 | 3d5eb6a (2026-09-12) | 3d5eb6a (2026-09-12) |
| `nb5_fresh_690x_6903` | dir | R31 | 2 | 2 | 3d5eb6a (2026-09-12) | 3d5eb6a (2026-09-12) |
| `nb5_fresh_690x_6904` | dir | R31 | 2 | 2 | 3d5eb6a (2026-09-12) | 3d5eb6a (2026-09-12) |
| `nb5_fresh_690x_6905` | dir | R31 | 2 | 2 | 3d5eb6a (2026-09-12) | 3d5eb6a (2026-09-12) |
| `nb5_manifests` | dir | R31 | 21 | 21 | f52b614 (2026-09-12) | d90459d (2026-09-12) |
| `nb5_p_6801_joint` | dir | R31 | 1 | 1 | 78f6b77 (2026-09-12) | 78f6b77 (2026-09-12) |
| `nb5_p_6801_reverse_v2` | dir | R31 | 1 | 1 | 360c603 (2026-09-12) | 360c603 (2026-09-12) |
| `nb5_rcsep_a_6801` | dir | R31 | 2 | 2 | 29b0760 (2026-09-12) | 29b0760 (2026-09-12) |
| `nb5_rcsep_a_6802` | dir | R31 | 2 | 2 | 29b0760 (2026-09-12) | 29b0760 (2026-09-12) |
| `nb5_rcsep_a_6803` | dir | R31 | 2 | 2 | 29b0760 (2026-09-12) | 29b0760 (2026-09-12) |
| `nb5_rcsep_a_6804` | dir | R31 | 2 | 2 | 29b0760 (2026-09-12) | 29b0760 (2026-09-12) |
| `nb5_rcsep_a_6805` | dir | R31 | 2 | 2 | 29b0760 (2026-09-12) | 29b0760 (2026-09-12) |
| `nb5_rcsep_b_fit_alg` | dir | R31 | 1 | 1 | abffc05 (2026-09-12) | abffc05 (2026-09-12) |
| `nb5_rcsep_b_gate_6801` | dir | R31 | 2 | 2 | 51272bf (2026-09-12) | 51272bf (2026-09-12) |
| `nb5_rcsep_b_gate_6802` | dir | R31 | 2 | 2 | 51272bf (2026-09-12) | 51272bf (2026-09-12) |
| `nb5_rcsep_b_gate_6803` | dir | R31 | 2 | 2 | 51272bf (2026-09-12) | 51272bf (2026-09-12) |
| `nb5_rcsep_b_gate_6804` | dir | R31 | 2 | 2 | 51272bf (2026-09-12) | 51272bf (2026-09-12) |
| `nb5_rcsep_b_gate_6805` | dir | R31 | 2 | 2 | 51272bf (2026-09-12) | 51272bf (2026-09-12) |
| `nb5_rcsep_b_gate_audit` | dir | R31 | 1 | 1 | 51272bf (2026-09-12) | 51272bf (2026-09-12) |
| `nb5_rcsep_c_nogate_hardptr_alg` | dir | R31 | 1 | 1 | 67233e8 (2026-09-12) | 67233e8 (2026-09-12) |
| `nb5_secondary` | dir | R31 | 1 | 1 | fec3a9f (2026-09-12) | fec3a9f (2026-09-12) |
| `nb5_stage_a_6801` | dir | R31 | 2 | 2 | 78f6b77 (2026-09-12) | 78f6b77 (2026-09-12) |
| `nb5_stage_a_6802` | dir | R31 | 2 | 2 | fec3a9f (2026-09-12) | fec3a9f (2026-09-12) |
| `nb5_stage_a_6803` | dir | R31 | 2 | 2 | fec3a9f (2026-09-12) | fec3a9f (2026-09-12) |
| `nb5_stage_a_6804` | dir | R31 | 2 | 2 | fec3a9f (2026-09-12) | fec3a9f (2026-09-12) |
| `nb5_stage_a_6805` | dir | R31 | 2 | 2 | fec3a9f (2026-09-12) | fec3a9f (2026-09-12) |
| `nb5_stage_b_6801` | dir | R31 | 2 | 2 | 78f6b77 (2026-09-12) | 78f6b77 (2026-09-12) |
| `nb5_stage_b_6802` | dir | R31 | 2 | 2 | fec3a9f (2026-09-12) | fec3a9f (2026-09-12) |
| `nb5_stage_b_6803` | dir | R31 | 2 | 2 | fec3a9f (2026-09-12) | fec3a9f (2026-09-12) |
| `nb5_stage_b_6804` | dir | R31 | 2 | 2 | fec3a9f (2026-09-12) | fec3a9f (2026-09-12) |
| `nb5_stage_b_6805` | dir | R31 | 2 | 2 | fec3a9f (2026-09-12) | fec3a9f (2026-09-12) |
| `omega_backend_quality_qualification` | dir | R47 | 33 | 836 | 6b61c04 (2026-09-28) | 6b61c04 (2026-09-28) |
| `omega_ce_only_baseline` | dir | R42 | 6 | 80 | 3d8428c (2026-09-19) | 6b61c04 (2026-09-28) |
| `omega_core_lm_0_autoregressive_generation_audit` | dir | R44 | 7 | 12 | 21fdd46 (2026-09-18) | ca8f4ad (2026-09-18) |
| `omega_core_lm_0_design_audit` | dir | R37 | 2 | 2 | d151c4a (2026-09-13) | d151c4a (2026-09-13) |
| `omega_core_lm_0_er32_design` | dir | R39 | 3 | 3 | ac8c2d5 (2026-09-17) | 0df64b5 (2026-09-17) |
| `omega_core_lm_0_er32_integration_and_cost_gate` | dir | R39 | 14 | 14 | f5499b5 (2026-09-17) | 34277cb (2026-09-17) |
| `omega_core_lm_0_er32_quality_scoping_a` | dir | R39 | 37 | 61 | fe26223 (2026-09-17) | 3ca4bf4 (2026-09-18) |
| `omega_core_lm_0_er64_quality_scoping_a` | dir | R40 | 48 | 60 | 543a1f5 (2026-09-18) | 12d46ea (2026-09-19) |
| `omega_core_lm_0_er64_scaling_gate` | dir | R40 | 9 | 15 | 36029b2 (2026-09-18) | ade11cf (2026-09-18) |
| `omega_core_lm_0_gpu_environment_preparation` | dir | R37 | 15 | 19 | 194aa46 (2026-09-13) | 6b61c04 (2026-09-28) |
| `omega_core_lm_0_r1_architecture_preflight` | dir | R37 | 1 | 1 | 1238ac3 (2026-09-13) | 1238ac3 (2026-09-13) |
| `omega_core_lm_0_r1_causal_training_conformance` | dir | R37 | 3 | 7 | a226489 (2026-09-14) | a226489 (2026-09-14) |
| `omega_core_lm_0_r1_cpu_fastpath_validation` | dir | R37 | 33 | 103 | bd42b21 (2026-09-14) | 49240cc (2026-09-18) |
| `omega_core_lm_0_r1_cpu_scoping_readiness` | dir | R37 | 52 | 56 | a05c130 (2026-09-14) | a05c130 (2026-09-14) |
| `omega_core_lm_0_r1_cuda_preflight` | dir | R37 | 10 | 10 | 7fb92b9 (2026-09-14) | dcf8b04 (2026-09-14) |
| `omega_core_lm_0_r1_fable_optimization_proposal` | dir | R46 | 5 | 5 | 6b3a9ad (2026-09-14) | 6b3a9ad (2026-09-14) |
| `omega_core_lm_0_r1_launch_readiness` | dir | R37 | 6 | 6 | 18240b8 (2026-09-14) | d8c921e (2026-09-14) |
| `omega_core_lm_0_r1_nominal_cost_and_device_preflight` | dir | R37 | 3 | 3 | 8d5c34d (2026-09-13) | 8d5c34d (2026-09-13) |
| `omega_core_lm_0_r1_pilot_execution_readiness` | dir | R37 | 12 | 12 | 6b61c04 (2026-09-28) | 6b61c04 (2026-09-28) |
| `omega_core_lm_0_r1_pilot_training_efficiency_gate` | dir | R37 | 4 | 8 | a05c130 (2026-09-14) | a05c130 (2026-09-14) |
| `omega_core_lm_0_r1_runtime_recovery_and_gpu_readiness` | dir | R37 | 5 | 5 | e2589be (2026-09-13) | e2589be (2026-09-13) |
| `omega_core_lm_0_r1_scientific_pilot_freeze` | dir | R37 | 2 | 2 | 2901e58 (2026-09-14) | 2901e58 (2026-09-14) |
| `omega_core_lm_0_r1_scientific_scoping_a` | dir | R38 | 218 | 344 | afe69f7 (2026-09-14) | 3c7de18 (2026-09-17) |
| `omega_core_lm_0_r1_training_technical_preflight` | dir | R37 | 2 | 2 | a89f7f4 (2026-09-13) | a89f7f4 (2026-09-13) |
| `omega_core_lm_0_r1_vps_builder_and_image_validation` | dir | R37 | 16 | 16 | a3d7b47 (2026-09-14) | a3d7b47 (2026-09-14) |
| `omega_expanded_frozen_validation` | dir | R43 | 4 | 8 | aa9826e (2026-09-18) | 6190a62 (2026-09-18) |
| `omega_hidden_cache_production_integration` | dir | R42 | 14 | 16 | 96e28e0 (2026-09-20) | 1532858 (2026-09-20) |
| `omega_minimum_language_competence` | dir | R55 | 16 | 82 | 3751494 (2026-09-28) | 015d916 (2026-09-29) |
| `omega_native_runtime_p0` | dir | R45 | 6 | 8 | 97aed42 (2026-09-20) | 363d1fe (2026-09-20) |
| `omega_native_runtime_p2r0` | dir | R45 | 62 | 5496 | 1a4103c (2026-09-20) | 6b61c04 (2026-09-28) |
| `omega_r1_k_curve_a_preflight` | dir | R53 | 15 | 788 | 6b61c04 (2026-09-28) | 6b61c04 (2026-09-28) |
| `omega_rank_svd_diagnostic` | dir | R43 | 5 | 9 | aa9826e (2026-09-18) | 6190a62 (2026-09-18) |
| `omega_readout_cache_survival_audit` | dir | R42 | 10 | 14 | 9e4a095 (2026-09-18) | b9774d8 (2026-09-18) |
| `omega_tbptt_directional_probe` | dir | R41 | 18 | 18 | 4b519c9 (2026-09-19) | ff0adff (2026-09-19) |
| `omega_teacher_hidden_cache_probe` | dir | R42 | 9 | 16 | da6f1e7 (2026-09-20) | 26a59e8 (2026-09-20) |
| `omega_teacher_logit_cache` | dir | R42 | 9 | 24 | 02e502e (2026-09-19) | 2c55aca (2026-09-20) |
| `omega_v2_0_conformance` | dir | R57 | 18 | 46 | c8de2a3 (2026-09-29) | 367d73a (2026-09-29) |
| `omega_v2_1_physical` | dir | R59 | 31 | 128 | 9742005 (2026-09-29) | a23ec1c (2026-09-29) |
| `omega_v2_1b_candidate_02` | dir | R61 | 11 | 108 | 2e8ac12 (2026-09-29) | 37bf659 (2026-09-29) |
| `omega_v2_1b_kernel_qualification` | dir | R60 | 66 | 252 | 6cd98b8 (2026-09-29) | 5ef5be6 (2026-09-29) |
| `omega_v2_1c_residency_only` | dir | R62 | 45 | 197 | 05b3bda (2026-09-29) | 1072b09 (2026-09-29) |
| `omega_v2_gpu_diag` | dir | R63 | 2 | 2 | e9f8786 (2026-09-29) | e9f8786 (2026-09-29) |
| `p2_replication` | dir | R11 | 25 | 25 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `pointer_chasing_p2_seed101` | dir | R11 | 4 | 4 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `pointer_chasing_seed101` | dir | R11 | 4 | 4 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `pointer_chasing_seed101_50k` | dir | R11 | 4 | 4 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `pointer_overfit_seed101` | dir | R11 | 6 | 6 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `runs` | dir | R11 | 400 | 400 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `sequential_update_su1_seed101` | dir | R11 | 4 | 4 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `sequential_update_su2_seed101` | dir | R11 | 16 | 16 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `sequential_update_su4_seed101` | dir | R11 | 4 | 4 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `sequential_update_su4_su5_replication` | dir | R11 | 29 | 29 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `sequential_update_su5_seed101` | dir | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `sequential_update_typed_seed101` | dir | R11 | 4 | 4 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `t1w_identity_replication` | dir | R12 | 21 | 21 | 80d6a6a (2026-09-04) | 80d6a6a (2026-09-04) |
| `t1w_workspace_identity_seed101` | dir | R12 | 4 | 4 | 80d6a6a (2026-09-04) | 80d6a6a (2026-09-04) |
| `t1w_workspace_seed101` | dir | R12 | 4 | 4 | 80d6a6a (2026-09-04) | 80d6a6a (2026-09-04) |
| `t1w_workspace_seed101_50k` | dir | R12 | 4 | 4 | 80d6a6a (2026-09-04) | 80d6a6a (2026-09-04) |
| `t2_i0_b_alg_m2_seed5601` | dir | R24 | 1 | 1 | ba6d594 (2026-09-09) | ba6d594 (2026-09-09) |
| `t2_i0_b_alg_seed5601` | dir | R24 | 1 | 1 | e3a4539 (2026-09-09) | e3a4539 (2026-09-09) |
| `t2_i0_b_r2_controls_seed5701` | dir | R24 | 1 | 1 | 741ec56 (2026-09-09) | 741ec56 (2026-09-09) |
| `t2_i0_b_r2_controls_seed5702` | dir | R24 | 1 | 1 | 1be6c8c (2026-09-09) | 1be6c8c (2026-09-09) |
| `t2_i0_b_r2_controls_seed5703` | dir | R24 | 1 | 1 | 41090ec (2026-09-09) | 41090ec (2026-09-09) |
| `t2_i0_b_r2_controls_seed5704` | dir | R24 | 1 | 1 | 12f98c2 (2026-09-09) | 12f98c2 (2026-09-09) |
| `t2_i0_b_r2_controls_seed5705` | dir | R24 | 1 | 1 | 19b35b0 (2026-09-09) | 19b35b0 (2026-09-09) |
| `t2_i0_b_r2_controls_smoke_seed5701` | dir | R24 | 1 | 1 | 715c689 (2026-09-09) | 715c689 (2026-09-09) |
| `t2_i0_b_r2_heldout_seed5701` | dir | R24 | 1 | 1 | 6c110c9 (2026-09-09) | 6c110c9 (2026-09-09) |
| `t2_i0_b_r2_heldout_seed5702` | dir | R24 | 1 | 1 | 1be6c8c (2026-09-09) | 1be6c8c (2026-09-09) |
| `t2_i0_b_r2_heldout_seed5703` | dir | R24 | 1 | 1 | 41090ec (2026-09-09) | 41090ec (2026-09-09) |
| `t2_i0_b_r2_heldout_seed5704` | dir | R24 | 1 | 1 | 12f98c2 (2026-09-09) | 12f98c2 (2026-09-09) |
| `t2_i0_b_r2_heldout_seed5705` | dir | R24 | 1 | 1 | 19b35b0 (2026-09-09) | 19b35b0 (2026-09-09) |
| `t2_i0_b_r2_heldout_smoke_seed5701` | dir | R24 | 1 | 1 | 715c689 (2026-09-09) | 715c689 (2026-09-09) |
| `t2_i0_b_r2_seed5701` | dir | R24 | 2 | 2 | 741ec56 (2026-09-09) | 741ec56 (2026-09-09) |
| `t2_i0_b_r2_seed5702` | dir | R24 | 2 | 2 | 1be6c8c (2026-09-09) | 1be6c8c (2026-09-09) |
| `t2_i0_b_r2_seed5703` | dir | R24 | 2 | 2 | 41090ec (2026-09-09) | 41090ec (2026-09-09) |
| `t2_i0_b_r2_seed5704` | dir | R24 | 2 | 2 | 12f98c2 (2026-09-09) | 12f98c2 (2026-09-09) |
| `t2_i0_b_r2_seed5705` | dir | R24 | 2 | 2 | 19b35b0 (2026-09-09) | 19b35b0 (2026-09-09) |
| `t2_i0_baseline_a_evaluation_seed4801` | dir | R24 | 1 | 1 | e42c2dd (2026-09-08) | e42c2dd (2026-09-08) |
| `t2_i0_baseline_a_preflight_seed4701` | dir | R24 | 1 | 1 | a6d715b (2026-09-08) | a6d715b (2026-09-08) |
| `t2_i0_baseline_a_seed4801` | dir | R24 | 5 | 5 | e42c2dd (2026-09-08) | e42c2dd (2026-09-08) |
| `t2_i0_baseline_b_evaluation_seed4901` | dir | R24 | 1 | 1 | 470592c (2026-09-08) | 470592c (2026-09-08) |
| `t2_i0_baseline_b_seed4901` | dir | R24 | 5 | 5 | 470592c (2026-09-08) | 470592c (2026-09-08) |
| `t2_i0_r1_1_baseline_a_seed5501` | dir | R24 | 2 | 2 | 05c4c3d (2026-09-08) | 05c4c3d (2026-09-08) |
| `t2_i0_r1_1_baseline_b_seed5601` | dir | R24 | 2 | 2 | 05c4c3d (2026-09-08) | 05c4c3d (2026-09-08) |
| `t2_i0_r1_1_controls_seed5501_5601` | dir | R24 | 1 | 1 | 05c4c3d (2026-09-08) | 05c4c3d (2026-09-08) |
| `t2_i0_r1_1_heldout_seed5501_5601` | dir | R24 | 1 | 1 | c1e57a9 (2026-09-08) | c1e57a9 (2026-09-08) |
| `t2_i0_r1_baseline_a_seed5301` | dir | R24 | 2 | 2 | c26d902 (2026-09-08) | c26d902 (2026-09-08) |
| `t2_i0_r1_baseline_b_seed5401` | dir | R24 | 2 | 2 | c26d902 (2026-09-08) | c26d902 (2026-09-08) |
| `t2_i0_r1_controls_seed5301_5401` | dir | R24 | 1 | 1 | c26d902 (2026-09-08) | c26d902 (2026-09-08) |
| `t2_i1_b_seed5801` | dir | R25 | 2 | 2 | 8acc41e (2026-09-09) | 8acc41e (2026-09-09) |
| `t2_i1_controls_seed5801` | dir | R25 | 1 | 1 | 8acc41e (2026-09-09) | 8acc41e (2026-09-09) |
| `t2_i1_heldout_seed5801` | dir | R25 | 1 | 1 | ac8a18d (2026-09-09) | ac8a18d (2026-09-09) |
| `t2_i2_r1_seed6101` | dir | R26 | 5 | 5 | 9719c0c (2026-09-10) | 9719c0c (2026-09-10) |
| `t2_i2_r2_alg_seed6201` | dir | R26 | 1 | 1 | b0ffb1b (2026-09-10) | b0ffb1b (2026-09-10) |
| `t2_i2_r2_seed6201` | dir | R26 | 7 | 7 | 146725e (2026-09-10) | 146725e (2026-09-10) |
| `t2_i2_r3_alg_seed6301` | dir | R26 | 1 | 1 | 812b596 (2026-09-11) | 812b596 (2026-09-11) |
| `t2_i2_r3_seed6301` | dir | R26 | 7 | 7 | 6e3ab28 (2026-09-10) | 6e3ab28 (2026-09-10) |
| `t2_i2_seed6101` | dir | R26 | 5 | 5 | b5f3c9c (2026-09-10) | b5f3c9c (2026-09-10) |
| `t2_i3_comp0_cal_alpha0125_seed6401` | dir | R27 | 3 | 3 | d10e676 (2026-09-12) | d10e676 (2026-09-12) |
| `t2_i3_comp0_g1_seed6401` | dir | R27 | 1 | 1 | 8070ffd (2026-09-11) | 8070ffd (2026-09-11) |
| `t2_i3_comp0_g2_seed6401` | dir | R27 | 1 | 1 | 8070ffd (2026-09-11) | 8070ffd (2026-09-11) |
| `t2_i3_comp0_g3_seed6401` | dir | R27 | 1 | 1 | 8070ffd (2026-09-11) | 8070ffd (2026-09-11) |
| `t2_i3_comp0_g4_reg_alg_seed6401` | dir | R27 | 1 | 1 | 3e1b605 (2026-09-12) | 3e1b605 (2026-09-12) |
| `t2_i3_comp0_g4_residual_alg_seed6401` | dir | R27 | 1 | 1 | 7774140 (2026-09-12) | 7774140 (2026-09-12) |
| `t2_i3_comp0_preflight_seed6401` | dir | R27 | 1 | 1 | 8070ffd (2026-09-11) | 8070ffd (2026-09-11) |
| `t2_i3_comp0_reg_alg_seed6401` | dir | R27 | 1 | 1 | 5caa0b3 (2026-09-12) | 5caa0b3 (2026-09-12) |
| `t2_i3_comp0_seed6401` | dir | R27 | 5 | 5 | 8070ffd (2026-09-11) | 8070ffd (2026-09-11) |
| `t2_i3_depth_g1_k0_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g1_k1_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g1_k2_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g1_k3_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g1_k4_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g1_k5_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g1_k6_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g1_k7_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g1_k8_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g2_k0_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g2_k1_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g2_k2_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g2_k3_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g2_k4_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g2_k5_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g2_k6_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g2_k7_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_depth_g2_k8_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_g1_k0_seed6401` | dir | R27 | 1 | 1 | 23760da (2026-09-11) | 23760da (2026-09-11) |
| `t2_i3_g1_k1_seed6401` | dir | R27 | 1 | 1 | 23760da (2026-09-11) | bc994c2 (2026-09-11) |
| `t2_i3_g1_k2_seed6401` | dir | R27 | 1 | 1 | 23760da (2026-09-11) | 23760da (2026-09-11) |
| `t2_i3_g2_k0_seed6401` | dir | R27 | 1 | 1 | 23760da (2026-09-11) | 23760da (2026-09-11) |
| `t2_i3_g2_k1_seed6401` | dir | R27 | 1 | 1 | 23760da (2026-09-11) | 23760da (2026-09-11) |
| `t2_i3_g2_k2_seed6401` | dir | R27 | 1 | 1 | 23760da (2026-09-11) | 23760da (2026-09-11) |
| `t2_i3_g3_seed6401` | dir | R27 | 1 | 1 | fe9617d (2026-09-11) | bc994c2 (2026-09-11) |
| `t2_i3_r3_f10_bind_alg_seed6401` | dir | R27 | 1 | 1 | b34230e (2026-09-12) | b34230e (2026-09-12) |
| `t2_i3_r3_residual_alg_seed6401` | dir | R27 | 1 | 1 | 0f59e29 (2026-09-11) | 0f59e29 (2026-09-11) |
| `t2_i3_seed6401` | dir | R27 | 3 | 3 | 23760da (2026-09-11) | bc994c2 (2026-09-11) |
| `t2_i3_think0_seed6401` | dir | R27 | 3 | 3 | 23760da (2026-09-11) | bc994c2 (2026-09-11) |
| `t2_i3_think1_diagnostic_seed6401` | dir | R27 | 1 | 1 | 9fbb9a1 (2026-09-11) | bc994c2 (2026-09-11) |
| `t2_i3_think2_depth_alg_seed6401` | dir | R27 | 1 | 1 | 1d729f6 (2026-09-11) | 1d729f6 (2026-09-11) |
| `t2_i3_think_alg_seed6401` | dir | R27 | 1 | 1 | 3ce51be (2026-09-11) | 3ce51be (2026-09-11) |
| `t2_i3_view0_seed6401` | dir | R27 | 0 | 0 | UNTRACKED |  |
| `t2_nobypass1_r0_seed6701` | dir | R30 | 5 | 5 | ef3e019 (2026-09-12) | ef3e019 (2026-09-12) |
| `t2_nobypass1_r11_kv_addr_alg` | dir | R30 | 1 | 1 | 2b81b92 (2026-09-12) | 2b81b92 (2026-09-12) |
| `t2_nobypass1_r11_kv_pos_alg` | dir | R30 | 1 | 1 | 9a777bc (2026-09-12) | 9a777bc (2026-09-12) |
| `t2_nobypass1_r11_kv_seed6703` | dir | R30 | 5 | 5 | 7972598 (2026-09-12) | 7972598 (2026-09-12) |
| `t2_nobypass1_r12_posfree_kv_rctx_alg` | dir | R30 | 1 | 1 | 1a754c2 (2026-09-12) | 1a754c2 (2026-09-12) |
| `t2_nobypass1_r12_posfree_kv_seed6704` | dir | R30 | 5 | 5 | dc05458 (2026-09-12) | dc05458 (2026-09-12) |
| `t2_nobypass1_r13_relkey_kv_readout_alg` | dir | R30 | 1 | 1 | 77c06a5 (2026-09-12) | 77c06a5 (2026-09-12) |
| `t2_nobypass1_r13_relkey_kv_seed6705` | dir | R30 | 6 | 6 | 882ff8c (2026-09-12) | 882ff8c (2026-09-12) |
| `t2_nobypass1_r13_zs_sharp_alg` | dir | R30 | 1 | 1 | dce9e16 (2026-09-12) | dce9e16 (2026-09-12) |
| `t2_nobypass1_r14_gate_geom_alg` | dir | R30 | 1 | 1 | f432bee (2026-09-12) | f432bee (2026-09-12) |
| `t2_nobypass1_r14_gate_signal_alg` | dir | R30 | 1 | 1 | 565907c (2026-09-12) | 565907c (2026-09-12) |
| `t2_nobypass1_r14_nc_gateonly_seed6706` | dir | R30 | 5 | 5 | 96f083a (2026-09-12) | 96f083a (2026-09-12) |
| `t2_nobypass1_r14_nc_residual_alg` | dir | R30 | 1 | 1 | 835abd5 (2026-09-12) | 835abd5 (2026-09-12) |
| `t2_nobypass1_r14_s_gateonly_seed6706` | dir | R30 | 4 | 4 | 28fc03e (2026-09-12) | 28fc03e (2026-09-12) |
| `t2_nobypass1_r14_s_seed6706` | dir | R30 | 5 | 5 | 7f0fbcd (2026-09-12) | 7f0fbcd (2026-09-12) |
| `t2_nobypass1_r1_attractor_alg` | dir | R30 | 1 | 1 | 29bca23 (2026-09-12) | 29bca23 (2026-09-12) |
| `t2_nobypass1_r1_mixpath_alg` | dir | R30 | 1 | 1 | 45e55a4 (2026-09-12) | 45e55a4 (2026-09-12) |
| `t2_nobypass1_r1_seed6702` | dir | R30 | 5 | 5 | 0eea2b1 (2026-09-12) | 0eea2b1 (2026-09-12) |
| `t2_nobypass1_staged_hardptr_alg` | dir | R30 | 1 | 1 | 67ca6f7 (2026-09-12) | 67ca6f7 (2026-09-12) |
| `t2_nobypass1_staged_hardptr_nb3_canon` | dir | R30 | 1 | 1 | b44b47b (2026-09-12) | b44b47b (2026-09-12) |
| `t2_nobypass1_staged_hardptr_nb4` | dir | R30 | 1 | 1 | b44b47b (2026-09-12) | b44b47b (2026-09-12) |
| `t2_nobypass1_staged_hardptr_nbf0` | dir | R30 | 2 | 2 | b44b47b (2026-09-12) | b44b47b (2026-09-12) |
| `t2_nobypass1_staged_nb3_canon` | dir | R30 | 1 | 1 | d985de7 (2026-09-12) | d985de7 (2026-09-12) |
| `t2_nobypass1_staged_nb4` | dir | R30 | 1 | 1 | a719bf1 (2026-09-12) | a719bf1 (2026-09-12) |
| `t2_nobypass1_staged_nb4_pairgeom_alg` | dir | R30 | 1 | 1 | c6799f4 (2026-09-12) | c6799f4 (2026-09-12) |
| `t2_nobypass1_staged_nbf0` | dir | R30 | 2 | 2 | d2b6533 (2026-09-12) | d2b6533 (2026-09-12) |
| `t2_nobypass2_final_closure` | dir | R31 | 1 | 1 | 3942431 (2026-09-12) | 3942431 (2026-09-12) |
| `t2_nobypass2_rcsep_nogate_freeze` | dir | R31 | 2 | 2 | 13b9eea (2026-09-12) | 13b9eea (2026-09-12) |
| `t2_nobypass_ab_seed6401` | dir | R30 | 1 | 1 | d4cbf33 (2026-09-12) | d4cbf33 (2026-09-12) |
| `t2_nobypass_b_all_seed6401` | dir | R30 | 1 | 1 | d54588d (2026-09-12) | d54588d (2026-09-12) |
| `t2_nobypass_b_mixed_alg_seed6401` | dir | R30 | 1 | 1 | 86629b5 (2026-09-12) | 86629b5 (2026-09-12) |
| `t2_nobypass_p0_seed6401` | dir | R30 | 1 | 1 | 7aebdf8 (2026-09-12) | 7aebdf8 (2026-09-12) |
| `t2_nobypass_ref_geom_alg` | dir | R30 | 1 | 1 | 9f1d3e8 (2026-09-12) | 9f1d3e8 (2026-09-12) |
| `t2_xf_clause_r1_1_controls_seed6001` | dir | R28 | 1 | 1 | 78ca247 (2026-09-09) | 78ca247 (2026-09-09) |
| `t2_xf_clause_r1_1_controls_seed6002` | dir | R28 | 1 | 1 | fdde50a (2026-09-09) | fdde50a (2026-09-09) |
| `t2_xf_clause_r1_1_controls_seed6003` | dir | R28 | 1 | 1 | f753ed4 (2026-09-09) | f753ed4 (2026-09-09) |
| `t2_xf_clause_r1_1_controls_seed6004` | dir | R28 | 1 | 1 | c934f1c (2026-09-09) | c934f1c (2026-09-09) |
| `t2_xf_clause_r1_1_controls_seed6005` | dir | R28 | 1 | 1 | ad8f8c3 (2026-09-09) | ad8f8c3 (2026-09-09) |
| `t2_xf_clause_r1_1_heldout_seed6001` | dir | R28 | 1 | 1 | 1598425 (2026-09-09) | 1598425 (2026-09-09) |
| `t2_xf_clause_r1_1_heldout_seed6002` | dir | R28 | 1 | 1 | fdde50a (2026-09-09) | fdde50a (2026-09-09) |
| `t2_xf_clause_r1_1_heldout_seed6002_with_failures` | dir | R28 | 1 | 1 | fdde50a (2026-09-09) | fdde50a (2026-09-09) |
| `t2_xf_clause_r1_1_heldout_seed6003` | dir | R28 | 1 | 1 | f753ed4 (2026-09-09) | f753ed4 (2026-09-09) |
| `t2_xf_clause_r1_1_heldout_seed6004` | dir | R28 | 1 | 1 | c934f1c (2026-09-09) | c934f1c (2026-09-09) |
| `t2_xf_clause_r1_1_heldout_seed6005` | dir | R28 | 1 | 1 | ad8f8c3 (2026-09-09) | ad8f8c3 (2026-09-09) |
| `t2_xf_clause_r1_1_seed6001` | dir | R28 | 2 | 2 | 78ca247 (2026-09-09) | 78ca247 (2026-09-09) |
| `t2_xf_clause_r1_1_seed6002` | dir | R28 | 2 | 2 | fdde50a (2026-09-09) | fdde50a (2026-09-09) |
| `t2_xf_clause_r1_1_seed6003` | dir | R28 | 2 | 2 | f753ed4 (2026-09-09) | f753ed4 (2026-09-09) |
| `t2_xf_clause_r1_1_seed6004` | dir | R28 | 2 | 2 | c934f1c (2026-09-09) | c934f1c (2026-09-09) |
| `t2_xf_clause_r1_1_seed6005` | dir | R28 | 2 | 2 | ad8f8c3 (2026-09-09) | ad8f8c3 (2026-09-09) |
| `t2_xf_clause_residual_audit_seed6001` | dir | R28 | 1 | 1 | 43d2b3c (2026-09-09) | 43d2b3c (2026-09-09) |
| `t2_xf_seq_r1_1_controls_seed6001` | dir | R29 | 1 | 1 | a19c2c1 (2026-09-09) | a19c2c1 (2026-09-09) |
| `t2_xf_seq_r1_1_controls_seed6002` | dir | R29 | 1 | 1 | 409cd11 (2026-09-09) | 409cd11 (2026-09-09) |
| `t2_xf_seq_r1_1_controls_seed6003` | dir | R29 | 1 | 1 | 409cd11 (2026-09-09) | 953fbce (2026-09-09) |
| `t2_xf_seq_r1_1_controls_seed6004` | dir | R29 | 1 | 1 | 917626d (2026-09-09) | 917626d (2026-09-09) |
| `t2_xf_seq_r1_1_controls_seed6005` | dir | R29 | 1 | 1 | 00e30a4 (2026-09-09) | 00e30a4 (2026-09-09) |
| `t2_xf_seq_r1_1_heldout_seed6001` | dir | R29 | 1 | 1 | f471fcc (2026-09-09) | f471fcc (2026-09-09) |
| `t2_xf_seq_r1_1_heldout_seed6002` | dir | R29 | 1 | 1 | 409cd11 (2026-09-09) | 409cd11 (2026-09-09) |
| `t2_xf_seq_r1_1_heldout_seed6003` | dir | R29 | 1 | 1 | 953fbce (2026-09-09) | 953fbce (2026-09-09) |
| `t2_xf_seq_r1_1_heldout_seed6004` | dir | R29 | 1 | 1 | 917626d (2026-09-09) | 917626d (2026-09-09) |
| `t2_xf_seq_r1_1_heldout_seed6005` | dir | R29 | 1 | 1 | 00e30a4 (2026-09-09) | 00e30a4 (2026-09-09) |
| `t2_xf_seq_r1_1_seed6001` | dir | R29 | 2 | 2 | a19c2c1 (2026-09-09) | a19c2c1 (2026-09-09) |
| `t2_xf_seq_r1_1_seed6002` | dir | R29 | 2 | 2 | 409cd11 (2026-09-09) | 409cd11 (2026-09-09) |
| `t2_xf_seq_r1_1_seed6003` | dir | R29 | 2 | 2 | 409cd11 (2026-09-09) | 953fbce (2026-09-09) |
| `t2_xf_seq_r1_1_seed6004` | dir | R29 | 2 | 2 | 917626d (2026-09-09) | 917626d (2026-09-09) |
| `t2_xf_seq_r1_1_seed6005` | dir | R29 | 2 | 2 | 00e30a4 (2026-09-09) | 00e30a4 (2026-09-09) |
| `t3_nobypass1_noop_distractor` | dir | R32 | 11 | 11 | 754b28f (2026-09-12) | 754b28f (2026-09-12) |
| `t4_final_closure` | dir | R33 | 1 | 1 | 8525a29 (2026-09-13) | 8525a29 (2026-09-13) |
| `t4_nobypass1_three_active_roles` | dir | R33 | 19 | 19 | 6b547ef (2026-09-12) | 49c47f3 (2026-09-13) |
| `t4_nobypass2_rcsep3_bg` | dir | R33 | 12 | 12 | ed1bdef (2026-09-13) | ed1bdef (2026-09-13) |
| `t4_nobypass3_fresh` | dir | R33 | 18 | 18 | 384c90d (2026-09-13) | ccbff5e (2026-09-13) |
| `t5_final_closure` | dir | R34 | 1 | 1 | 58ccb36 (2026-09-13) | 58ccb36 (2026-09-13) |
| `t5_n4_development` | dir | R34 | 11 | 11 | ce1fe26 (2026-09-13) | ce1fe26 (2026-09-13) |
| `t5_n4_fresh` | dir | R34 | 11 | 11 | 2a7e0e0 (2026-09-13) | 2a7e0e0 (2026-09-13) |
| `t5_n4_fresh_preparation` | dir | R34 | 6 | 6 | a7ccb09 (2026-09-13) | a7ccb09 (2026-09-13) |
| `t5_n4_preparation` | dir | R34 | 6 | 6 | b1b2dc4 (2026-09-13) | b1b2dc4 (2026-09-13) |
| `t5_nrole_design_audit` | dir | R34 | 1 | 1 | d6c98a8 (2026-09-13) | d6c98a8 (2026-09-13) |
| `t6_activeset_midpoint_development` | dir | R35 | 11 | 11 | bd54f53 (2026-09-13) | bd54f53 (2026-09-13) |
| `t6_activeset_midpoint_development_closed` | dir | R35 | 16 | 16 | bd54f53 (2026-09-13) | bd54f53 (2026-09-13) |
| `t6_activeset_midpoint_development_closed2` | dir | R35 | 16 | 16 | d6060e7 (2026-09-13) | d6060e7 (2026-09-13) |
| `t6_activeset_midpoint_development_complete` | dir | R35 | 16 | 16 | bd54f53 (2026-09-13) | bd54f53 (2026-09-13) |
| `t6_activeset_midpoint_development_corrected` | dir | R35 | 10 | 10 | bd54f53 (2026-09-13) | bd54f53 (2026-09-13) |
| `t6_activeset_midpoint_development_corrected2` | dir | R35 | 10 | 10 | bd54f53 (2026-09-13) | bd54f53 (2026-09-13) |
| `t6_activeset_midpoint_development_final` | dir | R35 | 11 | 11 | bd54f53 (2026-09-13) | bd54f53 (2026-09-13) |
| `t6_activeset_midpoint_fresh_development` | dir | R35 | 16 | 16 | 1bb3ecb (2026-09-13) | 1bb3ecb (2026-09-13) |
| `t6_activeset_midpoint_fresh_preparation` | dir | R35 | 8 | 8 | bd54f53 (2026-09-13) | bd54f53 (2026-09-13) |
| `t6_activeset_midpoint_preparation` | dir | R35 | 6 | 6 | e35dab4 (2026-09-13) | e35dab4 (2026-09-13) |
| `t6_final_closure` | dir | R35 | 1 | 1 | 286c838 (2026-09-13) | 286c838 (2026-09-13) |
| `t6_nrole_activeset_design` | dir | R35 | 1 | 1 | 22abe57 (2026-09-13) | 22abe57 (2026-09-13) |
| `t7_final_closure` | dir | R36 | 2 | 2 | 7c0dece (2026-09-13) | 7c0dece (2026-09-13) |
| `t7_noop_none_lexical_development_preparation` | dir | R36 | 6 | 6 | 1860fc5 (2026-09-13) | 26427d5 (2026-09-13) |
| `t7_noop_none_lexical_fresh_preparation` | dir | R36 | 8 | 8 | f33f525 (2026-09-13) | f33f525 (2026-09-13) |
| `t7_noop_none_lexical_paired_development` | dir | R36 | 6 | 11 | 60f8d86 (2026-09-13) | 60f8d86 (2026-09-13) |
| `t7_noop_none_lexical_paired_development_conformance` | dir | R36 | 7 | 12 | 2769776 (2026-09-13) | 2769776 (2026-09-13) |
| `t7_noop_none_lexical_paired_fresh` | dir | R36 | 11 | 16 | 96ab55a (2026-09-13) | 96ab55a (2026-09-13) |
| `t7_noop_none_lexical_stage_a_development` | dir | R36 | 11 | 11 | 13a123f (2026-09-13) | 13a123f (2026-09-13) |
| `t7_noop_none_lexical_stage_a_fresh` | dir | R36 | 12 | 12 | 5a77827 (2026-09-13) | 5a77827 (2026-09-13) |
| `t7_noop_none_lexical_stage_b_development` | dir | R36 | 11 | 11 | df37132 (2026-09-13) | df37132 (2026-09-13) |
| `t7_noop_none_lexical_stage_b_fresh` | dir | R36 | 12 | 12 | 5a4da75 (2026-09-13) | 5a4da75 (2026-09-13) |
| `t7_noop_none_lexical_stage_b_sanity_repair` | dir | R36 | 6 | 6 | fea9ea4 (2026-09-13) | fea9ea4 (2026-09-13) |
| `t7_noop_none_lexical_stage_c_development` | dir | R36 | 11 | 11 | 20b15e9 (2026-09-13) | 20b15e9 (2026-09-13) |
| `t7_noop_none_lexical_stage_c_fresh` | dir | R36 | 12 | 12 | 2ae3121 (2026-09-13) | 2ae3121 (2026-09-13) |
| `t7_noop_none_supervision_preflight` | dir | R36 | 1 | 1 | 8f22eee (2026-09-13) | 8f22eee (2026-09-13) |
| `t7_nrole_activeset_noop_distractor_design` | dir | R36 | 1 | 1 | b7fb256 (2026-09-13) | b7fb256 (2026-09-13) |
| `u0a_iso_clean_seed101_12000` | dir | R13 | 7 | 7 | 852a4cb (2026-09-05) | 852a4cb (2026-09-05) |
| `u0a_iso_clean_seed202_12000` | dir | R13 | 7 | 7 | 852a4cb (2026-09-05) | 852a4cb (2026-09-05) |
| `u0a_iso_clean_seed303_12000` | dir | R13 | 7 | 7 | 852a4cb (2026-09-05) | 852a4cb (2026-09-05) |
| `u0a_iso_clean_seed404_12000` | dir | R13 | 7 | 7 | 852a4cb (2026-09-05) | 852a4cb (2026-09-05) |
| `u0a_iso_clean_seed505_12000` | dir | R13 | 7 | 7 | 852a4cb (2026-09-05) | 852a4cb (2026-09-05) |
| `u0c_c0_oracle_seed101` | dir | R15 | 4 | 4 | fb14208 (2026-09-05) | fb14208 (2026-09-05) |
| `u0c_c0_oracle_seed101_network_trainable` | dir | R15 | 4 | 4 | c4a66d8 (2026-09-05) | a1588c4 (2026-09-05) |
| `u0c_c0_reader_seed101` | dir | R15 | 3 | 3 | a1588c4 (2026-09-05) | 9ceaaf4 (2026-09-05) |
| `u0c_c1_joint_seed101` | dir | R15 | 6 | 6 | 285368c (2026-09-05) | 08ba28a (2026-09-05) |
| `u0c_c1_lossnorm_anneal_seed101_12000` | dir | R15 | 27 | 27 | d78d2f0 (2026-09-05) | d78d2f0 (2026-09-05) |
| `u0c_c1_lossnorm_anneal_seed202_12000` | dir | R15 | 27 | 27 | 364d99c (2026-09-05) | 364d99c (2026-09-05) |
| `u0c_c1_lossnorm_anneal_seed303_12000` | dir | R15 | 27 | 27 | d78d2f0 (2026-09-05) | d78d2f0 (2026-09-05) |
| `u0c_c1_lossnorm_anneal_seed404_12000` | dir | R15 | 27 | 27 | d78d2f0 (2026-09-05) | d78d2f0 (2026-09-05) |
| `u0c_c1_lossnorm_anneal_seed505_12000` | dir | R15 | 27 | 27 | d78d2f0 (2026-09-05) | d78d2f0 (2026-09-05) |
| `u0c_c1_lossnorm_seed101` | dir | R15 | 27 | 27 | 7100832 (2026-09-05) | 7100832 (2026-09-05) |
| `u0c_c1_lossnorm_seed202_12000` | dir | R15 | 27 | 27 | a4c7be5 (2026-09-05) | a4c7be5 (2026-09-05) |
| `u0c_c1_lossnorm_seed303_12000` | dir | R15 | 27 | 27 | a4c7be5 (2026-09-05) | a4c7be5 (2026-09-05) |
| `u0c_c1_lossnorm_seed404_12000` | dir | R15 | 27 | 27 | a4c7be5 (2026-09-05) | a4c7be5 (2026-09-05) |
| `u0c_c1_lossnorm_seed505_12000` | dir | R15 | 27 | 27 | a4c7be5 (2026-09-05) | a4c7be5 (2026-09-05) |
| `u0c_c1_mix_o_composition_seed101_frozen` | dir | R16 | 2 | 2 | d8136a2 (2026-09-06) | 1911bf2 (2026-09-06) |
| `u0c_c1_mix_o_composition_seed202_frozen` | dir | R16 | 2 | 2 | d8136a2 (2026-09-06) | 1911bf2 (2026-09-06) |
| `u0c_c1_mix_o_composition_seed303_frozen` | dir | R16 | 2 | 2 | d8136a2 (2026-09-06) | 1911bf2 (2026-09-06) |
| `u0c_c1_mix_o_composition_seed404_frozen` | dir | R16 | 2 | 2 | d8136a2 (2026-09-06) | 1911bf2 (2026-09-06) |
| `u0c_c1_mix_o_composition_seed505_frozen` | dir | R16 | 2 | 2 | d8136a2 (2026-09-06) | 1911bf2 (2026-09-06) |
| `u0c_c1_mix_o_depth_seed101_frozen` | dir | R16 | 3 | 3 | acdc18a (2026-09-06) | acdc18a (2026-09-06) |
| `u0c_c1_mix_o_depth_seed202_frozen` | dir | R16 | 3 | 3 | acdc18a (2026-09-06) | acdc18a (2026-09-06) |
| `u0c_c1_mix_o_depth_seed303_frozen` | dir | R16 | 3 | 3 | acdc18a (2026-09-06) | acdc18a (2026-09-06) |
| `u0c_c1_mix_o_depth_seed404_frozen` | dir | R16 | 3 | 3 | acdc18a (2026-09-06) | acdc18a (2026-09-06) |
| `u0c_c1_mix_o_depth_seed505_frozen` | dir | R16 | 3 | 3 | acdc18a (2026-09-06) | acdc18a (2026-09-06) |
| `u0c_c1_mix_o_e_r_alu_read_set_seed101_frozen` | dir | R16 | 5 | 5 | 51b83fa (2026-09-06) | 51b83fa (2026-09-06) |
| `u0c_c1_mix_o_e_r_alu_seed101_activation_audit` | dir | R16 | 3 | 3 | c2187f2 (2026-09-06) | c2187f2 (2026-09-06) |
| `u0c_c1_mix_o_e_r_alu_seed101_frozen` | dir | R16 | 3 | 3 | 5b54410 (2026-09-06) | 0db782e (2026-09-06) |
| `u0c_c1_mix_o_memory_frozen` | dir | R16 | 13 | 13 | ce29d39 (2026-09-06) | 7e223e5 (2026-09-06) |
| `u0c_c1_mix_o_seed101_frozen` | dir | R16 | 2 | 2 | d3f6d3d (2026-09-06) | d3f6d3d (2026-09-06) |
| `u0c_c1_mix_o_seed101_frozen_read_set_explicit_1024_each` | dir | R16 | 2 | 2 | 51b83fa (2026-09-06) | 51b83fa (2026-09-06) |
| `u0c_c1_mix_o_seed101_frozen_regression_1024_each` | dir | R16 | 2 | 2 | 5b54410 (2026-09-06) | 5b54410 (2026-09-06) |
| `u0c_c1_read_set_validation` | dir | R16 | 4 | 4 | d8136a2 (2026-09-06) | d8136a2 (2026-09-06) |
| `u0c_c1_workspace_audit_seed101` | dir | R15 | 3 | 3 | 9c9ab4c (2026-09-05) | 9c9ab4c (2026-09-05) |
| `u0c_ctrl1_pilot_seed101_frozen` | dir | R17 | 16 | 16 | d945111 (2026-09-06) | d945111 (2026-09-06) |
| `u0c_ctrl1_preflight_seed101_frozen` | dir | R17 | 3 | 3 | b2953f3 (2026-09-06) | b2953f3 (2026-09-06) |
| `u0c_ctrl2_g_coverage_corrected` | dir | R18 | 3 | 3 | c3f0ebe (2026-09-06) | c3f0ebe (2026-09-06) |
| `u0c_ctrl2_g_coverage_frozen` | dir | R18 | 3 | 3 | 6d6eeac (2026-09-06) | 6d6eeac (2026-09-06) |
| `u0c_ctrl2_o_canon_diagnostic` | dir | R18 | 1 | 1 | 7b63a95 (2026-09-06) | 7b63a95 (2026-09-06) |
| `u0c_ctrl2_o_consistency_diagnostic` | dir | R18 | 1 | 1 | 0fe27bd (2026-09-06) | 0fe27bd (2026-09-06) |
| `u0c_ctrl2_o_consistency_pilot_seed2201_frozen` | dir | R18 | 18 | 18 | 0fe27bd (2026-09-06) | 0fe27bd (2026-09-06) |
| `u0c_ctrl2_o_pilot_seed2201_frozen` | dir | R18 | 18 | 18 | 1954e35 (2026-09-06) | 1954e35 (2026-09-06) |
| `u0c_ctrl2_o_replica_seed2202_canon` | dir | R18 | 1 | 1 | a2427cc (2026-09-06) | a2427cc (2026-09-06) |
| `u0c_ctrl2_o_replica_seed2202_frozen` | dir | R18 | 15 | 15 | a2427cc (2026-09-06) | a2427cc (2026-09-06) |
| `u0c_ctrl2_o_replica_seed2203_canon` | dir | R18 | 1 | 1 | a2427cc (2026-09-06) | a2427cc (2026-09-06) |
| `u0c_ctrl2_o_replica_seed2203_frozen` | dir | R18 | 15 | 15 | a2427cc (2026-09-06) | a2427cc (2026-09-06) |
| `u0c_ctrl2_o_replica_seed2204_canon` | dir | R18 | 1 | 1 | a2427cc (2026-09-06) | a2427cc (2026-09-06) |
| `u0c_ctrl2_o_replica_seed2204_frozen` | dir | R18 | 15 | 15 | a2427cc (2026-09-06) | a2427cc (2026-09-06) |
| `u0c_ctrl2_o_replica_seed2205_canon` | dir | R18 | 1 | 1 | a2427cc (2026-09-06) | a2427cc (2026-09-06) |
| `u0c_ctrl2_o_replica_seed2205_frozen` | dir | R18 | 15 | 15 | a2427cc (2026-09-06) | a2427cc (2026-09-06) |
| `u0c_ctrl2_o_replicas` | dir | R18 | 1 | 1 | a2427cc (2026-09-06) | a2427cc (2026-09-06) |
| `u0c_ctrl2_ordinal_audit` | dir | R18 | 7 | 7 | 30c20fa (2026-09-06) | 30c20fa (2026-09-06) |
| `u0c_ctrl2_pilot_seed2201_frozen` | dir | R18 | 17 | 17 | b5c5950 (2026-09-06) | b5c5950 (2026-09-06) |
| `u0c_ctrl2_preflight_seed101_frozen` | dir | R18 | 3 | 3 | b7a0bbb (2026-09-06) | b7a0bbb (2026-09-06) |
| `u0c_ctrl2_replica_seed2202` | dir | R18 | 17 | 17 | df34e14 (2026-09-06) | df34e14 (2026-09-06) |
| `u0c_ctrl2_replica_seed2202_g` | dir | R18 | 3 | 3 | df34e14 (2026-09-06) | df34e14 (2026-09-06) |
| `u0c_ctrl2_replica_seed2203` | dir | R18 | 17 | 17 | df34e14 (2026-09-06) | df34e14 (2026-09-06) |
| `u0c_ctrl2_replica_seed2203_g` | dir | R18 | 3 | 3 | df34e14 (2026-09-06) | df34e14 (2026-09-06) |
| `u0c_ctrl2_replica_seed2204` | dir | R18 | 17 | 17 | df34e14 (2026-09-06) | df34e14 (2026-09-06) |
| `u0c_ctrl2_replica_seed2204_g` | dir | R18 | 3 | 3 | df34e14 (2026-09-06) | df34e14 (2026-09-06) |
| `u0c_ctrl2_replica_seed2205` | dir | R18 | 17 | 17 | df34e14 (2026-09-06) | df34e14 (2026-09-06) |
| `u0c_ctrl2_replica_seed2205_g` | dir | R18 | 3 | 3 | df34e14 (2026-09-06) | df34e14 (2026-09-06) |
| `u0c_ctrl3_canonical_seed2201` | dir | R19 | 1 | 1 | a6c98be (2026-09-06) | fb4d91a (2026-09-06) |
| `u0c_ctrl3_canonical_seed2202` | dir | R19 | 1 | 1 | 3bf4a52 (2026-09-06) | 3bf4a52 (2026-09-06) |
| `u0c_ctrl3_canonical_seed2203` | dir | R19 | 1 | 1 | 3bf4a52 (2026-09-06) | 3bf4a52 (2026-09-06) |
| `u0c_ctrl3_canonical_seed2204` | dir | R19 | 1 | 1 | 3bf4a52 (2026-09-06) | 3bf4a52 (2026-09-06) |
| `u0c_ctrl3_canonical_seed2205` | dir | R19 | 1 | 1 | 3bf4a52 (2026-09-06) | 3bf4a52 (2026-09-06) |
| `u0c_ctrl3_real_r_seed2201` | dir | R19 | 2 | 2 | dc1b62f (2026-09-06) | fb4d91a (2026-09-06) |
| `u0c_ctrl3_real_r_seed2202` | dir | R19 | 1 | 1 | 3bf4a52 (2026-09-06) | 3bf4a52 (2026-09-06) |
| `u0c_ctrl3_real_r_seed2203` | dir | R19 | 1 | 1 | 3bf4a52 (2026-09-06) | 3bf4a52 (2026-09-06) |
| `u0c_ctrl3_real_r_seed2204` | dir | R19 | 1 | 1 | 3bf4a52 (2026-09-06) | 3bf4a52 (2026-09-06) |
| `u0c_ctrl3_real_r_seed2205` | dir | R19 | 1 | 1 | 3bf4a52 (2026-09-06) | 3bf4a52 (2026-09-06) |
| `u0c_ctrl4_evaluation_seed4401` | dir | R20 | 1 | 1 | 1f2f2c8 (2026-09-06) | 1f2f2c8 (2026-09-06) |
| `u0c_ctrl4_interop` | dir | R20 | 1 | 1 | b232158 (2026-09-06) | b232158 (2026-09-06) |
| `u0c_ctrl4_pilot_seed4401` | dir | R20 | 12 | 12 | 1f2f2c8 (2026-09-06) | 1f2f2c8 (2026-09-06) |
| `u0c_ctrl4_preflight_seed2201` | dir | R20 | 1 | 1 | 2cbfc13 (2026-09-06) | 2cbfc13 (2026-09-06) |
| `u0c_ctrl5_evaluation_seed4501` | dir | R21 | 1 | 1 | f382678 (2026-09-06) | f382678 (2026-09-06) |
| `u0c_ctrl5_pilot_seed4501` | dir | R21 | 13 | 13 | f382678 (2026-09-06) | f382678 (2026-09-06) |
| `u0c_ctrl5_preflight_seed2201` | dir | R21 | 1 | 1 | 1240f0c (2026-09-06) | 1240f0c (2026-09-06) |
| `u0c_ctrl6_clamp_evaluation_seed4601` | dir | R22 | 1 | 1 | 16a2da7 (2026-09-08) | 16a2da7 (2026-09-08) |
| `u0c_ctrl6_pilot_seed4601` | dir | R22 | 13 | 13 | e6f5a31 (2026-09-08) | e6f5a31 (2026-09-08) |
| `u0c_ctrl6_preflight_seed2201` | dir | R22 | 1 | 1 | 65f23ee (2026-09-08) | 65f23ee (2026-09-08) |
| `u0c_ctrl6_trained_evaluation_seed4601` | dir | R22 | 1 | 1 | e6f5a31 (2026-09-08) | e6f5a31 (2026-09-08) |
| `u0c_ctrl7_heldout_seed4701` | dir | R23 | 1 | 1 | aa9ce74 (2026-09-08) | aa9ce74 (2026-09-08) |
| `u0c_ctrl7_pilot_seed4701` | dir | R23 | 11 | 11 | 5635d74 (2026-09-08) | 5635d74 (2026-09-08) |
| `u0c_ctrl7_preflight_seed2201` | dir | R23 | 1 | 1 | b054450 (2026-09-08) | b054450 (2026-09-08) |
| `u0c_ctrl7_trained_evaluation_seed4701` | dir | R23 | 1 | 1 | 5635d74 (2026-09-08) | 5635d74 (2026-09-08) |
| `u0c_select_integration_seed101` | dir | R15 | 3 | 3 | 71413bb (2026-09-05) | 9ceaaf4 (2026-09-05) |
| `variable_binding_typed_replication` | dir | R11 | 21 | 21 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `variable_binding_typed_seed101` | dir | R11 | 4 | 4 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `wd_norm_fix` | dir | R11 | 45 | 45 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `affected_campaign.log` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `affected_pilot_complete.json` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `affected_plan.json` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `affected_process.stderr.log` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `affected_process.stdout.log` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `affected_state.json` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `analysis_summary.json` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `analysis_summary.stdout.log` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `campaign.log` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `nb5_fresh_690x_manifest_check.json` | file | R31 | 1 | 1 | 3d5eb6a (2026-09-12) | 3d5eb6a (2026-09-12) |
| `p0_oracle.json` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `p1_round_diagnostics.json` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `p1_round_diagnostics_p2.json` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `p2_closure_audit.json` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `p2_closure_audit.stdout.log` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `phase_a_audit.json` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `phase_a_audit.stdout.log` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `pilot_complete.json` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `plan.json` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `process.stderr.log` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `process.stdout.log` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `state.json` | file | R11 | 1 | 1 | 4fc5589 (2026-09-04) | 4fc5589 (2026-09-04) |
| `t2_i2_architecture_report.json` | file | R26 | 1 | 1 | b5f3c9c (2026-09-10) | b5f3c9c (2026-09-10) |
| `t2_i2_r1_architecture_report.json` | file | R26 | 1 | 1 | 9719c0c (2026-09-10) | 9719c0c (2026-09-10) |
| `t2_i2_r1_smoke.json` | file | R26 | 1 | 1 | 9719c0c (2026-09-10) | 9719c0c (2026-09-10) |
| `t2_i2_r2_architecture_report.json` | file | R26 | 1 | 1 | 146725e (2026-09-10) | 146725e (2026-09-10) |
| `t2_i2_r2_smoke.json` | file | R26 | 1 | 1 | 146725e (2026-09-10) | 146725e (2026-09-10) |
| `t2_i2_r3_architecture_report.json` | file | R26 | 1 | 1 | 6e3ab28 (2026-09-10) | 6e3ab28 (2026-09-10) |
| `t2_i2_r3_smoke.json` | file | R26 | 1 | 1 | 6e3ab28 (2026-09-10) | 6e3ab28 (2026-09-10) |
| `t2_i2_smoke.json` | file | R26 | 1 | 1 | b5f3c9c (2026-09-10) | b5f3c9c (2026-09-10) |
| `t2_i3_architecture_report.json` | file | R27 | 1 | 1 | 23760da (2026-09-11) | 23760da (2026-09-11) |
| `t2_i3_calibration_manifest.json` | file | R27 | 1 | 1 | 23760da (2026-09-11) | 23760da (2026-09-11) |
| `t2_i3_smoke.json` | file | R27 | 1 | 1 | 23760da (2026-09-11) | 23760da (2026-09-11) |
| `t3_nobypass1_noop_distractor_manifest_check.json` | file | R32 | 1 | 1 | d90459d (2026-09-12) | d90459d (2026-09-12) |
| `u0a_u0b_provenance_archive.json` | file | R13 | 1 | 1 | 88ecbc9 (2026-09-05) | 88ecbc9 (2026-09-05) |
| `u0b_b1_pointer_residual.json` | file | R14 | 1 | 1 | bcc4884 (2026-09-05) | bcc4884 (2026-09-05) |
| `u0b_b2_pointer_frozen.json` | file | R14 | 1 | 1 | 1a792c7 (2026-09-05) | 180d7b0 (2026-09-05) |
| `u0b_b3_write_e_disabled.json` | file | R14 | 1 | 1 | b1add52 (2026-09-05) | b1add52 (2026-09-05) |
| `u0b_b4_raw_register.json` | file | R14 | 1 | 1 | a5f3e06 (2026-09-05) | e442b11 (2026-09-05) |
| `u0b_b5_adapter_opcode_permutation.json` | file | R14 | 1 | 1 | e442b11 (2026-09-05) | fdf397f (2026-09-05) |
| `u0b_b5_head_permutation.json` | file | R14 | 1 | 1 | 7b555df (2026-09-05) | fdf397f (2026-09-05) |
| `u0b_b5_ledger.json` | file | R14 | 1 | 1 | fdf397f (2026-09-05) | fdf397f (2026-09-05) |
| `u0b_b6_workspace_replace.json` | file | R14 | 1 | 1 | 4e98a6c (2026-09-05) | 4e98a6c (2026-09-05) |
| `u0b_b7_workspace_frozen.json` | file | R14 | 1 | 1 | b828590 (2026-09-05) | b828590 (2026-09-05) |
| `u0b_b8_no_payload_workspace.json` | file | R14 | 1 | 1 | 7b9fde6 (2026-09-05) | 7b9fde6 (2026-09-05) |
| `u0b_b9_zero_reader_payload.json` | file | R14 | 1 | 1 | 57a1a9e (2026-09-05) | 57a1a9e (2026-09-05) |
| `u0c_correction_capacity_audit.json` | file | R15 | 1 | 1 | fdf397f (2026-09-05) | fb14208 (2026-09-05) |
| `h0-avx2-focus` | trash dir | R09 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `h0-parallel-avx2` | trash dir | R09 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `h0-parallel-avx2-physical4-repeated5` | trash dir | R09 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `h0-parallel-avx2-repeated5` | trash dir | R09 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-barrier-diagnostic` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-bridge1-literal-fused-d1472` | trash dir | R05 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-bridge1-literal-fused-threeway-d1472` | trash dir | R05 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-bridge1-literal-fused-threeway-rectangular` | trash dir | R05 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-bridge1-residency-d1472` | trash dir | R06 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-checksum-order-verification-20260902` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-fresh-verification-20260902` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-l1-combined-compare` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-l1-comparison` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-l1-direct-load-comparison` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-l1-row-simple-comparison` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-l1-spillfree-compare` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-legacy-compare-diagnostic` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-phase3` | trash dir | R02 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-phase3-median10` | trash dir | R02 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-phase3-median10-rerun` | trash dir | R01 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-r1-repeat-diagnostic` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-r16-alternating-order-diagnostic` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-r16-alternating-order-diagnostic-prebuild-stale-exe` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-r16-combined-diagnostic` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-r16-direct-load-diagnostic` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-r16-repeat-diagnostic` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-r16-row-simple-diagnostic` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-r16-self-test-gated-diagnostic` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-r16-spillfree-compare` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-recurrence-component-breakdown` | trash dir | R03 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-recurrence-component-breakdown-avx2` | trash dir | R03 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-recurrence-correction-evidence` | trash dir | R03 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-recurrence-performance-d512` | trash dir | R03 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-recurrence-performance-d512-avx2` | trash dir | R03 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-recurrence-residency-ab-d512-r16` | trash dir | R03 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-recurrence-vectorization-comparison` | trash dir | R03 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-recurrence-vectorization-comparison-exact` | trash dir | R03 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-sharding-diagnostic` | trash dir | R04 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0m-small-static-control-d512` | trash dir | R03 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0r-bclone-block-loop-timing-20260903` | trash dir | R08 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0r-bclone-control-original-20260903` | trash dir | R08 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0r-bclone-defender-excluded-20260903` | trash dir | R08 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0r-bclone-original-frozen-20260903` | trash dir | R08 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0r-bclone-ready-timing-20260903` | trash dir | R08 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0r-bclone-warmup-affinity-timing-20260903` | trash dir | R08 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `t0r-int8-sharded` | trash dir | R07 | see row | see row | 99121f6 (move) | 99121f6 (move) |
| `h0_probe.stderr.tmp` | trash file | R09 | 1 | 1 | 99121f6 (move) | 99121f6 (move) |
| `h0_probe.stdout.tmp` | trash file | R09 | 1 | 1 | 99121f6 (move) | 99121f6 (move) |
| `h0_sweep.csv` | trash file | R09 | 1 | 1 | 99121f6 (move) | 99121f6 (move) |
| `h0_sweep.stderr.log` | trash file | R09 | 1 | 1 | 99121f6 (move) | 99121f6 (move) |
| `h0_sweep.summary.txt` | trash file | R09 | 1 | 1 | 99121f6 (move) | 99121f6 (move) |
| `manual.stderr.tmp` | trash file | R09 | 1 | 1 | 99121f6 (move) | 99121f6 (move) |
| `manual.stdout.tmp` | trash file | R09 | 1 | 1 | 99121f6 (move) | 99121f6 (move) |

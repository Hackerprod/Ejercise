# OMEGA-V2-1 — Native Windows Physical T0

Authority: `MD/304.md` (full 2,200-line contract), the premeasurement clarifications in `MD/305.md`, validated V2-0 seal at `../omega_v2_0_conformance/V2_0_RESULT_SEAL.json`, and authorized snapshot merge-base `38061477d4c2b0c5c20d75d902b21dd1ef0a2611`. This isolated unit measures only physical T0 on the actual Windows host. It does not train, score language, invoke GPU/RunPod, or start T3.

## Frozen question and scope

Measure (A) shared resident Q4 core, (B) equal-valued but distinct untied storage rotated through a measured pool, and (C) A's exact storage after per-round cache eviction. The same native kernel, Q4 bytes/scales, activations, round order, worker pool, and QPC timer are used for all variants. Sweep d={512,640}, m={1 control,4,8,16}, K={1,4,8}, A/B/C: 72 cells, 105 measured samples/cell (5 blocks × 21), plus ≥10 warmups/cell. `schedule_seed=20260929` permutes A/B/C order inside each block.

Before H0, the coordinator is pinned to a single-thread E-core CPU-set, so it cannot contend with a P-core candidate. P/E identity is read on each pinned physical core from Intel CPUID leaf `0x1A` (`0x40` Core, `0x20` Atom); Windows EfficiencyClass is also recorded, not used as an unverified performance ordering. H0 measures each P-core candidate individually with one hardware thread and no worker on its SMT sibling, d512, the same fixed W_Q shard/Q4 weights/kernel, m={1,4,8,16}; every m has 10 warmups and 31 measured repetitions. E-core timings are not collected. For each P-core candidate, `v_i(m)=median(MAC/s)` over 31 samples for m={4,8,16}, and `v_i=(v_i(4)*v_i(8)*v_i(16))^(1/3)`. m1 is control-only and excluded from P-core selection and sharding. Choose the four P-core candidates with highest v_i; exact ties choose the lowest Windows CPU-set ID. The same v_i determines contiguous, non-overlapping output-row shards, rounded only to the kernel's declared output-row tile (1).

Primary workers are four distinct physical P-cores selected from measured H0 throughputs, pinned to fixed Windows CPU-set IDs without worker threads on SMT siblings. No complete core is replicated per worker. The B group pool must be at least `max(2.5 × measured LLC, 64 MiB)`; C uses CLFLUSH when available, otherwise a sweep buffer meeting that same size bound. Eviction effectiveness E must be ≥1.5 before the physical sweep.

Weights use fixed synthetic seed 20260929 and signed symmetric Q4, group size 32, FP16 scales in separate 64-byte-aligned buffers, no zero-point; activation and accumulation are FP32. A/B/C share identical post-Q4 values. Physical buffer allocation and alignment padding are measured and reported. The Q4 scalar/reference correctness test uses d32,m4,K2; the A/B/C full correctness gate uses d512,m4,K8 before timing.

## Native measurement

The executable is Windows x64 Release MSVC. QPC is the primary timer; frequency is recorded. A sample times each full recurrent round identically across A/B/C and accumulates `T_X(K)=Σ_r t_X,r`; setup/eviction is outside each round timer. The sweep retains every sample and discard reason. Python only orchestrates/analyses files and does not time kernels.

A timed round is valid when its QPC interval is positive, its resulting state is finite, and all fixed-affinity workers remain on their assigned CPU sets; a K-round sample is valid only when all K rounds are present and valid. Invalidate only for explicit MD/304 §43 instrumentation reasons (OS interruption when explicitly detected, timer error, affinity loss/core migration, non-finite output, thermal emergency when measurable, or measurement-process failure). Do not filter latency outliers after observing A/B/C. A timed heap allocation is recorded as a contractual test failure, not used as a post-hoc sample discard rule.

Forward matmul accounting is inherited from V2-0: per round MAC=`16md²+2m²d`; FLOPs=2×MAC. A/B/C compute counts must match exactly. Non-GEMM counts remain separate.

## Predeclared primary gates

- `rho_resident = c_A(8)/c_B(8)` at d512,m4: ≤0.50 minimum, ≤0.25 strong.
- `delta_CB = |c_C(8)-c_B(8)|/c_B(8)` at d512,m4: ≤0.25 minimum, ≤0.10 strong; also require c_C>c_A.
- `G_matrix = max(R_A(8),R_A(16))/R_A(1)` at d512,K8: ≥1.50 minimum, ≥2.00 strong; report m4/m1 too.
- d640 is mandatory frontier evidence but cannot block if d512 passes.
- The nine primary stability cells are d512,m4,A/B/C,K1 and K8 (six cells), plus d512,m1/m8/m16,A,K8 (three matrixization cells). For each, `R_MAD = MAD(T_sample)/median(T_sample)` uses all valid sample totals from all five blocks, where each `T_sample` sums the K timed rounds. A cell is noisy if R_MAD>0.15; two or more noisy cells produce `MEASUREMENT_STABILITY_HOLD`. d640, K4, and other cells are diagnostic and excluded from this denominator.
- Bootstrap 90% intervals are diagnostic only; point estimates from predeclared medians apply the gates.

The 37 contractual test names are recorded individually in `test_report.json`; any contractual SKIP forces `V2_1_ACCEPTANCE_HOLD`.

Instrumental invalidity, unavailable compiler/permissions, failed H0/Q4/correctness, incomplete sweep, or contractual test SKIP yields a concrete HOLD/measurement-invalid result; no scientific PASS/FAIL is declared from invalid measurement.

## Forbidden conclusions

No language or quality statement; no cache-residency statement beyond measured T0; no CPU speedup claim; no K-quality scaling; no external-memory claim; no GPU release; no V2-2 or T3 GO. Only a valid minimum/strong T0 result can be returned for independent review.

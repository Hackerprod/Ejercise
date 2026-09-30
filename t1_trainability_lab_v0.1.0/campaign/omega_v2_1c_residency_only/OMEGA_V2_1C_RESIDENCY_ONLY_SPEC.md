# OMEGA-V2-1c-RESIDENCY-ONLY

Authority: MD/310. This is a new, isolated physical-residency experiment. V2-1b remains closed with `PROJECT_NATIVE_SPEED_GATE_FAIL`; its `attempt_03` was not run. This unit does not alter or replace attempt_02 or candidate_02 KQ evidence.

## Frozen kernel and harness binding

- Candidate_02 kernel implementation is frozen. No edits, tuning, alternate kernels, or candidate rebuilds are permitted.
- The sealed candidate_02 KQ executable SHA-256 is `be5c195e701ccbbf4b606d39382d951ba887b41c1faf96370d2d0ac2a19d084f`.
- The frozen KQ executable is KQ-only; the physical sweep is performed by this companion harness, whose executable has its own hash. The harness links the byte-identical candidate_02 `q4_kernel_candidate2.cpp`, `full_block_candidate2.cpp`, and `kq_candidate2.hpp` translation units/header, with candidate_02 Release compiler flags/toolset recorded and checked against the candidate_02 build manifest. Candidate_02 files and executable remain unchanged.
- Before timing, compare the companion against all sealed KQ output files available (d512,m4,K1; d512,m16,K1; d512,m8,K4) bit-exactly. For d512, m={1,4,16}, K={1,4}, compare against the scalar reference at KQ tolerances (max_abs≤1e-5, max_rel≤1e-4) and run two identical companion executions requiring bit-exact determinism. The sealed KQ run did not retain outputs for m1,K1 or K4 at m={1,4,16}; this limitation is explicit and is not represented as a direct binary-output comparison.

## Scientific protocol

Repeat the 72 cells and attempt_02 controls exactly: d={512,640}; m={1,4,8,16} (m1 control); K={1,4,8}; variants A/B/C; 5 blocks × 21 measured samples per cell/variant, 10 warmups, schedule seed 20260929, same v_i values, four fixed P-core workers, QPC, B-pool, and eviction protocol. Candidate_02 is the sole kernel implementation. Preserve all attempt_02 artifacts byte-for-byte.

Primary gates use d512,m4,K8: rho_resident≤0.50; delta_CB≤0.25 and c_C>c_A; G_matrix≥1.50. Stability uses the nine MD/305 primary cells and holds at ≥2 noisy cells (R_MAD>0.15). m1 remains diagnostic-only. Report attempt_02↔V2-1c comparisons as non-gate diagnostics.

There is no PyTorch S_native gate; S_native is not a gate in this unit. This experiment makes claims only about physical residency on the measured CPU; it does not validate candidate_02 as a competitive backend, trainability, language quality, or T3.

## Contractual V2-1c tests

The runner uses the judge-approved test inventory recorded in `scripts/run_v2_1c.py`. All required tests must be reported individually; no candidate selection or kernel changes occur in this unit.

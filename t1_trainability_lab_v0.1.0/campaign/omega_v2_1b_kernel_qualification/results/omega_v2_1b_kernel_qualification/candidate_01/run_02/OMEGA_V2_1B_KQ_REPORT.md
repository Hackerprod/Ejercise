# OMEGA-V2-1b Kernel Qualification (KQ only)

- candidate: `KQ1_DEQUANT_ROW_REUSE`
- terminal_status: `KERNEL_SCIENTIFICALLY_QUALIFIED_PROJECT_NATIVE_SPEED_GATE_FAIL_CANDIDATE_2_ALLOWED`
- implementation_commit: `e285f4be1abfebd22c490f7be64d2e8580e0fb72`
- implementation_parent: `a70465f6c306efe5bff02302fa54f56cd999b851`
- results_root_abs: `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02`
- frozen_attempt02: `immutable; manifest_sha256=d17d5b79d43fe72eba644069e72b8539ce6990032579313d953190a7313f5287`
- executable_abs: `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\build\Release\omega_v2_1b_kq.exe`
- executable_sha256: `2fdeac276797899b473da0a6dcb17844a95cb915d47e5096d274cee43717f15f`
- attempt03_executed: `false`
- global_status: `CONFORMANCE_HOLD`; GPU/T3: `HOLD`

## Correctness and KQ gates
- vectorized_vs_scalar_d32_m4_k2: `{'d': 32, 'm': 4, 'K': 2, 'max_abs_error': 9.53674e-07, 'max_rel_error': 2.15388e-05, 'abs_tolerance': 1e-05, 'rel_tolerance': 0.0001, 'same_q4_weights': True, 'pass': True}`
- d512_full_finite_and_checksums_stable: `True` / `True`
- P_FMA: `2.04600195e+11` MAC/s; L1 tile: `43008` / `49152` bytes
- P_H0(4/16): `3.4836412e+10` / `7.29444174e+10` MAC/s
- P_FULL(4/16): `3.87663897e+10` / `5.70892365e+10` MAC/s
- E_Q4: `0.356522` (minimum 0.25): `True`
- E_FULL(4/16): `1.112812` / `0.782640` (each minimum 0.60): `True` / `True`
- S_native: `0.528578` (minimum 1.20): `False`
- scratch_per_worker_bytes: `16384` (limit 65,536)

## PyTorch source and correctness-side errors
- source: `V2-0 omega_v2.core.ContractualCoreBlock FP32 source tensors, pre-Q4`
- seed: `20260929`; source FP32 value SHA-256: `3a0117ae300ab98286ae840248f417ddd909ca3077c6f828aa0a83bebdc6dc6d`
- same source tensors in native-Q4 packer and PyTorch control: `True`
- dequantized-FP32 vs original-FP32 timing delta (diagnostic): `0.030310%`; >3%: `False`
- m16_k1: kernel error vs dequant-FP32 `{'max_abs': 1.9073486328125e-06, 'max_rel': 0.010499886237084866}`; quantization error vs original-FP32 `{'max_abs': 0.3294581472873688, 'max_rel': 4237.55029296875}`
- m4_k1: kernel error vs dequant-FP32 `{'max_abs': 2.086162567138672e-06, 'max_rel': 0.0015300051309168339}`; quantization error vs original-FP32 `{'max_abs': 0.4411609172821045, 'max_rel': 98.76214599609375}`
- m8_k4: kernel error vs dequant-FP32 `{'max_abs': 9.775161743164062e-06, 'max_rel': 0.017426272854208946}`; quantization error vs original-FP32 `{'max_abs': 1.6489014625549316, 'max_rel': 1784.092041015625}`

## Non-gate diagnostics
- machine balance: BW_DRAM `65770435634.83119` bytes/s; working set `134217728` bytes; M_machine `0.5894196881041521` MAC/byte; AI_m4 `7.118055555555555` MAC/byte; rho_optimistic `0.923526230124577`
- attempt_02 m1 d512 diagnostic rho_A/B `1.0014170017492448`; delta_CB `0.008252118403416739`; c_C>c_A `True`
- attempt_02 m1 d640 diagnostic rho_A/B `0.9975413338354469`; delta_CB `0.009477995118877832`; c_C>c_A `True`
- m1 diagnostics are non-gate and were not used to select/score the kernel candidate.

## Contractual KQ tests (17)
- PASS: `test_v2_1b_vectorized_q4_matches_scalar`
- PASS: `test_v2_1b_dequant_tile_reused_across_slots`
- PASS: `test_v2_1b_no_full_predequantized_weight_copy`
- PASS: `test_v2_1b_scratch_per_worker_le_64k`
- PASS: `test_v2_1b_fma_peak_fixture`
- PASS: `test_v2_1b_h0_efficiency_formula`
- PASS: `test_v2_1b_full_efficiency_formula`
- PASS: `test_v2_1b_kernel_quality_thresholds`
- PASS: `test_v2_1b_pytorch_reference_same_equations`
- PASS: `test_v2_1b_pytorch_speedup_formula`
- PASS: `test_v2_1b_candidate_limit_two`
- PASS: `test_v2_1b_no_abc_before_kernel_freeze`
- PASS: `test_v2_1b_attempt02_preserved`
- PASS: `test_v2_1b_attempt03_same_72_cells`
- PASS: `test_v2_1b_m1_residency_diagnostic`
- PASS: `test_v2_1b_machine_balance_diagnostic`
- PASS: `test_v2_1b_conformance_block`

## Source/build/artifact SHA-256
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_0_conformance\V2_0_RESULT_SEAL.json`: `95facae0865750783faca8f9c271dcd557b7d1338f9fce512b0974cfa4606623`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\artifact_hashes.json`: `d17d5b79d43fe72eba644069e72b8539ce6990032579313d953190a7313f5287`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\allocation_guard.cpp`: `fb0d47882cf84c490e7669b2c7afe771f398334156657bca8cfa321e7cb22439`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\cache_controls.cpp`: `54d4fe802ee2281dc35c064a7c7ea5b283bb110f80d4f576a8eebeb4b59efe54`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\hardware_topology.cpp`: `9bbf96f789955117d3dd27f3a86835ae45da79b6a8a5de07f582f42a3dc3917d`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\q4_layout.cpp`: `b3e829e3b2a392c4567a65020ee9037b17787e7f43ee877832c80c4abd2226b7`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\v2_1.hpp`: `96c77cfaf3e6b8e1a1661401411d8641e0e96a258e637c02615b232723bbb594`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\v2_full_block.cpp`: `c63dfa059b3e328938399146980132798733e1b4d521893fd8a54ffc70b71bdf`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\worker_pool.cpp`: `3549c67d69765beafbaabd821e49b94dbab47681d33485e2f3f10ecf2f2e6270`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\.gitignore`: `020c2606e8ed484089c2083e0be438b9d0736bee1558a6b76af0d58bd23f69cf`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\CMakeLists.txt`: `afb0281c3fdae5fb827412aebc180fd8dd3f7371e200808e9d27a0938532f5b8`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\OMEGA_V2_1B_KQ_SPEC.md`: `cb8f824936466b25c225795a888014dff847f780c485abda387983269c19e79d`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\build\Release\omega_v2_1b_kq.exe`: `2fdeac276797899b473da0a6dcb17844a95cb915d47e5096d274cee43717f15f`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\OMEGA_CONFORMANCE_BLOCK_V2_1B.yaml`: `0d2c78539676d55d5c28416d6391930e26995a3b33e992f6984da7d9ef17b0b2`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\attempt02_preservation.json`: `cafc6f751bb383954971aada4e53e26dc6fe60b38f58ade303bcd807281647ae`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\build_manifest.json`: `acc45007a39ca5895e63dfa81b3bbd1953d563db71850260266a37d852b4e8fa`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\native_correctness_outputs\candidate1_full_m16_k1.bin`: `30c06a5a84f0ef26fb63320df199859af6ad7976e859bcff548a3a4e13a78936`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\native_correctness_outputs\candidate1_full_m4_k1.bin`: `da784bfd81c11cd51530fc3b13ae08234b72841b3d15ffece5af6aa0d8f150a8`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\native_correctness_outputs\candidate1_full_m8_k4.bin`: `6079f41c1d16db9a1d9aa5fd17833cf479cef5bd8eed8e0a5dc1defb1dbb49d8`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\native_kq_candidate_01.json`: `7e1457890959d5edd7c3edbc17838c6140fce6a0677000b05298401d6738ef5b`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\native_kq_stderr.log`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\native_kq_stdout.log`: `7e1457890959d5edd7c3edbc17838c6140fce6a0677000b05298401d6738ef5b`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\pytorch_control.json`: `708ed80301f308c2de176d62a1eb28923f059aa573a8a27732fdfb83000f2ccc`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\pytorch_control_stderr.log`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\pytorch_control_stdout.log`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\source_weight_manifest.json`: `783f90b6b440f2d8e0aaa74cda4f689365d7fe31501f93ce670fb50d7492dedb`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\summary_metrics.json`: `300f671c1b81819e21da45bd68a318775064a7102d5b6719665653b0753cfcf4`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\test_report.json`: `a6edf0d88bdaf2780254d7868b3a7cb99317fb72cc03deeb60369dd88e917af9`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\scripts\finalize_kq_run.py`: `0e9fd777c404964f4387f1cec4fcfbed42ad1d851a65e3f17b43e93d71f0a3c8`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\scripts\pytorch_control.py`: `6230ad5aef0bb3d0f42987438a32713f2abfca780d306ddde91676b1402e8bf5`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\scripts\run_kq.py`: `34a56b20f436e1ad08b77e67115d63fecdd9c05770d67ef27d5c8e095cc67792`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\src\kq_candidate.hpp`: `107e2c0019def6d2ad35b5c0027e132db70c38a7cead9d2b9be19913ac465f68`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\src\main.cpp`: `190490d4b79931215d610387285af5fed01f13935f08b9782c6dd828859f412f`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\src\q4_kernel_candidate1.cpp`: `76f3269f6d89507572e65eec196f3375666b3be7996c35d121e3a3d6464da93a`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\tests\test_kq_contract.py`: `b57dc4f1a81934e24d05f415e29abffd4a9c77687403324d891be8335a2328f3`

All SHA-256 values are complete 64-hex digests. `artifact_hashes.json` also records byte sizes; attempt_01/attempt_02 artifacts were read-only.

STOP: this report covers kernel qualification only. No A/B/C sweep, attempt_03, GPU, training, language scoring, LN, or T3 ran.

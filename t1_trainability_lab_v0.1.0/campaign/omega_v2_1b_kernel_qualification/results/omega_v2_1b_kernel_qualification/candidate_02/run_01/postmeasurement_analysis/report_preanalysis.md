# OMEGA-V2-1b Candidate 02 Kernel Qualification (KQ only)

- candidate: `KQ2_ROW_TILE4_SLOT2_FUSED`
- terminal_status: `PROJECT_NATIVE_SPEED_GATE_FAIL`
- implementation_commit: `2e8ac1207e742e659aab34ef0ab4bf51d43dbe55`
- results_root_abs: `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01`
- attempt_02 immutable manifest SHA-256: `d17d5b79d43fe72eba644069e72b8539ce6990032579313d953190a7313f5287`
- candidate_01 immutable artifact manifest SHA-256: `d37737c68909e6d6cd5b1b08ff9c0e42fc00f5d6f1a8efeb504c5773e72d0b12`
- executable_abs: `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_candidate_02\build\Release\omega_v2_1b_candidate_02.exe`
- executable_sha256: `be5c195e701ccbbf4b606d39382d951ba887b41c1faf96370d2d0ac2a19d084f`
- attempt_03 executed: `false`; A/B/C executed: `false`; GPU/T3: `HOLD`

## Gate results
- correctness: `True`; toy d32,m4,K2 max_abs `8.34465027e-07`, max_rel `2.61931326e-05`
- exp approximation: dense 16,001-point grid on `[-80,80]`, observed max relative error `6.99936761e-08`, bound `2e-6`
- P_FMA reused from candidate_01: `2.04600195e+11` MAC/s
- P_H0(4/16): `4.35093776e+10` / `8.90510403e+10` MAC/s
- P_FULL(4/16): `5.84735376e+10` / `9.16487662e+10` MAC/s
- E_Q4 `0.435244` (min 0.25): `True`
- E_FULL(4/16) `1.343930` / `1.029171` (each min 0.60): `True` / `True`
- S_native `0.778487` (min 1.20): `False`
- scratch per worker `49152` bytes (limit 65,536)

## Candidate 01 → Candidate 02
- Candidate 01: E_Q4 `0.356522`, E_FULL(4) `1.112812`, E_FULL(16) `0.782640`, S_native `0.528578`; candidate_02 authorized: `True`.
- Candidate 02: E_Q4 `0.435244`, E_FULL(4) `1.343930`, E_FULL(16) `1.029171`, S_native `0.778487`.
- Candidate 02 kernel: output-row tile 4 × slot tile 2; fused QKV and gate/up; vectorized AVX2 attention and SwiGLU; Q4+FP16 scales remain the only persistent weight form.

## PyTorch control and quantization error
- source FP32 value SHA-256: `3a0117ae300ab98286ae840248f417ddd909ca3077c6f828aa0a83bebdc6dc6d`; same source tensors: `True`.
- PyTorch: `2.11.0+cpu`, threads `4`, interop `1`, four-core affinity `[10, 8, 2, 14]`.
- S_native cell: PyTorch original median `0.0013657` s, native Q4 median `0.0017543` s.
- m16_k1: kernel error vs dequant-FP32 `{'max_abs': 1.817941665649414e-06, 'max_rel': 0.009815110825002193}`; Q4 quantization error vs original FP32 `{'max_abs': 0.3294581472873688, 'max_rel': 4237.55029296875}`
- m4_k1: kernel error vs dequant-FP32 `{'max_abs': 3.814697265625e-06, 'max_rel': 0.0023800078779459}`; Q4 quantization error vs original FP32 `{'max_abs': 0.4411609172821045, 'max_rel': 98.76214599609375}`
- m8_k4: kernel error vs dequant-FP32 `{'max_abs': 8.58306884765625e-06, 'max_rel': 0.013404825702309608}`; Q4 quantization error vs original FP32 `{'max_abs': 1.6489014625549316, 'max_rel': 1784.092041015625}`

## Non-gate diagnostics
- machine balance reuses candidate_01's independent same-host/four-P-core DRAM stream measurement: BW_DRAM `6.57704356e+10` bytes/s; working set `134217728` bytes; M_machine `0.889055045`; AI_m4 `7.11805556`; rho_optimistic `0.888966808`.
- attempt_02 m1 diagnostics (non-gate, evaluated after candidate selection): d512 rho_A/B `1.001417`, delta_CB `0.0082521184`; d640 rho_A/B `0.997541334`, delta_CB `0.00947799512.
- No A/B/C run, attempt_03 run, residency-gate evaluation, or GPU work occurred.

## Contractual KQ tests (17)
- PASS: `test_v2_1b_vectorized_q4_matches_scalar`
- FAIL: `test_v2_1b_dequant_tile_reused_across_slots`
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

## Source, build, and artifact SHA-256
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_0_conformance\V2_0_RESULT_SEAL.json`: `95facae0865750783faca8f9c271dcd557b7d1338f9fce512b0974cfa4606623`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\artifact_hashes.json`: `d17d5b79d43fe72eba644069e72b8539ce6990032579313d953190a7313f5287`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\allocation_guard.cpp`: `fb0d47882cf84c490e7669b2c7afe771f398334156657bca8cfa321e7cb22439`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\hardware_topology.cpp`: `9bbf96f789955117d3dd27f3a86835ae45da79b6a8a5de07f582f42a3dc3917d`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\q4_layout.cpp`: `b3e829e3b2a392c4567a65020ee9037b17787e7f43ee877832c80c4abd2226b7`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\v2_1.hpp`: `96c77cfaf3e6b8e1a1661401411d8641e0e96a258e637c02615b232723bbb594`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\worker_pool.cpp`: `3549c67d69765beafbaabd821e49b94dbab47681d33485e2f3f10ecf2f2e6270`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_candidate_02\.gitignore`: `0cf256cc9234b415f912f5b0e9f8c9364688466041611047cd3f599c35f4efa6`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_candidate_02\CMakeLists.txt`: `b73e3214a2aa4dca4997a6a19acad482f9c36b89c5111dc9762eb02c41c68ce3`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_candidate_02\OMEGA_V2_1B_CANDIDATE_02_SPEC.md`: `0a36d87897ae0405d2dce342ebbd1641edeefc7dcabf7358286fb13283e5f980`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_candidate_02\build\Release\omega_v2_1b_candidate_02.exe`: `be5c195e701ccbbf4b606d39382d951ba887b41c1faf96370d2d0ac2a19d084f`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_candidate_02\build\Release\omega_v2_1b_candidate_02_correctness.exe`: `88195b6f835513fb3f2a406f75cefe51985946d852989945c1f18e65f88eece1`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_candidate_02\scripts\run_candidate02_kq.py`: `5fc228e463e45b20a317e7cab9bd362f75bac6137e94c94f624635f2d9c52ec2`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_candidate_02\src\correctness_candidate2.cpp`: `581f65bd623889cebe12988512bcbe25d38b61a75ce4f15fdfb0f855abed01e7`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_candidate_02\src\full_block_candidate2.cpp`: `68cfdbca0856ecfcc015f76fdc2613a2581636a6e76234c3897425f47772fc0b`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_candidate_02\src\kq_candidate2.hpp`: `77c6e0305a5eb87408b1838392376955fcbd563d6e8390ee1d6006d808f48bfd`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_candidate_02\src\main_candidate2.cpp`: `e899065526fc118d37acdcd196ac6ccde92b033d77a307229a586484291f24da`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_candidate_02\src\platform_candidate2.cpp`: `0de3e4ad1c9fd2a2be8d7ec569d684941c61b9b5d87abacc2328b5507f5a0990`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_candidate_02\src\q4_kernel_candidate2.cpp`: `c15e1618a25888a9b77dacee85ecc373215cdc03c764ca9445de2ade5a90594a`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\artifact_hashes.json`: `d37737c68909e6d6cd5b1b08ff9c0e42fc00f5d6f1a8efeb504c5773e72d0b12`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\native_kq_candidate_01.json`: `7e1457890959d5edd7c3edbc17838c6140fce6a0677000b05298401d6738ef5b`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02\summary_metrics.json`: `300f671c1b81819e21da45bd68a318775064a7102d5b6719665653b0753cfcf4`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\OMEGA_CONFORMANCE_BLOCK_V2_1B.yaml`: `e31853342a3af1818fad3cc57763f92413299540e4adbae0fffbf4423613dd26`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\attempt02_preservation.json`: `cafc6f751bb383954971aada4e53e26dc6fe60b38f58ade303bcd807281647ae`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\build_manifest.json`: `00fe34988e76f466729bdafb92f6021fb93c80535729ab20a6ce15da9ccb01a4`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\native_correctness_outputs\candidate2_full_m16_k1.bin`: `f1784bbc741db613a072f54629072c191c513a935ed9f91bcc6bd220e4702eea`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\native_correctness_outputs\candidate2_full_m4_k1.bin`: `3789d3093759ea620a297510a09a0ed75b8d762fca47cae1c5735c3dd9696ebc`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\native_correctness_outputs\candidate2_full_m8_k4.bin`: `c68ddd5d0a6e7c68d49fb7895b8a079111b6aeb29af675425966c9d733b3e560`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\native_kq_candidate_02.json`: `230dce1ccae58696f38610077459e2c6781b749330298b4617df9b3561da7b6c`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\native_kq_stderr.log`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\native_kq_stdout.log`: `230dce1ccae58696f38610077459e2c6781b749330298b4617df9b3561da7b6c`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\pytorch_control.json`: `17f36c13b4bef5b8cc0c26cdb04f5f3d80512306c2834bb08dd0abc70a4b7faf`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\pytorch_control_stderr.log`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\pytorch_control_stdout.log`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\source_weight_manifest.json`: `c3e285ef422fd284f071beb8c5377f9e4111031b737989e37737185485843ebc`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\summary_metrics.json`: `da90fbf5d97b3c169223fcb13b35f6a823b71f65d37745c94bc196f34c1955cd`
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_02\run_01\test_report.json`: `91787a04058327d278b48d1923d088a90f2c771d5b3627cb9fbfc706639beb0d`

STOP: candidate_02 is the final permitted kernel candidate. attempt_03 is permitted only if all five frozen gates pass.

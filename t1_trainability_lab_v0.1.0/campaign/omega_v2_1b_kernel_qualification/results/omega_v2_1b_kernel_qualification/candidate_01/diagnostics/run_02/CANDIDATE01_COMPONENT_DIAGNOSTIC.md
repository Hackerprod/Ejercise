# Candidate-01 Component Diagnostic (diagnostic only)

- terminal_status: `CANDIDATE01_COMPONENT_DIAGNOSTIC_COMPLETE`
- candidate_01_KQ_result_abs: `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\run_02`
- diagnostic_result_abs: `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\diagnostics\run_02`
- candidate_01 executable: `not executed; diagnostic build used a separate executable`
- attempt_02/candidate_01 artifacts: `hash-verified and unchanged`
- A/B/C: `not run`; attempt_03: `not run`; rho: `not read or calculated`

## Requested component timing breakdown

- dequant-only full K4, including checksum and 28 dispatches: `0.002576000 s`
- 28 empty dispatches: `0.000096500 s`
- estimated dequant+checksum, subtracting empty dispatches: `0.002479500 s`
- row-timed dequant wall estimate (dispatch/checksum excluded): `0.000551900 s`
- row-timed checksum wall estimate (reported separately): `0.001723100 s`
- dequant+FMA Q4 matmuls full K4, including 28 dispatches: `0.002012100 s`
- estimated dequant+FMA, subtracting empty dispatches: `0.001915600 s`
- full-block QKVO: `0.000739900 s`
- full-block SwiGLU: `0.001734000 s`
- RMSNorm/residual: `0.000046500 s`
- QKV matmuls / attention / W_O: `0.000424300` / `0.000170800` / `0.000145600 s`
- gate+up / SiLU+Hadamard / down: `0.000965900` / `0.000262200` / `0.000503500 s`

## Method note

Dequant-only consumes each transient row tile into a checksum to keep decode writes observable. It reports the full wall interval, empty-dispatch-subtracted interval, and row-QPC dequant/checksum estimates separately. The q4_matmul batch includes dequant plus FMA for seven matrices over four rounds; its empty-dispatch-subtracted value is an estimate. Component timers are diagnostic, not gates.

## SHA-256 (complete artifact table)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\allocation_guard.cpp`: `fb0d47882cf84c490e7669b2c7afe771f398334156657bca8cfa321e7cb22439` (2225 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\cache_controls.cpp`: `54d4fe802ee2281dc35c064a7c7ea5b283bb110f80d4f576a8eebeb4b59efe54` (6878 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\hardware_topology.cpp`: `9bbf96f789955117d3dd27f3a86835ae45da79b6a8a5de07f582f42a3dc3917d` (27187 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\q4_layout.cpp`: `b3e829e3b2a392c4567a65020ee9037b17787e7f43ee877832c80c4abd2226b7` (13580 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\v2_1.hpp`: `96c77cfaf3e6b8e1a1661401411d8641e0e96a258e637c02615b232723bbb594` (8853 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\worker_pool.cpp`: `3549c67d69765beafbaabd821e49b94dbab47681d33485e2f3f10ecf2f2e6270` (6886 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\CMakeLists.txt`: `afb0281c3fdae5fb827412aebc180fd8dd3f7371e200808e9d27a0938532f5b8` (1508 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\OMEGA_V2_1B_KQ_SPEC.md`: `cb8f824936466b25c225795a888014dff847f780c485abda387983269c19e79d` (3644 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\diagnostics\candidate01\.gitignore`: `904dfd7e0e58758dd7441b227cbbe0f13ccef5caf57b5effb26231bbc36efdd1` (23 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\diagnostics\candidate01\CMakeLists.txt`: `7ca77ee139db9b4ffb669fc7f5993859a456d8c83197dd0c1657307ade173714` (1585 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\diagnostics\candidate01\build\Release\omega_v2_1b_candidate01_diag.exe`: `5ca06709f187627c3b81919dc957b6f6a4da1eb78871c31e1a9834b1c3c9c083` (159744 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\diagnostics\candidate01\component_diagnostic.cpp`: `ca51ec030107e49d6fd2d779176aa97f38afc06dc0e4e189014d9160497f7749` (27904 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\diagnostics\candidate01\finalize_candidate01_diagnostic.py`: `07998e39f0ac02d70bdd5abea2811d1f625835561a269ad7d2784650d21c6acf` (3120 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\diagnostics\candidate01\run_candidate01_diagnostic.py`: `ebf1f596c1b59373503cf8adb61c8223ed088a6611f68e161bc0ebb439e5fda5` (18053 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\diagnostics\run_02\build_manifest.json`: `a07b8159a99735f6c58988196994e72763e285a408757a3aa74ee65ce27e05a0` (5961 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\diagnostics\run_02\candidate01_component_diagnostic.json`: `61c5ff334d352b57c45cd62d640f8f916b1447dbe6064088ad2408ecdef9d349` (13210 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\diagnostics\run_02\native_stderr.log`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (0 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\diagnostics\run_02\native_stdout.log`: `61c5ff334d352b57c45cd62d640f8f916b1447dbe6064088ad2408ecdef9d349` (13210 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\results\omega_v2_1b_kernel_qualification\candidate_01\diagnostics\run_02\summary_metrics.json`: `f5f9310702ca94e31ab9de19d7ac02fc0fd3bc5bc22245fe30e0da2a9cf0d629` (17839 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\scripts\pytorch_control.py`: `6230ad5aef0bb3d0f42987438a32713f2abfca780d306ddde91676b1402e8bf5` (9957 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\scripts\run_kq.py`: `34a56b20f436e1ad08b77e67115d63fecdd9c05770d67ef27d5c8e095cc67792` (45589 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\src\kq_candidate.hpp`: `107e2c0019def6d2ad35b5c0027e132db70c38a7cead9d2b9be19913ac465f68` (762 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\src\main.cpp`: `190490d4b79931215d610387285af5fed01f13935f08b9782c6dd828859f412f` (34311 bytes)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1b_kernel_qualification\src\q4_kernel_candidate1.cpp`: `76f3269f6d89507572e65eec196f3375666b3be7996c35d121e3a3d6464da93a` (8774 bytes)

`artifact_hashes.json` excludes its own self-hash to avoid a circular digest; all measured inputs, outputs, source files, binaries, and this analysis tool are listed.

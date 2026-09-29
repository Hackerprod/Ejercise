# OMEGA-V2-1 Physical T0 Report

- terminal_status: `RESIDENCY_GATE_FAIL`
- terminal_reason: `rho_resident=0.9985306340340571 > 0.50`
- repo_root_abs: `D:\Install\Dev\Projects\IA\Ejercise`
- unit_root_abs: `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical`
- implementation_commit: `18f98a1d83f8dfe021e7bbe9943d44e95dc79859`
- implementation_parent: `974200500e12ba7ae5b36fc500f54636fa245b2c`
- branch: `main`
- git_status_porcelain: ``
- global_status: `CONFORMANCE_HOLD`
- T3: `HOLD`

## Native hardware preflight
- cpu_model: `13th Gen Intel(R) Core(TM) i7-13700F`
- physical/logical processors: `16` / `24`
- P/E cores measured/classified: `8` / `8`
- LLC bytes: `31457280`
- cache line bytes: `64`
- selected CPU sets: `[266, 264, 258, 270]`
- H0 P-core classification: `Intel CPUID leaf 0x1A core type 0x40 identifies P-cores; H0 v_i ranks only those candidates; Windows EfficiencyClass is recorded`
- eviction method/E: `CLFLUSH` / `6`

## Primary gates (point estimate; bootstrap90 diagnostic only)
- c_A_K8: `0.010795385714285715`
- c_B_K8: `0.010811271428571429`
- c_C_K8: `0.010860057142857144`
- rho_resident: `0.9985306340340571`
- rho_CB: `1.0045124863072799`
- delta_CB: `0.004512486307279879`
- c_C_gt_c_A: `True`
- R_A_m1: `836566358.0054586`
- R_A_m4: `1561296695.506656`
- R_A_m8: `1988888244.7245903`
- R_A_m16: `2296183780.568959`
- G_matrix: `2.7447718385945143`
- noisy_primary_stability_cells: `0`
- measurement_stability_hold: `False`
- bootstrap90_diagnostics: `{"G_matrix": {"lower90": 2.7328474026947283, "replicates": 5000, "upper90": 2.767904502446708}, "delta_CB": {"lower90": 0.0004710545262171826, "replicates": 5000, "upper90": 0.009490451153984785}, "rho_resident": {"lower90": 0.9929598593019002, "replicates": 5000, "upper90": 1.004203514808015}}`

## Tests (all required names)
- PASS: `test_v2_1_reference_commit_and_v2_0_seal`
- PASS: `test_v2_1_native_windows_required`
- PASS: `test_v2_1_hardware_topology_complete`
- PASS: `test_v2_1_four_primary_workers_are_distinct_pcores`
- PASS: `test_v2_1_no_smt_sibling_in_primary_workers`
- PASS: `test_v2_1_q4_group_size_32`
- PASS: `test_v2_1_q4_pack_unpack_signed_nibbles`
- PASS: `test_v2_1_q4_fp16_scale_layout`
- PASS: `test_v2_1_q4_no_zeropoint`
- PASS: `test_v2_1_q4_physical_byte_ledger`
- PASS: `test_v2_1_full_block_scalar_reference`
- PASS: `test_v2_1_v2_0_equation_structure`
- PASS: `test_v2_1_abc_numerical_identity`
- PASS: `test_v2_1_a_reuses_same_storage`
- PASS: `test_v2_1_b_round_storages_are_disjoint`
- PASS: `test_v2_1_b_values_equal_a`
- PASS: `test_v2_1_b_pool_exceeds_2p5_llc`
- PASS: `test_v2_1_c_uses_a_storage`
- PASS: `test_v2_1_eviction_probe_effectiveness`
- PASS: `test_v2_1_eviction_outside_timed_region`
- PASS: `test_v2_1_worker_shards_cover_all_rows`
- PASS: `test_v2_1_worker_shards_do_not_overlap`
- PASS: `test_v2_1_no_full_weight_replication_per_worker`
- PASS: `test_v2_1_no_heap_allocation_in_timed_kernel`
- PASS: `test_v2_1_persistent_thread_pool`
- PASS: `test_v2_1_fixed_affinity_preserved`
- PASS: `test_v2_1_qpc_monotonic`
- PASS: `test_v2_1_qpc_overhead_recorded`
- PASS: `test_v2_1_flop_mac_ledger_matches_v2_0`
- PASS: `test_v2_1_same_compute_abc`
- PASS: `test_v2_1_sweep_contains_m1_control`
- PASS: `test_v2_1_sweep_completeness`
- PASS: `test_v2_1_rho_resident_formula_fixture`
- PASS: `test_v2_1_c_vs_b_formula_fixture`
- PASS: `test_v2_1_matrixization_formula_fixture`
- PASS: `test_v2_1_conformance_block`
- PASS: `test_v2_1_report_contract_paths_and_hashes`

## SHA-256 (complete)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_0_conformance\V2_0_RESULT_SEAL.json`: `95facae0865750783faca8f9c271dcd557b7d1338f9fce512b0974cfa4606623` (1312 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_0_conformance\V2_0_RESULT_SEAL.json`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\.gitignore`: `020c2606e8ed484089c2083e0be438b9d0736bee1558a6b76af0d58bd23f69cf` (8 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\.gitignore`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\CMakeLists.txt`: `97f1e3db561224856c909552944da4ae30fabbec8936c73661acc4075bfd594a` (1200 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\CMakeLists.txt`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\OMEGA_V2_1_PHYSICAL_SPEC.md`: `a972a86ddd9b2fc616d164fa17587ec7b167dfe8c3f03ee2346a086bc1d197c5` (5605 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\OMEGA_V2_1_PHYSICAL_SPEC.md`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\build\Release\omega_v2_1_bench.exe`: `b113ae09c6664234055d6d78397987b4c6f15ff4213bea259040b761221fb37f` (265216 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\build\Release\omega_v2_1_bench.exe`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\OMEGA_CONFORMANCE_BLOCK.yaml`: `bc641e746961424ae0509d88c3665fd4c77372a1562796cd50cbecf15ad4edeb` (30935 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\OMEGA_CONFORMANCE_BLOCK.yaml`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\benchmark_config.json`: `1ef5d47c0dabcea11438d5f9047717693c56c1e74b81fa38eda443da1177396c` (4215 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\benchmark_config.json`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\build_manifest.json`: `3470b0edb81fce10eec4d34e230be2da6812947c1eb04b167a66b9c7428a8b35` (3622 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\build_manifest.json`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\hardware_preflight.json`: `cabfecc90cc2b7d9aeaa119e1c0b6920d0982fe3487c547407faddd61d89dbcb` (40591 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\hardware_preflight.json`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\native_run_status.json`: `44bf82014b04f0a989a22740005732a5c0848095986d0a66876aeff0029d1dc9` (68 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\native_run_status.json`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\native_stderr.log`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (0 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\native_stderr.log`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\native_stdout.log`: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` (0 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\native_stdout.log`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\native_test_report.json`: `2514863d246f3b93f7ba074e22bdb26206dc316dc5dad9127fb690bfb0b48653` (725 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\native_test_report.json`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\q4_physical_ledger.json`: `5b8bc8319cbbf1165305110a57b41db84ee631d027501e0d8240ecf15a222416` (5014 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\q4_physical_ledger.json`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\raw_measurements.csv`: `993830bc8ad997ad3e8259397b967618d3f6e0f7ffaeebad3f1395268eea219b` (8417691 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\raw_measurements.csv`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\summary_metrics.json`: `9fa32d347b0cf8ed3c6d427995e90cf28a593d688d3c960365df085a5a2d475f` (223151 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\summary_metrics.json`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\test_report.json`: `d3b2ae668dc742bd552abe2168a208926a382de093275cafdac95b3e7dacec85` (4228 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\test_report.json`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\worker_shard_manifest.json`: `174aab12738a0e2110e65cdc4ce891f9ed236f0c1f945094735960a5de6382d8` (14585 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\results\omega_v2_1_physical\attempt_02\worker_shard_manifest.json`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\scripts\analyze_v2_1.py`: `4a6eb9841a07ee0de522f9679ade731436d7866f449131368882b70a56726651` (62109 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\scripts\analyze_v2_1.py`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\allocation_guard.cpp`: `fb0d47882cf84c490e7669b2c7afe771f398334156657bca8cfa321e7cb22439` (2225 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\allocation_guard.cpp`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\benchmark.cpp`: `b6d37b370c8504f6d5b2a1499e67bb11ed1aad53515b59951b8b77c4b76c1e44` (18178 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\benchmark.cpp`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\cache_controls.cpp`: `54d4fe802ee2281dc35c064a7c7ea5b283bb110f80d4f576a8eebeb4b59efe54` (6878 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\cache_controls.cpp`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\hardware_topology.cpp`: `9bbf96f789955117d3dd27f3a86835ae45da79b6a8a5de07f582f42a3dc3917d` (27187 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\hardware_topology.cpp`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\main.cpp`: `286207f2b65516e6f6f672fabad44a096da4948c96d142234c1fe7343920f568` (35931 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\main.cpp`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\q4_kernel.cpp`: `9feb55c4ccc58204d9b830396f713807686138ba9768d6a08e32c1e4a272f74e` (4468 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\q4_kernel.cpp`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\q4_layout.cpp`: `b3e829e3b2a392c4567a65020ee9037b17787e7f43ee877832c80c4abd2226b7` (13580 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\q4_layout.cpp`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\v2_1.hpp`: `96c77cfaf3e6b8e1a1661401411d8641e0e96a258e637c02615b232723bbb594` (8853 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\v2_1.hpp`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\v2_full_block.cpp`: `c63dfa059b3e328938399146980132798733e1b4d521893fd8a54ffc70b71bdf` (9879 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\v2_full_block.cpp`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\worker_pool.cpp`: `3549c67d69765beafbaabd821e49b94dbab47681d33485e2f3f10ecf2f2e6270` (6886 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\src\worker_pool.cpp`)
- `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\tests\test_analyze_v2_1.py`: `690bbc22ccda0ed75abf157df21458774d5d3ba8949423552beecc40c9dcb9ea` (9049 bytes; `D:\Install\Dev\Projects\IA\Ejercise\t1_trainability_lab_v0.1.0\campaign\omega_v2_1_physical\tests\test_analyze_v2_1.py`)

STOP: no language training, GPU, LN, T3, or OMEGA-trained scoring/generation was run.

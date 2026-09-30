# V2-1c Offline Seal Recovery Note

## Binding preflight recovery

The companion had already completed the binding run with return code 0. The saved `build/preflight/binding_companion.json` (SHA-256 `e7637df1a32ef5a33af3309abdaa816ef3222e9b1da91d0b0871877c651e522a`) contained `pass=true`, the exact-output anchor results, CPU-set IDs, timing samples, and `timing_protocol.no_timed_allocations=true`. The parser expected the last key at the top level and raised `KeyError`.

In a Python recovery process, `execute_equivalence` was monkeypatched in memory to return the already-saved JSON, and `compare_binding_timing` was monkeypatched to read the key at its actual location and calculate ratios offline against `native_kq_candidate_02.json` (SHA-256 `230dce1ccae58696f38610077459e2c6781b749330298b4617df9b3561da7b6c`). `binding_stage()` then validated the inputs and sealed the saved evidence. The resulting timing-sanity JSON has SHA-256 `7e0f3ef5fb1b6cab2782f550132a1441b2e0aaaae1066f4e55066817b5d5608a`; the report has SHA-256 `478f985238f42babb9bfd33d64f6578a4441e9989b01957ceb7c173ad9759191`; the artifact manifest has SHA-256 `1ffc581ba5e7c3c49d3e3fe4aad22bbdb2257beb2b5afcbe92d8b92e5221a74d`.

The companion executable was not invoked during recovery.

## H0 core-selection recovery

The companion had already written `build/preflight/core_selection_preflight.json` (SHA-256 `e0ca7b7bffae9242ade1b5b5981e82d5b26c68c0be5f18e9b341bae04e5a08a4`). The parser failed because the inherited attempt_02 `worker_shard_manifest.json` (SHA-256 `174aab12738a0e2110e65cdc4ce891f9ed236f0c1f945094735960a5de6382d8`) uses `cpu_set_id`, not `windows_cpu_set_id`, in each matrix shard.

In memory, a `windows_cpu_set_id` alias was added to each shard object after reading it. `subprocess.run` was monkeypatched to return a `CompletedProcess` containing the saved H0 JSON, without executing the EXE. `core_selection_stage()` validated and sealed the saved H0 report. The resulting report has SHA-256 `09faba6e82a632bd6e2ac4a6e78c2d75de5cbc520709868828f464df1673ca02`; the artifact manifest has SHA-256 `2dbf73d67eeb0e897fb301a21251275da28495a8533321279d50353bc653a7b7`.

The H0 measurement was not repeated. During recovery, `subprocess.run` returned a `CompletedProcess` containing the saved H0 JSON; the companion EXE was not invoked. No rebuild, relink, or reselection occurred.

## Temporary files and commit

Neither `v2_0_fp32_source_weights_correctness.tmp` nor `v2_0_fp32_source_weights.tmp` is included in local commit `832c0ff`. Both are temporary files under `build/preflight`, created after that commit. The commit contains only the ten intended V2-1c source, specification, and test files.

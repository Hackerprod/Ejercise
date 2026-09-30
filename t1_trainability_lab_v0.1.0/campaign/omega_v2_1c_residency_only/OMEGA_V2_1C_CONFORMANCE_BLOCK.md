# OMEGA V2-1c Conformance Block

```yaml
status: CONFORMANCE_HOLD
terminal_classification: V2_1C_INVALID_PREFLIGHT
failure_stage: CORRECTNESS_PREFLIGHT
scientific_timing_executed: false
residency_verdict: NOT_ESTABLISHED
protocol_deviation:
  occurred: false
correctness_failure:
  frozen_abs_threshold: 1e-5
  frozen_rel_threshold: 1e-4
  relative_floor: 1e-12
  result: FAIL
```

```yaml
candidate_02_identity: SOURCE_AND_BEHAVIOR_BOUND_COMPANION
original_KQ_executable:
  immutable: true
  not_used_for_V2_1c: true
scientific_kernel: no_compute_source_changes
```

## Correctness values

| Comparison | Cell | max_abs_error | max_rel_error |
|---|---|---:|---:|
| Full-size K1 | d512 m1 K1 | 4.5299530029296875e-06 | 0.00020438118949852289 |
| Full-size K1 | d512 m4 K1 | 2.7418136596679688e-06 | 0.00086370703057522889 |
| Full-size K1 | d512 m8 K1 | 2.6822090148925781e-06 | 0.0025169574676848256 |
| Full-size K1 | d512 m16 K1 | 2.9653310775756836e-06 | 0.042236024844720499 |
| Full-size K1 | d640 m1 K1 | 3.6954879760742188e-06 | 0.00038979650329607336 |
| Full-size K1 | d640 m4 K1 | 2.5033950805664062e-06 | 0.0010542635658914729 |
| Full-size K1 | d640 m8 K1 | 3.2186508178710938e-06 | 0.20722433460076045 |
| Full-size K1 | d640 m16 K1 | 3.3974647521972656e-06 | 0.0047066815468969868 |
| Toy recurrence | d32 m4 K1 | 3.5762786865234375e-07 | 0.00010574876736593039 |
| Toy recurrence | d32 m4 K4 | 1.9073486328125e-06 | 0.00017395237184059003 |
| Toy recurrence | d32 m4 K8 | 3.814697265625e-06 | 3.7548333493113476e-05 |
| Candidate_02 KQ | d512 m1 K1 | 2.1457672119140625e-06 | 0.022508038585209004 |
| Candidate_02 KQ | d512 m1 K4 | 7.152557373046875e-06 | 0.0023835674058841397 |
| Candidate_02 KQ | d512 m4 K1 | 3.874301910400391e-06 | 0.0007047689364701145 |
| Candidate_02 KQ | d512 m4 K4 | 1.049041748046875e-05 | 0.003707004099091071 |
| Candidate_02 KQ | d512 m16 K1 | 2.86102294921875e-06 | 0.01465076660988075 |
| Candidate_02 KQ | d512 m16 K4 | 1.0609626770019531e-05 | 0.0012047412218612287 |

Full-size K4 results are diagnostic and are not gates.

## offline seal recovery (disclosed)

Binding input `build/preflight/binding_companion.json` SHA-256: `e7637df1a32ef5a33af3309abdaa816ef3222e9b1da91d0b0871877c651e522a`. KQ input `native_kq_candidate_02.json` SHA-256: `230dce1ccae58696f38610077459e2c6781b749330298b4617df9b3561da7b6c`. Binding timing-sanity JSON SHA-256: `7e0f3ef5fb1b6cab2782f550132a1441b2e0aaaae1066f4e55066817b5d5608a`; binding report SHA-256: `478f985238f42babb9bfd33d64f6578a4441e9989b01957ceb7c173ad9759191`; binding artifact manifest SHA-256: `1ffc581ba5e7c3c49d3e3fe4aad22bbdb2257beb2b5afcbe92d8b92e5221a74d`. The in-memory recovery returned the saved binding JSON and read `timing_protocol.no_timed_allocations` at its recorded location. The companion EXE was not invoked during recovery.

H0 input `build/preflight/core_selection_preflight.json` SHA-256: `e0ca7b7bffae9242ade1b5b5981e82d5b26c68c0be5f18e9b341bae04e5a08a4`. Attempt_02 shard-manifest input SHA-256: `174aab12738a0e2110e65cdc4ce891f9ed236f0c1f945094735960a5de6382d8`. An in-memory `windows_cpu_set_id` alias was added to the read shard objects; `subprocess.run` returned a `CompletedProcess` containing the saved H0 JSON. H0 report SHA-256: `09faba6e82a632bd6e2ac4a6e78c2d75de5cbc520709868828f464df1673ca02`; H0 artifact manifest SHA-256: `2dbf73d67eeb0e897fb301a21251275da28495a8533321279d50353bc653a7b7`. The companion EXE was not invoked during recovery.

Neither binding nor H0 measurements were repeated. No rebuild, relink, or reselection occurred.

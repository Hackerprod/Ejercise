# CPU-Native Architecture

Local Windows validation workspace for the T0 synthetic Q4 microkernel gate.

## Scope

- No model training.
- Native Windows timing with `QueryPerformanceCounter`.
- Fixed affinity probe for each selected logical CPU, with processor-group reporting.
- Deterministic scalar packed-Q4 matrix-vector throughput probe.
- Runtime-dispatched AVX2+FMA Q4 path with scalar fallback and correction gate.
- Reproducible H0 sweep across logical CPUs 0..7, target working-set sizes, depths, and variants A/B/C.
- T0-M Phase 1-3 int8 fused/repeat correction and AVX2 measurement executable; Phase 4 remains blocked.

## Hardware Evidence

Confirmed from `D:\ASUS.json` (authoritative export supplied for this gate):

- AMD Ryzen AI 5 330, Krackan2E.
- 1 Classic + 3 Compact physical cores.
- 8 logical CPUs.
- 1 MiB L2 per core, shared 8 MiB L3.
- 28 W STAPM, 35 W PL1, 40 W PL2.
- Observed 4541.8 MHz.
- CPPC performance order 1-4.

Checked-in `hardware/cpuz_report.txt` provides corroborating CPU-Z cache/topology evidence. Runtime results still depend on current Windows power, thermal, and scheduler state.

## Build

```powershell
cmake -S . -B build -G Ninja
cmake --build build
```

Target: `cpu_native_q4_probe`. CTest includes deterministic smoke paths for all variants:

```powershell
ctest --test-dir build --output-on-failure
.\build\cpu_native_q4_probe.exe --self-test
```

Run all active logical CPUs. CSV goes to stdout; summary goes to stderr:

```powershell
.\build\cpu_native_q4_probe.exe
```

Run one global logical CPU and tune probe:

```powershell
.\build\cpu_native_q4_probe.exe --cpu 0 --m 1024 --K 4096 --depth 4 --iterations 2 --repetitions 5 --warmup 2
```

Options: `m` output rows, `K` input columns, `target-kib` derived target size, `depth` repeated matrix-vector passes, `variant` A/B/C, `kernel` `scalar`, `avx2`, or `auto`, `iterations` kernel invocations per timed repetition, `repetitions` timed repetitions, and `warmup` untimed repetitions. Defaults remain `m=1024`, `K=4096`, `depth=4`, `variant=A`, `kernel=scalar`, `iterations=2`, `repetitions=5`, and `warmup=2`. `--m` and `--target-kib` are mutually exclusive.

## Target-size Mapping

`--target-kib` requires `--K 512`. The probe computes actual bytes as:

```text
packed_Q4_bytes = ceil(m*K/2)
scale_bytes = 4*ceil(m*K/32)
actual_weight_bytes_per_block = packed_Q4_bytes + scale_bytes
```

It chooses the largest positive `m` whose actual bytes do not exceed `target_kib*1024`. With `K=512`, each row uses 320 bytes. The H0 points therefore map as follows:

| target KiB | m | actual bytes per block |
|---:|---:|---:|
| 256 | 819 | 262080 |
| 384 | 1228 | 392960 |
| 512 | 1638 | 524160 |
| 640 | 2048 | 655360 |
| 768 | 2457 | 786240 |
| 896 | 2867 | 917440 |
| 1024 | 3276 | 1048320 |
| 1280 | 4096 | 1310720 |

Direct `--m` remains available for custom shapes. CSV reports `target_kib=0` for direct shapes.

## Variants

- A reuses one deterministic packed-Q4/scales block for every depth pass.
- B allocates one deterministic distinct block per depth pass and selects blocks by `pass % depth` inside the timed kernel. CSV reports total `allocated_weight_bytes`.
- C uses A plus a deterministic 64 MiB eviction buffer touched between depth passes inside the timed QPC region. CSV reports `eviction_bytes=67108864`.

All allocations and warmup happen before timing. Compiler-visible checksums retain kernel and eviction work.

## AVX2 Path

`--kernel avx2` requests the vector path; `--kernel auto` selects it when runtime CPUID confirms AVX2, FMA, OSXSAVE, and XMM/YMM state support. Otherwise both modes fall back to scalar. The vector path loads 16 packed bytes, uses SIMD mask/shift/unpack operations to form 32 signed Q4 values, converts eight values at a time to FP32, and performs `_mm256_fmadd_ps`; horizontal reduction happens once per output row. Every AVX2 measurement first runs a same-data scalar/vector checksum correction test with relative tolerance `1e-3`.

MSVC builds use `/O2 /EHsc`. `/arch:AVX2` is intentionally not enabled globally: doing so could auto-vectorize the scalar fallback and execute AVX2 on unsupported CPUs before runtime dispatch. AVX2/FMA intrinsics generate the guarded instructions explicitly; the CSV records `kernel_requested`, `kernel_used`, `avx2_supported`, and `fma_supported`.

## H0 Sweep

After building, run the exact 8 CPU x 8 size x 5 depth x 3 variant sweep:

```powershell
.\scripts\run_h0_sweep.ps1
```

The script assumes `build\cpu_native_q4_probe.exe` already exists and installs nothing. It uses `iterations=1`, `repetitions=2`, and `warmup=1`. Outputs are written under `sweep-output\`:

- `h0_sweep.csv`: one header and one raw row per invocation, stdout only.
- `h0_sweep.stderr.log`: probe diagnostics and exact commands, stderr only.
- `h0_sweep.summary.txt`: row count, mapping/configuration, and 256 KiB variant A classification.

The summary reports the fastest logical CPU at 256 KiB variant A as an empirical Classic candidate. This is inference from measured throughput, not a hard-coded CPU identity.

## Parallel Contention Sweep

The isolated sweep above cannot expose shared DRAM contention. Run the focused AVX2 batch sweep to start eight logical workers simultaneously, each pinned to logical CPUs `0..7`, with one wall-clock measurement for the complete batch:

```powershell
.\scripts\run_h0_parallel_sweep.ps1
```

It covers `target_kib={128,192,512,1024,1280}`, `depth={4,8,16}`, and variants A/B. The 128/192 KiB points fit within the confirmed 256 KiB private L2; 512 KiB and above test shared-L3/DRAM behavior. Each batch row reports aggregate `batch_mac_per_second`, with `B_over_A` comparisons in `h0_parallel.summary.txt`. Workers allocate independent deterministic data before the synchronized wall-clock start, so A/B batches compete for shared cache and memory rather than sharing one weight allocation.

For four physical cores without SMT siblings, pass the topology-confirmed mask explicitly:

```powershell
.\scripts\run_h0_parallel_sweep.ps1 -Workers 4 -LogicalCpuIndices @(0,2,4,6) -OutputDirectory .\sweep-output\h0-parallel-avx2-physical4
```

## Interpretation and Limitations

- `mac_count` counts scalar multiply-accumulate pairs; `mac_per_second` is measured throughput for the timed region.
- `affinity_succeeded=true` means worker was pinned with `SetThreadGroupAffinity`; previous thread group/affinity is restored by RAII before worker exits.
- Kernel deterministically generates packed 4-bit signed nibbles and FP32 scales, then dequantizes scalars inside dot product. It is a throughput probe, not an exact Q4 model-kernel equivalence claim.
- No AVX, AVX2, AVX-512, FMA, or VNNI assumption is made. Scalar reference path is intentional for T0.
- Results are per logical CPU, not a four-physical-core aggregate. SMT siblings, Windows background load, boost state, thermal limits, and processor-group topology can change results.
- Harness does not prove cache residency, DRAM traffic, model quality, training stability, or equivalence with a Transformer. Use hardware counters and later A/B/C implementations for those claims.
- Target sizes are approximate resident packed-Q4 plus scale bytes; allocator/vector overhead and input/output buffers are not included in `actual_weight_bytes_per_block`.
- This remains a scalar probe. SMT sibling contention, Windows scheduler placement, boost state, thermal limits, affinity failures, and background load can change rankings. Classic identification is empirical.
- Do not compare elapsed values across machines without recording power mode, temperature, build compiler, and exact command line.

## T0-M Phase 1-2

`t0m_int8_probe` is isolated from `int8_probe.cpp` and uses unambiguous names: `D` is internal state dimension, `S` is slot count, `R` is recurrent depth, `O_i` is rows assigned to physical worker `i`, and `B_i` is weight bytes assigned to physical worker `i`. Variant A shares one weight block across `R`; variant B uses distinct blocks by recurrent depth.

The `fused` kernel tiles output rows and slots, loads each weight tile once, feeds that tile to every slot in the slot tile, then stores outputs. `repeat` is the S-independent-GEMV control. Supported `S` values are `1,2,4,8,16`; runtime `S_tile` candidates are `2,4,8`.

Every invocation runs correction first. Correction covers all four shards, positive/negative int8 data, non-multiple dimensions and tile sizes, exact reference/fused/repeat equality, checksum coverage for every `Y[S x O_i]` cell, and one-worker versus four-worker accounting. Speed CSV is emitted only when this gate passes.

Focused verification only:

```powershell
cmake --build build --target t0m_int8_probe
ctest --test-dir build -R t0m_int8_correction --output-on-failure
.\build\t0m_int8_probe.exe --D 64 --S 4 --R 2 --mode fused --variant A --S-tile 4 --workers 1 --rows-per-worker 128 --timed-repetitions 2
```

## T0-M Phase 3

Phase 3 runs only the approved D=512 matrix: target sizes `384/512/640/768 KiB` per worker, `S={1,2,4,8,16}`, `R={1,2,4,8,16}`, fused/repeat, and variants A/B. Four equal shards use physical logical CPUs `0,2,4,6`; `O_i=floor(target_bytes/(D bytes/int8))`, with `B_i=O_i*D` for A and C and `B_i=O_i*D*R` for B. C controls cover only `512/768 KiB`, `R=16`, `S={1,8,16}` and use shared A weights with eviction and `clflush` outside timed kernel units.

The one-shot runner first measures `S_tile={2,4,8}` on `D=512,S=8,R=16` for A/B and fused/repeat, selects fastest by median fused throughput across pilot repeats and A/B mean, then executes 400 A/B rows plus 12 C controls:

```powershell
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File scripts\run_t0m_phase3.ps1
```

Outputs are written under `sweep-output\t0m-phase3\`: `t0m_phase3.machine.csv`, exact `t0m_phase3.commands.log`, `t0m_phase3.stderr.log`, and `t0m_phase3.summary.txt`. Every speed invocation must pass correction, AVX2, four-worker affinity, exact repetition, and row-shape validation. The runner does not start Phase 4 and refuses to repeat a completed campaign.

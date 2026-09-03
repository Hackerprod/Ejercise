# T0-R Provenance Archive

Phase 0 archive only. Nothing under `sweep-output` was modified or deleted. T0-M is out of scope.

## Invocation

Exact invocation used:

```text
powershell.exe -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File scripts\run_t0r_int8_sharded_sweep.ps1
```

Internal defaults: `RepeatCount=5`, `Repetitions=8`, `Warmup=2`; target sizes `384/512/640/768 KiB per worker`; depths `1/4/8/16`; variants `A/B/C`; physical logical CPUs `0,2,4,6`.

## Topology Evidence

Physical CPU mapping used: `0,2,4,6`.

Prior evidence:

- `D:\ASUS.json`: one processor package, 4 physical cores, 8 logical CPUs, and `1 x Classic, 3 x Compact` cores with 2 Classic and 6 Compact logical CPUs.
- `D:\Install\Dev\Projects\IA\Ejercise\cpu-native-arch\hardware\cpuz_report.txt`: CPU Group 0 has 8 threads and mask `0xFF`; CPU-Z reports 4 cores, 8 threads, and two core sets (`P-Cores`: 1 core/2 threads; `E-Cores`: 3 cores/6 threads).
- `D:\Install\Dev\Projects\IA\Ejercise\cpu-native-arch\scripts\get_physical_topology.ps1`: prior `GetLogicalProcessorInformationEx` evidence mapped physical cores as `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`; non-SMT worker selection is therefore `0,2,4,6`.

## Calibration

Requested calibration values, recorded separately: `cpu0=19.3`, `cpu2=18.1`, `cpu4=10.9`, `cpu6=17.0 GMAC/s`.

Archived `int8_calibration.csv` actual values (raw `mac_per_second`, with GMAC/s conversion):

| CPU | Raw MAC/s | Actual GMAC/s |
|---:|---:|---:|
| 0 | 24345900000 | 24.3459 |
| 2 | 14743800000 | 14.7438 |
| 4 | 17636500000 | 17.6365 |
| 6 | 18119500000 | 18.1195 |

## Corrected Sweep

The corrected sweep has 240 valid rows and uses proportional row allocation in CPU order `0,2,4,6`:

| Target KiB/worker | Total rows | Rows per worker (CPU 0,2,4,6) |
|---:|---:|---|
| 384 | 3072 | 999,605,724,744 |
| 512 | 4096 | 1332,807,965,992 |
| 640 | 5120 | 1665,1009,1206,1240 |
| 768 | 6144 | 1999,1210,1448,1487 |

DRAM ceiling: `32.93 GB/s` (measured source value `32.9295 GB/s`).

Corrected formula: `MAC_total = sum_i(O_i*D*S*R*iterations*timed_repetitions)`, with parallel phase wall timing. The timed metric is one synchronized main-thread wall-clock interval around each kernel phase; preparation is excluded.

## Repetition Bug

The prior `run_parallel` path executed one timed repetition but multiplied MAC count by `Repetitions=8`, inflating throughput by approximately 8x. Detection came from comparing `run_single` and `run_parallel` accounting and was confirmed by the physically impossible parallel result. The corrected path times all repeated phases and uses synchronized parallel wall time. Source: `D:\Install\Dev\Projects\IA\Ejercise\cpu-native-arch\int8_probe.cpp`.

## Sources

- `D:\Install\Dev\Projects\IA\Ejercise\cpu-native-arch\build\int8_probe.exe`
- `D:\Install\Dev\Projects\IA\Ejercise\cpu-native-arch\sweep-output\t0r-int8-sharded\t0r_int8_sharded.csv`
- `D:\Install\Dev\Projects\IA\Ejercise\cpu-native-arch\sweep-output\t0r-int8-sharded\int8_calibration.csv`
- `D:\Install\Dev\Projects\IA\Ejercise\cpu-native-arch\sweep-output\t0r-int8-sharded\t0r_int8_sharded.stderr.log`
- `D:\Install\Dev\Projects\IA\Ejercise\cpu-native-arch\sweep-output\t0r-int8-sharded\t0r_int8_sharded.summary.txt`
- `D:\Install\Dev\Projects\IA\Ejercise\cpu-native-arch\scripts\run_t0r_int8_sharded_sweep.ps1`

## Archive Files

- `int8_probe.exe` - copied current build executable.
- `t0r_int8_sharded.csv` - complete corrected sweep CSV.
- `int8_calibration.csv` - calibration CSV, copied unchanged.
- `t0r_int8_sharded.stderr.log` - stderr and command log, copied unchanged.
- `t0r-provenance-manifest.md` - this manifest.

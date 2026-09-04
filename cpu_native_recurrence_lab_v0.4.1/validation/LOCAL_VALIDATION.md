# Local validation of CNRL v0.4.1

This validation was executed in the Linux build environment used to prepare the package. It verifies compilation, correctness, sanitizers, source invariants and ELF assembly. It does **not** replace the Windows/Ryzen bare-metal run.

## Build matrix

| Toolchain | Configuration | CTest |
|---|---|---|
| GCC 14 | Release, `-O3 -mavx2 -mfma`, warnings as errors | 4/4 PASS |
| Clang 17 | Release, `-O3 -mavx2 -mfma`, warnings as errors | 4/4 PASS |
| GCC 14 | Debug, ASan + UBSan, warnings as errors | 4/4 PASS |

The four tests are:

1. C++ unit/oracle tests;
2. gate CSV analyzer/tamper tests;
3. end-to-end gate CLI→CSV→strict analyzer;
4. transition benchmark chain/reset/accounting→strict analyzer.

## Static/source audit

`tools/audit_source.py`: **44/44 PASS**.

## Assembly audit

For both GCC and Clang objects:

- `fused4` contains signed int8 expansion and `vpmaddwd`;
- `fused4` has no XMM/YMM stack access in its arithmetic body;
- transition objects contain AVX2 scale/shift, saturated packing and RMS vector paths.

## Deterministic fixed-point calibration smoke

At `D=1472,R=8`, four equal shards, default seed:

| Transition | Shift | S=1 clipping | S=8 clipping | S=16 clipping |
|---|---:|---:|---:|---:|
| fixed-point | 12 | 25.433% | 25.053% | 25.411% |
| fixed-point | 13 | 3.252% | 2.900% | 3.083% |
| fixed-point | 14 | 0.000% | 0.000% | 0.0016% |
| fixed-point | 15 | 0.000% | 0.000% | 0.000% |
| group-RMS | 12 | 0.000% | 0.0053% | 0.0053% |
| global-RMS | 12 | 0.000% | 0.0032% | 0.0042% |

These clipping counts are deterministic numerical checks. Their Linux throughput ratios are not portable to the target laptop.

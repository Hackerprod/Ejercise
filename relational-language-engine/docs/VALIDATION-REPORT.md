# Validation Report

Generated: 2026-08-15T15:49:49Z

| Gate | Result | Scope |
|---|---:|---|
| Required project inventory | **PASS** | Core modules, tests, deployment and operations assets |
| Debug compile + CTest | **PASS** | Native integration and functional/concurrency unit tests |
| Clean-prefix CMake install | **PASS** | Installability outside the source tree |
| End-to-end CLI smoke | **PASS** | Import, train, persist, infer, inspect, checkpoint, expire and benchmark |
| Release + LTO + CTest | **PASS** | Optimized-build parity |
| ASan + UBSan + CTest | **PASS** | Memory/lifetime/undefined-behaviour instrumentation |

Unresolved source markers outside the copied reference specification: **3**.

## Test-covered architectural guarantees

1. Frozen embedding round-trip, version/checksum validation and immutable reopening.
2. Dimensional token relationships and deterministic relation observations.
3. Physical CLEAN isolation before Top-K; no M1 ghost branch or zero-operator substitute.
4. Lane-local scoring/composition and immutable inference controller state.
5. WAL replay, torn-tail repair and deterministic batch idempotence.
6. Replay storage bounded by depth × beam × candidates rather than a symbolic k^D tree.
7. Exact audit reopen and  when configured limits prevent proof.
8. Recoverable M1→M2 promotion and TTL pinning during audit/promotion.
9. Resume-safe streaming corpus training.
10. Independent FULL/CLEAN execution under multi-client stress and timeout.
11. Dual-lane versus single-lane runtime measurement and relational versus trigram accuracy.

## Toolchain

- c++ (Debian 14.2.0-19) 14.2.0
- cmake version 3.31.6
- Ninja 1.12.1
- Linux 6.18.35 x86_64 GNU/Linux

## Raw logs

, , , , and  are retained at the repository root.

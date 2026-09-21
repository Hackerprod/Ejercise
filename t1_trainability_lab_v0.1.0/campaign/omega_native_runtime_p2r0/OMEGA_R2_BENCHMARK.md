# OMEGA Native Runtime P2-R2

`omega_recurrent_production_bridge.py` is the clean autograd bridge. It sets
`instrumentation=0`, passes null certification pointers, borrows parameters,
and leases full-BPTT workspaces from a config/size keyed pool.

`run_omega_native_runtime_r2_benchmark.py` exposes four fresh-process-compatible
routes: `pytorch-k1`, `pytorch-k4`, `native-k1`, and `native-k4`.

Default command runs synthetic canaries only:

```text
python run_omega_native_runtime_r2_benchmark.py --phase canary
```

Real phases are explicit and authorization-gated:

```text
python run_omega_native_runtime_r2_benchmark.py --phase preflight --route native-k1 --confirm-real-execution
python run_omega_native_runtime_r2_benchmark.py --phase stable --route native-k1 --confirm-real-execution --manifest ... --cache-file ...
python run_omega_native_runtime_r2_benchmark.py --phase profile --route native-k1 --confirm-real-execution --manifest ... --cache-file ... --output results/r2_profile_native_k1.json
```

Stable separates `--warmup-updates` (default 1, excluded from totals) from
`--updates` measured updates. Aggregation consumes measured records only.

Stable records total update time, separate clean-bridge forward/backward
boundaries, C-ABI forward/backward time, bridge overhead (inclusive bridge
boundary minus C-ABI time), workspace allocation/reuse, and RSS. No small-GEMM
timers are used. Aggregation reports `R_K = T_NATIVE,K/T_PYTORCH,K` and
`R_joint = (T_NATIVE,K1+T_NATIVE,K4)/(T_PYTORCH,K1+T_PYTORCH,K4)`.

Addendum 240 classification is candidate/baseline: STRONG requires
`R_joint <= 0.80` and every K `<= 1.05`; PASS requires `R_joint <= 0.90` and
every K `<= 1.05`; NEUTRAL requires `0.90 < R_joint <= 1.05` and every K
`<= 1.10`; REGRESSION requires `R_joint > 1.05` or any K `> 1.10`.
Uncovered tradeoff combinations are reported as `UNCLASSIFIED`.

Native setup validates the fused R1 prelude state projection once before
measurement and fails closed if it is non-contiguous. It does not perform a
per-update projection copy.

## Temporary Native Profile

Build profile-enabled DLL explicitly; ordinary builds use
`-DOMEGA_PROFILE_INTERNAL` absent and contain no profile ABI or timer scopes:

```text
cmake -S native -B native/build-profile -DCMAKE_BUILD_TYPE=Release -DOMEGA_PROFILE_INTERNAL=ON
cmake --build native/build-profile --config Release
```

The `profile` phase is restricted to fresh native K1 and fixed at one warmup
plus two measured updates. It reuses stable setup, but never schedules K4 or
stable 8+32. Native stage timers are compiled only behind
`OMEGA_PROFILE_INTERNAL` and disabled at runtime by default; default builds
omit profile symbols, clock reads, and profile aggregation. Enabling profile
adds `std::chrono::steady_clock` reads and mutex-protected aggregate updates,
so totals are diagnostic and not performance evidence. Timers cover only
high-level QKV projection; attention scores/softmax/mixing; out projection;
FC1/GELU; FC2; RMSNorm/gates; state/prelude; depth-embedding pairwise
reduction and carry; and practical history-buffer reads/writes. No individual
multiply/GEMM is timed.

Run profile output through files, not a null pipe:

```powershell
python .\run_omega_native_runtime_r2_benchmark.py --phase profile --route native-k1 --confirm-real-execution --manifest <manifest> --cache-file <cache> --output .\results\native_k1_profile_report.json *> .\results\native_k1_profile.stdout.log
```

The result records exact loaded DLL path, SHA256, size, mtime, Release flags,
profile-build state, and native ISA/threading metadata at measurement time.

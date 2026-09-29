# OMEGA Fable Performance Delivery Closeout

**Date:** 2026-09-24  
**Cycle status:** CLOSED  
**Scope:** performance-candidate disposition and evidence index. Scientific adoption status is unchanged.

## Final performance candidate

`be37623d7e69b65551ad9c306092f328cc250576a56642fc34a0068df00600cc` is final Fable performance candidate, status **NATIVE-PERF-STRONG**. Candidate is on `omega/diagnostics-out-state`:

- Code commit: `18e7225b819af61304189eff8537d602f729cf2f`
- Correctness-coverage commit: `d993b3c1f1e19e4bffa32d7c4be35687257b28c6`
- DLL: `C:\Users\danil\bpf2a\t1_trainability_lab_v0.1.0\campaign\omega_native_runtime_p2r0\native\build-diagnostics-out-state-candidate-verified\python\omega_recurrent.dll`
- SHA-256: `be37623d7e69b65551ad9c306092f328cc250576a56642fc34a0068df00600cc`

Final ratios combine sums from two separately reported stable blocks (original route order and reverse route order). Do not average block ratios.

| Depth | PyTorch total, both blocks | Native total, both blocks | Native/PyTorch | Faster |
|---|---:|---:|---:|---:|
| K1 | 157.7019354004 s | 120.5649839001 s | 0.764512 | 23.55% |
| K4 | 229.6457892999 s | 182.9933213005 s | 0.796850 | 20.31% |
| Combined | 387.3477247003 s | 303.5583052006 s | 0.783684 | 21.63% |

Each block used fresh processes, 8 excluded warmups +32 measured updates per route, native runtime threads=4, PyTorch intra/inter-op=4/1, and canonical KMP/OMP/MKL overrides unset.

| Stable block | Route order | R_K1 | R_K4 | R_joint | Verdict |
|---|---|---:|---:|---:|---|
| Original | PyTorch-K1 → Native-K1 → PyTorch-K4 → Native-K4 | 0.7605233617 | 0.7959252328 | 0.7814874389 | STRONG |
| Reverse | Native-K4 → PyTorch-K4 → Native-K1 → PyTorch-K1 | 0.7685122244 | 0.7977729003 | 0.7858801915 | STRONG |

Original route totals: PyTorch K1 78.9700976005 s, Native K1 60.0586041005 s, PyTorch K4 114.6667714999 s, Native K4 91.2661768004 s. Reverse route totals: Native K4 91.7271445001 s, PyTorch K4 114.9790178000 s, Native K1 60.5063797996 s, PyTorch K1 78.7318377999 s.

## Correctness and performance evidence

The chosen candidate passed all Fable correctness gates:

- Historical diagnostic-mask equivalence: 2,048 masks at each of K1/K4; 4,096/4,096 byte-exact.
- 32 named diagnostic masks: direct and runtime DLL paths, K1/K4, workers 1/2/4; byte-exact, including changed-data reset checks.
- Four instrumented-vs-clean golden comparisons and four shared-DLL golden cases passed.
- Runtime determinism passed at workers 1/2/4; matrix-view and stride-validation tests passed.
- False-diagnostics assembly inspection confirmed optional diagnostic traversal absent; assembly-build `.text` matched candidate `.text` hash `4b0898b3d4cc3ab1a037fa3280eb5f96b11d87fd4cd2f895fb89aaeab3d5d360`.
- Backward-only comparison against rollback 279e: three favorable pairs, median improvement 19.45%.
- Short E2E comparison against 279e: K1 R=0.9460352611 (3/3 favorable), K4 R=0.9158358276 (3/3 favorable).

Evidence files and SHA-256:

| Evidence | Artifact | SHA-256 |
|---|---|---|
| Full candidate provenance and correctness/performance gates | `C:\Users\danil\AppData\Local\Temp\opencode\omega-out-state-provenance.json` | `7eacd948ac50f69d7a762eb97168526327753f0d8dd36965b180b7bc4c7f7f7b` |
| Gate disposition and correctness summary | `C:\Users\danil\AppData\Local\Temp\opencode\omega-out-state-results-20260924.json` | `df54bd3804aa2f4e3426eff909eef441b330384c092846f5d057558c3b055db9` |
| Original stable block summary | `C:\Users\danil\AppData\Local\Temp\opencode\omega-out-state-stable-8p32-20260924-summary.json` | `7c16f38bdcabf1b3b858f30aa8d8f00f93426146fdb96659eb18af88a579bfff` |
| Reverse stable block summary | `C:\Users\danil\AppData\Local\Temp\opencode\omega-newprimary-r2-reverse-8p32-20260924-summary.json` | `f79b409f35093e6a356dcc0405c1b4bd96965bf1f713f1f0b6cf707cf4c1d5f1` |
| Short E2E gate aggregate | `C:\Users\danil\AppData\Local\Temp\opencode\omega-out-state-e2e-short-20260924\aggregate_report.json` | `4b86ae0b46d95fb4ef19a8ba63cc6710bcf9155e73b9875e0bfebc6128c4ff36` |

## Rollback candidates

| Role | DLL and SHA-256 | Provenance / status |
|---|---|---|
| Previous certified candidate | `C:\Users\danil\bpf2a\t1_trainability_lab_v0.1.0\campaign\omega_native_runtime_p2r0\native\build-qkv-fc2-fastpath-candidate\python\omega_recurrent.dll` — `279e342e88ff76293514252fcaf497206d885fa72485cf6a2ad271bc0bc01556` | QKV+FC2 candidate from `omega/diagnostics-selected`, source commit `fddcb74cb693fa8ccf74e99d738b37e00e8d1a23`; certified PASS. Its prior stable characterization was K1 R=0.8012185547, K4 R=0.8614847979, joint R=0.8368760347. |
| Pre-QKV/FC2 baseline | `C:\Users\danil\AppData\Local\Temp\opencode\omega-backward-diagnostics-fastpath\python\omega_recurrent.dll` — `6f38e3b1dcada32c98aef50e870e201510bab9c75b875438823915ba53b854d5` | Clean source baseline at commit `64a95e2adeeccf5e630987f6079e32b459b65eb1`; preserved as earlier rollback/reference. |

Both DLLs remain available and unchanged. `279e` is preferred rollback from the selected candidate; `6f38` is the earlier pre-QKV/FC2 baseline.

## Rejected / not-selected variants

| Variant | Identity | Reason for NOT_SELECTED | Evidence |
|---|---|---|---|
| Attention backward AVX2 (DCC25A5B) | DLL SHA `dcc25a5b9a8548863d4ae2300925193a0cbec84313f092b1dbd91f1ecd8f94e1`; experiment commit `dc81e7f8493523d84a29c1e3d8e03c40b18ec5d5` | Backward-only passed 3/3, median +3.60%, but E2E gains were only K1 +0.76% and K4 +0.43%, with 2/3 favorable pairs each; too weak for selection. | DLL `C:\Users\danil\AppData\Local\Temp\opencode\omega-attention-avx2-candidate\python\omega_recurrent.dll`; raw evidence remains in Temp. |
| FMA pilot | DLL SHA `bb23bff709095074c9e8c004919f4d8069e1e464dca8a01223d2773ada94b4c4`; experiment commit `fd024cc459b65190bf039645584b03148fe6b483` | Numeric/correctness and backward gates passed, but E2E K4 regressed: K1 R=0.9849208 (+1.508%), K4 R=1.0100148 (-1.001%) with only 1/3 K4 pairs favorable. | DLL `C:\Users\danil\AppData\Local\Temp\opencode\omega-attention-fma-fc1-fc2-qkv-candidate\python\omega_recurrent.dll`; raw FMA matrices remain in Temp. |
| Unit A: FC1-bias diagnostic gating + proven Q/K/V initial-clear removal + div/mod loop | Candidate SHA `9f52d2b0be50f1ac09cffa8b54b29a1c4a75d83f623772c6005216ac1ef505dc`; uncommitted branch `omega/diagnostics-fable-a` based at `d993b3c1...` | Correctness, poison, and reset-removal gates passed; E2E failed performance threshold. K1 1/3 favorable, median -0.14%; K4 3/3 favorable but median +0.55% (<2%). Not frozen or committed. | Diff preserved in `stash@{0}` in `C:\Users\danil\bpf2a`; aggregate `C:\Users\danil\AppData\Local\Temp\opencode\omega-fable-a-unitA-20260924T064035-aggregate.json` (SHA `1661b937e11202b5effadefb560ad28b799ac02306c3e27ff2736969e68eae2a`). |
| Unit B: backward erf reuse | Candidate SHA `c26910562df236c232b8422faf08c7f0bfd61e3ff2fdeb199188c036ae89a1c8`; uncommitted branch `omega/backward-erf-reuse` based directly at `d993b3c1...` | Lifetime point 7 was proven safe and exactness passed, but E2E medians did not reach 2%: K1 +1.37%, K4 +1.69%, despite 3/3 favorable pairs at both depths. Not committed or merged. | Diff remains on unmerged branch in `C:\Users\danil\bpf2a`; aggregate `C:\Users\danil\AppData\Local\Temp\opencode\omega-fable-b-unitB-20260924T072017-aggregate.json` (SHA `064eccbfd05d6ac1b9da62c2e56eea4e725b76530de40624bdb037bdd44436b2`). |

## Scientific status — unchanged

- `strict_trajectory_proximity = FAIL_RECORDED`
- `training_quality_equivalence = NOT_ESTABLISHED`

These remain open scientific-adoption items, separate from native performance. This closeout does not resolve, revise, or relabel either status.

## Closure

Fable optimization cycle is closed. `be376` remains final performance candidate; `279e` and `6f38` remain rollback/reference options. Attention AVX2, FMA, Unit A, and Unit B remain preserved as NOT_SELECTED evidence. No further optimization, training change, or adoption action is authorized by this closeout.

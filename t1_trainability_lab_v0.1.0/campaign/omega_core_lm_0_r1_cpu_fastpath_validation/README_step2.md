# OMEGA R1 CPU Fastpath Validation, Step 2

This unit measures four complete FP32/eager/CPU update routes after Step 1:
`R` corrected reference, `F` local fast model with original loss, `L` local
fast model with LSE loss, and `C` the faster measured `F`/`L` route with a
bounded FP32 teacher cache.

## Quick Path

```powershell
python run_step2_benchmark.py
python -m pytest -q test_step2_benchmark.py
python -m py_compile run_step2_benchmark.py test_step2_benchmark.py omega_fast_candidate.py
```

Each invocation creates a new discardable directory under
`results/step2_benchmark/<run-id>/`; existing reports are never overwritten.

## Measurement Contract

- CPU only, eager only, FP32 only, one trainer sequentially.
- Fixed `B=8`, window length `256`, `D=128`, `S=8`, vocabulary `50257`.
- Eight exact approved WikiText-2 train documents and pinned teacher route:
  window 0 context `[0:256]`, window 1 context `[0:512]`.
- Each config and variant runs six updates with schedule `0,1,0,1,0,1`.
  First pair is warmup; final four are measured. Total is exactly 48 updates.
- Timed update includes teacher/cache read, student path, loss, backward,
  clipping, AdamW, and final ledger write. L target preparation is timed;
  C cache read is timed.
- Timings include per-component, warmup/measured, per-window, and `t_K1`/
  `t_K4` totals. Memory records before, after, and every sampled update.
- C selection uses measured joint `shared_K1 + shared_K4` cost. Ties choose L.

## Cache

Cache is created only after F/L selection, stored and read back as FP32, and
records creation time, format, dtype, bytes, and verification path. F stores
teacher logits. L stores teacher probabilities plus negative entropy.

## Scope Guard

No SCOPE-A runner/results are touched or relaunched. No validation/test split,
projection, GPU, compile, mixed precision, CUDA graph, vmap, parallel trainer,
thread experiment, Fable optimization, or quality interpretation is used.
The Fable proposal module remains read-only reference material.

# OMEGA Native Runtime P0

Python/PyTorch cost profile for the current R1/F production route after hidden-cache adoption.

Scope:

- FP32, CPU, batch 8, full BPTT256, K1 and K4.
- CE+KL distillation with RAM-preloaded hidden cache and exact teacher LM head online.
- 20 updates per K by default, first 4 warmup and remaining 16 measured.
- Timers for embedding/prelude, recurrent forward, student readout, hidden fetch, teacher LM head, CE/KL, student vocabulary projection, one backward, gradient clipping, and AdamW.
- Backward remains one `loss.backward()` call. Boundary hooks estimate vocabulary/readout, recurrent, and embedding/prelude segments. If hooks cannot segment safely, report keeps one backward total and marks it `unsegmented`.

No cache rebuild, C++, CUDA, AVX, or native implementation is included.

Run only after explicit authorization:

```powershell
python run_omega_native_runtime_p0.py --profile --confirm-real-execution
```

Report: `results/profile_report.json`, self-hashed with `report_self_hash`.

# OMEGA Teacher Hidden Cache Probe

Follow-up probe after `OMEGA-TEACHER-LOGIT-CACHE`.

- Stores raw FP32 DistilGPT2 transformer hidden states, not vocabulary logits.
- Backing store target is `C:\omega_cache\teacher_hidden.fp32` on NVMe; code and
  reports remain in this campaign directory on D:.
- Shape is `[window, document, token, hidden] = [2, 602, 256, 768]`.
- Hidden-cache size is `2 * 602 * 256 * 768 * 4 = 946,864,128` bytes.
- Primary benchmark preloads the complete backing store into RAM once per child.
  `cache_load_seconds` is reported separately and never enters per-update timing.
- All real entrypoints require explicit authorization. Tests use only tiny
  tensors/models and never construct the real cache.

## Guarded commands

```text
python run_omega_teacher_hidden_cache_probe.py --feasibility --confirm-real-execution
python run_omega_teacher_hidden_cache_probe.py --build --confirm-real-execution
python run_omega_teacher_hidden_cache_probe.py --correctness --confirm-real-execution --cache-manifest <path>
python run_omega_teacher_hidden_cache_probe.py --benchmark --confirm-real-execution --cache-manifest <path>
```

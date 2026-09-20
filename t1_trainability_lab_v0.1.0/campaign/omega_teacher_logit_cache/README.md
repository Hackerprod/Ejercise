# OMEGA Teacher Logit Cache

Implementation unit for the persistent, sealed DistilGPT2 raw-logit cache.

## Contract

- Reuses the frozen CE-only train manifest: 602 documents, 1000 cyclic pairs,
  physical batch 8, and 513 retained GPT-2 tokens per document.
- One memory-mapped FP32 file has conceptual shape
  `[window, document, token, vocab] = [2, 602, 256, 50257]`.
- Each entry is exactly `256 * 50257 * 4 = 51,463,168` bytes; the complete
  cache is `61,961,654,272` bytes (about 57.7 GiB).
- Cache keys contain document identity, window, teacher/tokenizer/dataset
  identity, context range, representation, and dtype. K, seed, student
  architecture, and temperature are deliberately absent.
- Build seals the manifest as `CACHE_SEALED / READ_ONLY`; readers open the
  mmap read-only and never load or call the teacher.
- Phases A-D are guarded. No build, correctness run, benchmark, or training
  runs from tests or from the default command.

## Commands

```text
python run_omega_teacher_logit_cache.py --feasibility --confirm-real-execution
python run_omega_teacher_logit_cache.py --build --confirm-real-execution
python run_omega_teacher_logit_cache.py --correctness --confirm-real-execution --cache-manifest <path>
python run_omega_teacher_logit_cache.py --benchmark --confirm-real-execution --cache-manifest <path>
```

Benchmark primary region is updates 40-151 (112 updates). Ratios always use
`R_cost = T_CANDIDATE / T_BASELINE`.

# OMEGA R1 CPU Fastpath Validation, Step 3

Step3 isolates exact SCOPE-A cycle-frequency simulation from a targeted real
teacher cache miss/hit trial. It does not modify or relaunch SCOPE-A, rewrite
Step1/Step2 artifacts, calculate projection, or load validation/test data.

## Quick Path

```powershell
python run_step3_cache_extension.py
python -m pytest -q test_step3_cache_extension.py
python -m py_compile run_step3_cache_extension.py test_step3_cache_extension.py
```

Every invocation creates a unique directory under
`results/step3_cache_extension/<run-id>/`. The parent writes
`full_traversal_simulation.json`, two child results, and `step3_report.json`.

## Simulation

- Uses pinned WikiText-2 train data and pinned tokenizer revision only.
- Reconstructs level-one documents, applies the existing eligibility rule, and
  deduplicates by full-text and retained-token hashes without retaining full
  corpus tokens.
- Traverses exactly 1000 pairs and 2000 logical window updates using the same
  modulo-8 schedule and wrap behavior as `build_pair_manifest`.
- Simulates a persistent on-disk LRU for C+L with separate document/window
  entries and a hard 4 GiB limit. It reports frequency estimates only.

## Block Trial

- Parent launches exactly two fresh sequential subprocesses: `F`, then `C+L`.
- Each child uses CPU, FP32, eager execution, `B=8`, `window=256`, `D=128`,
  `S=8`, `V=50257`, fixed threads, and the existing AdamW settings.
- Each variant runs pair 0 and the derived first repeated pair, windows 0 then
  1, for four updates per variant and eight updates per route.
- F uses the corrected local fast candidate with original loss and current real
  teacher route. C+L uses the same candidate with LSE loss and FP32 cached
  probabilities plus negative entropy.
- Timings separate teacher/cache access, transform, cache write/read,
  recurrence/forward, loss, backward, clipping, AdamW, and ledger phases.
- A 1 GiB available-memory reserve and 4 GiB new-cache-file limit are checked
  before and after phases. Violations stop a child fail-closed.

## Interpretation Guard

`block_trial_costs` contains measured targeted costs. `full_traversal_simulation`
contains metadata-only frequency estimates. They are deliberately separate and
are never combined into a SCOPE-A projection. The formula remains uncomputed:

`T_SCOPE-A = T_preparación + Σ(rutas/variantes)(N_aciertos×t_acierto + N_fallos×t_fallo) + T_validación/checkpoints/E-S`

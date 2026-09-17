# ER32 Quality Scoping A

Phase 1 contains a synthetic CPU-only smoke runner and tests. It uses tiny
fixtures and never loads the real corpus or teacher.

Run Phase 1 checks from this directory:

```text
python -m pytest test_er32_quality_scoping_a.py -q
python -m py_compile run_er32_quality_scoping_a.py
python run_er32_quality_scoping_a.py --smoke --output-dir <synthetic-output>
```

Real Phase 2 execution is intentionally explicit:

```text
python run_er32_quality_scoping_a.py --full --confirm-quality-scoping-a --output-dir <real-output>
```

That command is not part of Phase 1 verification. It reuses the frozen R1
validation curves and the scientific corpus path only after separate approval.
See `CONTRACT.md` for the complete scope and classification rules.

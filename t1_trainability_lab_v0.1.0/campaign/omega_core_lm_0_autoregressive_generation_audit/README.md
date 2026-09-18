# Autoregressive Generation Audit

Phase 1 uses synthetic tiny models only:

```text
python -m pytest test_autoregressive_generation_audit.py -q
python -m py_compile run_autoregressive_generation_audit.py
python run_autoregressive_generation_audit.py --smoke --output-dir <synthetic-output>
```

Real inference is explicitly gated and was not run during implementation:

```text
python run_autoregressive_generation_audit.py --full --confirm-autoregressive-generation-audit
```

See `CONTRACT.md`. No training occurs and no existing checkpoint is modified.

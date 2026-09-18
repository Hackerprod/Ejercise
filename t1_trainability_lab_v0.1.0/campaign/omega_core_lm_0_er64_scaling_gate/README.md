# ER64 Scaling Gate — Stage 1

Run structural verification only:

```text
python -m py_compile omega_fast_er64.py test_er64_scaling_gate.py
python -m pytest test_er64_scaling_gate.py -q
```

This directory is separate from ER32 historical artifacts. Stage 2 cost work
and Stage 3 quality work require separate authorization.

# OMEGA Rank SVD Diagnostic

Run synthetic checks only:

```text
python -m pytest test_omega_rank_svd_diagnostic.py -q
python run_omega_rank_svd_diagnostic.py --smoke --output-dir <synthetic-output>
```

The real checkpoint/corpus evaluation is intentionally blocked and was not
run in Unit 1 implementation verification. See `CONTRACT.md`.

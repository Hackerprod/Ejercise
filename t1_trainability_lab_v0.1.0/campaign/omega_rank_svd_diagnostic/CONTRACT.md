# OMEGA Rank SVD Diagnostic

Unit 1 is a no-training diagnostic over the four frozen R1 checkpoints at
update 2000:

- `shared_K1_seed_20260913/checkpoint_02000.pt`
- `shared_K4_seed_20260913/checkpoint_02000.pt`
- `shared_K1_seed_20260914/checkpoint_02000.pt`
- `shared_K4_seed_20260914/checkpoint_02000.pt`

For each tied `[50257,128]` embedding, the runner computes all 128 usable
singular directions, then builds ranks 32 and 64:

```text
E = P Sigma Q^T
C_r = P_r Sigma_r^(1/2)
U_r = Sigma_r^(1/2) Q_r^T
E_r = C_r U_r
```

Only the tied input/output lexical interface is replaced. All nonlexical
state is copied from the frozen R1 model. No factor is trainable.

Primary evaluation is wired to the unchanged Scientific Scoping A evaluator:
the frozen eight-document validation set, first 513 tokens per document,
blocked unless both flags are supplied:

```text
python run_omega_rank_svd_diagnostic.py --full --confirm-omega-rank-svd-diagnostic
```

No formal PASS/FAIL decision is emitted. The self-hashed artifact status is
`DIAGNOSTIC_COMPLETE`.

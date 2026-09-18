# OMEGA-CORE-LM-0-ER64-SCALING-GATE — Stage 1

Stage 1 verifies rank-64 structural correctness only. No corpus, training,
cost gate, quality gate, or historical ER32 file is modified.

- `C`: `[50257, 64]`, standard deviation `1`.
- `U`: `[64, 128]`, standard deviation `1/sqrt(64)`.
- Core: inherited directly from unchanged F implementation.
- Initialization: `from_f_reference(..., experimental_seed=seed)`.
- Efficient and explicit paths must agree at `atol=rtol=1e-4`.
- No dense `[V,128]` lexical parameter may be constructed.
- Stage 1 does not emit scientific PASS/FAIL decisions.

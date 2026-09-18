# OMEGA Expanded Frozen Validation

Secondary-only validation over all eligible WikiText-2 validation documents.
It does not replace or reclassify primary R1/ER32 gate results.

## Quick path

```text
python -m pytest test_omega_expanded_frozen_validation.py -q
python run_omega_expanded_frozen_validation.py --smoke --output-dir <synthetic-output>
```

Smoke uses synthetic documents and an in-memory model. It never loads WikiText-2
or historical checkpoints.

## Real path

Real evaluation is blocked unless both flags are supplied after separate approval:

```text
python run_omega_expanded_frozen_validation.py --full --confirm-secondary-frozen-validation --output-dir <real-output>
```

The real path reads WikiText-2 validation, reconstructs and deduplicates documents
using the R1 rule, retains the first 513 GPT-2 tokens of every eligible document,
then evaluates frozen R1 and ER32 K1/K4 checkpoints for seeds `20260913` and
`20260914` at update `2000`.

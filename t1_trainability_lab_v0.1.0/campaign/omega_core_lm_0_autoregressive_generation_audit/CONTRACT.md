# OMEGA-CORE-LM-0-AUTOREGRESSIVE-GENERATION-AUDIT

Documentary inference audit only. It does not alter scientific classifications,
replace NLL, or emit quality PASS/FAIL results. Formal result is `AUDIT_COMPLETE`
only when all 64 generations are produced correctly.

## Frozen inputs

- Checkpoints: R1/F and ER32, K1/K4, seeds 20260913/20260914, all at update 2000.
- Excluded: R1 update 5000 and seed 20260915.
- Prompts: first 32 token IDs from the eight frozen R1 validation documents.
- Decode: CPU FP32 eager, greedy `argmax`, `model.eval()`, inference mode,
  zero recurrent state per prompt, free-running token-by-token continuation.
- Maximum new tokens: 64. GPT-2 EOS `50256` stops generation and is recorded.

## Outputs

- `generation_results.json`: complete provenance, token IDs, decoded text, and
  descriptive repetition metrics.
- `blind_review.md`: continuations only, identified by opaque generation IDs.
- `blind_identity_map.json`: separate mapping from opaque IDs to model identity
  and prompt identity.

Metrics are observations, not gates: unique-token proportion, distinct-1/2,
maximum same-token run, repeated 2/3/4-gram frequencies, exact short cycles,
and EOS position.

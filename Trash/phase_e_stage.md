# Phase E Stage 1

Stage 1 benchmark is bounded to 5,000 TinyStories source stories. It does not
run the 10,000-story resume or any Phase E Stage 2 work.

## Policy

- Source boundaries use `<|endoftext|>` delimiters.
- Stage selects first 5,000 source stories by corpus order.
- Each story is tokenized with existing `mrdl.language.tokenize` and capped at
  8 tokens including `<bos>` and `<eos>`. This bounds stage cost; it is not
  learned behavior or a corpus rewrite.
- Truncated stories replace final retained token with `<eos>`.
- Split is whole-story and deterministic: `sha256(split_seed:story_id) mod 5`,
  bucket `0` test and all other buckets train. Story IDs cannot overlap.
- Training observes only train stories. Test stories are evaluated only after
  training and promotion/controller updates.
- Both existing frozen embedding modes run. Transport remains pipeline default
  `cosine`; existing controller, PromotionStore, FULL/CLEAN lanes, VSA, gate,
  and generation code are not changed.
- Stage scoring uses existing composition depth 4. Reachability always runs
  exact d1..d4 frontiers, with one frontier computation reused for repeated
  source tokens; this changes measurement cost only, not learned behavior.
- Stage-only VSA dimension is 8 with deterministic seed 1729. VSA operations
  and implementation remain unchanged; dimension is recorded in results and
  checkpoints.

## Metrics

Primary reachability checks FULL exact propagation frontiers at d1, d2, d3,
and d4. It reports first target appearance, not decoder argmax:

- `never_reachable`
- `first_reachable_d_le_2`
- `first_reachable_d_ge_3`
- exact `first_reachable_d1` through `first_reachable_d4`

Secondary metrics report graph loss/accuracy and trigram loss/accuracy on the
non-overlapping test partition.

## Run

```bash
python -m benchmarks.phase_e_stage --stories 5000 --story-token-limit 8 \
  --output results/phase_e_stage1.json
```

The run writes one JSON result and one checkpoint per embedding mode. Each
checkpoint contains pipeline config, frozen embedding vectors, edge records
with M1/M2 levels and support/confidence/relation state, controller weights and
update count, causal/promotion counters, split manifest, corpus cursor, and a
round-trip restore check. The tool refuses `--stories > 5000`.

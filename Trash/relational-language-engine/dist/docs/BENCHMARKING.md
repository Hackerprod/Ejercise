# Benchmarking protocol

The built-in benchmark deliberately reports the two questions that can invalidate the architecture in practice:

1. Does the relational predictor beat or lose to a trivial trigram on held-out tokens?
2. What is the permanent runtime tax of FULL+CLEAN versus a single FULL lane?

```bash
rlmctl benchmark --config CONFIG --corpus EVAL.txt --samples 10000
```

Output fields:

- `relational_accuracy`: next-token top-1 from FULL at depth 1;
- `trigram_accuracy`: deterministic trigram→bigram→unigram backoff;
- `relational_eval_seconds`;
- `full_only_seconds`;
- `full_clean_seconds`;
- `dual_lane_runtime_ratio`;
- `trigram_states`.

## Correct protocol

- Use a held-out corpus not used to choose scoring weights.
- Freeze embeddings and M2 before the evaluation.
- Report vocabulary coverage and unknown-token rate separately.
- Run at least five times after one warm-up run.
- Record CPU model, vCPU count, RAM, filesystem and storage latency.
- Compare `parallel_lanes=true` and `false`.
- Do not combine training and inference timing.
- Report both quality and runtime; a memory bound alone does not justify permanent duplicate inference.

The benchmark baseline is intentionally strong enough to expose a graph/scoring regression but simple enough to audit. Passing it on a fixture is not evidence of broad language quality.

# Changelog

## 1.0.0

- Frozen Q8 mmap embedding store with checksummed versioned format.
- Dimensional token-to-token relation repository with physically separate M1 and M2 tiers.
- FULL/CLEAN lane isolation before Top-K selection.
- Pure scoring/composition interfaces and bounded beam search.
- Bounded replay DAG, pruning certificates and exact counterfactual audit with `UNKNOWN` fallback.
- Durable, recoverable M1→M2 promotion journal and TTL pinning.
- Exactly-once resumable corpus batches using deterministic IDs and WAL recovery.
- Independent bounded FULL/CLEAN execution without cross-lane barriers.
- Native CLI/HTTP service, Prometheus metrics, trigram baseline and runtime benchmark.
- Debug, Release/LTO, sanitizer, smoke, deployment and VPS operations assets.

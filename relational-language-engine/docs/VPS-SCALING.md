# Scaling on a 4 vCPU / 8 GB VPS

This profile prioritizes correctness and bounded memory. It does not assume that adding lanes is free: FULL+CLEAN performs two searches and the benchmark reports its runtime ratio against one lane.

## Recommended starting profile

Use `config/vps-4vcpu-8gb.toml` and keep the process count at one. The process owns one bounded worker pool; starting multiple service processes duplicates mmap page tables, caches, queues and compaction pressure.

- Four execution workers maximum, with a bounded queue.
- One trainer/maintenance stream at a time.
- Memory-mapped frozen embeddings; do not copy the full vector table per request.
- Compact Q8 vectors for embeddings and relations.
- Bounded beam/depth/candidates and bounded replay retention.
- WAL durability selected explicitly; stronger fsync policy costs throughput.
- HTTP bound to loopback, with Nginx handling TLS, body limits and request rate.

## Capacity method

Measure, do not extrapolate from a toy corpus:

1. Import the real frozen embedding table and record its mapped size.
2. Train a representative corpus shard and record M1/M2 bytes per million observations.
3. Run one-lane inference to establish the CPU baseline.
4. Run FULL+CLEAN with identical requests and record the dual/single runtime ratio.
5. Increase concurrency until p95 latency rises sharply or the queue begins rejecting/inline-running work.
6. Leave RAM for the kernel page cache; sustained swap activity invalidates the result.
7. Repeat after restart to distinguish warm-page-cache behavior from cold-start performance.

## Operational limits

Set limits from measured memory, not only from request count:

- maximum request body and token count;
- search depth, beam width and candidates per expansion;
- replay records/bytes and TTL;
- M1 records/bytes and TTL;
- queued jobs and open connections;
- WAL/snapshot disk budget and minimum free space.

The service should reject work before it crosses these limits. Linux OOM termination is not backpressure.

## CPU placement

On a four-vCPU VM, let the kernel schedule the single process first. Pinning can hurt when the provider migrates virtual CPUs or shares cores. Use `taskset` only after repeated measurements show lower variance. Do not enable more worker threads than runnable vCPUs for CPU-bound beam expansion.

## Storage

Place WAL, snapshots and promotion journal on a local filesystem with reliable rename and fsync semantics. Network/object storage may be used for backups, not as the live repository. Maintain enough free space for the current snapshot, the next snapshot and WAL growth during checkpointing.

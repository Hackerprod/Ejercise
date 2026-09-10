# Debugging and Fault Isolation

## Reproduce before inspecting

Use the smallest persisted inputs that still reproduce the issue:

- embedding file and checksum;
- repository epoch and M1/M2 snapshots;
- request token IDs and lane;
- search configuration;
- bounded replay trace;
- promotion/training WAL tail.

A bug report without the embedding checksum and repository epoch is not considered reproducible because candidate geometry may differ.

## Isolation order

1. **Embedding lookup:** verify token ID, dimension, checksum and normalized vector.
2. **Relation retrieval:** inspect raw per-tier candidates before scoring. CLEAN must contain no M1 provenance.
3. **Scorer:** feed a captured branch/candidate pair directly; compare finite score components.
4. **Composer:** verify dimensional output, lane preservation and parent index.
5. **Beam:** run with one worker and fixed limits; compare trace ordering and pruning certificates.
6. **Replay/audit:** replay the captured trace with and without the target edge. A cap must produce `UNKNOWN`.
7. **Promotion/persistence:** inspect PREPARE/COMMIT sequence and reopen the repository in a fresh process.
8. **Concurrency:** repeat the same deterministic query with `parallel_lanes=false`, then true, under a timeout.

## Failure classes

| Symptom | First module to inspect | Evidence |
|---|---|---|
| CLEAN result changes after an unpromoted observation | relation repository / controller | candidate provenance and controller mutability |
| FULL and CLEAN hang together but work separately | executor / lock order | thread dump and timeout stress test |
| promoted edge disappears after restart | promotion journal / M2 WAL | last valid CRC records |
| support doubles after restart | trainer batch dedupe | deterministic batch ID and committed set |
| audit promotes only at narrow beam | pruning certificate/reopen | original and reopened trace |
| memory grows exponentially with depth | replay representation | step/candidate counts versus `D·B·k` |
| expired edge remains forever | pin lifecycle / sweeper | active pin count and promotion state |

## Locking discipline

- Never wait on another lane while holding a repository, shard, replay or promotion lock.
- Repository visibility lock precedes tier/shard locks.
- Search copies candidate pages and releases repository locks before expansion.
- Promotion serialization precedes the atomic repository visibility transition.
- TTL collects candidate IDs first and does not call promotion state while a tier lock is held.
- The bounded executor runs a task inline on saturation to eliminate nested-submit starvation.

## Durable-state diagnosis

WAL readers validate magic/version, payload length and CRC. A torn final record is truncated to the last valid offset. A malformed record in the middle is corruption and must stop startup; it is not skipped. Keep the original files before attempting repair.

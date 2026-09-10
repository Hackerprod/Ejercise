# Failure model and recovery guarantees

| Failure point | Recovery behavior |
|---|---|
| partial WAL header/payload at EOF | tail truncated to prior complete record |
| CRC mismatch inside WAL | startup fails with `DATA_LOSS`; no speculative repair |
| crash after batch WAL, before trainer checkpoint | batch is reread; deterministic `BatchId` deduplicates it |
| crash after snapshot rename, before WAL reset | old WAL replays idempotently on top of snapshot |
| crash after promotion PREPARE | recovery completes M2 upsert, M1 erase and COMMIT |
| crash after M2 upsert, before M1 erase | same recovery path removes M1 duplicate |
| crash after M1 erase, before COMMIT | complete edge in PREPARE proves what belongs in M2; COMMIT is written |
| M2 content conflicts with PREPARE | startup fails with `DATA_LOSS` |
| M1 TTL while audit is active | edge is pinned; expiration skips it |
| graph mutation during search | search retries; result is marked unstable if three attempts cannot obtain a stable epoch |
| graph epoch changed since evidence | replay returns `UNKNOWN`; no promotion |
| omitted candidate cannot be certified | replay reopens; reaches `UNKNOWN` at configured hard limit |
| worker queue saturation | task executes in caller; no blocking enqueue and no worker starvation |
| second process opens same state | nonblocking process lock returns `UNAVAILABLE` |
| changed frozen embeddings | checksum mismatch invalidates checkpoint/replay |

## Durability modes

- `none`: fastest, intended only for disposable benchmarks; OS buffers may be lost.
- `data`: `fdatasync` records before acknowledgement; default for VPS.
- `full`: `fsync` file and directory metadata for the strongest atomic-replace guarantee.

`data` is the recommended balance for the 4-vCPU VPS. Use `full` when the filesystem/controller cannot preserve rename metadata without a directory sync.

## Backups

Run `rlmctl checkpoint` before copying state. `scripts/backup.sh` performs that sequence and creates a compressed archive. The embedding file should be backed up with the state because its checksum binds replay and trainer checkpoints.

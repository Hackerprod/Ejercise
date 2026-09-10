# Architecture

## 1. Immutable semantic substrate

`FrozenEmbeddingStore` is the only source of base token semantics. Its file is opened with `mmap(PROT_READ)` and exposes no write operation. Rows are Q8 with a per-row scale; the complete payload and vocabulary have independent checksums. Every replay trace records the embedding checksum, so an audit cannot silently replay against a different semantic substrate.

A relation is not a scalar adjacency entry. For source token `s` and target token `t`, its structural vector is:

```text
r(s,t) = Q8(normalize(E[t] - E[s]))
```

The trainer may aggregate confidence, support and evidence around that connection, but it never trains or rewrites `E`.

## 2. Evidence tiers and lane semantics

M1 and M2 use separate directories, snapshots and WALs:

```text
state/m1/snapshot.bin + changes.wal
state/m2/snapshot.bin + changes.wal
```

`RelationRepository::outgoing` obeys these definitions:

```text
FULL(s)  = TopK(M2(s) ∪ M1(s))
CLEAN(s) = TopK(M2(s))
```

CLEAN does not compute `TopK(M2 ∪ M1)` and remove M1 afterward. It never asks M1 for candidates. Therefore M1 cannot consume a CLEAN candidate slot, influence a pruning certificate or create a branch whose operator is later zeroed.

## 3. Branch composition

A `SpectralBranch` contains the last token, a bounded float state, cumulative score and the selected path. The default composer applies the dimensional connection directly:

```text
state' = normalize(
    state_decay  · state
  + relation_mix · r(s,t)
  + target_mix   · E[t]
)
```

The controller is immutable. It scores confidence/support and semantic consistency without updating shared parameters. This closes the control-parameter leakage path where FULL/M1 writes could otherwise alter future CLEAN behavior.

## 4. Search and exactness certificates

Beam search is bounded by depth `D`, beam width `B` and candidate count `k`. A trace stores at most `D·B·k` candidate records, not a recursively copied `k^D` proof tree.

Candidate retrieval is ordered by confidence. For a truncated page, the controller computes an upper bound for every omitted candidate using the first omitted confidence and maximum possible values of the remaining bounded features. The certificate is safe when that upper bound cannot beat the current beam cutoff.

During counterfactual replay:

1. replay with the original `k`;
2. if any certificate is unsafe, reopen with `2k`;
3. repeat up to `replay_reopen_limit`;
4. return `UNKNOWN` if exactness still cannot be proved.

No approximate replay can promote an edge.

## 5. Promotion transaction

Promotion holds an M1 pin across audit and commit. TTL cannot mark that edge as expiring while the pin exists.

The journal protocol is:

```text
PREPARE(tx, complete M1 edge)
M2.upsert(edge as M2)
M1.erase(edge)
COMMIT(tx)
```

Recovery replays any PREPARE without COMMIT. All storage actions are idempotent. A conflicting M2 vector is treated as data loss rather than overwritten.

## 6. Locking model

- process lock: one writer process owns a state directory;
- mutation mutex: serializes each tier's WAL and in-memory application;
- shard locks: short-lived adjacency map access only;
- index locks: released before acquiring shard locks on read paths;
- promotion commit mutex: serializes cross-tier commits;
- pin registry: independent from relation shard locks;
- inference: never holds a graph lock while scoring/composing;
- FULL and CLEAN: no shared barrier or cross-wait.

The lock order deliberately prevents the earlier `promote()` ↔ TTL cycle: TTL consults only the independent pin registry before entering an M1 erase; it never queries promotion state while holding a graph lock.

## 7. Persistence formats

Every persisted component is little-endian, versioned and bounded before allocation:

- embeddings: full-file CRC plus vocabulary hash;
- WAL: header CRC and payload CRC per record;
- relation snapshots: full-file CRC;
- replay records: validated `D·B·k` bound;
- trainer checkpoint: corpus hash, embedding checksum and CRC;
- promotion journal: complete edge in PREPARE.

A partial WAL tail is truncated to the last complete record. A CRC mismatch inside the valid prefix is a hard `DATA_LOSS` error.

# Relational Language Engine

Native C++20 implementation of the **Relational Language Architecture** described in `docs/REFERENCE-SPEC.md`. The non-negotiable core is preserved:

1. **Frozen embeddings** encode the base semantic knowledge of tokens.
2. **Directed token-to-token connections carry dimensional relation vectors** derived from those embeddings.
3. FULL/CLEAN lanes, M1/M2 evidence, replay, counterfactual audit, TTL and promotion are replaceable layers around that core. They cannot mutate the frozen embedding store.

This repository is an executable system rather than a notebook or simulated prototype: memory-mapped Q8 embeddings, physically separate evidence stores, CRC-protected WALs, atomic checkpoints, deterministic resumable corpus training, bounded replay, transactional promotion, a fixed worker pool, an HTTP service, metrics, tests and deployment files are included.

## Build and validate

Requirements: Linux, CMake 3.20+, a C++20 compiler and POSIX threads.

```bash
./scripts/build.sh
./scripts/smoke_test.sh
```

The build runs the invariant and concurrency suite with a 120-second timeout. The smoke test builds a frozen fixture, trains from a corpus, executes parallel FULL/CLEAN inference, compares against a trigram baseline, checkpoints and reopens the state.

## Quick start

```bash
mkdir -p data
./build/rlmctl embeddings build \
  --input fixtures/embeddings.txt \
  --output data/embeddings.rle

./build/rlmctl train \
  --config config/vps-4vcpu-8gb.toml \
  --corpus fixtures/corpus.txt

./build/rlmctl infer \
  --config config/vps-4vcpu-8gb.toml \
  --text "the cat" \
  --lane both \
  --depth 1
```

Start the service:

```bash
./build/rlmctl serve --config config/vps-4vcpu-8gb.toml
curl -sS http://127.0.0.1:9087/healthz
curl -sS -X POST 'http://127.0.0.1:9087/v1/infer?lane=both&depth=1' \
  -H 'Content-Type: text/plain' \
  --data-binary 'the cat'
```

Expose it through Nginx or another TLS reverse proxy; the native server intentionally speaks plain HTTP on the private interface.

## Architecture at a glance

```text
FrozenEmbeddingStore (read-only mmap, Q8)
        │
        ├── relation_vector(source,target) = normalize(E[target] - E[source])
        │
RelationRepository
   ├── state/m1/   provisional evidence, TTL-managed, promotion-pinnable
   └── state/m2/   promoted evidence, physically separate
        │
BeamSearch + immutable controller + vector composer
   ├── FULL enumerates M2 ∪ M1 before Top-K
   └── CLEAN enumerates M2 only; no M1 branch is instantiated
        │
ReplayStore: O(D·B·k) trace DAG + pruning certificates
        │
CounterfactualAuditor: exact/reopened replay or UNKNOWN
        │
PromotionManager: PREPARE → M2 upsert → M1 erase → COMMIT
```

The CLEAN lane never retrieves M1 and then filters it. This matters: post-Top-K filtering could allow a high-ranked M1 edge to displace a valid M2 edge and create the prohibited “ghost branch.” In this implementation, absence in CLEAN means **the branch does not exist**; no zero monomial is constructed.

## Modularity and replacement boundaries

Public interfaces are under `include/rlm/`:

| Interface | Default implementation | Safe replacement scope |
|---|---|---|
| `IEmbeddingStore` | `FrozenEmbeddingStore` | File format/quantization, while preserving immutable lookup and checksum semantics |
| `IRelationRepository` | `RelationRepository` | Sharding/index backend, while preserving physical lane semantics |
| `IScoringController` | `FrozenLinearController` | Candidate scoring; must remain immutable during inference |
| `IBranchComposer` | `VectorRelationComposer` | Relation-vector composition |
| `ISearchStrategy` | `BeamSearch` | Search policy and replay reopening contract |
| `IReplayStore` | `ReplayStore` | Replay persistence/backend |
| `ICounterfactualAuditor` | `CounterfactualAuditor` | Promotion policy; must return UNKNOWN when exactness cannot be established |

No implementation module reaches into another module's private state. Persistent formats are versioned and CRC-protected. Strict configuration rejects unknown keys instead of silently ignoring misspellings.

## Corpus training and crash semantics

Training is deterministic and resumable:

- corpus and frozen-embedding fingerprints bind a checkpoint to its inputs;
- each corpus batch receives a deterministic `BatchId`;
- all relation observations for that batch are stored as one WAL record;
- the M1 store records applied batch IDs;
- replaying a batch after a crash is idempotent;
- relation/replay/promotion WALs are flushed before the trainer checkpoint is atomically replaced.

A crash after the relation WAL append but before the trainer checkpoint causes the batch to be read again, but the persisted `BatchId` prevents support/confidence from being counted twice.

## FULL/CLEAN concurrency

`parallel_lanes=true` uses two independent tasks. There is no barrier, cross-lane lock or worker waiting on another worker. The bounded queue executes a task inline when saturated, which prevents nested queue starvation. Search reads do not hold repository locks while scoring or composing branches.

Running both lanes permanently costs more compute than one lane. Use the built-in benchmark to measure the real tax on the VPS:

```bash
./build/rlmctl benchmark \
  --config config/vps-4vcpu-8gb.toml \
  --corpus /path/to/evaluation.txt \
  --samples 1000
```

It reports relational accuracy, trigram accuracy, FULL-only runtime, FULL+CLEAN runtime and their ratio. The design does not claim that dual-lane inference is free.

## Operational commands

```text
embeddings build   Convert text float rows into the immutable mmap Q8 format
 doctor             Validate configuration, storage, lock ownership and embeddings
 train              Stream and resume corpus training
 infer              Run FULL, CLEAN or both
 inspect            Report dimensions, vocabulary, edge counts and epoch
 checkpoint         Atomically compact relation stores and rotate WALs
 expire             Apply M1 TTL without touching pinned promotions
 metrics            Print Prometheus exposition
 benchmark          Compare against trigram and measure dual-lane overhead
 serve              Run the bounded HTTP service
```

## Realistic boundary

The fixture proves mechanics, invariants and recovery; it is not a trained general-purpose language model. Language quality depends on the frozen embedding vocabulary, corpus coverage, scoring calibration and promoted M2 graph. “Production ready” here means the runtime and state transitions are engineered for deployment and failure recovery—not that a tiny fixture has general intelligence.

See:

- `docs/ARCHITECTURE.md`
- `docs/FAILURE-MODEL.md`
- `docs/OPERATIONS.md`
- `docs/API.md`
- `docs/BENCHMARKING.md`
- `docs/SPEC-COMPLIANCE.md`

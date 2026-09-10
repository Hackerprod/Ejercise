# Compliance Matrix

| Requirement from the design | Implementation boundary | Verification |
|---|---|---|
| Frozen embeddings are basic knowledge and never trained in-place | `IEmbeddingStore` / `FrozenEmbeddingStore`; read-only mmap Q8 format | Embedding round-trip, checksum and reopen tests |
| Token relationships are dimensional vectors, not scalar n-grams | `RelationEdge::direction`, embedding delta builder, vector composer | Relation/vector tests and corpus-training smoke |
| Core can survive replacement of outer algorithms | Interfaces for embeddings, relation repository, scorer, composer and search strategy | Dependency injection in `Engine` and compile-time interface boundaries |
| FULL may use M1+M2; CLEAN can use only M2 | Physically separate tier stores and lane-specific retrieval | CLEAN isolation/no-ghost-branch test |
| M1 absence in CLEAN is structural, not a zero monomial | CLEAN never creates an M1 candidate | Candidate provenance assertions |
| Evidence cannot be laundered by composition/folding/routing | Every branch has immutable lane/provenance and composition is lane-local | Mixed-lane rejection and CLEAN non-interference tests |
| Shared controller state cannot leak M1 into CLEAN | Immutable inference controller; training only mutates relation storage | Parallel lane stress and deterministic CLEAN result checks |
| Replay is O(G·D·B·k), not k^D | Bounded step/candidate DAG with parent indices | Replay bound test |
| Uncertified pruning cannot cause promotion | Exact replay reopens the affected step; cap yields UNKNOWN | Reopen/UNKNOWN audit tests |
| M1 cannot expire during audit/promotion | RAII pin registry checked by TTL sweeper | TTL pin test |
| Promotion survives process failure | PREPARE → durable M2 write → M1 removal → COMMIT journal | Promotion recovery test |
| Corpus resume is exactly-once at relation-batch level | Deterministic batch IDs + WAL deduplication + corpus checkpoint | Trainer resume/idempotence test |
| FULL/CLEAN concurrency must not deadlock | Independent jobs, no cross-lane barrier; bounded pool executes inline when saturated | Multi-client parallel stress with timeout |
| Dual-lane cost is measured explicitly | Benchmark reports dual-lane/single-lane runtime ratio | `rlmctl benchmark` and smoke script |
| Baseline comparison remains honest | Separate deterministic trigram/bigram/unigram baseline | Benchmark reports both accuracies |

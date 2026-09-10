# Module Replacement Contract

The frozen embedding store and the dimensional token-relation model are the architectural core. Everything else is replaceable through narrow interfaces, but replacements must preserve the contracts below.

## Stable boundaries

| Module | Interface | May read | May mutate | Forbidden |
|---|---|---|---|---|
| Embedding store | `IEmbeddingStore` | immutable vocabulary and Q8 vectors | nothing after open | training in place; returning mutable views |
| Relation repository | repository API | M1/M2 snapshots | relation tiers through WAL-backed transactions | exposing M1 to CLEAN; unjournaled durable writes |
| Scoring controller | `IScoringController` | branch state, frozen embeddings, edge metadata | local stack state only | parameters learned from M1 and later reused by CLEAN |
| Branch composer | `IBranchComposer` | a branch and one same-lane edge | newly returned branch | composing branches from different lanes |
| Search strategy | `ISearchStrategy` | immutable query snapshot | bounded trace/result objects | holding repository locks during beam expansion |
| Replay store | replay API | bounded trace DAG | replay WAL/retention state | materializing a symbolic `k^D` evidence tree |
| Auditor | audit API | FULL/CLEAN replay and target edge | verdict/evidence attachment | promoting on approximate or uncertified replay |
| Promotion manager | promotion API | M1 edge and durable journal | atomic M1→M2 transition | deleting M1 before durable M2 visibility |
| Trainer | trainer API | corpus and embeddings | relation observations and checkpoint | changing frozen embeddings; replaying a committed batch twice |

## Procedure for replacing a module

1. Implement the relevant interface in a separate translation unit. Do not modify consumers first.
2. Run the existing contract tests against both the current implementation and the replacement.
3. Compare deterministic traces on the same fixture: candidate IDs, lane, score, parent indices, pruning certificates and verdict.
4. Inject the replacement through `Engine` construction/configuration. Keep the original selectable until the differential gate passes.
5. For a persistent-format change, increment the format version and ship an offline migration. A reader must reject unknown versions and checksum mismatches.
6. Run Debug, Release, ASan/UBSan, crash-recovery and parallel-lane gates before making the replacement the default.

## Non-negotiable invariants

- `CLEAN candidates = topK(M2 candidates)`, never `filter_M2(topK(M1 ∪ M2))`.
- An M1 edge is absent from CLEAN; it is not represented by a zero operator.
- Scoring/composition are pure for inference. No controller cache may be trained by FULL and consumed by CLEAN.
- Trace storage is bounded by configured depth, beam width and candidates per node.
- Any audit that cannot exactly reproduce the decisive path returns `UNKNOWN`.
- A pinned edge cannot expire.
- Promotion has one externally visible semantic transition, even though its journal contains multiple durable steps.

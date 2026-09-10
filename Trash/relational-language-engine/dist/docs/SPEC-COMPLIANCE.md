# Specification compliance map

This map ties the implementation to the design constraints in `REFERENCE-SPEC.md` and the validation clarifications made before implementation.

| Requirement | Implementation |
|---|---|
| frozen embeddings are basic knowledge | read-only `FrozenEmbeddingStore`, mmap/Q8, checksum-bound replay/checkpoints |
| token connections are dimensional vectors | `RelationEdge::relation`; derived from normalized target-source embedding delta |
| no combinatorial evidence tree | replay stores at most `D·B·k` candidate records and validates that bound on load |
| FULL and CLEAN cannot wash evidence across lanes | repository candidate enumeration is lane-specific before Top-K; composer accepts only returned edges |
| CLEAN has no literal zero monomial for M1 | an M1 branch is absent from CLEAN; no branch/operator instance is created |
| M1 and M2 physically distinct | separate snapshot/WAL directories and independent tier stores |
| shared-controller leakage forbidden | controller and composer configurations are immutable during inference/training writes |
| counterfactual replay exactness | pruning certificates, geometric reopen, hard `UNKNOWN` limit |
| approximate audit must not promote | auditor promotes only exact cases; insufficient exactness is `UNKNOWN` |
| TTL must not orphan active promotion | `EdgePin` spans audit+commit; expiration atomically marks only unpinned edges |
| promotion survives process failure | complete edge in PREPARE journal; idempotent recovery to M2 then erase M1 |
| batch resume cannot double-count | deterministic batch IDs persisted with atomic M1 batch WAL records |
| parallel lanes must not deadlock | no barrier/cross-wait; bounded pool inline fallback; dedicated stress test |
| dual-lane compute cost must be measured | benchmark reports FULL+CLEAN runtime versus FULL-only |
| module can be replaced without rewriting core | explicit interfaces and pimpl boundaries under `include/rlm/` |

## Test coverage

`tests/test_main.cpp` covers:

- frozen embedding roundtrip/checksum;
- CLEAN physical isolation and no ghost branch;
- batch idempotence, WAL recovery and partial-tail repair;
- replay `D·B·k` bound and serialization;
- unfinished promotion journal recovery;
- TTL pin behavior;
- deterministic trainer resume;
- concurrent parallel FULL/CLEAN stress.

# OMEGA Hidden Cache Production Integration

Small end-to-end integration gate for the sealed hidden cache. This unit is
not a quality campaign and does not run 152/2000-update experiments.

Requirements enforced before cached training:

- fail-closed manifest, cache, teacher/tokenizer/dataset, and lm-head
  provenance verification;
- no DistilGPT2 transformer load or forward in production cached route;
- K1/K4 smoke schedule covering window 0 -> window 1, pair transition, and
  fresh-process checkpoint resume;
- bit-exact direct oracle comparison;
- cache preload memory/lock safety.

Real entrypoint requires explicit authorization and uses the already-built
`C:\omega_cache\teacher_hidden.fp32`; it never rebuilds that cache.

# ER64 Scaling Gate — Stage 2 Phase 1

This phase implements synthetic verification for the short ER64 cost gate. It
does not load the technical corpus or teacher and does not execute the 24 real
updates. Real execution requires separate authorization.

- Training ratio: `T_ER64_total / T_F_total`.
- Inference ratios: `ER64_time / F_time`, using `ms_per_token` for mode A and
  `mean_seconds_per_window` for mode B.
- Threshold: `1.25` for both timing gates.
- RSS anomaly: ER64 may not exceed F by more than `128 MiB` for either K.
- Reports use write-once, reread, self-hashed JSON.

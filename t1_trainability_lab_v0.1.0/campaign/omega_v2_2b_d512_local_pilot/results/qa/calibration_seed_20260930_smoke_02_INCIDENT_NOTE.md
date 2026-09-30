# V2-2B calibration smoke_02 - incident note (MD/327)

- terminal = V2_2B_CALIBRATION_QA_HARNESS_HOLD; scientific_verdict = NONE; official_attempt_consumed = false; held-out seeds untouched.
- hard_stop: cell D7_20260930_R4: TypeError: Object of type Tensor is not JSON serializable.
- root cause: metrics.py optimizer_state_canonical_sha256, param_groups comprehension used `else value` (variable leaked from the preceding state loop, a Tensor) instead of `else values`.
- D7 R4 completed 20 updates + L20; failure occurred while building the optimizer canonical hash.
- Evidence: D1,D2,D4,D5,D6 PASS; D3 A/B/C PASS 7/7 families within frozen thresholds; max peak_allocated 478,680,064 B; wall_gate 11.67 s.
- Source snapshot used: CALIBRATION_SOURCE_SNAPSHOT.json committed in bb64499 (sha256 F7FC5A98A870BE91151C0F0EF35C244BA8C4E7A27C1694FF46B093122B9DCABD).
- CALIBRATION_SMOKE_RESULT.json sha256 D2F710E298BC52008AB56746E415460760E0A08B171B8FF1233D48DE7B82C2F8.
- Launch: command.txt/stdout.log/stderr.log in calibration_seed_20260930_smoke_02_launch_logs (external to slot).

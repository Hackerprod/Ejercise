# PHASE B — OMEGA-ABSOLUTE-LM-CALIBRATION + MINIMUM-LANGUAGE-COMPETENCE instrument preflight

## Provenance

- Main HEAD: `e4d4322dc082b2c7de6e06fb0d3076234afbeffa`
- Parent: `6b61c04cb40642e8dcbf18ba1a0a5524e54346f3`
- Conversacion LN.md blob: `fc750a2ae9fb7d3933c54fb91f09ce5568d035ec`

## Data freeze

- Candidate stream: 4,096,000 target presentations; 3,869 unique chunks; 1,980,928 unique target positions.
- VALL: 452 chunks; 231,424 targets.
- Training-observed token types: 35,825.
- VALL unseen token types: 560; target occurrences: 1,198.

No OOV mapping, `<UNK>` substitution, or probability prior was specified. NLL scoring was stopped before any baseline NLL was produced.

- Phase-B manifest self-hash: `69a376d08efa7d02513fb1b4e865c02c589a0727a3ce27491aa5e69d81e6791d`
- Stream manifest self-hash: `6f50fed30fc2de6adc9e8466932dde7f637e35a4af431544033420a62a03ebc0`
- OOV report self-hash: `fe53771638570db6200373363d8bd00803ef9fe2173bf8e6f68f7fd24a038170`

## Not run due to the OOV hold

- Unigram, modified KN5, DistilGPT2 NLL/PPL: `NOT_COMPUTED`
- L0/L1 KN5/teacher scoring and instrument validity: `NOT_RUN`
- Degeneration reference/control, FP, U_FP, threshold_auto: `NOT_RUN`
- L0/L1 bank and leakage artifacts: `NOT_BUILT`
- OMEGA update>0 checkpoints: not loaded, scored, or generated
- Training updates: `0`; test split: not loaded

## Human panel

- H1: `user`
- H2: `TBD`
- H3: `TBD`
- `RATER_PANEL_READY`: `false`
- Pilot/main human evaluation: `NOT_RUN`
- Status artifact: `human_rater_status.json`

ABSOLUTE_CALIBRATION_HOLD

# OMEGA Fable closeout scope note — loss contract reconciliation

This note supplements, and does not edit or supersede, `campaign/omega_native_runtime_p2r0/OMEGA_FABLE_CLOSEOUT.md`. Fable remains formally CLOSED. Selected `be376` DLL SHA-256 remains `be37623d7e69b65551ad9c306092f328cc250576a56642fc34a0068df00600cc`; no DLL or recurrent kernel source was changed for this unit.

The discovered defect is in the historical Python loss callable that supplies recurrent backward gradients. It changes CE/KL mixture because CE is token-mean while KL is sequence-summed under `batchmean` on `[B,L,V]`. This is not evidence of a recurrent-kernel defect and does not justify reopening kernel optimization.

Disposition:

- Preserve historical Fable closeout, candidate/rollback DLLs, reports, source hashes, and all existing tests/fixtures.
- Historical performance measurements remain valid for the exact executed objective and workload. Do not claim those percentages as canonical-R1-loss performance; Step 3's separately authorized 8+32 recaracterization must establish that anew.
- Kernel correctness/equivalence evidence remains scoped to recorded tensor inputs and upstream gradients. Recurrent replay fixtures remain valid VJP fixtures for their recorded `G_R`; they are not fixtures of canonical R1 loss gradients.
- Historical optimizer/clipping effects are classified per consumer in `LOSS_CONSUMER_IMPACT_AUDIT.md`; no global checkpoint invalidation or global clearance is inferred.
- `strict_trajectory_proximity=FAIL_RECORDED` and `training_quality_equivalence=NOT_ESTABLISHED` remain unchanged. Hidden-state/logit cache provenance is not implicated.

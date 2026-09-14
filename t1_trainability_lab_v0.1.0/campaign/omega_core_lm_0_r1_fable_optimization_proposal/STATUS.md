# Status of this proposal (judge notes, not part of Fable's original analysis)

This directory contains an external optimization analysis of the OMEGA-CORE-LM-0-R1
CPU training/inference stack, produced by a separate agent ("Fable") that cloned this
public repository independently and analyzed `WorkspaceUpdateBlock`,
`OmegaCoreLM0R1Technical`, and the `omega_core_lm_0_r1_scientific_scoping_a` runner
at commit `afe69f7`.

## What the judge has independently verified so far

- **The `_batch_loss` batch=1 loop bug it identified is real.** The judge confirmed it
  by rereading the runner code, then empirically confirmed it live: the actual running
  SCOPE-A campaign process (PID 16976) was measured at ~37.9 s/update wall-clock against
  a T0-TR-P-verified ~10.65-11.53 s/update, a ~3.4x slowdown matching Fable's own
  6.69s-vs-2.03s (3.3x) benchmark almost exactly. The process was halted and the fix
  (real batch=8 forward instead of an 8x Python loop) was applied and independently
  verified by the judge: diff read in full, reference-loop equivalence test added and
  rerun (loss/gradients/states/post-optimizer-step parameters within atol/rtol 1e-5),
  9/9 tests pass, and a bounded real-teacher timing probe confirms 9.003132 s/update
  (shared_K1) / 10.245438 s/update (shared_K4) — back in line with the T0-TR-P budget.
  This fix is committed; the full SCOPE-A campaign has not yet been relaunched pending
  the user's explicit go-ahead.

## What has NOT yet been independently verified

- `omega_fast.py`'s deeper claims: fused RMSNorm/SDPA/QKV kernel, hoisted readout,
  logsumexp-rewritten distillation loss, the claimed 5.7e-7 gradient-relative-error
  equivalence, and the claimed 5.2x additional speedup on top of the batch fix. These
  numbers were measured by Fable on a different sandbox (Linux, 1 core) than the
  Windows/Zen 5 laptop used for every other measurement in this campaign.
- The teacher double-computation optimization (window 0 recomputed inside window 1's
  0:512 context) and the teacher-logit caching proposal.
- Everything in the GPU section (CUDA Graphs, TF32, vmap-stacked seeds) — not
  applicable until/unless a GPU confirmatory campaign is authorized.
- The architectural alternatives (linear-carry/minGRU-Mamba-2 style, DEER/Newton
  parallel-time solving, early-exit depth gating) are explicitly flagged by Fable
  itself as *new variants*, not equivalent rewrites of R1 -- they would change the
  model, not just its implementation, and are out of scope for anything already
  authorized.

This directory is preserved verbatim as received for Sol's and the judge's review.
Nothing in `omega_fast.py` has been wired into any authorized runner.

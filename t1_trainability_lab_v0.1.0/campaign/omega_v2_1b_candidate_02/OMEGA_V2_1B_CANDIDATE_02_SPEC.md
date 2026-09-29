# OMEGA-V2-1b — Candidate 02 KQ-only

Authority: `MD/306.md` §§4–14, 26–27 and the freeze clarification in `MD/307.md`. Candidate 01 and V2-1 attempt_01/02 are immutable evidence. This isolated unit is the one permitted KQ-only candidate_02; it does not run A/B/C or attempt_03.

## Candidate 02 changes

- Q4 output-row tile 4 × slot tile 2. Each output-row tile is dequantized once into transient per-worker scratch, then reuses each input vector across four output rows.
- Q/K/V are processed behind one worker-pool dispatch; W_gate/W_up are processed behind one dispatch. W_O and W_down each retain their own dispatch.
- Attention Q·K uses AVX2/FMA dot products; attention-value mixing uses AVX2/FMA across features.
- SwiGLU computes SiLU, gate×up, and the Hadamard product in one AVX2 pass. The exp approximation is range-reduced to |r|≤ln(2)/2 and uses a degree-7 Taylor polynomial; its documented approximation bound is checked densely against scalar `std::exp` over [-80,80].
- Maximum candidate scratch is 49,152 bytes/worker (≤64 KiB). Persistent model storage remains Q4 packed weights plus FP16 group-32 scales only.

## KQ and project gate

Use attempt_02 CPU-set IDs/shard weights, the same V2-0 FP32 source tensors (seed 20260929), Q4 pack/scales, fixed inputs, and four P-cores. Measure P_H0(4/16), P_FULL(4/16), the d512,m8,K4 PyTorch FP32 control and candidate Q4 control. Compare against the already measured candidate-01 P_FMA roofline baseline. All timing is QPC; each timed cell uses 10 warmups/31 samples. Do not invoke A/B/C helpers or calculate the m4 residency rho while selecting the candidate.

Candidate 02 passes only with correctness, E_Q4≥0.25, E_FULL(4)≥0.60, E_FULL(16)≥0.60, and S_native≥1.20. A pass freezes this candidate and permits one attempt_03. A scientific pass with S_native<1.20 is terminal project speed-gate failure; any scientific KQ failure is `V2_1B_KERNEL_QUALIFICATION_FAIL`. In either STOP case, do not run attempt_03.

## Run sequence

After committing candidate_02 sources on `main`, use the Python 3.14 interpreter with the sealed PyTorch 2.11 CPU environment. First run `py -3.14 scripts/run_candidate02_kq.py --preflight-only`. This verifies V2-0, candidate_01, and attempt_02 seals, builds Release executables, and runs the d32/m4/K2 offline scalar-reference plus dense exp-approximation check without opening a measurement result directory. Only after preflight passes, run `py -3.14 scripts/run_candidate02_kq.py` once. Its sole measurement slot is `candidate_02/run_01`; never rerun or overwrite that slot.

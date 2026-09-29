# OMEGA-CORE-LM-0-R1 Pilot Execution Readiness

Local readiness artifacts only. This unit proves recovery and statistical
bookkeeping on CPU; it does not perform scientific updates, GPU training, or
spend.

## Quick Path

1. Run `python -m pytest t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_pilot_execution_readiness -q`.
2. Run `python t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_pilot_execution_readiness/provenance.py --root .`.
3. Reserve `python t1_trainability_lab_v0.1.0/campaign/omega_core_lm_0_r1_pilot_execution_readiness/verify_image_contents.py IMAGE_REF` for an authorized image check.
4. Treat image verification, remote measurement, image rebuild, platform selection, and execution as pending.

## Resume Semantics

Resume is valid only from a verified scientific checkpoint. A checkpoint is
canonical only when its update is a multiple of 500 and the resumed run keeps
the same seed, logical run, image digest, GPU backend, code, configuration, and
data. Restore all of the following, not model weights alone:

- model state
- AdamW optimizer state
- update boundary
- Torch RNG state
- data position
- checkpoint hash and provenance

Ledger recovery is append-only. A resume creates a new `execution_segment` and
must include `parent_checkpoint_hash`; it must never silently reuse or rewrite
an earlier segment. Interrupted evidence remains preserved in the ledger but is
non-canonical until verified against a checkpoint. Example: an interruption at
update 12,347 selects canonical update 12,000, discards updates 12,001 through
12,347 as non-canonical evidence, and resumes at update 12,000 with window 0.
Update 500 satisfies `500 mod 2 = 0`; the next canonical update window is 0.
The fixture models update labels as the next update at a checkpoint boundary,
which makes this resume boundary explicit.

## Paired CI

For each of five paired seeds, compute exactly:

```text
d_i = NLL_K1,i - NLL_K4,i
mean(d) +/- t(0.975, 4) * sample_std(d) / sqrt(5)
```

The fixed critical value is `t(0.975, 4) = 2.7764451051977987`. This is a
paired five-delta confidence interval, not ten independent NLL observations.
The scientific gate remains unchanged: compare the mean depth delta with
`0.05 nats/token`; CI calculation and gate evaluation are separate operations.

## GPU Selection And Parallelism

GPU selection uses cost per completed update, not NLL:

```text
cost_per_completed_update = hourly_price_usd * steady_state_seconds_per_update / 3600
```

Campaign cost must use this exact accounting boundary:

```text
C_campaign = t_update × p_GPU × N_updates + evaluation + I/O + overhead
```

Collect hourly price, allocatable capacity, steady-state seconds/update,
p95 update time, startup time, peak GPU memory, utilization, throttling or
failure rate, and reproducibility flags. NLL is a scientific result and is not
a GPU-selection criterion.

Execution shape is one run = one GPU = one process. Five identical GPUs run
the five paired seeds for one variant in parallel; `shared_K1`, `shared_K4`,
and `untied_K4` run as three sequential variant batches. Existing reference
timings are 3.532970, 8.927244, and 8.934740 seconds/update respectively;
these are prior reference measurements, not new work in this unit. One seed
totals 118.9 hours / 4.95 days of compute; serial execution totals 24.8 days.
Parallel seeds use five identical GPUs and have the same total compute cost as
serial execution. There is no 15-GPU policy.

```text
T_total = 5 * [20,000*t_shared_K1 + 20,000*t_shared_K4 + 20,000*t_untied_K4]
```

Use measured wall time for `t_variant` when platform benchmarking is
authorized. CPU timing is pending because it was not measured here.

## Evaluation And Storage Budget

Validation runs occur at update 0 and every 500 updates through 20,000:
`41 evaluations/run * 15 runs = 615 evaluations`. Evaluation budget is
separate from training time and cost.

There are 615 scientific checkpoint boundaries, but not 615 full AdamW
copies. Scientific checkpoints store the immutable selection evidence needed
for evaluation. Recovery checkpoints, written according to interruption and
operational policy, store model, optimizer, RNG, data position, provenance,
and parent hash. `q4t3-vol` 50GB target sizing and byte sizing for both classes
are pending a remote read-only measurement; no remote measurement was
performed here.

## Provenance

`provenance.py` verifies expected SHA-256 values for the runner,
`requirements.lock`, `model.py`, and the immutable Dockerfile snapshot. Frozen
identity values are `source_commit =
2901e58834bc0d8d8225ccaf51ca136901468698` and `R1_identity =
269a4d79b5a1e6df8c230962e2b8c9e237095f18`; `R1_identity` is the frozen code
commit, not a human-readable label.

`Dockerfile.frozen.6f2ddb87d7485411abc5d77084caa749c0b3a02531724a5cedc4902a4d260432`
remains byte-identical and is the expected Dockerfile hash authority. The
live build Dockerfile now adds only the verifier and this snapshot to the
image, so the live Dockerfile is intentionally not treated as that frozen
snapshot.

## Container Verification Protocol

`verify_image_contents.py IMAGE_REF` invokes:

```text
docker run --rm --entrypoint python IMAGE_REF /opt/omega/verify_image_contents.py --in-image
```

The command computes hashes from paths inside the image for the runner,
`requirements.lock`, `model.py`, and the immutable Dockerfile snapshot. The
verifier reads and checks the exact `source_commit` and `R1_identity` values
from the in-image `readiness_report.json` manifest. `--entrypoint python` is
required because the normal image entrypoint starts `sshd`.

Run this protocol twice when authorized:

- Before publishing: verify the newly built image, then record its immutable digest.
- After pulling by digest and clean boot: run the same verifier against that digest and require every check to pass.

This local readiness report does not claim either image check executed. No
Docker command, image rebuild, pull, or remote measurement was performed here.

## Out Of Scope And Pending

- No scientific updates, GPU training, or spend.
- Remote q4t3-vol sizing measurement.
- q4t3-vol 50GB sizing confirmation.
- Authorized GPU benchmark and cost-per-update selection.
- Derived image rebuild and immutable image digest verification.
- Pilot execution authorization and all 15 scientific runs.

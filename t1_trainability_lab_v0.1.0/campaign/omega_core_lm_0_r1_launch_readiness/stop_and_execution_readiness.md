# Stop and Execution Readiness

**Status: prepared only.** No Pod was created, no spend occurred, and CUDA preflight was not executed.

## Review Path

1. Confirm image publication and later spend authorization.
2. Request exactly one NVIDIA L4 in SECURE at US-MO-2 with existing `q4t3-vol` (`6yrppoqpkz`) attached without expansion.
3. Observe provider allocation and host facts before execution.
4. Run preflight only under explicit authorization:

```text
python omega_nominal_microbatch_runner.py preflight --cache-root <cache-root> --output-root <output-root> --device cuda --updates N
```

The command is recorded as prepared, not executed. R1 architecture, CE/KL losses, effective batch, and microbatch recipe are unchanged: effective batch 8, physical microbatch 2, four microbatches per update, with the existing shared/untied R1 variants.

## Budget Controller Authority

`budget_controller.py` is simulation-only. `intent=request_stop` records an intent; it does not stop billing. An authorized external operator must request the provider stop and independently verify provider state. No provider lifecycle action is available from this repository.

## Stop Versus Terminate

| Action | Meaning | Storage consequence |
| --- | --- | --- |
| Stop | Stop compute while preserving restartability subject to provider semantics. | Provider storage behavior must be verified; network volume remains retained. |
| Terminate | Permanently remove Pod. | Pod-local/container storage is lost; network volume remains retained. |

RunPod documents container disks as temporary storage and network volumes as persistent, portable storage. The exact post-stop state still requires provider-state verification; no Pod exists in this preparation unit. Termination is not equivalent to stop: it removes the Pod and its Pod-local storage, while the existing network volume remains an independently retained resource.

`q4t3-vol` must never be deleted under either flow. No replacement Pod or volume mutation is allowed by this readiness package.

## Blockers

- Derived image is published at `ghcr.io/hackerprod/omega-core-lm-0-r1:a3d7b47` with digest matching local identity `sha256:b6834274f18d9dceaa2f5983a9ebfa099d13a2265a6e607303ca1324284ce4e7`; private visibility and separate-read pull remain pending.
- L4 secure stock is Low and allocatable count is not exposed by read-only results.
- Exact host CPU/model and host RAM are post-allocation facts.

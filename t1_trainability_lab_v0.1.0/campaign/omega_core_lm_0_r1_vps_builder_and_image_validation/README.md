# OMEGA CORE-LM-0 R1 VPS builder and image validation

This artifact records one authorized CPU-only VPS builder unit. It must not
create RunPod Pods, attach `q4t3-vol`, spend on GPU runtime, or publish to a
public registry.

Build source is a filtered context containing only Dockerfile inputs:
requirements, Docker helper files, OMEGA preparation code/tests, the frozen
technical preflight script and its design-audit dependency, and the minimal
`t1_trainability` model package surface. No checkout metadata, credentials,
checkpoints, corpus, caches, historical JSONL, or persistent volume content is
copied.

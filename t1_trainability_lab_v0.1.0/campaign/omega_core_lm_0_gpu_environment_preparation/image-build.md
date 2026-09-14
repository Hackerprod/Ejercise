# Derived image build and verification

Build context is `t1_trainability_lab_v0.1.0`; no build may run on the
persistent volume. The Dockerfile preserves the CUDA base digest, installs the
locked dependencies, copies OMEGA code under `/opt/omega`, runs `pip check`,
checks import/version compatibility, and runs both synthetic suites during the
image build.

```bash
docker build \
  --file campaign/omega_core_lm_0_gpu_environment_preparation/Dockerfile \
  --tag <authorized-private-image>:omega-core-lm-0-r1 \
  .
```

After build, run `docker/verify-image.sh <authorized-private-image>:omega-core-lm-0-r1`.
It starts two fresh containers from the same image, generates absent SSH host
keys at runtime, and requires distinct fingerprints. The image must not be
published to a public registry without explicit visibility authorization.

`/workspace/omega` is an ordinary directory, not a Docker `VOLUME`; RunPod may
mount `q4t3-vol` there without a nested Docker volume hiding the parent mount.

#!/usr/bin/env bash
set -u

context=/root/omega-core-lm-0-r1-vps-builder/context
log=/root/omega-core-lm-0-r1-vps-builder/logs/build.log
status=/root/omega-core-lm-0-r1-vps-builder/logs/build.status

cd "$context"
exec > >(tee "$log") 2>&1
set -o pipefail
echo "build_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker buildx build \
  --builder omega-core-lm-0-r1 \
  --progress=plain \
  --file campaign/omega_core_lm_0_gpu_environment_preparation/Dockerfile \
  --tag omega-core-lm-0-r1:validated \
  --load \
  .
rc=$?
printf 'exitCode=%s\n' "$rc" > "$status"
echo "build_finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) exitCode=$rc"
exit "$rc"

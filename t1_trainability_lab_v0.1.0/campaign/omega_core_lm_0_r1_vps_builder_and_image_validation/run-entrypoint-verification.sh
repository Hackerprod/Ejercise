#!/usr/bin/env bash
set -u

log=/root/omega-core-lm-0-r1-vps-builder/logs/entrypoint-verification.log
status=/root/omega-core-lm-0-r1-vps-builder/logs/entrypoint-verification.status
exec > >(tee "$log") 2>&1
set -o pipefail
echo "verification_started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
bash /root/omega-core-lm-0-r1-vps-builder/context/campaign/omega_core_lm_0_gpu_environment_preparation/docker/verify-image.sh omega-core-lm-0-r1:validated
rc=$?
printf 'exitCode=%s\n' "$rc" > "$status"
echo "verification_finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ) exitCode=$rc"
exit "$rc"

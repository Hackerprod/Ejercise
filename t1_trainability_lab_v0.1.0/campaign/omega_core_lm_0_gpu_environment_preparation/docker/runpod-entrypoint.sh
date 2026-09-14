#!/usr/bin/env bash
set -euo pipefail

mkdir -p /run/sshd /root/.ssh /workspace/omega
chmod 0700 /root/.ssh

# RunPod-managed templates commonly inject PUBLIC_KEY. Preserve any key already
# installed by the platform and append only a non-empty managed-key variable.
append_key() {
    local key="$1"
    if [[ -n "${key}" ]] && ! grep -Fqx -- "${key}" /root/.ssh/authorized_keys 2>/dev/null; then
        printf '%s\n' "${key}" >> /root/.ssh/authorized_keys
    fi
}

touch /root/.ssh/authorized_keys
append_key "${PUBLIC_KEY:-}"
append_key "${RUNPOD_SSH_PUBLIC_KEY:-}"
chmod 0600 /root/.ssh/authorized_keys

ssh-keygen -A
exec /usr/sbin/sshd -D -e

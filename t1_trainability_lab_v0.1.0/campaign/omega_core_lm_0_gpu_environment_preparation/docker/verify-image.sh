#!/usr/bin/env bash
set -euo pipefail

image="${1:?usage: verify-image.sh IMAGE}"
root="$(mktemp -d)"
offline_container=""
container_a=""
container_b=""
tmp_key="$root/test-host-key"
workspace_a="$root/workspace-a"
workspace_b="$root/workspace-b"
mkdir -p "$workspace_a" "$workspace_b"
cleanup() {
    test -z "$container_a" || docker rm -f "$container_a" >/dev/null 2>&1 || true
    test -z "$container_b" || docker rm -f "$container_b" >/dev/null 2>&1 || true
    test -z "$offline_container" || docker rm "$offline_container" >/dev/null 2>&1 || true
    rm -rf "$root"
}
trap cleanup EXIT

echo "image=$image"

# Offline image check: create does not start the image process. Export the
# filesystem and reject any private host key baked into the image layer.
offline_container="$(docker create --entrypoint /bin/sh "$image" -c 'true')"
docker export "$offline_container" -o "$root/image.tar"
if tar -tf "$root/image.tar" | grep -Eq '(^|/)etc/ssh/ssh_host_[^/]+_key$'; then
    echo "baked-ssh-host-private-key: FAIL"
    exit 1
fi
echo "offline-image-host-key-check: PASS"

ssh-keygen -t ed25519 -N '' -f "$tmp_key" >/dev/null
public_key="$(cat "$tmp_key.pub")"

# Normal startup: preserve image ENTRYPOINT (tini -> omega entrypoint -> sshd).
# Docker allocates high loopback-only host ports; no public port or firewall is
# changed. Temporary workspace mounts never reference the prepared network volume.
container_a="$(docker run -d --publish 127.0.0.1::22 --env PUBLIC_KEY="$public_key" --volume "$workspace_a:/workspace" "$image")"
container_b="$(docker run -d --publish 127.0.0.1::22 --env RUNPOD_SSH_PUBLIC_KEY="$public_key" --volume "$workspace_b:/workspace" "$image")"

wait_for_ssh() {
    local container="$1"
    local port
    port="$(docker port "$container" 22/tcp | sed -E 's/.*://')"
    for _ in $(seq 1 30); do
        if docker exec "$container" sh -c 'test -s /etc/ssh/ssh_host_ed25519_key && pgrep -x sshd >/dev/null' 2>/dev/null; then
            ssh-keyscan -T 2 -p "$port" 127.0.0.1 > "$root/$container.keyscan" 2>/dev/null
            if test -s "$root/$container.keyscan"; then
                printf '%s %s\n' "$container" "$port"
                return 0
            fi
        fi
        sleep 1
    done
    echo "SSH did not become ready for $container" >&2
    docker logs "$container" >&2 || true
    return 1
}

read -r _ port_a < <(wait_for_ssh "$container_a")
read -r _ port_b < <(wait_for_ssh "$container_b")
fingerprint_a="$(ssh-keygen -lf "$root/$container_a.keyscan" | awk 'NR == 1 {print $2}')"
fingerprint_b="$(ssh-keygen -lf "$root/$container_b.keyscan" | awk 'NR == 1 {print $2}')"
test -n "$fingerprint_a" && test -n "$fingerprint_b"
test "$fingerprint_a" != "$fingerprint_b"
echo "normal-entrypoint-ssh-host-key-isolation: PASS ($fingerprint_a != $fingerprint_b)"

for container in "$container_a" "$container_b"; do
    docker exec "$container" sh -c 'test -f /opt/omega/omega_nominal_microbatch_runner.py && test -d /workspace/omega && ! mountpoint -q /workspace/omega'
done
echo "workspace-overlay-and-code-access: PASS"

#!/usr/bin/env bash
set -euo pipefail

image="${1:?usage: verify-image.sh IMAGE}"
tmp_a="$(mktemp)"
tmp_b="$(mktemp)"
cleanup() {
    rm -f "$tmp_a" "$tmp_b"
}
trap cleanup EXIT

container_check='set -eu
rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub
ssh-keygen -A >/dev/null
test -n "$(find /etc/ssh -maxdepth 1 -type f -name "ssh_host_*_key" -print -quit)"
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key | awk "{print \$2}"
'

docker run --rm --entrypoint /bin/bash "$image" -c "$container_check" >"$tmp_a"
docker run --rm --entrypoint /bin/bash "$image" -c "$container_check" >"$tmp_b"

test -s "$tmp_a"
test -s "$tmp_b"
test "$(cat "$tmp_a")" != "$(cat "$tmp_b")"
echo "host-key-isolation: PASS"

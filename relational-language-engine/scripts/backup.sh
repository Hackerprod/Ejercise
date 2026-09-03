#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:?usage: backup.sh CONFIG [DESTINATION]}"
DESTINATION="${2:-./backups}"
RLMCTL="${RLMCTL:-rlmctl}"
STATE_DIR="$(awk -F'=' '/^[[:space:]]*state_dir[[:space:]]*=/{gsub(/[ \t\"]/,"",$2); print $2; exit}' "$CONFIG")"
if [[ -z "$STATE_DIR" ]]; then echo "state_dir not found" >&2; exit 1; fi
if [[ "$STATE_DIR" != /* ]]; then STATE_DIR="$(cd "$(dirname "$CONFIG")" && pwd)/$STATE_DIR"; fi
"$RLMCTL" checkpoint --config "$CONFIG"
mkdir -p "$DESTINATION"
archive="$DESTINATION/rlm-state-$(date -u +%Y%m%dT%H%M%SZ).tar.zst"
tar --zstd -C "$(dirname "$STATE_DIR")" -cf "$archive" "$(basename "$STATE_DIR")"
echo "$archive"

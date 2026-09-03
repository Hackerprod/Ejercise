#!/usr/bin/env bash
set -Eeuo pipefail

CONFIG="${1:-/etc/mrdl/mrdl.ini}"
ROOT="${2:-/var/backups/mrdl}"
MRDL_BIN="${MRDL_BIN:-mrdl}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$ROOT"
DEST="$(mktemp -d "$ROOT/mrdl-$STAMP-XXXXXX")"
cleanup_on_error() { rm -rf -- "$DEST"; }
trap cleanup_on_error ERR INT TERM
"$MRDL_BIN" backup --config "$CONFIG" --output "$DEST" --json
trap - ERR INT TERM
printf '%s\n' "$DEST"

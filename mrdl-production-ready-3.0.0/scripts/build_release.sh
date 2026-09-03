#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
JOBS="${JOBS:-$(nproc)}"
if (( JOBS > 4 )); then JOBS=4; fi

cd "$ROOT_DIR"
cmake --preset release-vps
cmake --build --preset release-vps --parallel "$JOBS"
ctest --preset release-vps
printf 'Release build ready: %s\n' "$ROOT_DIR/build/release/mrdl"

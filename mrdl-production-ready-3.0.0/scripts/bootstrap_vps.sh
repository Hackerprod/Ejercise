#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ${EUID:-$(id -u)} -eq 0 ]]; then
  SUDO=()
else
  command -v sudo >/dev/null || { echo 'sudo is required when not running as root' >&2; exit 1; }
  SUDO=(sudo)
fi

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y build-essential cmake ninja-build libsqlite3-dev sqlite3 ca-certificates
"$ROOT_DIR/scripts/build_release.sh"

if [[ "${INSTALL:-0}" == "1" ]]; then
  "${SUDO[@]}" cmake --install "$ROOT_DIR/build/release" --prefix "${PREFIX:-/usr/local}"
fi

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-/opt/relational-language-engine}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
"$ROOT/scripts/build.sh"
sudo install -d -m 0750 "$PREFIX/bin" "$PREFIX/config" "$PREFIX/data" "$PREFIX/log"
sudo install -m 0755 "$BUILD_DIR/rlmctl" "$PREFIX/bin/rlmctl"
sudo install -m 0640 "$ROOT/config/vps-4vcpu-8gb.toml" "$PREFIX/config/production.toml"
echo "Install frozen embeddings at $PREFIX/data/embeddings.rle before starting the service."

#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT/config/production.toml}"
CORPUS="${2:-$ROOT/fixtures/corpus.txt}"
SAMPLES="${SAMPLES:-1000}"
"${RLMCTL:-$ROOT/build/rlmctl}" benchmark --config "$CONFIG" --corpus "$CORPUS" --samples "$SAMPLES"

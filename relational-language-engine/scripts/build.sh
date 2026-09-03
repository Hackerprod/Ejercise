#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DRLM_BUILD_TESTS=ON
cmake --build "$BUILD_DIR" --parallel "${JOBS:-$(nproc)}"
ctest --test-dir "$BUILD_DIR" --output-on-failure

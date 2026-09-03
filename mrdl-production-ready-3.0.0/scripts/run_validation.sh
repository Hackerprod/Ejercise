#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT_DIR/artifacts/validation}"
JOBS="${JOBS:-$(nproc)}"
if (( JOBS > 4 )); then JOBS=4; fi
mkdir -p "$OUT_DIR"

cd "$ROOT_DIR"
{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  uname -a
  command -v c++ >/dev/null && c++ --version | head -1
  cmake --version | head -1
  sqlite3 --version 2>/dev/null || true
} | tee "$OUT_DIR/environment.txt"

cmake --preset debug-sanitize | tee "$OUT_DIR/configure-debug.log"
cmake --build --preset debug-sanitize --parallel "$JOBS" | tee "$OUT_DIR/build-debug.log"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1 \
  "$ROOT_DIR/build/debug/mrdl_tests" | tee "$OUT_DIR/tests-sanitized.log"

cmake --preset release-vps | tee "$OUT_DIR/configure-release.log"
cmake --build --preset release-vps --parallel "$JOBS" | tee "$OUT_DIR/build-release.log"
ctest --preset release-vps | tee "$OUT_DIR/ctest-release.log"

"$ROOT_DIR/build/release/mrdl_bench" --quick --suite dual \
  | tee "$OUT_DIR/benchmark-dual.csv"
"$ROOT_DIR/build/release/mrdl_bench" --quick --suite dual --json \
  | tee "$OUT_DIR/benchmark-dual.jsonl"
"$ROOT_DIR/build/release/mrdl_bench" --quick --suite monomial --json \
  | tee "$OUT_DIR/benchmark-monomial.jsonl"

"$ROOT_DIR/scripts/smoke_test.sh" "$ROOT_DIR/build/release/mrdl" \
  | tee "$OUT_DIR/smoke-test.log"

(
  cd "$OUT_DIR"
  rm -f SHA256SUMS
  find . -maxdepth 1 -type f ! -name SHA256SUMS -printf '%f\n' | sort | xargs -r sha256sum > SHA256SUMS
)
printf 'Validation artifacts: %s\n' "$OUT_DIR"

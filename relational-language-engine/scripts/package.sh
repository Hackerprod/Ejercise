#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT=${OUT_DIR:-$(dirname "$ROOT")}
NAME=relational-language-engine-production-ready
cd "$(dirname "$ROOT")"
rm -f "$OUT/$NAME.zip" "$OUT/$NAME.tar.gz"
zip -qr "$OUT/$NAME.zip" "$(basename "$ROOT")" \
  -x "$(basename "$ROOT")/build/*" \
     "$(basename "$ROOT")/build-debug/*" \
     "$(basename "$ROOT")/build-release/*" \
     "$(basename "$ROOT")/build-sanitize/*" \
     "$(basename "$ROOT")/runtime/*" \
     "$(basename "$ROOT")/.git/*"
tar --exclude="$(basename "$ROOT")/build" \
    --exclude="$(basename "$ROOT")/build-debug" \
    --exclude="$(basename "$ROOT")/build-release" \
    --exclude="$(basename "$ROOT")/build-sanitize" \
    --exclude="$(basename "$ROOT")/runtime" \
    --exclude="$(basename "$ROOT")/.git" \
    -czf "$OUT/$NAME.tar.gz" "$(basename "$ROOT")"
sha256sum "$OUT/$NAME.zip" "$OUT/$NAME.tar.gz" > "$OUT/relational-language-engine-CHECKSUMS.txt"
printf '%s\n' "$OUT/$NAME.zip" "$OUT/$NAME.tar.gz"

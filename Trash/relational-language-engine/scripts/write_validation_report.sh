#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
status() {
  local name=$1
  [[ -f ${name}_OK ]] && { printf PASS; return; }
  [[ -f ${name}_FAILED ]] && { printf FAIL; return; }
  printf NOT_RUN
}
BUILD_STATUS=$(status BUILD)
INSTALL_STATUS=$(status INSTALL)
SMOKE_STATUS=$(status SMOKE)
RELEASE_STATUS=$(status RELEASE)
SANITIZER_STATUS=$(status SANITIZER)

required=(
  CMakeLists.txt CMakePresets.json Makefile VERSION README.md
  include/rlm/embedding_store.hpp include/rlm/relation_store.hpp
  include/rlm/search.hpp include/rlm/replay.hpp include/rlm/audit.hpp
  include/rlm/promotion.hpp include/rlm/trainer.hpp include/rlm/engine.hpp
  src/embedding_store.cpp src/relation_store.cpp src/search.cpp src/replay.cpp
  src/audit.cpp src/promotion.cpp src/trainer.cpp src/engine.cpp
  apps/rlmctl.cpp tests/test_main.cpp
  config/production.toml config/vps-4vcpu-8gb.toml
  docs/REFERENCE-SPEC.md docs/ARCHITECTURE.md docs/FAILURE-MODEL.md
  docs/MODULE-REPLACEMENT.md docs/DEBUGGING.md docs/VPS-SCALING.md
  scripts/build.sh scripts/smoke_test.sh scripts/validate.sh
  deploy/relational-language.service
)
missing=()
for f in "${required[@]}"; do [[ -f $f ]] || missing+=("$f"); done
if ((${#missing[@]} == 0)); then INVENTORY=PASS; else INVENTORY=FAIL; fi
TODO_COUNT=$(grep -RInE --exclude-dir='build*' --exclude='*.log' --exclude='REFERENCE-SPEC.md' '(TODO|FIXME|HACK|XXX)' include src apps tests scripts config deploy docs 2>/dev/null | wc -l | tr -d ' ')
NOW=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
COMPILER=$(c++ --version 2>/dev/null | head -n1 || true)
CMAKE=$(cmake --version 2>/dev/null | head -n1 || true)
NINJA=$(ninja --version 2>/dev/null || true)
KERNEL=$(uname -srmo 2>/dev/null || true)

cat > docs/VALIDATION-REPORT.md <<REPORT
# Validation Report

Generated: ${NOW}

| Gate | Result | Scope |
|---|---:|---|
| Required project inventory | **${INVENTORY}** | Core modules, tests, deployment and operations assets |
| Debug compile + CTest | **${BUILD_STATUS}** | Native integration and functional/concurrency unit tests |
| Clean-prefix CMake install | **${INSTALL_STATUS}** | Installability outside the source tree |
| End-to-end CLI smoke | **${SMOKE_STATUS}** | Import, train, persist, infer, inspect, checkpoint, expire and benchmark |
| Release + LTO + CTest | **${RELEASE_STATUS}** | Optimized-build parity |
| ASan + UBSan + CTest | **${SANITIZER_STATUS}** | Memory/lifetime/undefined-behaviour instrumentation |

Unresolved source markers outside the copied reference specification: **${TODO_COUNT}**.

## Test-covered architectural guarantees

1. Frozen embedding round-trip, version/checksum validation and immutable reopening.
2. Dimensional token relationships and deterministic relation observations.
3. Physical CLEAN isolation before Top-K; no M1 ghost branch or zero-operator substitute.
4. Lane-local scoring/composition and immutable inference controller state.
5. WAL replay, torn-tail repair and deterministic batch idempotence.
6. Replay storage bounded by depth × beam × candidates rather than a symbolic k^D tree.
7. Exact audit reopen and `UNKNOWN` when configured limits prevent proof.
8. Recoverable M1→M2 promotion and TTL pinning during audit/promotion.
9. Resume-safe streaming corpus training.
10. Independent FULL/CLEAN execution under multi-client stress and timeout.
11. Dual-lane versus single-lane runtime measurement and relational versus trigram accuracy.

## Toolchain

- ${COMPILER}
- ${CMAKE}
- Ninja ${NINJA}
- ${KERNEL}

## Raw logs

`BUILD.log`, `INSTALL.log`, `SMOKE.log`, `RELEASE.log`, and `SANITIZER.log` are retained at the repository root.
REPORT

if ((${#missing[@]})); then
  printf '\n## Missing paths\n\n' >> docs/VALIDATION-REPORT.md
  printf -- '- `%s`\n' "${missing[@]}" >> docs/VALIDATION-REPORT.md
fi

for spec in 'BUILD:BUILD.log:Debug/CTest' 'INSTALL:INSTALL.log:Install' 'SMOKE:SMOKE.log:Smoke' 'RELEASE:RELEASE.log:Release' 'SANITIZER:SANITIZER.log:Sanitizer'; do
  IFS=: read -r marker log title <<<"$spec"
  if [[ -f ${marker}_FAILED && -f $log ]]; then
    {
      printf '\n## %s failure tail\n\n```text\n' "$title"
      tail -n 160 "$log"
      printf '\n```\n'
    } >> docs/VALIDATION-REPORT.md
  fi
done

cat > /mnt/data/relational-language-engine-validation-status.txt <<STATUS
inventory=${INVENTORY}
debug=${BUILD_STATUS}
install=${INSTALL_STATUS}
smoke=${SMOKE_STATUS}
release=${RELEASE_STATUS}
sanitizer=${SANITIZER_STATUS}
source_markers=${TODO_COUNT}
generated=${NOW}
STATUS

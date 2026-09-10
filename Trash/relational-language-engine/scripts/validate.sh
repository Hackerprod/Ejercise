#!/usr/bin/env bash
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
JOBS=${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf 2)}
(( JOBS > 4 )) && JOBS=4
SKIP_SANITIZERS=0
[[ ${1:-} == --skip-sanitizers ]] && SKIP_SANITIZERS=1

run_gate() {
  local name=$1 log=$2; shift 2
  rm -f "${name}_OK" "${name}_FAILED"
  "$@" >"$log" 2>&1
  local rc=$?
  if (( rc == 0 )); then touch "${name}_OK"; else touch "${name}_FAILED"; fi
  return 0
}

run_gate BUILD BUILD.log bash -lc "
  cmake -S . -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRLM_ENABLE_TESTS=ON -DRLM_ENABLE_SANITIZERS=OFF &&
  cmake --build build-debug --parallel ${JOBS} &&
  timeout 240s ctest --test-dir build-debug --output-on-failure
"

if [[ -f BUILD_OK ]]; then
  run_gate INSTALL INSTALL.log bash -lc '
    rm -rf /tmp/rlm-install &&
    cmake --install build-debug --prefix /tmp/rlm-install &&
    test -x /tmp/rlm-install/bin/rlmctl &&
    /tmp/rlm-install/bin/rlmctl --help
  '
  run_gate SMOKE SMOKE.log timeout 360s bash scripts/smoke_test.sh build-debug
else
  printf 'SKIPPED: debug gate failed.\n' > INSTALL.log; touch INSTALL_FAILED
  printf 'SKIPPED: debug gate failed.\n' > SMOKE.log; touch SMOKE_FAILED
fi

run_gate RELEASE RELEASE.log bash -lc "
  cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release -DRLM_ENABLE_TESTS=ON -DRLM_ENABLE_LTO=ON &&
  cmake --build build-release --parallel ${JOBS} &&
  timeout 240s ctest --test-dir build-release --output-on-failure
"

if (( SKIP_SANITIZERS == 0 )); then
  run_gate SANITIZER SANITIZER.log bash -lc "
    cmake -S . -B build-sanitize -G Ninja -DCMAKE_BUILD_TYPE=Debug -DRLM_ENABLE_TESTS=ON -DRLM_ENABLE_SANITIZERS=ON &&
    cmake --build build-sanitize --parallel ${JOBS} &&
    ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 timeout 360s ctest --test-dir build-sanitize --output-on-failure
  "
else
  rm -f SANITIZER_OK SANITIZER_FAILED
  printf 'SKIPPED by request.\n' > SANITIZER.log
fi

bash scripts/write_validation_report.sh

failed=0
for gate in BUILD INSTALL SMOKE RELEASE; do [[ -f ${gate}_OK ]] || failed=1; done
if (( SKIP_SANITIZERS == 0 )); then [[ -f SANITIZER_OK ]] || failed=1; fi
exit "$failed"

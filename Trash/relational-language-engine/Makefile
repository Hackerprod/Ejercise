SHELL := /usr/bin/env bash
.DEFAULT_GOAL := debug

.PHONY: debug release sanitize test smoke install clean package preflight

debug:
	cmake --preset debug
	cmake --build --preset debug

release:
	cmake --preset release
	cmake --build --preset release

sanitize:
	cmake --preset sanitize
	cmake --build --preset sanitize
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 ctest --preset sanitize

test: debug
	ctest --preset debug

smoke: debug
	./scripts/smoke_test.sh build/debug

install: release
	cmake --install build/release --prefix "$${PREFIX:-/usr/local}"

preflight:
	./scripts/preflight_vps.sh "$${DATA_DIR:-./runtime}"

clean:
	rm -rf build build-debug build-release build-sanitize runtime/smoke

package:
	./scripts/package.sh

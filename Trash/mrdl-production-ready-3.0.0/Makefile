.PHONY: configure build test sanitize validate bench smoke install package clean

JOBS ?= 4
PREFIX ?= /usr/local

configure:
	cmake --preset release-vps

build: configure
	cmake --build --preset release-vps --parallel $(JOBS)

test: build
	ctest --preset release-vps

sanitize:
	cmake --preset debug-sanitize
	cmake --build --preset debug-sanitize --parallel $(JOBS)
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 ./build/debug/mrdl_tests

validate:
	./scripts/run_validation.sh

bench: build
	./build/release/mrdl_bench --quick --suite all

smoke: build
	./scripts/smoke_test.sh ./build/release/mrdl

install: build
	cmake --install build/release --prefix $(PREFIX)

package: build
	cpack --config build/release/CPackConfig.cmake

clean:
	rm -rf build artifacts

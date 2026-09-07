#!/bin/sh
# Local check runner.
#
#   sh tools/ci.sh            all stages
#   sh tools/ci.sh native     one stage: native | make | search | python
#
# Requires: cmake >= 3.20, ctest, a C11/C++17 toolchain, libcurl (search),
# python3 (tooling and standard-library tests).

set -eu

STAGE="${1:-all}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

run_native() {
    cmake -S native -B build/native -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DBUILD_TESTING=ON
    cmake --build build/native --config "$BUILD_TYPE"
    ctest --test-dir build/native -C "$BUILD_TYPE" --output-on-failure
}

run_make() {
    make -C native lib
    make -C native test
}

run_search() {
    cmake -S search -B build/search -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    cmake --build build/search --config "$BUILD_TYPE"
    ctest --test-dir build/search -C "$BUILD_TYPE" --output-on-failure
}

run_python() {
    python3 -m compileall -q tools neutral scripts src tests
    python3 -m unittest discover -s tests -p 'test_*.py'
    python3 tools/tests/test_convert_gguf.py
    python3 tools/tests/test_kquants.py
}

case "$STAGE" in
    native) run_native ;;
    make)   run_make ;;
    search) run_search ;;
    python) run_python ;;
    all)
        run_native
        run_make
        run_search
        run_python
        ;;
    *)
        echo "unknown stage: $STAGE (native|make|search|python|all)" >&2
        exit 2
        ;;
esac

echo "ok: $STAGE"

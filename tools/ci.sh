#!/bin/sh
set -eu

STAGE="${1:-all}"
BUILD_TYPE="${BUILD_TYPE:-Release}"

run_native() {
    cmake -S native -B build/native -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DBUILD_TESTING=ON
    cmake --build build/native --config "$BUILD_TYPE"
    ctest --test-dir build/native -C "$BUILD_TYPE" --output-on-failure
}

run_search() {
    cmake -S search -B build/search -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    cmake --build build/search --config "$BUILD_TYPE"
    ctest --test-dir build/search -C "$BUILD_TYPE" --output-on-failure
}

run_storage() {
    cmake -S storage -B build/storage -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    cmake --build build/storage --config "$BUILD_TYPE"
    ctest --test-dir build/storage -C "$BUILD_TYPE" --output-on-failure
}

run_python() {
    python3 -m compileall -q tools scripts src tests
    python3 -m unittest discover -s tests -p 'test_*.py'
    python3 tools/tests/test_convert_gguf.py
    python3 tools/tests/test_kquants.py
}

case "$STAGE" in
    native)  run_native ;;
    search)  run_search ;;
    storage) run_storage ;;
    python)  run_python ;;
    all)
        run_native
        run_search
        run_storage
        run_python
        ;;
    *)
        echo "unknown stage: $STAGE (native|search|storage|python|all)" >&2
        exit 2
        ;;
esac

echo "ok: $STAGE"

#!/usr/bin/env sh
set -eu

usage() {
    echo "usage: scripts/verify.sh [release|sanitize]" >&2
}

MODE="${1:-release}"
case "$MODE" in
    release|sanitize) ;;
    *)
        usage
        exit 2
        ;;
esac

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
JOBS="${NIYAH_BUILD_JOBS:-2}"

case "$MODE" in
    release)
        BUILD_DIR="${NIYAH_BUILD_DIR:-$ROOT_DIR/build/verify-release}"
        cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
            -DNIYAH_BUILD_TESTS=ON \
            -DCMAKE_BUILD_TYPE=Release
        cmake --build "$BUILD_DIR" --config Release --parallel "$JOBS"
        ctest --test-dir "$BUILD_DIR" -C Release --output-on-failure
        ;;
    sanitize)
        BUILD_DIR="${NIYAH_BUILD_DIR:-$ROOT_DIR/build/verify-sanitize}"
        cmake -S "$ROOT_DIR" -B "$BUILD_DIR" \
            -DNIYAH_BUILD_TESTS=ON \
            -DNIYAH_ENABLE_SANITIZERS=ON \
            -DCMAKE_BUILD_TYPE=Debug
        cmake --build "$BUILD_DIR" --config Debug --parallel "$JOBS"
        ASAN_OPTIONS="${ASAN_OPTIONS:-halt_on_error=1}" \
        UBSAN_OPTIONS="${UBSAN_OPTIONS:-halt_on_error=1:print_stacktrace=1}" \
            ctest --test-dir "$BUILD_DIR" -C Debug --output-on-failure
        ;;
esac

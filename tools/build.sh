#!/usr/bin/env sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

printf '%s\n' '[1/2] regular repository checks'
sh tools/ci.sh all

printf '%s\n' '[2/2] native ASan + UBSan'
cmake     -S native     -B build/native-sanitize     -DNIYAH_SANITIZE=ON     -DBUILD_TESTING=ON     -DCMAKE_BUILD_TYPE=Debug

cmake --build build/native-sanitize --parallel

ASAN_OPTIONS=detect_leaks=1:abort_on_error=1 UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1 ctest     --test-dir build/native-sanitize     --output-on-failure

printf '%s\n' 'BUILD=PASS'

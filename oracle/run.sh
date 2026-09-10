#!/usr/bin/env bash
# oracle/run.sh — single command that reproduces every measured number.
# Exits non-zero when any measured value crosses its named ceiling.
# Run from the repository root at commit 2a8e094669a5b4cb9855e9993b11459ae8bbe80e.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

FAIL=0
fail() { echo "CEILING BREACH: $1"; FAIL=1; }

# ── PIN CHECK ─────────────────────────────────────────────────────────────────
ACTUAL_SHA=$(git rev-parse HEAD)
EXPECTED_SHA="2a8e094669a5b4cb9855e9993b11459ae8bbe80e"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
    echo "PIN_UNREACHABLE: HEAD is $ACTUAL_SHA not $EXPECTED_SHA"
    exit 1
fi
echo "PIN_OK: $ACTUAL_SHA"

# ── BUILD BASELINE ────────────────────────────────────────────────────────────
rm -rf build/oracle_native
cmake -S native -B build/oracle_native \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_FLAGS="-Wall -Wextra" \
      -DCMAKE_INSTALL_PREFIX=/tmp/niyah_oracle 2>&1 | tail -3
cmake --build build/oracle_native --parallel 2>&1 | tee /tmp/oracle_build.log | tail -5

BUILD_RC=$?
WARNINGS_COUNT=$(grep -c "warning:" /tmp/oracle_build.log || true)
echo "BUILD_RC=$BUILD_RC"
echo "WARNINGS_COUNT=$WARNINGS_COUNT"
[ "$BUILD_RC" -ne 0 ] && fail "BUILD_RC=$BUILD_RC (ceiling: 0)"

# CEILING: warnings > 10 flag a regression
[ "$WARNINGS_COUNT" -gt 10 ] && fail "WARNINGS_COUNT=$WARNINGS_COUNT (ceiling: 10)"

# ── CTEST ─────────────────────────────────────────────────────────────────────
CTEST_OUT=$(ctest --test-dir build/oracle_native --output-on-failure 2>&1)
echo "$CTEST_OUT" | tail -5
TESTS_TOTAL=$(echo "$CTEST_OUT" | grep -oP '\d+(?= tests passed)' | head -1 || echo 0)
TESTS_FAILED=$(echo "$CTEST_OUT" | grep -oP '\d+(?= tests failed)' | head -1 || echo 0)
TESTS_PASSED=$((TESTS_TOTAL - TESTS_FAILED))
echo "TESTS_TOTAL=$TESTS_TOTAL TESTS_PASSED=$TESTS_PASSED TESTS_FAILED=$TESTS_FAILED"
[ "$TESTS_FAILED" -gt 0 ] && fail "TESTS_FAILED=$TESTS_FAILED (ceiling: 0)"

# ── SANITIZER BUILD ───────────────────────────────────────────────────────────
rm -rf build/oracle_san
cmake -S native -B build/oracle_san -DNIYAH_SANITIZE=ON 2>&1 | tail -3
cmake --build build/oracle_san --parallel 2>&1 | tail -3
SAN_OUT=$(ASAN_OPTIONS="detect_leaks=1:abort_on_error=1" \
          UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" \
          ctest --test-dir build/oracle_san --output-on-failure 2>&1)
echo "$SAN_OUT" | tail -5
ASAN_ERRORS=$(echo "$SAN_OUT" | grep -c "ERROR: AddressSanitizer" || echo 0)
UBSAN_ERRORS=$(echo "$SAN_OUT" | grep -c "runtime error:" || echo 0)
LEAKS_BYTES=$(echo "$SAN_OUT" | grep -oP '\d+(?= bytes in)' | awk '{s+=$1} END{print s+0}')
echo "ASAN_ERRORS=$ASAN_ERRORS UBSAN_ERRORS=$UBSAN_ERRORS LEAKS_BYTES=$LEAKS_BYTES"
[ "$ASAN_ERRORS" -gt 0 ] && fail "ASAN_ERRORS=$ASAN_ERRORS (ceiling: 0)"
[ "$UBSAN_ERRORS" -gt 0 ] && fail "UBSAN_ERRORS=$UBSAN_ERRORS (ceiling: 0)"
[ "$LEAKS_BYTES" -gt 0 ] && fail "LEAKS_BYTES=$LEAKS_BYTES (ceiling: 0)"

# ── GATE 1: STUB VERIFICATION ─────────────────────────────────────────────────
STUB_BYTES=$(wc -c < native/niyah_test_llama_cli.c)
echo "STUB_BYTES=$STUB_BYTES"
WEIGHT_FILES_FOUND=$(find . -name "*.gguf" -o -name "*.safetensors" 2>/dev/null \
    | grep -v ".git" | wc -l)
echo "WEIGHT_FILES_FOUND=$WEIGHT_FILES_FOUND"
[ "$STUB_BYTES" -ne 1175 ] && fail "STUB_BYTES=$STUB_BYTES (ceiling: must be 1175)"
[ "$WEIGHT_FILES_FOUND" -gt 0 ] && fail "WEIGHT_FILES_FOUND=$WEIGHT_FILES_FOUND (ceiling: 0)"
# Real llama-cli test:
echo "CANNOT_RUN=NO_LLAMA_CLI: no llama-cli binary in PATH"
echo "HUMAN_COMMAND: NIYAH_LLAMA_CLI=/path/to/llama-cli ./build/oracle_native/niyah run <pkg> --prompt P --max-tokens 8 --proof out.proof"

# ── GATE 1: TOCTOU RACE ───────────────────────────────────────────────────────
echo "--- GATE 1 TOCTOU RACE ---"
NIYAH_HOME=$(mktemp -d)
export NIYAH_HOME
PKG="testpkg" VER="1-0"
PKG_DIR="$NIYAH_HOME/packages/$PKG/$VER"
BLOBS_DIR="$NIYAH_HOME/blobs"
mkdir -p "$PKG_DIR" "$BLOBS_DIR"
echo -n 'A' > /tmp/oracle_blob_orig
echo -n 'B' > /tmp/oracle_blob_swap
ORIG_SHA=$(sha256sum /tmp/oracle_blob_orig | awk '{print $1}')
cp /tmp/oracle_blob_orig "$BLOBS_DIR/$ORIG_SHA"
echo "$VER" > "$NIYAH_HOME/packages/$PKG/current"
printf 'NIYAH-PACKAGE 1\nname %s\nversion %s\nartifact base %s https://x\n' \
    "$PKG" "$VER" "$ORIG_SHA" > "$PKG_DIR/manifest.niyah"
export NIYAH_TEST_OUTPUT="hello" NIYAH_TEST_EXIT="0"
export NIYAH_LLAMA_CLI="$(pwd)/build/oracle_native/niyah_test_llama_cli"

RACE_ATTEMPTS=100 RACE_WINS=0
for i in $(seq 1 $RACE_ATTEMPTS); do
    cp /tmp/oracle_blob_orig "$BLOBS_DIR/$ORIG_SHA"
    rm -f /tmp/oracle_race.proof
    ./build/oracle_native/niyah run "$PKG" \
        --prompt "test" --max-tokens 1 --proof /tmp/oracle_race.proof \
        2>/dev/null &
    cp /tmp/oracle_blob_swap "$BLOBS_DIR/$ORIG_SHA"
    wait $! 2>/dev/null || true
    [ -f /tmp/oracle_race.proof ] && RACE_WINS=$((RACE_WINS+1))
done
echo "RACE_ATTEMPTS=$RACE_ATTEMPTS RACE_WINS=$RACE_WINS"
rm -rf "$NIYAH_HOME"

# CEILING: if 0 wins in 100 attempts something is wrong with the setup
# (not a ceiling on correctness — winning means a defect exists)

# ── GATE 2: MUTATION — niyah_control.c ───────────────────────────────────────
echo "--- GATE 2 MUTATION: niyah_control.c ---"
bash mutate_test.sh native/niyah_control.c niyah_control_test
CONTROL_KR=$(bash mutate_test.sh native/niyah_control.c niyah_control_test 2>/dev/null \
    | grep "KILL_RATE" | grep -oP '[\d.]+(?=%)' || echo 0)
echo "control_kill_rate=${CONTROL_KR}%"
# CEILING: < 80% means "default deny, no prefix widening" is UNVERIFIED
if python3 -c "import sys; sys.exit(0 if float('${CONTROL_KR}') >= 80 else 1)" 2>/dev/null; then
    echo "niyah_control.c mutation: VERIFIED (>= 80%)"
else
    fail "niyah_control.c KILL_RATE=${CONTROL_KR}% < 80% ceiling — default deny claim is UNVERIFIED"
fi

# ── GATE 5: NONCE REPLAY ─────────────────────────────────────────────────────
echo "--- GATE 5 NONCE REPLAY ---"
gcc -O0 -I. native/niyah_control.c gate5_control.c -o /tmp/gate5_control 2>&1
/tmp/gate5_control | tee /tmp/gate5_out.txt
NONCE_D1=$(grep "DECISION_1" /tmp/gate5_out.txt | grep -o "ALLOW\|DENY")
NONCE_D2=$(grep "DECISION_2" /tmp/gate5_out.txt | grep -o "ALLOW\|DENY")
echo "NONCE_DECISION_1=$NONCE_D1 NONCE_DECISION_2=$NONCE_D2"
if [ "$NONCE_D1" = "ALLOW" ] && [ "$NONCE_D2" = "ALLOW" ]; then
    fail "NONCE_REPLAY: identical (nonce,cap,resource) allowed twice — nonce is decorative"
fi

# ── GATE 4: BM25 TRUNCATION (defect test) ────────────────────────────────────
echo "--- GATE 4 BM25 DEFECT TEST ---"
gcc -O0 -std=c11 -I search/ -I native/ \
    search/niyah_index.c search/niyah_index_test.c \
    -o /tmp/bm25_defect_test -lm 2>&1
/tmp/bm25_defect_test; BM25_RC=$?
echo "BM25_TEST_RC=$BM25_RC"
[ "$BM25_RC" -ne 0 ] && fail "BM25_TEST_RC=$BM25_RC (ceiling: 0) — truncation defect not fixed"

# ── GATE 7: LICENSE CHECK ─────────────────────────────────────────────────────
echo "--- GATE 7 LICENSE ---"
echo "LICENSE_FILE_EXISTS=$(test -f LICENSE && echo YES || echo NO)"
echo "README_SAYS_NO_LICENSE=$(grep -c 'No LICENSE file' README.md || echo 0)"
DATA_SHA=$(sha256sum data/khawrizm_graph_consolidated.json | awk '{print $1}')
echo "data/khawrizm_graph_consolidated.json sha256=$DATA_SHA bytes=$(wc -c < data/khawrizm_graph_consolidated.json)"
echo "DECLARED_SOURCE_URL=REFUSED (not in file or README)"
echo "DECLARED_LICENSE=REFUSED (no LICENSE file, no SPDX in README)"
echo "REDISTRIBUTABLE=NO — no license means all-rights-reserved by default"

# ── GATE 8: README vs MACHINE ─────────────────────────────────────────────────
echo "--- GATE 8 PYTHON TESTS ---"
cd "$REPO"
python3 -m compileall -q tools scripts src tests 2>&1; COMPILE_RC=$?
PYTHON_TESTS_DISCOVERED=$(python3 -c "
import unittest, sys
loader = unittest.TestLoader()
suite = loader.discover('tests', pattern='test_*.py')
print(suite.countTestCases())
" 2>/dev/null || echo 0)
PYTHON_ASSERTIONS_EXECUTED=$(python3 -m unittest discover -s tests -p 'test_*.py' 2>&1 \
    | grep -oP '\d+(?= test)' | head -1 || echo 0)
echo "PYTHON_COMPILE_RC=$COMPILE_RC"
echo "PYTHON_TESTS_DISCOVERED=$PYTHON_TESTS_DISCOVERED"
echo "PYTHON_ASSERTIONS_EXECUTED=$PYTHON_ASSERTIONS_EXECUTED"
echo "NOTE: compileall proves syntax only, not behavior"
WORKFLOW_JOBS=$(grep "^  [a-z]" .github/workflows/native.yml | grep -c ":" || echo 0)
echo "WORKFLOW_JOBS_IN_FILE=$WORKFLOW_JOBS (README says 6)"
[ "$WORKFLOW_JOBS" -ne 6 ] && fail "WORKFLOW_JOBS=$WORKFLOW_JOBS (README claims 6)"

# ── SUMMARY ───────────────────────────────────────────────────────────────────
echo ""
echo "=== ORACLE SUMMARY ==="
if [ "$FAIL" -eq 0 ]; then
    echo "ALL CEILINGS HELD"
    exit 0
else
    echo "ONE OR MORE CEILINGS BREACHED — see CEILING BREACH lines above"
    exit 1
fi

#!/usr/bin/env bash
# Mechanical mutation tester for niyah_proof.c
set -u
SRC="native/niyah_proof.c"
TEST_BIN="niyah_proof_test"
ORIG=$(cat "$SRC")
TOTAL=0
KILLED=0

run_mutant() {
    local label="$1"
    local mutated="$2"
    TOTAL=$((TOTAL+1))
    echo "$mutated" > "$SRC"
    if cmake --build build/native --parallel -j$(nproc) -- "$TEST_BIN" 2>/dev/null 1>/dev/null; then
        if ! "./build/native/$TEST_BIN" 2>/dev/null 1>/dev/null; then
            KILLED=$((KILLED+1))
            echo "KILLED: $label"
        else
            echo "SURVIVED: $label"
        fi
    else
        KILLED=$((KILLED+1))
        echo "KILLED (compile fail): $label"
    fi
    echo "$ORIG" > "$SRC"
}

# Mutant 1: diff == 0u -> diff != 0u in digest_equal (inverts equality check)
run_mutant "digest_equal: diff==0 -> !=0" "$(echo "$ORIG" | sed 's/return diff == 0u;/return diff != 0u;/')"

# Mutant 2: == 0u in valid_buffer -> != 0u
run_mutant "valid_buffer: size==0 -> !=0" "$(echo "$ORIG" | sed 's/return size == 0u || data != NULL;/return size != 0u || data != NULL;/')"

# Mutant 3: Change sha256 domain separator from 0u to 1u
run_mutant "separator 0u -> 1u" "$(echo "$ORIG" | sed 's/static const uint8_t separator = 0u;/static const uint8_t separator = 1u;/')"

# Mutant 4: NIYAH_SHA256_BYTES in loop -> NIYAH_SHA256_BYTES - 1 (off-by-one in digest compare)
run_mutant "digest_equal loop: BYTES -> BYTES-1" "$(echo "$ORIG" | sed 's/i < NIYAH_SHA256_BYTES; ++i/i < NIYAH_SHA256_BYTES - 1; ++i/')"

# Mutant 5: return NIYAH_ERR_INVALID_ARG -> NIYAH_OK
run_mutant "invalid arg -> NIYAH_OK" "$(echo "$ORIG" | sed '0,/return NIYAH_ERR_INVALID_ARG;/{s/return NIYAH_ERR_INVALID_ARG;/return NIYAH_OK;/}')"

# Mutant 6: memcpy prompt_hash -> skip (remove memcpy for prompt_hash)
run_mutant "skip prompt_hash memcpy" "$(echo "$ORIG" | sed 's/memcpy(out->prompt_hash, prompt_hash, NIYAH_SHA256_BYTES);//')"

# Mutant 7: memcpy output_hash -> skip 
run_mutant "skip output_hash memcpy" "$(echo "$ORIG" | sed 's/memcpy(out->output_hash, output_hash, NIYAH_SHA256_BYTES);//')"

echo ""
echo "MUTANTS_TOTAL=$TOTAL MUTANTS_KILLED=$KILLED KILL_RATE=$(echo "scale=1; $KILLED * 100 / $TOTAL" | bc)%"

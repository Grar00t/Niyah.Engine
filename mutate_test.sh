#!/usr/bin/env bash
# Mechanical mutation tester for C files
# Usage: ./mutate_test.sh <source_file> <test_binary_name>
set -u
SRC="$1"
TEST_BIN="$2"
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

# Mutant 1: == -> != in capability_is_single_known: `capability == 0u` -> `capability != 0u`
run_mutant "cap==0 -> cap!=0" "$(echo "$ORIG" | sed 's/capability == 0u || /capability != 0u || /')"

# Mutant 2: != -> == in (capability & ~known) != 0u -> (capability & ~known) == 0u
run_mutant "(cap&~known)!=0 -> ==0" "$(echo "$ORIG" | sed 's/(capability & ~known) != 0u/(capability \& ~known) == 0u/')"

# Mutant 3: == -> != in single-bit check: (capability & (capability - 1u)) == 0u -> != 0u
run_mutant "single-bit ==0 -> !=0" "$(echo "$ORIG" | sed 's/(capability & (capability - 1u)) == 0u/(capability \& (capability - 1u)) != 0u/')"

# Mutant 4: Delete single-bit check line entirely (replace with return 1)
run_mutant "delete single-bit check" "$(echo "$ORIG" | sed 's/return (capability & (capability - 1u)) == 0u;/return 1;/')"

# Mutant 5: >= -> > in length bound: ++length >= NIYAH_CONTROL_RESOURCE_MAX -> ++length > NIYAH_CONTROL_RESOURCE_MAX
run_mutant "length >= -> >" "$(echo "$ORIG" | sed 's/++length >= NIYAH_CONTROL_RESOURCE_MAX/++length > NIYAH_CONTROL_RESOURCE_MAX/')"

# Mutant 6: Delete length bound entirely (comment out)
run_mutant "delete length bound" "$(echo "$ORIG" | sed 's/if (++length >= NIYAH_CONTROL_RESOURCE_MAX) {/if (0) {/')"

# Mutant 7: && -> || in resource_id_valid isalnum check
run_mutant "isalnum && -> ||" "$(echo "$ORIG" | sed "s/!(isalnum(\*p) || \*p == '_' || \*p == '-' || \*p == '.' || \*p == ':')/!isalnum(*p) \&\& *p != '_' \&\& *p != '-' \&\& *p != '.' \&\& *p != ':'/")"

# Mutant 8: return NIYAH_ERR_INVALID_ARG -> return NIYAH_OK in policy_add_grant
run_mutant "invalid arg ret -> OK" "$(echo "$ORIG" | sed '0,/return NIYAH_ERR_INVALID_ARG;/{s/return NIYAH_ERR_INVALID_ARG;/return NIYAH_OK;/}')"

# Mutant 9: verdict = NIYAH_CONTROL_DENY -> NIYAH_CONTROL_ALLOW in initialize
run_mutant "init deny -> allow" "$(echo "$ORIG" | sed 's/decision.verdict = NIYAH_CONTROL_DENY;/decision.verdict = NIYAH_CONTROL_ALLOW;/')"

# Mutant 10: nonce == 0u check: == -> !=
run_mutant "nonce==0 -> !=0" "$(echo "$ORIG" | sed 's/request->nonce == 0u/request->nonce != 0u/')"

echo ""
echo "MUTANTS_TOTAL=$TOTAL MUTANTS_KILLED=$KILLED KILL_RATE=$(echo "scale=1; $KILLED * 100 / $TOTAL" | bc)%"

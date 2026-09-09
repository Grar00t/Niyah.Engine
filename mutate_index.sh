#!/usr/bin/env bash
# Mutation testing for search/niyah_index.c
set -u
SRC="search/niyah_index.c"
TEST_SRC="search/niyah_index_test.c"
TEST_BIN="build/niyah_index_test_mut"
ORIG=$(cat "$SRC")
TOTAL=0
KILLED=0

compile_and_run() {
    gcc -O0 -std=c11 -I search/ -I native/ \
        "$SRC" "$TEST_SRC" \
        -o "$TEST_BIN" -lm 2>/dev/null
}

run_mutant() {
    local label="$1"
    local mutated="$2"
    TOTAL=$((TOTAL+1))
    echo "$mutated" > "$SRC"
    if compile_and_run 2>/dev/null; then
        if ! "./$TEST_BIN" 2>/dev/null 1>/dev/null; then
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

# Mutant 1: hit_compare: < -> <= for score (invert sort order)
run_mutant "hit_compare: score < -> <=" "$(echo "$ORIG" | sed 's/if (ha->score < hb->score) return 1;/if (ha->score <= hb->score) return 1;/')"

# Mutant 2: bm25 idf: log1p -> log (changes scoring formula)
run_mutant "bm25 idf: log1p -> log" "$(echo "$ORIG" | sed 's/const double idf = log1p(/const double idf = log(/')"

# Mutant 3: bm25 return 0.0 for term_frequency==0 -> skip check
run_mutant "bm25: skip tf==0 guard" "$(echo "$ORIG" | sed 's/if (!index || !entry || term_frequency == 0 || document_length == 0 ||/if (!index || !entry || 0 || document_length == 0 ||/')"

# Mutant 4: tokenize: < -> <= for max_tokens (off-by-one)
run_mutant "tokenize: count < max_tokens -> <=" "$(echo "$ORIG" | sed 's/while (\*cursor && count < max_tokens)/while (*cursor \&\& count <= max_tokens)/')"

# Mutant 5: token_byte: c >= 0x80u -> c > 0x80u (off-by-one high byte)
run_mutant "token_byte: >=0x80 -> >0x80" "$(echo "$ORIG" | sed 's/c >= 0x80u/c > 0x80u/')"

# Mutant 6: average_document_length update: wrong denominator
run_mutant "avg_len: document_count -> document_count+1" "$(echo "$ORIG" | sed 's|/ (double)index->document_count;$|/ (double)(index->document_count + 1);|')"

# Mutant 7: bm25 score: k1+1 -> k1 (factor off)
run_mutant "bm25: k1+1 -> k1" "$(echo "$ORIG" | sed 's/(index->k1 + 1\.0)/(index->k1)/')"

# Mutant 8: NIYAH_DOCUMENT_TOKEN_LIMIT 1024 -> 10 (drastically change limit)
run_mutant "doc_limit: 1024 -> 10" "$(echo "$ORIG" | sed 's/#define NIYAH_DOCUMENT_TOKEN_LIMIT 1024u/#define NIYAH_DOCUMENT_TOKEN_LIMIT 10u/')"

# Mutant 9: == in find_term: strcmp == 0 -> != 0 (never finds terms)
run_mutant "find_term: ==0 -> !=0" "$(echo "$ORIG" | sed 's/if (strcmp(index->terms\[i\].term, term) == 0) return/if (strcmp(index->terms[i].term, term) != 0) return/')"

# Mutant 10: ++entry->document_frequency -> no-op
run_mutant "skip document_frequency increment" "$(echo "$ORIG" | sed 's/++entry->document_frequency;//')"

echo ""
echo "MUTANTS_TOTAL=$TOTAL MUTANTS_KILLED=$KILLED KILL_RATE=$(echo "scale=1; $KILLED * 100 / $TOTAL" | bc)%"

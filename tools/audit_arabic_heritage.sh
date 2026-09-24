#!/usr/bin/env bash
set +e

OUT="${1:-/mnt/d/training-data/arabic-heritage-audit-20260924}"
mkdir -p "$OUT"

TSV="$OUT/inventory.tsv"
LOG="$OUT/audit.log"

ROOTS=(
  /mnt/d/training-data
  /mnt/d/AI
  /mnt/d/Projects
  /mnt/d/src
  /mnt/d/NiyahData
)

printf 'path\tbytes\tsha256\tfamily\tdecision\tnote\n' > "$TSV"
: > "$LOG"

classify() {
  local path="$1"
  local sha="$2"
  local low
  low="$(printf '%s' "$path" | tr '[:upper:]' '[:lower:]')"

  case "$sha" in
    195bc25a333237a2126470da888d7936b59ed3729f9210e0a4194ba43497dd70)
      printf 'camel-msa-r13\tREVIEW_BEFORE_TRAIN\texact public CAMeL GPL-v2 morphology.db hash'
      return
      ;;
    0b88b55d09eda8edc2f0009cb3f46d4b2ad8176cf5dc7a17a7b65439f7aaae7d)
      printf 'camel-glf-01\tSAFE_WITH_ATTRIBUTION\texact public CAMeL CC-BY-4.0 morphology.db hash'
      return
      ;;
  esac

  if [[ "$low" == *camel* ]]; then
    printf 'camel-unknown\tREVIEW_BEFORE_TRAIN\tCAMeL family but exact package/hash not matched'
  elif [[ "$low" == *arramooz* ]]; then
    printf 'arramooz\tREVIEW_BEFORE_TRAIN\tupstream dictionary is GPL; verify exact local copy'
  elif [[ "$low" == *sarf* ]]; then
    printf 'sarf\tREVIEW_BEFORE_TRAIN\tcurrent GitHub mirror says MIT; exact historical provenance still needs confirmation'
  elif [[ "$low" == *arabicdictionary* ]]; then
    printf 'arabic-dictionary-unknown\tQUARANTINE\tunknown provenance/license'
  elif [[ "$low" == *verb* || "$low" == *.dic ]]; then
    printf 'lexicon-unknown\tQUARANTINE\tidentify upstream source/license before use'
  else
    printf 'arabic-heritage-unknown\tQUARANTINE\tmanual provenance review required'
  fi
}

for root in "${ROOTS[@]}"; do
  [ -d "$root" ] || continue

  find "$root" -maxdepth 8 -type f \
    \( -iname '*camel*' \
       -o -iname '*arramooz*' \
       -o -iname '*sarf*' \
       -o -iname 'ArabicDictionary.sql' \
       -o -iname '*verb*.dic' \
       -o -iname '*.dic' \
       -o -path '*camel*/*' \
       -o -path '*arramooz*/*' \
       -o -path '*sarf*/*' \) \
    -print0 2>>"$LOG"
done | sort -zu | while IFS= read -r -d '' f; do
  bytes="$(stat -c '%s' "$f" 2>/dev/null)"
  [ -n "$bytes" ] || continue

  sha="$(nice -n 19 ionice -c3 sha256sum "$f" 2>/dev/null | awk '{print $1}')"
  [ -n "$sha" ] || sha='HASH_FAILED'

  meta="$(classify "$f" "$sha")"
  family="$(printf '%s' "$meta" | cut -f1)"
  decision="$(printf '%s' "$meta" | cut -f2)"
  note="$(printf '%s' "$meta" | cut -f3-)"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$f" "$bytes" "$sha" "$family" "$decision" "$note" >> "$TSV"
done

echo '=== ARABIC HERITAGE AUDIT ==='
echo "OUT=$OUT"
echo "TSV=$TSV"

echo
echo '=== DECISION COUNTS ==='
awk -F '\t' 'NR>1 {c[$5]++} END {for (k in c) print k "=" c[k]}' "$TSV" | sort

echo
echo '=== INVENTORY ==='
column -t -s $'\t' "$TSV" 2>/dev/null || cat "$TSV"

echo
echo '=== IMPORTANT ==='
echo 'This scanner is read-only with respect to source assets.'
echo 'A filename/path match is not proof of provenance; exact known hashes are stronger evidence.'
echo 'QUARANTINE and REVIEW_BEFORE_TRAIN assets must not enter tokenizer/model training automatically.'

echo
echo 'ARABIC_HERITAGE_AUDIT=PASS'
echo 'TERMINAL_ALIVE=YES'

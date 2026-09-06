#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-legacy-model/legacy-model-2.5-7B-Instruct}"
MANIFEST_FILE="${MANIFEST_FILE:-${ROOT_DIR}/corpus/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/legacy-model_neutral}"

usage() {
    printf 'usage: %s validate | train | infer <prompt>\n' "$0" >&2
    exit 2
}

case "${1:-}" in
    validate)
        python3 "${ROOT_DIR}/neutral/validate_manifest.py" "${MANIFEST_FILE}"
        ;;
    train)
        python3 "${ROOT_DIR}/neutral/train.py" \
            --model "${MODEL}" \
            --data "${MANIFEST_FILE}" \
            --output "${OUTPUT_DIR}" \
            --epochs "${EPOCHS:-1.0}" \
            --max-seq-length "${MAX_SEQ_LENGTH:-2048}"
        ;;
    infer)
        shift
        [ "$#" -gt 0 ] || usage
        python3 "${ROOT_DIR}/neutral/inference.py" \
            --model "${OUTPUT_DIR}" \
            --prompt "$*" \
            --max-new-tokens "${MAX_NEW_TOKENS:-256}"
        ;;
    *)
        usage
        ;;
esac

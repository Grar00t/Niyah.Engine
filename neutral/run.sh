#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-openai/gpt-oss-20b}"
MANIFEST_FILE="${MANIFEST_FILE:-${ROOT_DIR}/corpus/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/gpt_oss_20b_adapter}"

usage() {
    printf 'usage: %s validate | train | infer <prompt>\n' "$0" >&2
    exit 2
}

infer_model() {
    if [[ -n "${INFER_MODEL:-}" ]]; then
        printf '%s\n' "$INFER_MODEL"
        return
    fi

    if [[ -f "${OUTPUT_DIR}/adapter_config.json" ]]; then
        printf '%s\n' "$OUTPUT_DIR"
        return
    fi

    printf '%s\n' "$MODEL"
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
            --max-seq-length "${MAX_SEQ_LENGTH:-2048}" \
            --learning-rate "${LEARNING_RATE:-2e-4}" \
            --lora-r "${LORA_R:-8}" \
            --lora-alpha "${LORA_ALPHA:-16}"
        ;;
    infer)
        shift
        [[ "$#" -gt 0 ]] || usage
        if [[ -n "${AUDIT_LOG:-}" ]]; then
            python3 "${ROOT_DIR}/neutral/inference.py" \
                --model "$(infer_model)" \
                --prompt "$*" \
                --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
                --temperature "${TEMPERATURE:-0.0}" \
                --top-p "${TOP_P:-0.95}" \
                --audit-log "$AUDIT_LOG"
        else
            python3 "${ROOT_DIR}/neutral/inference.py" \
                --model "$(infer_model)" \
                --prompt "$*" \
                --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
                --temperature "${TEMPERATURE:-0.0}" \
                --top-p "${TOP_P:-0.95}"
        fi
        ;;
    *)
        usage
        ;;
esac

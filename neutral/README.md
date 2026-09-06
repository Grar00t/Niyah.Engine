# neutral/

Optional model-hub training and inference utilities. This directory is separate from the native C11 runtime in `native/`.

## What is implemented

- `clean_corpus.py` converts approved local files into JSONL records and hashes the original file bytes with SHA-256.
- `validate_manifest.py` validates corpus records before training.
- `model_profiles.py` selects model-specific runtime/training behavior. `gpt-oss` uses its native MXFP4 checkpoint path and attention-only LoRA targets (`q_proj`, `k_proj`, `v_proj`, `o_proj`). Other causal LMs use the existing bitsandbytes NF4 path.
- `train.py` performs LoRA causal-language-model domain adaptation on the corpus. It does not implement reinforcement learning, truth scoring, LVU, peer prediction, or factuality guarantees.
- `inference.py` runs generation from a base model or a saved local PEFT adapter and can append input/output hashes to a plain JSONL audit file. For `gpt-oss`, it requires the tokenizer chat template rather than inventing a prompt format.
- `run.sh` is a dispatcher for validate, train, and infer. Its default base model is `openai/gpt-oss-20b`; if a saved adapter exists in `OUTPUT_DIR`, inference uses it automatically.

The model process is a text generator. This directory does not grant it filesystem, process, or network authority. A model output is not an authorization decision.

## Corpus preparation

Source metadata is supplied explicitly by the operator; the repository does not download or certify datasets for you.

```bash
python3 neutral/clean_corpus.py \
  --input ./raw/rfc \
  --output ./corpus/rfc.jsonl \
  --source-name IETF \
  --source-url-prefix https://www.rfc-editor.org/rfc \
  --domain networking \
  --language en \
  --license 'operator-verified-license'
```

Merge approved JSONL files into the manifest you intend to train on, then validate it:

```bash
python3 neutral/validate_manifest.py ./corpus/manifest.jsonl
```

## Install

```bash
python3 -m pip install -r neutral/requirements.txt
```

The GPT-OSS path depends on current Transformers MXFP4 support (`transformers`, `kernels`, and Triton). If MXFP4 cannot be used by the installed CUDA stack, Transformers may require substantially more memory for a higher-precision fallback.

## Weights-first inference

```bash
./neutral/run.sh infer 'Explain the indexed material.'
```

Equivalent direct invocation:

```bash
python3 neutral/inference.py \
  --model openai/gpt-oss-20b \
  --prompt 'Explain the indexed material.'
```

A model-hub model id may cause network access and cache files. Set `MODEL` or `INFER_MODEL` to a local model path when offline execution is required.

## LoRA domain adaptation

```bash
MODEL=openai/gpt-oss-20b \
MANIFEST_FILE=./corpus/manifest.jsonl \
OUTPUT_DIR=./gpt_oss_20b_adapter \
./neutral/run.sh train
```

This is domain adaptation over source text. It is not training a foundation model from scratch, and it is not a claim that the adapter fixes instruction/data conflation inside the transformer.

## Inference with a saved adapter

`run.sh infer` automatically selects `OUTPUT_DIR` when `adapter_config.json` exists. To force a specific base or adapter path:

```bash
INFER_MODEL=./gpt_oss_20b_adapter \
./neutral/run.sh infer 'Explain the indexed material.'
```

Optional plain JSONL audit:

```bash
AUDIT_LOG=./inference_audit.jsonl \
./neutral/run.sh infer 'Explain the indexed material.'
```

The audit records hashes and timestamps only. It is not a Merkle tree, does not prove correctness, does not establish provenance of generated claims, and does not make the log immutable.

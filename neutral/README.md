# neutral/

Optional Hugging Face training and inference utilities. This directory is separate from the native C11 runtime in `native/`.

## What is implemented

- `clean_corpus.py` converts approved local files into JSONL records and hashes the original file bytes with SHA-256.
- `validate_manifest.py` validates corpus records before training.
- `train.py` performs 4-bit QLoRA causal-language-model domain adaptation on the corpus. It does not implement reinforcement learning, truth scoring, LVU, peer prediction, or factuality guarantees.
- `inference.py` runs generation from a base model or a saved local PEFT adapter and can append input/output hashes to a plain JSONL audit file. The JSONL file is not a Merkle tree or cryptographic append-only log.
- `run.sh` is a small dispatcher for validate, train, and infer.

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

## Training

Install the optional dependencies:

```bash
python3 -m pip install -r neutral/requirements.txt
```

QLoRA training requires a CUDA-capable environment supported by `bitsandbytes`:

```bash
python3 neutral/train.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data ./corpus/manifest.jsonl \
  --output ./qwen_neutral
```

A Hugging Face model id may cause network access. Use a local model path when offline execution is required.

## Inference

```bash
python3 neutral/inference.py \
  --model ./qwen_neutral \
  --prompt 'Explain the indexed material.'
```

Optional plain JSONL audit:

```bash
python3 neutral/inference.py \
  --model ./qwen_neutral \
  --prompt 'Explain the indexed material.' \
  --audit-log ./inference_audit.jsonl
```

The audit records hashes and timestamps only. It does not prove correctness, provenance of generated claims, or immutability.

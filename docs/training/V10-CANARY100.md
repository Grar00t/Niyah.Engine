# Niyah V10 CUDA Canary100

Status: **PASS**

This document records a fresh 100-update CUDA training canary over the
V10 clean corpus. It is an execution/reproducibility receipt, not a
claim that the resulting checkpoint is a production-quality language
model.

## Source

- Engine source commit: `3ace25669b2481161b2125a94562f537d2c4e534`
- Mode: `new`
- Backend: CUDA
- CUDA toolkit: 12.4 / V12.4.131
- GPU: NVIDIA GeForce RTX 3060 12 GiB

## Data lineage

- Tokenizer SHA-256:
  `d47a69a1ec180f577aeebe3905d6aed22d4b8ea3408b66fe4b8571fccc2fa841`
- Shards: `33`
- Shard hash gate: PASS
- Source fence: PASS

## Model

- Vocabulary: 12,288
- Context: 256
- Embedding dimension: 512
- Layers: 8
- Attention heads: 8
- KV heads: 4
- FFN hidden dimension: 1,536
- Tied word embeddings: yes

## Optimization

- Updates: 100
- Batch size: 4
- Accumulation steps: 1
- Learning rate: 1e-4
- Model seed: 20260926
- Data seed: 20260926
- Optimizer step: 100
- Cursor position: 400

## Loss receipt

- First loss: `9.447824480`
- Final loss: `8.428232190`
- First-10 mean: `9.413444996`
- Last-10 mean: `8.491244984`
- Mean delta: `-0.922200012`
- Overall mean loss: `8.76192188`
- Finite loss gate: PASS

## Artifacts

Checkpoint:

`model-0100.ckpt`

- Bytes: `377592032`
- SHA-256:
  `4127fbdef90b5aa36ad1098e3439c8dd4ef2e779bec21e783ac4506fb2679a90`

Cursor:

`cursor-0100.bin`

- Bytes: `120`
- SHA-256:
  `43de8ce5fa076104adf84c83f8b3777e0087c86cbb2761f527fb29777241668f`

The binary checkpoint and cursor are intentionally not committed to
the source repository.

## Gates

- Training return code: 0
- Checkpoint load gate: PASS
- Tokenizer changed: NO
- Shards changed: NO
- Source code changed: NO
- `NIYAH_V10_CANARY100=PASS`

## Interpretation

The canary establishes that the clean V10 lineage can execute fresh
CUDA training, update optimizer state, advance the data cursor, emit a
loadable checkpoint, and reduce training loss over the first 100
updates.

It does not establish final generation quality.

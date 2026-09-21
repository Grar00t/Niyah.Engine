# Verification Status

This document separates repository-backed implementation evidence from local diagnostic model-quality evidence.

**Documentation snapshot:** 2026-09-21  
**Documentation branch base:** `732f84fc34b2b5cad2ac40b4195e293fc2fad9aa`

## 1. Current repository/CI evidence

The documentation branch was created from `main` at commit:

```text
732f84fc34b2b5cad2ac40b4195e293fc2fad9aa
Merge pull request #57
```

For that exact `main` commit, GitHub Actions reports:

| Gate | Result |
|---|---|
| `core-ci` push workflow | PASS |
| Ubuntu Release configure/build/test job | PASS through successful workflow completion |
| Windows Release configure/build/test job | PASS through successful workflow completion |
| Ubuntu ASan/UBSan configure/build/test job | PASS through successful workflow completion |
| GitHub CodeQL push analysis | PASS |

The repository workflow defines Ubuntu and Windows Release build/test matrix jobs plus an Ubuntu sanitizer job. Successful workflow completion establishes that those configured jobs completed successfully for this commit.

## 2. Implemented native lifecycle

On the documented `main` lineage, the repository contains native code for:

```text
corpus preparation
  → tokenizer persistence/identity
  → tokenizer-bound dataset shard
  → fresh training
  → checkpoint + dataset cursor
  → resume
  → incremental inference
  → diagnostic probing
```

The model core includes Transformer forward execution, explicit backward gradients, AdamW optimization, KV-cache decode, sampling, and read-only evaluation APIs.

## 3. CLI surface verified from source

The current `tools/niyah.c` usage contract exposes:

```text
niyah prepare
niyah run
```

The current `tools/niyah_train.c` usage contract exposes:

```text
niyah-train new
niyah-train resume
```

The current standalone diagnostic tool exposes:

```text
niyah_probe --tokenizer TOK
            [--shard SHARD]
            [--checkpoint CKPT --prompt TEXT [--topk N] [--trace-steps N]]
```

A first-class `niyah eval` CLI is not part of this documented `main` revision. Evaluation exists as a native API and has been exercised through diagnostic helpers.

## 4. Training/resume diagnostic evidence

A supplied local diagnostic run resumed from optimizer step 1270 to 1905 for 635 requested updates.

Observed completion record:

```text
mode=resume
shards=1
samples=10157
updates=635
batch_size=8
accumulation_steps=2
optimizer_step=1905
cursor_epoch=3
cursor_position=9
mean_loss=3.74314785
RESUME_EXIT=0
```

Produced artifacts:

```text
model-1905.ckpt  = 13,383,392 bytes
cursor-1905.bin  = 120 bytes
```

This establishes successful continuation and persistence for the supplied run. It does not establish corpus quality or broad model capability.

> [!IMPORTANT]
> The pasted local diagnostic transcript does not include the exact repository SHA used to build that training binary. Therefore the model-quality numbers below must not be represented as reproduced specifically at `732f84f` unless that SHA is independently recorded for the run.

## 5. Held-out diagnostic evidence

A persistent evaluation helper was built successfully and evaluated four checkpoints against the same held-out records:

```text
BUILD_EVAL_EXIT=0
HELDOUT records=30 tokens=17584 samples=292 sequence_length=64
```

Measured checkpoint results:

| Checkpoint | Samples | Evaluated tokens | Mean loss | Perplexity |
|---|---:|---:|---:|---:|
| `model-0200.ckpt` | 292 | 17,554 | 5.581078354 | 265.357600904 |
| `model-0635.ckpt` | 292 | 17,554 | 4.589545587 | 98.449683132 |
| `model-1270.ckpt` | 292 | 17,554 | 3.942194185 | 51.531547081 |
| `model-1905.ckpt` | 292 | 17,554 | 3.643954027 | 38.242751050 |

```text
EVAL_EXIT=0
```

The measured trajectory is monotonically improving across these four checkpoints.

<p align="center">
  <img src="assets/heldout-learning-curve.svg" alt="Held-out learning trajectory" width="100%" />
</p>

## 6. Evidence-supported interpretation

The held-out trajectory establishes that later checkpoints predict the supplied held-out distribution better than earlier checkpoints under the measured objective.

From step 1270 to 1905:

```text
mean_loss:   3.942194185 -> 3.643954027
perplexity: 51.531547081 -> 38.242751050
relative perplexity reduction ≈ 25.79%
```

This provides direct evidence of learning beyond training-loss reduction alone.

## 7. Validation-set boundary

The same 30-record set has been used to make continuation decisions. It should therefore be treated as a **validation set** for future reporting, not a pristine final test set.

A final quality characterization requires an additional untouched test set evaluated only after training/checkpoint-selection decisions are frozen.

## 8. Corpus-quality boundary

The current v6 pilot corpus is recorded as web/Hugging Face sourced and known to contain quality defects. Therefore the observed loss/perplexity improvement does not establish:

- factual correctness of the learned content;
- high-quality conversational behavior;
- strong Arabic or English fluency across domains;
- reasoning capability;
- suitability of the corpus for a final assistant model.

The engine can optimize against a low-quality distribution successfully.

## 9. Diagnostic generation evidence

The repository includes `niyah_probe`, added through PR #57, to inspect next-token logits and autoregressive cache movement without modifying generation/training behavior.

The PR's documented representative diagnostic showed:

```text
prompt: A sequence is
first top token: byte-space token (id 32)
subsequent greedy trace: repeated id 32
KV cache: advanced on every step
```

This established that the repeated-space behavior was produced by the model's changing next-token distribution while the incremental decode path continued to advance. It did not establish the ultimate cause of the model preference.

## 10. Explicit non-claims

Current evidence does **not** establish:

- production readiness;
- broad conversational competence;
- broad reasoning capability;
- factual reliability;
- safety/alignment quality;
- distributed training behavior;
- mixed-precision training correctness;
- CPU/CUDA bitwise equivalence;
- superiority or parity with Qwen, Llama, GPT-family, or another established model family;
- legal/regulatory compliance.

## 11. Open evidence boundary

PR #56 (`feat: extract evidence v1 core from P9 chain`) remains an open draft at the time of this documentation snapshot. Its IR/receipt/evidence-v1 additions must not be described as merged `main` functionality until that PR is actually merged.

## 12. Verification rule

Future capability/status changes should attach, where applicable:

```text
repository SHA
exact command
build/test workflow result
model/tokenizer/dataset identities
training configuration and seeds
checkpoint identity
evaluation-set identity
metric output
exit status
```

Documentation should distinguish implementation facts, reproduced evidence, local diagnostic evidence, inference, and unestablished claims.

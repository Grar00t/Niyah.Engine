# Niyah.Engine

[![core-ci](https://github.com/Grar00t/Niyah.Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Grar00t/Niyah.Engine/actions/workflows/ci.yml)
![Language](https://img.shields.io/badge/language-C11-5c6bc0)
![Reference backend](https://img.shields.io/badge/reference-CPU%20FP32-4f7cac)
![Stage](https://img.shields.io/badge/stage-research%20runtime-c58b39)

**Niyah.Engine** is a native C11 language-model implementation built from first principles around one canonical `NiyahModel` weight layout. The same model state is used for training, checkpoint/resume, evaluation, and autoregressive inference. No hosted-model API or external LLM runtime is required by the model core.

<p align="center">
  <img src="docs/assets/niyah-engine-architecture.svg" alt="Niyah.Engine end-to-end architecture" width="100%" />
</p>

## Design goals

- **One model, one weight layout.** Training and inference operate on the same canonical weights.
- **Native implementation.** Tokenization, Transformer math, backward gradients, optimization, persistence, and generation live in this repository.
- **Deterministic lifecycle.** Fixed seeds, dataset cursor state, tokenizer identity, checkpoint identity, and explicit failure handling are first-class concerns.
- **CPU reference first.** CPU FP32 defines the reference behavior. CUDA is an optional build-time backend, not a dependency of the core library.
- **Evidence before claims.** Passing tests and successful runtime paths establish implementation behavior; they do not by themselves establish production readiness or useful language capability.

## Current implementation

| Area | Current state |
|---|---|
| Model core | Canonical contiguous FP32 weights, RMSNorm, RoPE, causal GQA, SwiGLU, final norm, LM head |
| Tokenizer | Native deterministic byte-level BPE with persistence and SHA-256 identity |
| Dataset | Tokenizer-bound `NIYAHSRD` shards, deterministic cursor/order persistence, explicit sample geometry |
| Training | Cross-entropy, explicit backward gradients, global clipping, AdamW, gradient accumulation |
| Persistence | Versioned checkpoint save/load plus separately persisted dataset cursor state |
| Resume | Checkpoint + cursor compatibility checks and continued optimizer stepping |
| Inference | KV cache, incremental decode, deterministic greedy/seeded sampling, autoregressive generation |
| Evaluation | Read-only token-weighted mean cross-entropy and perplexity |
| CUDA | Optional backend behind `NIYAH_ENABLE_CUDA`; CPU FP32 remains the reference |
| CI | Ubuntu + Windows Release build/test, plus Ubuntu ASan/UBSan job |

The repository also contains optional segment/role embedding support in the training API. Segment IDs are a modeling signal, not a security boundary.

## What this repository does **not** claim

The current implementation should not be described as production-ready solely because the native pipeline builds and tests successfully. The repository does not currently establish:

- production-scale training behavior;
- mixed-precision training;
- production orchestration or distributed training;
- final Arabic or English language capability;
- final held-out quality for a validation set known to share the current tokenizer semantics;
- compatibility between historical raw K11 weights and the current checkpoint/tokenizer contracts.

## Build

Requirements:

- CMake 3.20+
- a C11 compiler
- optional CUDA toolkit only when building the CUDA backend

Configure and build out-of-source:

```sh
cmake -S . -B build -DNIYAH_BUILD_TESTS=ON
cmake --build build --config Release
```

Run the regression suite:

```sh
ctest --test-dir build -C Release --output-on-failure
```

The `--config Release` argument is relevant to multi-config generators such as Visual Studio; it is harmless to omit for single-config generators configured with `CMAKE_BUILD_TYPE`.

## Prepare a corpus

```sh
niyah prepare \
  --corpus corpus.txt \
  --tokenizer-out tok.bin \
  --shard-out shard.bin \
  --target-vocab 269 \
  --min-pair-frequency 2 \
  --sequence-length 64
```

The base tokenizer vocabulary is 258 tokens: raw bytes `0..255`, BOS `256`, and EOS `257`. Learned BPE tokens begin at `258`.

`prepare` also supports boundary-aware records and supervised prompt/response preprocessing through `--record-mode blank-line` and one or more `--response-delimiter` values.

## Train from scratch

```sh
niyah-train new \
  --tokenizer tok.bin \
  --shard shard.bin \
  --checkpoint-out model.ckpt \
  --cursor-out cursor.bin \
  --updates 100 \
  --batch-size 32 \
  --accumulation-steps 4 \
  --model-seed 42 \
  --data-seed 42 \
  --context-length 64 \
  --embedding-dim 128 \
  --layers 4 \
  --heads 4 \
  --kv-heads 2 \
  --ffn-hidden-dim 512 \
  --rms-norm-eps 1e-5 \
  --tie-word-embeddings 1 \
  --learning-rate 0.0005 \
  --beta1 0.9 \
  --beta2 0.999 \
  --epsilon 1e-8 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0
```

`niyah-train` prints per-update progress to stderr and a structured summary on successful completion. Multiple `--shard` arguments are accepted in an ordered collection.

## Resume training

```sh
niyah-train resume \
  --tokenizer tok.bin \
  --shard shard.bin \
  --checkpoint-in model.ckpt \
  --cursor-in cursor.bin \
  --checkpoint-out model-next.ckpt \
  --cursor-out cursor-next.bin \
  --updates 1 \
  --batch-size 32 \
  --accumulation-steps 4
```

Resume outputs must be new paths. Inputs are never overwritten. Before training continues, the runtime validates tokenizer/checkpoint compatibility and the persisted dataset/cursor identities required by the current format.

## Run inference

```sh
niyah run \
  --tokenizer tok.bin \
  --checkpoint model-next.ckpt \
  --prompt "Hello" \
  --max-new-tokens 32 \
  --temperature 0 \
  --seed 42 \
  --backend cpu
```

When compiled with `NIYAH_ENABLE_CUDA=ON`, the CLI can also select `--backend cuda`. CUDA support is optional; CPU remains the reference path.

## Architectural boundary

Niyah.Engine intentionally keeps the model core independent from application infrastructure. The following are **not required dependencies** of the native model runtime:

- hosted model APIs;
- Llama / Qwen / Mistral runtimes;
- llama.cpp or Hugging Face Transformers runtime;
- RAG or vector databases;
- PostgreSQL or document services;
- agent frameworks, GUI shells, or HTTP serving layers.

Those systems may be integrated externally when a concrete application requires them, but they are not prerequisites for building, training, loading, or running the model.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) — model layout, data flow, training/inference paths, persistence boundaries
- [Training lifecycle](docs/TRAINING.md) — prepare, new training, resume, evaluation, and failure boundaries
- [Verification status](docs/VERIFICATION.md) — what has been demonstrated and what remains unestablished

## Repository layout

```text
include/niyah/       Public C API
src/                 Core implementation
tools/               CLI and benchmark programs
tests/               Native regression tests
docs/                Architecture and lifecycle documentation
.github/workflows/   CI configuration
```

## North star

The project exists to answer a narrow systems question rigorously: **can one small, native, auditable C implementation own the complete language-model lifecycle without delegating its core semantics to an external LLM runtime?**

Niyah.Engine treats that as an engineering problem: explicit formats, deterministic state, executable tests, and progressively stronger evidence.
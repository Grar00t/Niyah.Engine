# Niyah.Engine

Niyah.Engine is a native local language model implementation built from scratch in C11, with optional CUDA acceleration planned after the CPU reference path and training lifecycle are established.

## North star

The repository exists to build one canonical language model whose **same weights** are used for training and autoregressive inference.

Core path:

```text
text
  -> Niyah tokenizer
  -> token ids
  -> embeddings
  -> Transformer blocks
  -> final RMSNorm
  -> LM head
  -> logits
  -> sampler
  -> autoregressive generation
```

Training path:

```text
token ids
  -> same NiyahModel weights
  -> forward
  -> cross entropy
  -> backward
  -> gradients
  -> [planned] AdamW
  -> [planned] same weights updated
  -> [planned] checkpoint / resume
```

## Core rules

- Native C11 implementation.
- CPU FP32 is the current reference path; optional CUDA acceleration is planned later.
- One canonical `NiyahModel` layout for training and inference.
- Deterministic tests currently cover model math, tokenizer behavior, forward/decode parity, gradients, generation, and the tokenizer-to-model text pipeline.
- Optimizer and checkpoint tests will be added with those implementations; they do not exist yet.
- No hidden telemetry.
- No hosted-model API dependency.
- No external LLM runtime or model dependency.

The following are **not** core dependencies: Qwen, Mistral, Llama, llama.cpp, Hugging Face Transformers runtime, OpenAI/Anthropic/Google model APIs, PostgreSQL, RAG, evidence graphs, or constraint engines.

PostgreSQL may be used later only as an optional external service for metadata or tooling if a concrete use case justifies it. It must never be required to build, train, load, or run the model.

## Initial implementation order

Implemented:

1. Canonical model/config and tensor layout.
2. Tokenizer runtime + tokenizer training.
3. RMSNorm, RoPE, attention/GQA, SwiGLU, residual path.
4. KV cache and autoregressive generation.
5. Cross-entropy and explicit backward gradients on the canonical weights.

Planned:

6. AdamW, gradient clipping, and deterministic checkpoint/resume.
7. Evaluation/perplexity.
8. Optional CUDA kernels and residency.

## Status

The current implementation is the native CPU reference path through explicit backward gradients. AdamW, checkpoint/resume, tokenizer persistence, production dataset/training tooling, held-out evaluation, mixed precision, and CUDA are not implemented yet.

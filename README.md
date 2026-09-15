# Niyah.Engine

Niyah.Engine is a native local language model implementation built from scratch in C11, with an optional CUDA backend.

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
  -> AdamW
  -> same weights updated
  -> checkpoint / resume
```

## Core rules

- Native C11 implementation.
- Optional CUDA acceleration; CPU remains a first-class backend.
- One canonical `NiyahModel` layout for training and inference.
- Deterministic tests for math, tokenizer, forward, gradients, optimizer, checkpointing, and generation.
- No hidden telemetry.
- No hosted-model API dependency.
- No external LLM runtime or model dependency.

The following are **not** core dependencies: Qwen, Mistral, Llama, llama.cpp, Hugging Face Transformers runtime, OpenAI/Anthropic/Google model APIs, PostgreSQL, RAG, evidence graphs, or constraint engines.

PostgreSQL may be used later only as an optional external service for metadata or tooling if a concrete use case justifies it. It must never be required to build, train, load, or run the model.

## Initial implementation order

1. Canonical model/config and tensor layout.
2. Tokenizer runtime + tokenizer training.
3. RMSNorm, RoPE, attention/GQA, SwiGLU, residual path.
4. KV cache and autoregressive generation.
5. Cross-entropy and explicit backward gradients on the canonical weights.
6. AdamW and deterministic checkpoint/resume.
7. Evaluation/perplexity.
8. Optional CUDA kernels and residency.

## Status

Fresh rebuild. The previous multi-component repository was intentionally not restored as the architecture baseline.

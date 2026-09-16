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
  -> global gradient clipping
  -> AdamW
  -> same canonical weights updated
  -> [planned] checkpoint / resume
```

## Core rules

- Native C11 implementation.
- CPU FP32 is the current reference path; optional CUDA acceleration is planned later.
- One canonical `NiyahModel` layout for training and inference.
- Deterministic tests currently cover model math, tokenizer behavior, forward/decode parity, gradients, generation, the tokenizer-to-model text pipeline, AdamW arithmetic, global clipping, and a tiny backward-to-AdamW training chain.
- The tiny deterministic training-chain tests verify executable integration and loss decrease on a synthetic task; they do not establish real-corpus convergence or Arabic/English model capability.
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
6. Native reference AdamW with robust global gradient clipping over the canonical gradient vector.

Planned:

7. Versioned checkpoint/resume foundation.
8. Tokenizer persistence/identity binding and deterministic dataset/training lifecycle.
9. Held-out validation/perplexity.
10. Optional CUDA kernels and residency.

## Status

The current implementation is the native CPU reference path through explicit backward gradients, robust global gradient clipping, and AdamW updates to the same canonical FP32 model weights.

Checkpoint/resume, tokenizer persistence, production dataset preprocessing/sharding, a production training executable/loop, true mini-batches, gradient accumulation, held-out validation/perplexity, mixed precision, CUDA, instruction tuning, and conversational tuning are not implemented. Real-corpus convergence, Arabic model capability, and English model capability have not been demonstrated.

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
  -> versioned model + AdamW checkpoint / resume foundation
```

## Core rules

- Native C11 implementation.
- CPU FP32 is the current reference path; optional CUDA acceleration is planned later.
- One canonical `NiyahModel` layout for training and inference.
- Deterministic tests currently cover model math, tokenizer behavior, forward/decode parity, gradients, generation, the tokenizer-to-model text pipeline, AdamW arithmetic, global clipping, the tiny backward-to-AdamW training chain, and checkpoint roundtrip/resume, corruption rejection, and failure-atomic behavior.
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
7. Versioned model + AdamW checkpoint/resume foundation.

Planned:

8. Deterministic dataset/training lifecycle: cursor/order persistence, the reference training loop, and deterministic per-sample gradient accumulation with one averaged optimizer update are implemented; production preprocessing/sharding, a training executable, and true tensor mini-batching remain.
9. Held-out validation/perplexity.
10. Optional CUDA kernels and residency.

## Status

The current implementation is the native CPU reference path through explicit backward gradients, robust global gradient clipping, AdamW updates to the same canonical FP32 model weights, and versioned persistence of the canonical model plus AdamW training state.

Tokenizer persistence V1 saves and loads tokenizer state with a stable SHA-256 identity. Checkpoint V2 can bind model and AdamW state to that tokenizer identity while Checkpoint V1 remains supported. Dataset cursor state is persisted separately by the dataset lifecycle API; checkpoint persistence does not yet include that cursor state, scheduler state, gradient accumulation state, mixed-precision/CUDA state, or other training-path RNG state.

Production dataset preprocessing/sharding, a production training executable, true tensor mini-batching, held-out validation/perplexity, mixed precision, CUDA, instruction tuning, and conversational tuning are not implemented. Real-corpus convergence, Arabic model capability, and English model capability have not been demonstrated.

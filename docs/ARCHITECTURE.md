# Canonical LLM Architecture

## Identity

Niyah.Engine is one native language model implementation. Training and inference operate on the same `NiyahModel` weight layout.

There is no second "mini" model family and no external model runtime behind generation.

## Canonical weight order

All model parameters live in one contiguous FP32 array:

```text
token_embedding [vocab, dim]
for each layer:
  attn_norm      [dim]
  wq             [dim, dim]
  wk             [kv_dim, dim]
  wv             [kv_dim, dim]
  wo             [dim, dim]
  ffn_norm       [dim]
  w_gate         [ffn, dim]
  w_up           [ffn, dim]
  w_down         [dim, ffn]
final_norm        [dim]
lm_head           [vocab, dim]   # omitted as separate storage when tied
```

`head_dim = dim / n_heads` and `kv_dim = head_dim * n_kv_heads`.

This ordering is the contract for full forward, incremental decode, backward gradients, and future optimizer/checkpoint work.

## Model invariants

- Configuration validation rejects invalid head/GQA dimensions, including RoPE head dimensions smaller than 2 or not divisible by 2.
- Weight-size arithmetic is overflow-checked.
- Tied embeddings make the LM head reference the embedding offset rather than allocating duplicate parameters.
- Untied embeddings allocate a real independent LM head.
- Parameter initialization is deterministic for a fixed seed.
- CPU FP32 math is the current reference implementation.

## Tokenizer contract

The tokenizer is implemented inside Niyah.Engine; it is not delegated to an external model or tokenizer runtime.

- Byte tokens `0..255` guarantee lossless coverage of arbitrary input bytes and UTF-8 text.
- `256` is BOS and `257` is EOS.
- Learned tokens begin at `258`.
- Training is deterministic byte-level BPE with explicit merge order and deterministic tie-breaking.
- `target_vocab_size` is an upper training target; model configuration for a trained tokenizer uses the realized tokenizer vocabulary size.
- Runtime encoding starts from bytes and applies learned merge rules in training order.
- Decoding reconstructs original bytes exactly while ignoring BOS/EOS control tokens.
- Arabic and English round-trip behavior is covered by native tests.
- Generic generation remains tokenizer-independent; EOS is an explicit caller-provided token ID.

## Implemented now

- canonical contiguous FP32 `NiyahModel` weights;
- deterministic native parameter initialization;
- native byte-level BPE tokenizer and tokenizer training;
- RMSNorm;
- RoPE;
- causal grouped-query attention;
- SwiGLU feed-forward path;
- full-sequence Transformer forward;
- KV cache;
- incremental single-token decode;
- deterministic greedy/seeded sampler;
- autoregressive generation;
- cross-entropy objective;
- explicit CPU backward gradients over the canonical model weights;
- tokenizer -> realized vocabulary -> model -> incremental decode -> generation -> tokenizer decode integration coverage;
- Ubuntu and Windows Release CI;
- Ubuntu Debug AddressSanitizer + UndefinedBehaviorSanitizer CI.

## Not implemented yet

- AdamW optimizer;
- gradient clipping;
- model checkpoint save/load;
- optimizer checkpoint state;
- tokenizer persistence;
- dataset preprocessing and binary sharding;
- production training executable/loop;
- gradient accumulation;
- true mini-batch training;
- held-out validation loss/perplexity tooling;
- mixed precision;
- CUDA backend;
- instruction-tuning pipeline;
- conversational-tuning pipeline.

## Current scaling risks

- Tokenizer BPE training processes the corpus in memory and repeatedly rebuilds/sorts pair arrays.
- Full training currently materializes `token_count * vocab_size` logits and corresponding `dlogits`.
- Transformer mathematics is duplicated across full forward, incremental decode, and the cached forward used by backward. Parity tests reduce drift risk but do not remove this duplication.

These are later engineering targets. P5.1 does not refactor the three Transformer execution paths.

## Architectural boundary

The model core contains tokenizer, model layout/weights, Transformer math, inference primitives, and training primitives. Optimizer/checkpoint/dataset/evaluation work belongs in later trainer-facing code. Tool use, planning, shell/files/git/search orchestration, persistent task state, GUI, HTTP serving, RAG, databases, and agent frameworks are outside the model core.

## Out of core

RAG, evidence systems, graph reasoning, PostgreSQL, document services, hosted model APIs, external LLM runtimes, and agent frameworks are not model-core dependencies. Niyah.Core must remain buildable and usable without those systems.

# Canonical LLM Architecture

## Identity

Niyah.Engine is one native language model implementation. Training and inference operate on the same `NiyahModel` weight layout.

There is no second "mini" model family and no external model runtime behind generation.

## Canonical weight order

All parameters live in one contiguous FP32 array during P0:

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

This ordering is the contract for both future forward/backward training code and autoregressive inference.

## P0 invariants

- Configuration validation rejects invalid head/GQA dimensions.
- Weight-size arithmetic is overflow-checked.
- Tied embeddings make the LM head reference the embedding offset rather than allocating duplicate parameters.
- Untied embeddings allocate a real independent LM head.
- Parameter initialization is deterministic for a fixed seed.
- CPU math primitives are the reference implementation.

## Build order

P0 establishes model layout and reference math. Subsequent patches add, in order:

1. tokenizer runtime and trainer;
2. RoPE, attention/GQA, SwiGLU and Transformer forward;
3. KV cache and generation;
4. cross-entropy and explicit backward gradients over this exact layout;
5. AdamW, gradient clipping and checkpoint/resume;
6. held-out loss/perplexity;
7. optional CUDA acceleration without changing model semantics.

## Out of core

RAG, evidence systems, graph reasoning, PostgreSQL, document services and external LLM APIs are not part of the model core. They must not become dependencies of build, training, checkpoint loading, or generation.

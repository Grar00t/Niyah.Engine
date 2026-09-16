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

This ordering is the contract for full forward, incremental decode, backward gradients, AdamW updates, and checkpoint persistence.

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

## AdamW training primitive

The CPU reference optimizer updates the same canonical FP32 weight storage used by forward, decode, and backward.

- One robust global L2 norm is computed over the complete canonical gradient vector.
- Clipping is optimizer-internal and leaves caller-owned gradients unchanged.
- Adam first and second moments are stored as FP32 arrays with one element per canonical model weight.
- Bias correction is evaluated for the next optimizer step using double-precision calculations.
- Weight decay is decoupled from the gradient/moment path.
- Token embeddings, projection/feed-forward matrices, and an untied LM head are decay-enabled.
- Attention RMSNorm, feed-forward RMSNorm, and final RMSNorm scales are decay-exempt.
- Tied token-embedding/LM-head storage is processed once because it is one physical canonical span.
- Structural, numerical, step-overflow, aliasing, and FP32-representability checks complete before the two-pass commit mutates weights or optimizer state.
- Optimizer model/storage pointer binding is an in-process compatibility check only; it is not checkpoint identity.

## Implemented now

- canonical contiguous FP32 `NiyahModel` weights;
- deterministic native parameter initialization;
- native byte-level BPE tokenizer and tokenizer training;
- tokenizer persistence V1 with stable SHA-256 tokenizer identity;
- tokenizer-bound Checkpoint V2 while Checkpoint V1 remains supported;
- deterministic dataset sample ordering with resumable cursor persistence;
- deterministic single-sample reference training loop over dataset cursor, backward gradients, and AdamW;
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
- robust global gradient clipping;
- native reference AdamW over the canonical model weights;
- versioned checkpoint save/load for canonical model weights and AdamW `m`, `v`, step, and hyperparameters;
- explicit little-endian checkpoint wire format with streaming CRC-32 corruption detection;
- two-pass checkpoint load with structural validation before reconstructed state is committed to caller outputs;
- optimizer tests covering arithmetic, decay policy, state validation, failure atomicity, and tied/untied storage;
- deterministic tiny backward -> clipping -> AdamW training-chain coverage for tied and untied models;
- tokenizer -> realized vocabulary -> model -> incremental decode -> generation -> tokenizer decode integration coverage;
- Ubuntu and Windows Release CI;
- Ubuntu Debug AddressSanitizer + UndefinedBehaviorSanitizer CI.

## Not implemented yet

- dataset preprocessing and binary sharding;
- production training executable;
- gradient accumulation;
- true mini-batch training;
- held-out validation loss/perplexity tooling;
- mixed precision;
- CUDA backend;
- instruction-tuning pipeline;
- conversational-tuning pipeline.

## Not demonstrated by current tests

- real-corpus language-model convergence;
- Arabic model capability;
- English model capability;
- production-scale training behavior.

The tiny deterministic training-chain tests prove only that the currently implemented forward, objective, backward, clipping, and AdamW components form a coherent executable update path whose synthetic loss decreases under the tested configuration.

## Current scaling risks

- Tokenizer BPE training processes the corpus in memory and repeatedly rebuilds/sorts pair arrays.
- Full training currently materializes `token_count * vocab_size` logits and corresponding `dlogits`.
- Transformer mathematics is duplicated across full forward, incremental decode, and the cached forward used by backward. Parity tests reduce drift risk but do not remove this duplication.

These are later engineering targets. Checkpoint persistence does not refactor the three Transformer execution paths.

## Architectural boundary

The model core contains tokenizer, model layout/weights, Transformer math, inference primitives, backward gradients, global clipping, the reference AdamW optimizer, versioned checkpoint persistence, and deterministic dataset cursor sequencing/persistence. Production dataset preprocessing/sharding, a production training executable, true mini-batching, evaluation, and accelerator work remain later model-training lifecycle work. Tool use, planning, shell/files/git/search orchestration, persistent task state, GUI, HTTP serving, RAG, databases, and agent frameworks are outside the model core.

## Out of core

RAG, evidence systems, graph reasoning, PostgreSQL, document services, hosted model APIs, external LLM runtimes, and agent frameworks are not model-core dependencies. Niyah.Core must remain buildable and usable without those systems.

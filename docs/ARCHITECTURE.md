# Niyah.Engine Architecture

Niyah.Engine is organized around a single architectural invariant: **training, persistence, evaluation, and inference operate on one canonical `NiyahModel` layout**. There is no second model family hidden behind generation and no required external LLM runtime.

<p align="center">
  <img src="assets/niyah-engine-architecture.svg" alt="Full Niyah.Engine architecture" width="100%" />
</p>

## 1. System boundary

The native model core owns:

- model configuration and canonical parameter layout;
- tokenizer training, encoding, decoding, persistence, and identity;
- dataset shard construction, loading, identity, and deterministic cursor state;
- full-sequence Transformer forward execution;
- incremental decode with KV cache;
- causal language-model loss;
- explicit backward gradients;
- robust global gradient clipping;
- AdamW optimizer state and updates;
- checkpoint persistence and compatibility validation;
- read-only evaluation;
- autoregressive generation and sampling.

Application concerns such as RAG, databases, hosted APIs, agent orchestration, GUIs, HTTP servers, and document systems remain outside this boundary.

## 2. Canonical model representation

`NiyahModel` owns a configuration, a computed layout, one contiguous FP32 weight array, and the total weight count.

Conceptually, the canonical parameter order is:

```text
token_embedding [vocab, dim]
optional segment_embedding [segments, dim]
for each transformer layer:
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
lm_head           [vocab, dim]   # separate only when embeddings are untied
```

where:

```text
head_dim = embedding_dim / n_heads
kv_dim   = head_dim * n_kv_heads
```

This order is the contract shared by forward execution, backward gradients, AdamW state, persistence, and inference.

### Model invariants

- configuration validation rejects incompatible head/GQA geometry;
- weight-size arithmetic is overflow-checked;
- tied embeddings reuse physical storage rather than duplicating the LM head;
- initialization is deterministic for a fixed seed;
- CPU FP32 is the reference implementation;
- optional segment embeddings are a trainable modeling signal, not an isolation or security primitive.

## 3. Tokenizer contract

The tokenizer is native to Niyah.Engine.

### Base vocabulary

```text
0..255  raw byte tokens
256     BOS
257     EOS
258+    learned BPE merge tokens
```

The base vocabulary size is therefore 258. `target_vocab_size` is an upper training target; the realized vocabulary depends on how many valid merges are learned from the corpus.

### Properties

- arbitrary byte input is representable without an unknown-token fallback;
- UTF-8 text is handled as bytes and can round-trip losslessly;
- BPE merge learning and application are deterministic;
- tokenizer persistence carries a stable SHA-256 identity;
- tokenizer identity is used by checkpoint and dataset compatibility checks.

## 4. Dataset architecture

The dataset subsystem has two separate responsibilities: **content persistence** and **sequencing state**.

### Dataset shards

`NiyahDatasetShard` persists tokenized training content and sample geometry. Current code supports versioned `NIYAHSRD` persistence and validates the tokenizer identity when loading a shard.

The preparation path supports:

- continuous text streams;
- boundary-aware blank-line records;
- supervised prompt/response records with a response delimiter and response-only objective masking.

A causal-LM sample exposes shifted views:

```text
input : t0 t1 t2 ... tN-1
target: t1 t2 t3 ... tN
```

### Dataset cursor

`NiyahDatasetCursor` tracks deterministic sample ordering and progress independently of the shard contents. Persisted cursor state can bind to:

- sample count;
- dataset collection identity;
- checkpoint identity;
- deterministic epoch/order position.

This separation lets checkpoint/model state and dataset sequencing state evolve as an explicit pair rather than hiding sequencing state inside the model object.

## 5. Transformer execution

A full forward pass follows:

```text
token ids
  → token embeddings
  → optional segment embeddings
  → N × Transformer layer
      → RMSNorm
      → RoPE
      → causal grouped-query attention
      → residual
      → RMSNorm
      → SwiGLU feed-forward
      → residual
  → final RMSNorm
  → LM head
  → logits
```

### Grouped-query attention

The model allows `n_kv_heads <= n_heads`. Query heads are mapped onto the smaller K/V head set through GQA geometry derived from `head_dim` and `kv_dim`.

### Incremental decode

Inference does not recompute the entire prefix on every generated token. The decode path uses a KV cache and advances token-by-token after the prompt has been encoded.

## 6. Training lifecycle

`niyah-train` exposes two modes: `new` and `resume`.

### New training

```text
tokenizer + ordered shard collection
  → model initialization
  → AdamW state initialization
  → deterministic dataset cursor
  → select samples
  → forward / cross entropy
  → backward
  → accumulate gradients
  → average accumulated gradients
  → global L2 clipping
  → AdamW update
  → repeat
  → save checkpoint
  → compute checkpoint identity
  → bind cursor to checkpoint identity
  → save cursor
```

The effective number of sample consumptions per optimizer update is:

```text
batch_size × accumulation_steps
```

The current implementation performs deterministic sequential per-sample gradient accumulation and then commits one optimizer update for the accumulated group. This is not the same as a vectorized tensor mini-batch kernel.

### Progress reporting

After each completed optimizer update, `niyah-train` emits:

```text
update=I/N loss=L
```

to stderr. The final structured run summary is emitted after successful completion.

## 7. Resume contract

Resume is intentionally strict. It loads and checks the persisted state rather than silently starting a new run.

Conceptually:

```text
tok.bin
  + model.ckpt
  + cursor.bin
  + ordered shard collection
      ↓
load tokenizer
load checkpoint bound to tokenizer
load cursor
compute dataset collection identity
verify sample count / dataset identity
verify checkpoint identity
verify context/model compatibility
      ↓
continue optimizer steps
      ↓
write new checkpoint + new cursor
```

Resume inputs are not overwritten; output paths must be new paths.

## 8. Checkpoint architecture

A checkpoint persists the state required to reconstruct the trainable model and optimizer lifecycle, including:

- canonical model configuration;
- canonical model weights;
- AdamW first moments;
- AdamW second moments;
- optimizer step;
- optimizer hyperparameters;
- tokenizer compatibility metadata in tokenizer-bound formats.

Checkpoint persistence is versioned, explicitly serialized, corruption-checked, and loaded through validation before reconstructed state is published to the caller.

A separately persisted dataset cursor completes the resumable training-state pair.

## 9. Evaluation

`niyah_evaluate()` is a read-only evaluation primitive over caller-owned token/target samples.

It reports:

```text
sample_count
token_count
mean_loss   # token-weighted mean cross entropy / mean token NLL
perplexity  # exp(mean_loss)
```

Evaluation does not mutate the model weights. A meaningful quality claim still requires a genuinely held-out dataset that is semantically compatible with the tokenizer used to train the checkpoint.

## 10. Inference

`niyah run` performs:

```text
tok.bin + checkpoint
  → tokenizer load
  → tokenizer-bound checkpoint load
  → prompt encoding
  → BOS + prompt tokens
  → KV-cache generation
  → sampler
  → decoded output bytes
```

The runtime validates context capacity before generation.

### Sampling

The CLI supports deterministic generation at `temperature=0` and seeded stochastic generation for positive temperatures.

## 11. CPU and CUDA

### CPU

CPU FP32 is the reference path and the baseline for correctness.

### CUDA

CUDA is optional and enabled at configure time:

```sh
cmake -S . -B build-cuda -DNIYAH_ENABLE_CUDA=ON
cmake --build build-cuda --config Release
```

When enabled, CMake builds the optional CUDA backend and links CUDA support into the `niyah` CLI. The CLI can then select:

```text
--backend cpu
--backend cuda
```

The existence of an optional CUDA path does not by itself establish CUDA training parity, mixed precision, or production accelerator readiness.

## 12. CI and failure boundaries

The repository CI currently exercises:

- Ubuntu Release build/test;
- Windows Release build/test;
- Ubuntu Debug with AddressSanitizer and UndefinedBehaviorSanitizer.

Core code is compiled with warnings treated as errors in the configured toolchains.

The design prefers explicit failure over silent state corruption. Examples include validation of:

- model geometry;
- non-finite numerical states;
- tokenizer/shard compatibility;
- checkpoint structure and corruption checks;
- cursor/dataset/checkpoint identities;
- output-path collisions;
- context capacity;
- optimizer state compatibility.

## 13. Repository map

```text
include/niyah/
  niyah.h              Canonical model/config/layout API
  tokenizer.h          Tokenizer API and token constants
  dataset.h            Shards, cursor, identities
  train.h              Loss/backward interfaces
  training_loop.h      Accumulation/update lifecycle
  optimizer.h          AdamW and clipping
  checkpoint.h         Checkpoint persistence/identity
  eval.h               Read-only evaluation
  decode.h             KV-cache decode
  generate.h           Autoregressive generation

src/
  niyah_model.c
  niyah_math.c
  niyah_tokenizer.c
  niyah_dataset.c
  niyah_dataset_shard.c
  niyah_transformer.c
  niyah_decode.c
  niyah_sampler.c
  niyah_generate.c
  niyah_train.c
  niyah_backward.c
  niyah_training_loop.c
  niyah_optimizer.c
  niyah_checkpoint.c
  niyah_eval.c

tools/
  niyah.c               prepare + run CLI
  niyah_train.c         new + resume training CLI
  niyah_cuda_generation_bench.c

tests/
  native regression coverage
```

## 14. Architectural non-goals

The following are intentionally not required model-core dependencies:

- external LLM runtimes;
- hosted model APIs;
- RAG/vector stores;
- relational or document databases;
- agent/tool orchestration frameworks;
- GUI shells;
- web serving infrastructure.

These may exist in applications around Niyah.Engine, but they must not become prerequisites for the native language-model lifecycle.

## 15. Evidence boundary

Architecture documentation describes what the implementation is designed to do. It must not be read as evidence that the model has already achieved production-scale convergence or useful language capability.

For the current verification boundary, see [VERIFICATION.md](VERIFICATION.md).
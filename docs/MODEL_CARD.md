# Niyah.Engine Model & Runtime Card

## Project status

**Niyah.Engine** is a research language-model engine implemented primarily in C11. The repository contains a native tokenizer, Transformer model, training/backward path, AdamW optimizer, checkpoint/resume persistence, evaluation primitive, KV-cache decode path, sampler, and autoregressive generation.

This document describes the current implementation and the diagnostic model evidence available as of **2026-09-21**. It is not a production model release card.

## System classification

| Property | Current state |
|---|---|
| Model family | Autoregressive causal Transformer |
| Core implementation | Native C11 |
| Reference numerical path | CPU FP32 |
| Optional accelerator path | CUDA build-time backend |
| Attention | Causal grouped-query attention (GQA) |
| Position encoding | RoPE |
| Normalization | RMSNorm |
| Feed-forward | SwiGLU |
| Tokenizer | Native deterministic byte-level BPE |
| Base vocabulary | 258 tokens: bytes `0..255`, BOS `256`, EOS `257` |
| Training objective | Causal cross-entropy; response-masked supervised path is supported |
| Optimizer | AdamW with global gradient clipping |
| Decode | Incremental KV-cache autoregressive decode |
| Persistence | Versioned model checkpoint + separate dataset cursor |

## Architectural intent

The central invariant is that training, persistence, evaluation, and inference operate on one canonical `NiyahModel` weight layout. The project does not require PyTorch, Transformers, llama.cpp, a hosted model API, a database, RAG system, or agent framework to execute the native model lifecycle.

That design choice describes ownership of the runtime path; it is not a claim that Niyah.Engine matches the capability, scale, throughput, or maturity of established model families.

## Demonstrated implementation behavior

The repository has executable evidence for the following classes of behavior:

- native corpus preparation into tokenizer-bound shards;
- deterministic tokenizer persistence and identity;
- causal Transformer forward execution;
- explicit backward gradients and AdamW updates;
- response-masked supervised training mechanics;
- checkpoint save/load and optimizer-state persistence;
- dataset cursor persistence and resume validation;
- incremental generation with KV cache;
- deterministic greedy and seeded stochastic sampling;
- read-only cross-entropy/perplexity evaluation;
- native regression testing on supported CI environments;
- standalone logit/token diagnostic probing.

## Diagnostic learning evidence

A current pilot run was evaluated at four checkpoints on the same held-out record set:

| Checkpoint | Optimizer step | Mean loss | Perplexity |
|---|---:|---:|---:|
| `model-0200.ckpt` | 200 | 5.581078354 | 265.357600904 |
| `model-0635.ckpt` | 635 | 4.589545587 | 98.449683132 |
| `model-1270.ckpt` | 1270 | 3.942194185 | 51.531547081 |
| `model-1905.ckpt` | 1905 | 3.643954027 | 38.242751050 |

Evaluation run metadata:

```text
heldout_records          = 30
prepared_tokens          = 17584
samples                  = 292
sequence_length          = 64
evaluated_target_tokens  = 17554
eval_exit                = 0
```

The trajectory is monotonically improving on this evaluation distribution through step 1905.

<p align="center">
  <img src="assets/heldout-learning-curve.svg" alt="Held-out loss and perplexity trajectory through optimizer step 1905" width="100%" />
</p>

### What this establishes

The result is evidence that the trained model is learning statistical structure that transfers to the supplied held-out records better at later checkpoints than at earlier checkpoints.

### What this does not establish

It does not establish:

- broad conversational competence;
- broad Arabic or English language quality;
- reasoning capability;
- factual reliability;
- robustness outside the evaluated distribution;
- production-scale convergence;
- safety or alignment quality;
- superiority to another model family.

Because the same 30-record set has been used to make continuation decisions, it should now be treated operationally as a **validation set**, not as a pristine final test set.

## Data quality boundary

The current diagnostic training corpus is not treated as a gold-quality model dataset. The run record identifies known corpus-quality concerns, including web/Hugging Face sourced material and known factual defects in at least part of the collected material.

A declining loss therefore establishes optimization/learning behavior, not endorsement of the facts, style, or conversational quality present in the corpus.

See [DATA.md](DATA.md) for the dataset evidence boundary.

## Current quality status

| Capability | Status |
|---|---|
| Generate tokens through the native runtime | Demonstrated |
| Produce some coherent English/Arabic text | Observed in diagnostic runs |
| Held-out loss improvement on current validation distribution | Demonstrated |
| Stable broad chat quality | Unestablished |
| General reasoning | Unestablished |
| Reliable factual answering | Unestablished |
| Long-context capability beyond the configured context | Unestablished |
| Production readiness | Unestablished |

## Intended use

Current intended uses are research and systems engineering tasks such as:

- studying a small native language-model lifecycle;
- validating tokenizer/model/checkpoint contracts;
- testing deterministic training and resume behavior;
- examining optimization and held-out learning trajectories;
- experimenting with small-model data quality and architecture decisions;
- investigating native inference and accelerator boundaries.

## Out-of-scope interpretations

The repository should not be represented as a deployed general-purpose assistant, a production medical/legal/financial system, or a proven replacement for mature open-weight or commercial model families based on the evidence currently available.

## Reproducibility expectation

A capability statement should ideally identify:

```text
repository commit
training corpus identity/provenance
tokenizer identity
model configuration
training command and seeds
checkpoint identity
evaluation set identity
evaluation command/tool
metric output
exit status
```

For current evidence details, see [VERIFICATION.md](VERIFICATION.md) and [EVALUATION.md](EVALUATION.md).

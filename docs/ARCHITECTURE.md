# Architecture

## Core invariants

1. All persistent integer fields are written little-endian explicitly.
2. Dataset identity is SHA-256 over the canonical little-endian token stream.
3. A checkpoint is bound to exactly one dataset identity.
4. Resume refuses a different dataset with `NIYAH_ERR_MISMATCH`.
5. Training consumes exactly one adjacent-token transition per update.
6. `cursor` is monotonic and determines the next transition modulo dataset length minus one.
7. Checkpoints include an integrity SHA-256 over cursor, dataset identity, transition count, and model counts.

## Data flow

```text
UTF-8 text
   |
byte tokenizer
   |
NIYDS1 shard + SHA-256
   |
new/resume trainer
   |
NIYCK1 checkpoint + dataset binding + integrity hash
   |
   +--> evaluator
   +--> greedy generator
```

## Current model

The baseline model is a 256 x 256 transition-count matrix. Add-1 smoothing gives every next byte non-zero probability. This model is deliberately small and deterministic so the repository can establish reproducible contracts before larger model architectures are introduced.

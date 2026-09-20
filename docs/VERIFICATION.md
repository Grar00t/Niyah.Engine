# Verification Status

This document separates **demonstrated behavior** from **unestablished capability claims**.

Verification snapshot date: **2026-09-20**  
Repository base: **`14d3ae6c2ef3136deea1dbe9d93297dfaabde2e4`**

## 1. Demonstrated repository/runtime behavior

The following behaviors have been exercised against the current native pipeline.

| Gate | Status | Evidence |
|---|---|---|
| Source/build recovery | PASS | Fresh repository build from the stated base SHA |
| CPU Release build | PASS | CMake + MSVC Release build completed |
| Native regression suite | PASS | 27/27 tests passed repeatedly in the exercised build |
| Corpus preparation | PASS | Native `niyah prepare` created tokenizer + shard |
| Real training execution | PASS | 100-update native training run completed |
| Checkpoint persistence | PASS | Model checkpoint created and SHA-256 recorded |
| Cursor persistence | PASS | Dataset cursor created and SHA-256 recorded |
| Resume | PASS | One additional optimizer update resumed from persisted checkpoint/cursor |
| CPU inference | PASS | Resumed checkpoint loaded and generated output through `niyah run` |
| Evaluation primitive | PASS | `NIYAH_EVAL_P6_G=PASS`, exit 0 |

These results establish an executable native lifecycle. They do not by themselves establish production readiness or useful language capability.

## 2. Preparation evidence

An exercised corpus preparation run reported:

```text
P8C_PREPARE=PASS
vocab=269
merges=11
tokens=1417854
samples=22154
```

This establishes that the current tokenizer-to-shard path can process a real corpus into native persisted artifacts.

## 3. Training evidence

The exercised small-model configuration used:

```text
context_length      = 64
embedding_dim       = 128
layers              = 4
heads               = 4
kv_heads            = 2
ffn_hidden_dim      = 512
batch_size          = 32
accumulation_steps  = 4
learning_rate       = 0.0005
model_seed          = 42
data_seed           = 42
```

Observed training loss decreased during the bounded run. Example observations included:

```text
update=1/100  loss=5.60521841
update=20/100 loss=4.24649668
update=35/100 loss=3.78623343
```

A falling training loss is evidence that the exercised forward/backward/update path is changing the model in a direction that reduces the observed training objective. It is **not** evidence of held-out generalization.

## 4. Persistence evidence

The completed 100-update run produced:

```text
model-0100.ckpt
size   = 12,223,712 bytes
SHA256 = 7F91995692172CE0E9D84935E608C4BFCD9C87E792184CC2A5991AFF6A02A54D

cursor-0100.bin
size   = 120 bytes
SHA256 = 011D4EE53D9293152344E55264274D5536D46BC5778431AC10B781FC2E3CB084
```

## 5. Resume evidence

A one-update resume from the exact persisted pair completed successfully:

```text
update=1/1 loss=2.95031023
mode=resume
updates=1
batch_size=32
accumulation_steps=4
optimizer_step=101
cursor_epoch=0
cursor_position=12928
mean_loss=2.95031023
RESUME_EXIT=0
```

The cursor position is internally consistent with the exercised update geometry:

```text
101 optimizer steps × 32 batch size × 4 accumulation steps
= 12,928 sample consumptions
```

The resumed run produced new artifacts:

```text
model-0101-verify.ckpt
size   = 12,223,712 bytes
SHA256 = 28B712388C824613A920525DECD1FDE360B60F1C9E58468AED114474E3722BAD

cursor-0101-verify.bin
size   = 120 bytes
SHA256 = 1ED88C34FB5B71D383054CEA96D7C06A0DA5B33D16413F3F75F848E0E4043C1A
```

This demonstrates checkpoint load, optimizer-state continuity, dataset-cursor continuity, continued training, and persistence to new output paths for the exercised run.

## 6. Inference evidence

The resumed checkpoint was loaded using the current tokenizer and CPU backend:

```text
prompt: Hello
max_new_tokens: 16
temperature: 0
seed: 42
backend: cpu
```

Observed generated text:

```text
r the the the the sent th
```

The process exited successfully:

```text
INFERENCE_EXIT=0
```

This demonstrates load + generation path execution. The generated text is not presented as evidence of useful model quality.

## 7. Evaluation evidence and current limitation

The native evaluation regression executable passed:

```text
NIYAH_EVAL_P6_G=PASS
EXIT=0
```

Historical validation files were also inspected. Files such as `val-windows-clean.bin` are raw `uint32` token windows rather than current `NIYAHSRD` shards. Their token IDs fall within the numeric range of the current vocabulary, but numeric range compatibility does **not** prove that historical token IDs carry the same tokenizer semantics as the current `tok.bin`.

A raw-window diagnostic therefore must not be promoted to authoritative held-out quality evidence until tokenizer provenance is established or a validation set is rebuilt from held-out text using the current tokenizer semantics.

## 8. Current established chain

The following native path is demonstrated end-to-end for the exercised configuration:

```text
corpus
  → tokenizer
  → dataset shard
  → fresh training
  → checkpoint + cursor
  → resume
  → new checkpoint + cursor
  → CPU inference
```

## 9. Explicit non-claims

The evidence above does **not** establish:

- production readiness;
- production-scale convergence;
- useful Arabic language capability;
- useful English language capability;
- held-out generalization under a currently proven validation corpus;
- CUDA training parity;
- mixed-precision correctness;
- distributed training behavior;
- compatibility between legacy K11 raw weights and the current checkpoint/tokenizer contract.

## 10. Verification rule

Future status changes should be tied to reproducible evidence: repository SHA, exact command, artifact identities, test/evaluation output, and clearly scoped conclusions. Documentation should distinguish implementation facts from quality claims.
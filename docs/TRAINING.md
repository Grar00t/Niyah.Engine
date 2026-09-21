# Training

Niyah.Engine implements a native training lifecycle from tokenizer-bound dataset shards through checkpoint/cursor persistence and resume.

> [!NOTE]
> This document describes training mechanics. Model quality is evaluated separately in [EVALUATION.md](EVALUATION.md).

## Lifecycle

```text
raw corpus
  ↓
niyah prepare
  ├─ tok.bin
  └─ shard.bin
       ↓
niyah-train new
  ├─ model.ckpt
  └─ cursor.bin
       ↓
niyah-train resume
  ├─ model-next.ckpt
  └─ cursor-next.bin
       ↓
niyah run / niyah_evaluate() / niyah_probe
```

Tokenizer, dataset, checkpoint, and cursor identities form explicit compatibility boundaries. Do not mix artifacts from unrelated preparation/training runs unless compatibility is established.

## Corpus preparation

Basic preparation:

```sh
niyah prepare \
  --corpus corpus.txt \
  --tokenizer-out tok.bin \
  --shard-out shard.bin \
  --target-vocab 269 \
  --min-pair-frequency 2 \
  --sequence-length 64
```

Record-aware preparation:

```sh
niyah prepare \
  --corpus instruction-data.txt \
  --tokenizer-out tok.bin \
  --shard-out shard.bin \
  --target-vocab 269 \
  --min-pair-frequency 2 \
  --sequence-length 64 \
  --record-mode blank-line \
  --response-delimiter "Assistant:"
```

The response delimiter is corpus-specific. Niyah.Engine does not hard-code a universal chat format.

### Supervised objective geometry

For response-masked records:

```text
BOS + prompt + response + EOS
```

Prompt tokens remain causal context. Direct objective supervision begins at the response target boundary and continues through the following supervised targets.

## Fresh training

Example exercised small-model configuration:

```sh
niyah-train new \
  --tokenizer tok.bin \
  --shard shard.bin \
  --checkpoint-out model-0100.ckpt \
  --cursor-out cursor-0100.bin \
  --updates 100 \
  --batch-size 32 \
  --accumulation-steps 4 \
  --model-seed 42 \
  --data-seed 42 \
  --context-length 64 \
  --embedding-dim 128 \
  --layers 4 \
  --heads 4 \
  --kv-heads 2 \
  --ffn-hidden-dim 512 \
  --rms-norm-eps 1e-5 \
  --tie-word-embeddings 1 \
  --learning-rate 0.0005 \
  --beta1 0.9 \
  --beta2 0.999 \
  --epsilon 1e-8 \
  --weight-decay 0.01 \
  --max-grad-norm 1.0
```

These values document an exercised small configuration. They are not universal recommended hyperparameters.

## Update semantics

One optimizer update consumes:

```text
batch_size × accumulation_steps
```

sample slots through deterministic sequential gradient accumulation.

For example:

```text
32 × 4 = 128 sample consumptions per optimizer update
```

This is an accumulation contract, not a claim of vectorized tensor mini-batching.

With response masking, the number of supervised target tokens can be lower than:

```text
sample_consumptions × sequence_length
```

because prompt positions may contribute causal context without direct objective loss.

## Optimizer path

Conceptually:

```text
sample
  ↓
Transformer forward
  ↓
causal / masked cross entropy
  ↓
explicit backward gradients
  ↓
gradient accumulation
  ↓
average accumulated gradients
  ↓
global L2 clipping
  ↓
AdamW update
```

The trainable state and optimizer state are persisted in the checkpoint contract used by resume.

## Progress output

After each completed optimizer update, `niyah-train` emits to stderr:

```text
update=I/N loss=L
```

The per-update value can be noisy because it reflects the current accumulated update. Do not use one final update value as the primary model-quality metric.

On successful completion, the CLI emits a structured summary including fields such as mode, shard/sample counts, update count, optimizer step, cursor state, mean loss, and output paths.

## Checkpoint and cursor

Training produces two distinct persistence artifacts.

### Checkpoint

The model checkpoint carries the trainable model/optimizer state required by the current format.

### Cursor

The dataset cursor carries deterministic sequencing state and identity bindings.

Treat them as a pair for resume. The cursor can be bound to the exact checkpoint identity and ordered dataset collection identity.

## Resume

```sh
niyah-train resume \
  --tokenizer tok.bin \
  --shard shard.bin \
  --checkpoint-in model-0100.ckpt \
  --cursor-in cursor-0100.bin \
  --checkpoint-out model-0200.ckpt \
  --cursor-out cursor-0200.bin \
  --updates 100 \
  --batch-size 32 \
  --accumulation-steps 4
```

Resume validates persisted state required by the current formats before continuing.

Important behavior:

- resume inputs are not overwritten;
- output paths must be new;
- incompatible tokenizer/checkpoint/dataset/cursor state is rejected rather than silently ignored.

## Current diagnostic continuation record

A supplied local run completed a 635-update resume stage from optimizer step 1270 to 1905:

```text
updates=635
batch_size=8
accumulation_steps=2
optimizer_step=1905
cursor_epoch=3
cursor_position=9
mean_loss=3.74314785
RESUME_EXIT=0
```

Output artifacts were recorded as:

```text
model-1905.ckpt  13,383,392 bytes
cursor-1905.bin  120 bytes
```

The exact repository SHA used to build that local training binary is not present in the supplied transcript. Therefore this experiment record is kept separate from the repository CI snapshot.

## Inference verification

A produced checkpoint can be loaded through the native generation path:

```sh
niyah run \
  --tokenizer tok.bin \
  --checkpoint model-0200.ckpt \
  --prompt "Hello" \
  --max-new-tokens 16 \
  --temperature 0 \
  --seed 42 \
  --backend cpu
```

A successful exit establishes load + generation execution for that tokenizer/checkpoint pair. It does not establish useful output quality.

## Evaluation discipline

Training loss and held-out loss answer different questions.

```text
training loss   → how well current updates fit the training objective
validation loss → how well the checkpoint predicts a fixed unseen selection used for decisions
final test loss → held untouched until training decisions are frozen
```

The current 30-record pilot held-out set has been used to decide whether training should continue, so it is now validation-like.

See [EVALUATION.md](EVALUATION.md) for the measured checkpoint trajectory.

## Data quality

Optimization quality and corpus quality are separate.

A model can successfully lower training and validation loss while learning poor facts or poor conversational style if those patterns exist in both training and validation distributions.

The current diagnostic corpus has known quality concerns. See [DATA.md](DATA.md).

## Failure handling

Important explicit failure classes include:

- malformed/corrupt shard or checkpoint data;
- unsupported format versions;
- tokenizer identity mismatch;
- invalid model geometry/configuration;
- non-finite numerical states;
- output-path collisions;
- cursor/dataset/checkpoint identity mismatch;
- context-capacity violations.

Do not bypass a failed compatibility check merely to continue a run. Identify the incompatible artifact or contract.

## Reproducibility record

For a meaningful training experiment, capture at minimum:

```text
repository commit SHA
tokenizer identity / file SHA-256
shard identity / file SHA-256
training command
model configuration
model seed
data seed
optimizer hyperparameters
checkpoint SHA-256
cursor SHA-256
exit status
validation dataset identity/provenance
validation output
```

A local result without a repository SHA can still be useful diagnostic evidence, but it should be labeled as such rather than attached retroactively to a commit.

## Scaling work not established by the reference lifecycle

The verified native lifecycle does not by itself establish:

- production-scale orchestration;
- distributed training;
- true vectorized tensor mini-batching;
- mixed precision;
- accelerator parity beyond specifically tested paths;
- final instruction/chat quality;
- broad generalization.

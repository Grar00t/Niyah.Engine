# Training Lifecycle

This document describes the native training lifecycle implemented by Niyah.Engine: corpus preparation, fresh training, checkpoint/cursor persistence, resume, inference verification, and evaluation.

## 1. Lifecycle overview

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
niyah run / niyah_evaluate()
```

The tokenizer, shard collection, checkpoint, and cursor form explicit compatibility boundaries. Do not mix artifacts from unrelated preprocessing/training runs unless compatibility has been demonstrated.

## 2. Prepare

Example:

```sh
niyah prepare \
  --corpus corpus.txt \
  --tokenizer-out tok.bin \
  --shard-out shard.bin \
  --target-vocab 269 \
  --min-pair-frequency 2 \
  --sequence-length 64
```

### Outputs

`tok.bin`
: Persisted tokenizer state with a stable identity used by downstream compatibility checks.

`shard.bin`
: Tokenizer-bound dataset shard containing the token stream and sample geometry used by the training loop.

### Record-aware preparation

Preparation can preserve record boundaries:

```sh
niyah prepare \
  --corpus instruction-data.txt \
  --tokenizer-out tok.bin \
  --shard-out shard.bin \
  --target-vocab 269 \
  --min-pair-frequency 2 \
  --sequence-length 64 \
  --record-mode blank-line
```

For supervised records, one or more `--response-delimiter` values can define the prompt/response split. Prompt tokens remain context while direct objective supervision begins at the response target.

## 3. Fresh training

Example reference configuration:

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

The numeric values above are an exercised small-model configuration, not universal recommended hyperparameters.

### Update semantics

One optimizer update consumes:

```text
batch_size × accumulation_steps
```

samples through deterministic sequential gradient accumulation.

For example:

```text
32 × 4 = 128 sample consumptions per optimizer update
```

This is an accumulation contract, not a claim of vectorized tensor mini-batching.

### Progress output

After every completed update, the CLI emits:

```text
update=I/N loss=L
```

The reported value is training loss for the completed accumulated update. It is **not** held-out validation loss.

## 4. Checkpoint/cursor pair

Successful training produces two distinct artifacts.

### Checkpoint

The model checkpoint carries the canonical trainable state required by the current persistence contract, including model weights and AdamW state/configuration.

### Cursor

The dataset cursor carries deterministic sequencing state and persisted identity bindings.

Treat the checkpoint and cursor as a pair for resume purposes. A cursor may be bound to the exact checkpoint bytes through checkpoint identity.

## 5. Resume

Example:

```sh
niyah-train resume \
  --tokenizer tok.bin \
  --shard shard.bin \
  --checkpoint-in model-0100.ckpt \
  --cursor-in cursor-0100.bin \
  --checkpoint-out model-0101.ckpt \
  --cursor-out cursor-0101.bin \
  --updates 1 \
  --batch-size 32 \
  --accumulation-steps 4
```

### Resume safety properties

The current CLI is designed to reject incompatible or ambiguous persisted state rather than silently continuing. Resume validates the state required by the format, including relevant tokenizer, dataset, checkpoint, sample-count, and model-context compatibility.

Resume inputs are never overwritten. Output paths must be new.

## 6. Inference verification

After a checkpoint is created or resumed, verify that it can be loaded by the runtime:

```sh
niyah run \
  --tokenizer tok.bin \
  --checkpoint model-0101.ckpt \
  --prompt "Hello" \
  --max-new-tokens 16 \
  --temperature 0 \
  --seed 42 \
  --backend cpu
```

A successful exit establishes that the tokenizer/checkpoint pair can enter the native generation path. It does not establish model quality.

## 7. Evaluation

The public evaluation API is read-only:

```c
NiyahStatus niyah_evaluate(
    const NiyahModel *model,
    const NiyahEvaluationSample *samples,
    size_t sample_count,
    NiyahEvaluationMetrics *out_metrics);
```

Metrics are:

- `sample_count`
- `token_count`
- token-weighted mean cross-entropy (`mean_loss`)
- perplexity (`exp(mean_loss)`)

### Validation requirements

A defensible held-out result requires all of the following:

1. validation text that was not included in the training corpus;
2. tokenization semantics known to match the tokenizer used by the checkpoint;
3. objective/sample geometry compatible with the evaluation being reported;
4. no mutation of model weights during evaluation.

Raw token-ID windows from an older preprocessing pipeline are not automatically semantically compatible merely because every ID falls inside the current vocabulary range.

## 8. Failure handling

Do not reinterpret a rejected artifact as usable data. Important failure classes include:

- malformed or corrupt shard/checkpoint data;
- unsupported format versions;
- tokenizer identity mismatch;
- invalid model configuration;
- output-path collisions;
- non-finite numerical states;
- cursor/dataset/checkpoint identity mismatch;
- context-capacity violations.

The expected response to these failures is to identify the incompatible artifact or contract, not to bypass validation.

## 9. Reproducibility checklist

For a meaningful training record, capture at minimum:

```text
repository commit SHA
tokenizer SHA-256
shard SHA-256
training command
model seed
data seed
optimizer hyperparameters
model geometry
checkpoint SHA-256
cursor SHA-256
exit status
held-out dataset identity and provenance
```

This separates executable evidence from later interpretations about model capability.

## 10. Next-stage scaling work

Current architectural work that remains distinct from the verified reference lifecycle includes:

- production-scale orchestration;
- true vectorized tensor mini-batching;
- mixed precision;
- accelerator parity beyond the reference CPU path;
- larger-corpus validation methodology;
- instruction/conversation tuning quality evaluation.
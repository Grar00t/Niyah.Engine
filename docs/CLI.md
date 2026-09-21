# CLI Reference

This page documents command surfaces present on the current repository `main` lineage used for this documentation refresh.

## `niyah`

The primary CLI exposes three commands:

```text
niyah prepare
niyah eval
niyah run
```

### `niyah prepare`

```text
niyah prepare --corpus FILE --tokenizer-out TOK --shard-out SHARD
    --target-vocab N --min-pair-frequency N --sequence-length N
    [--record-mode stream|blank-line]
    [--response-delimiter TEXT ...]
```

`--response-delimiter` is accepted only with `--record-mode blank-line`.

Purpose:

- train/build the native tokenizer from the supplied corpus;
- persist tokenizer state;
- tokenize and persist a dataset shard;
- optionally preserve record boundaries;
- optionally split supervised prompt/response records using explicit delimiters.

Example:

```sh
niyah prepare \
  --corpus corpus.txt \
  --tokenizer-out tok.bin \
  --shard-out shard.bin \
  --target-vocab 269 \
  --min-pair-frequency 2 \
  --sequence-length 64 \
  --record-mode blank-line \
  --response-delimiter "Assistant:"
```

### `niyah eval`

```text
niyah eval --tokenizer TOK --checkpoint CKPT --heldout FILE
    [--format text|shard]
    [--sequence-length N]
```

`--format` defaults to `text`.

Text mode:

- reads the held-out file as raw bytes;
- builds a tokenizer-bound evaluation shard in memory;
- defaults `--sequence-length` to the checkpoint context length when omitted;
- reports model cross-entropy/perplexity and a sparse add-1 bigram baseline on the same token stream;
- reports tokenizer, checkpoint, and held-out identities;
- additionally reports bits per byte for the raw held-out bytes.

Shard mode:

- loads an existing tokenizer-compatible dataset shard;
- uses the shard's stored sequence length;
- rejects an explicit `--sequence-length`;
- reports the shard semantic identity;
- omits bits-per-byte metrics because no raw-text byte denominator is implied by the shard contract;
- rejects V3 loss-masked/supervised shards until the evaluator has an explicit loss-mask-aware scoring path.

For record-based V2 shards, the add-1 bigram baseline excludes persisted `EOS -> BOS` adjacencies between records so its scored transitions match the record-bounded model evaluation geometry.

Example:

```sh
niyah eval \
  --tokenizer tok.bin \
  --checkpoint model.ckpt \
  --heldout heldout.txt \
  --format text \
  --sequence-length 64
```

Successful output is machine-readable `key=value` text and terminates with:

```text
EVAL_EXIT=0
```

Evaluation is read-only with respect to checkpoint bytes. The CLI rejects missing or incompatible tokenizer/checkpoint/shard inputs and returns non-zero on failure.

### `niyah run`

```text
niyah run --tokenizer TOK --checkpoint CKPT --prompt TEXT
    --max-new-tokens N [--temperature F] [--seed N]
    [--backend cpu|cuda]
```

Purpose:

- load the tokenizer;
- load the tokenizer-compatible checkpoint;
- encode `BOS + prompt`;
- perform autoregressive KV-cache generation;
- sample tokens and decode output bytes.

Example:

```sh
niyah run \
  --tokenizer tok.bin \
  --checkpoint model.ckpt \
  --prompt "Hello" \
  --max-new-tokens 32 \
  --temperature 0 \
  --seed 42 \
  --backend cpu
```

`--backend cuda` requires a build configured with the optional CUDA backend.

## `niyah-train`

Training has two modes:

```text
niyah-train new
niyah-train resume
```

### `niyah-train new`

```text
niyah-train new --tokenizer TOK --shard SHARD [--shard SHARD ...]
    --checkpoint-out CKPT --cursor-out CURSOR
    --updates N --batch-size N --accumulation-steps N
    --model-seed N --data-seed N --context-length N --embedding-dim N
    --layers N --heads N --kv-heads N --ffn-hidden-dim N
    --rms-norm-eps F --tie-word-embeddings 0|1
    --learning-rate F --beta1 F --beta2 F --epsilon F
    --weight-decay F --max-grad-norm F
```

The command initializes a new model/optimizer/cursor state and writes new checkpoint/cursor artifacts after the requested updates complete.

### `niyah-train resume`

```text
niyah-train resume --tokenizer TOK --shard SHARD [--shard SHARD ...]
    --checkpoint-in CKPT --cursor-in CURSOR
    --checkpoint-out CKPT --cursor-out CURSOR
    --updates N --batch-size N --accumulation-steps N
```

Resume validates persisted compatibility state and continues optimizer stepping. Output paths must be new; resume inputs are not overwritten.

### Progress contract

During training, completed updates are reported to stderr:

```text
update=I/N loss=L
```

This per-update value is training loss, not held-out evaluation loss.

## `niyah_probe`

The standalone diagnostic probe does not change the training algorithm, sampler semantics, or generation implementation.

Usage:

```text
niyah_probe --tokenizer TOK
            [--shard SHARD]
            [--checkpoint CKPT --prompt TEXT [--topk N] [--trace-steps N]]
```

A tokenizer path is required for all modes, but tokenizer-only invocation currently emits no standalone report. At least `--shard`, or both `--checkpoint` and `--prompt`, are needed for useful output.

Capabilities include:

- shard token-count/unigram statistics;
- prompt token IDs;
- top-k next-token logits/probabilities;
- autoregressive greedy trace;
- KV-cache position movement;
- decoded bytes for selected tokens.

Example:

```sh
niyah_probe \
  --tokenizer tok.bin \
  --shard shard.bin \
  --checkpoint model.ckpt \
  --prompt "A sequence is" \
  --topk 10 \
  --trace-steps 8
```

## Exit handling

Commands return non-zero exit status on rejected configuration, malformed/incompatible state, IO errors, or other explicit failure paths. Automation should gate on the process exit code rather than parsing human-readable text alone.

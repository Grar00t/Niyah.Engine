# CLI Reference

This page documents the local review working tree inspected on 2026-10-13.
Some persistence and repetition changes are unpublished; see the exact revision
boundary in the [local audit](LOCAL_AUDIT_2026-10-13.md).

## `niyah`

The primary CLI exposes four commands:

```text
niyah prepare
niyah shard
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

### `niyah shard`

```text
niyah shard --tokenizer TOK --corpus FILE --shard-out SHARD
    --sequence-length N
    [--record-mode stream|blank-line]
    [--response-delimiter TEXT ...]
```

This uses an existing tokenizer to prepare another tokenizer-bound shard.
It does not retrain the tokenizer. The response delimiter requires blank-line
record mode. Output publication uses a new path.

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
- evaluates the all-token objective;
- reports model cross-entropy/perplexity and a sparse add-1 bigram baseline on the same scored sample geometry;
- reports tokenizer, checkpoint, and held-out identities;
- additionally reports bits per byte for the raw held-out bytes.

Shard mode:

- loads an existing tokenizer-compatible dataset shard;
- uses the shard's stored sequence length;
- rejects an explicit `--sequence-length`;
- reports the shard semantic identity;
- evaluates V1/V2 shards with the all-token objective;
- evaluates V3 supervised shards with the persisted `loss_start` objective, preserving prompt tokens as causal context while scoring only response/EOS targets;
- uses the same scored transitions for the add-1 bigram baseline;
- omits bits-per-byte metrics because no raw-text byte denominator is implied by the shard contract.

Machine-readable output includes:

```text
objective=all_tokens
```

or, for a persisted supervised loss mask:

```text
objective=loss_masked
```

`token_count` is the number of targets that contribute to the reported loss. For a loss-masked shard it therefore excludes prompt-only targets before each sample's `loss_start`.

The `tokenizer_sha256` field is a structured tokenizer identity. `checkpoint_sha256` hashes exact checkpoint-file bytes. `heldout_sha256` hashes exact text bytes in text mode, but is a structured shard identity in shard mode; `*_identity_kind` fields make these distinctions explicit. These different identity types must not be compared for equality.

The add-one bigram baseline is fitted and scored on the same evaluated supervised transitions; it is an in-sample descriptive baseline, not one trained independently on the training corpus. Text-mode bits per byte divides the scored negative log likelihood (including EOS, excluding BOS as a target) by every original input byte. The final short sample is included, with no discarded token tail. Context resets for each sample, so this metric describes that segmented evaluation geometry rather than an uninterrupted full-document likelihood. Shard mode has no asserted byte denominator.

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
    [--prompt-prefix TEXT] [--prompt-suffix TEXT]
    [--backend cpu|cuda]
    [--repetition-penalty F] [--repetition-window N]
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

Repetition defaults are penalty `1.0` (disabled) and window `64`. The penalty
must be finite and at least `1`; the window must be a positive decimal integer.
History contains emitted tokens only, excluding the prompt. Each distinct
token in the trailing window is penalized once. Penalty `1` delegates to the
legacy generation path. These options are rejected by `niyah eval`. See
[Repetition generation](REPETITION_GENERATION.md) for workspace and test scope.

If `--prompt-prefix` or `--prompt-suffix` is omitted, `niyah run` looks for sidecars named `<checkpoint>.prompt-prefix` and `<checkpoint>.prompt-suffix`. Any applied sidecar requires `<checkpoint>.prompt-sha256`; a missing, unreadable, malformed, or mismatched binding fails closed. Empty prefix/suffix sidecars are valid. Explicit CLI flags override the corresponding sidecars; pass an explicit empty string to disable one sidecar. Checkpoints without sidecars keep the previous raw-prompt behavior.

V2 metadata binds the exact checkpoint bytes and the exact prefix/suffix bytes, using lowercase SHA-256 hex and LF or CRLF separators:

```text
NIYAH_PROMPT_V2
<checkpoint SHA-256>
<prefix SHA-256, including the empty-string digest when absent>
<suffix SHA-256, including the empty-string digest when absent>
```

The optional last newline is ignored. A CLI override disables content validation only for the overridden component; any remaining applied sidecar is still checked. The legacy single checkpoint digest remains supported with a `template_content_integrity=unverified` diagnostic: it protects checkpoint selection but does not bind template contents. Hashes detect inconsistency, not malicious replacement of both data and metadata.

Generation writes the tokenizer's exact decoded bytes and checks the final stdout flush. Byte-level tokens can form invalid UTF-8; consumers needing text must decode strictly and retain raw bytes on failure instead of silently replacing bytes. The Windows command-line wrapper converts UTF-16 arguments to UTF-8 with strict conversion flags. This is independent of generated-byte validity and terminal rendering.

Paths passed to the native tokenizer, shard, checkpoint, cursor, and publication functions are UTF-8 on Windows. The CLI wrappers convert Windows arguments consistently and the filesystem layer uses wide Windows APIs. On POSIX, path strings retain native filesystem-byte semantics.

`niyah run` and `niyah_probe` reject models with nonzero `n_segments`: these CLI paths have no per-token segment input. Segment-conditioned forward/decode/backward remain available through the explicit native APIs.

`niyah_probe --tokenizer TOK --checkpoint CKPT --prompt TEXT --topk N --trace-steps N` prints escaped token bytes and a bounded greedy diagnostic trace. The cache is primed with BOS and every prompt token. The trace deliberately continues after EOS and labels `trace_stop_on_eos=0`; `niyah run` generation stops on EOS and retains its existing stops for eight repeated identical tokens or entropy below 0.05 nats. Those stops do not establish semantic quality. The shard report's standalone token ID 32 count excludes spaces inside merged tokens.

## `niyah-train`

Training has three modes:

```text
niyah-train new
niyah-train finetune
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
    [--warmup-steps N] [--progress-details] [--backend cpu|cuda]
```

The command initializes a new model/optimizer/cursor state and writes new checkpoint/cursor artifacts after the requested updates complete.

### `niyah-train finetune`

```text
niyah-train finetune --tokenizer TOK --shard SHARD [--shard SHARD ...]
    --checkpoint-in CKPT --checkpoint-out CKPT --cursor-out CURSOR
    --updates N --batch-size N --accumulation-steps N --data-seed N
    --learning-rate F --beta1 F --beta2 F --epsilon F
    --weight-decay F --max-grad-norm F
    [--warmup-steps N] [--progress-details] [--backend cpu|cuda]
```

Fine-tuning loads compatible model weights, creates fresh AdamW state and a
new dataset cursor, and uses the supplied optimizer settings. Its optimizer
step count starts from zero. It does not continue the input checkpoint's
optimizer/cursor state or require a checkpoint/cursor completion pair.

### `niyah-train resume`

```text
niyah-train resume --tokenizer TOK --shard SHARD [--shard SHARD ...]
    --checkpoint-in CKPT --cursor-in CURSOR
    --checkpoint-out CKPT --cursor-out CURSOR
    --updates N --batch-size N --accumulation-steps N
    [--progress-details] [--backend cpu|cuda]
```

Resume validates persisted compatibility state and continues optimizer stepping. Output paths must be new; resume inputs are not overwritten.

All three modes publish checkpoint/cursor pairs under the local
[completion protocol](PERSISTENCE_PROTOCOL.md). A successful publication keeps
the `.pending` reservation and adds `.complete`; pending alone is insufficient
for resume. Absence of both sidecars selects legacy binding checks.

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

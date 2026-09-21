# Quickstart

This guide exercises the native Niyah.Engine lifecycle without introducing an external model runtime.

> [!NOTE]
> Niyah.Engine is a research runtime. A successful build or training run does not imply useful language quality or production readiness.

## 1. Requirements

- CMake 3.20+
- a C11 compiler
- optional CUDA toolkit only when configuring `NIYAH_ENABLE_CUDA=ON`

## 2. Configure and build

```sh
cmake -S . -B build -DNIYAH_BUILD_TESTS=ON
cmake --build build --config Release
```

Run the regression suite:

```sh
ctest --test-dir build -C Release --output-on-failure
```

The build does not install the executables or add them to `PATH`.

On single-config generators, executables are normally emitted directly under `build/`. For a POSIX shell, make the later bare commands runnable with:

```sh
export PATH="$PWD/build:$PATH"
```

On Visual Studio/multi-config builds, Release executables normally appear under `build/Release/`. In PowerShell, use:

```powershell
$env:Path = "$(Resolve-Path .\build\Release);$env:Path"
```

Alternatively, invoke each executable by its explicit build path, for example `./build/niyah` on a single-config POSIX build or `.\build\Release\niyah.exe` on a Visual Studio Release build.

## 3. Prepare a corpus

Basic stream-mode preparation:

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
  --corpus corpus.txt \
  --tokenizer-out tok.bin \
  --shard-out shard.bin \
  --target-vocab 269 \
  --min-pair-frequency 2 \
  --sequence-length 64 \
  --record-mode blank-line
```

For supervised prompt/response records, `--response-delimiter` is valid only with `--record-mode blank-line`. Example:

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

One or more response delimiters may be supplied. Each exact delimiter must match the corpus format. Niyah.Engine does not hard-code a language-specific chat schema.

## 4. Train a small model

The following is an exercised small-model shape, not a universal recommendation:

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

During training, progress is written to stderr as:

```text
update=I/N loss=L
```

The final successful summary is emitted after the requested updates complete.

## 5. Resume training

Checkpoint state and dataset cursor state are persisted separately and validated together during resume:

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

Resume output paths must be new paths. The CLI does not overwrite resume inputs.

## 6. Evaluate on held-out text

Use the same tokenizer and a checkpoint against a held-out text file:

```sh
niyah eval \
  --tokenizer tok.bin \
  --checkpoint model-0200.ckpt \
  --heldout heldout.txt \
  --format text \
  --sequence-length 64
```

The command reports model mean loss, perplexity, bits per token, and a sparse add-1 bigram baseline on the same evaluation geometry. Text mode also reports bits per byte. Successful output ends with:

```text
EVAL_EXIT=0
```

Evaluation is read-only with respect to checkpoint bytes. For an already prepared compatible non-loss-masked shard, use `--format shard` and omit `--sequence-length`. V3 supervised/loss-masked shards are currently rejected rather than scored with the wrong all-token objective.

For metric interpretation and the current pilot trajectory, see [EVALUATION.md](EVALUATION.md).

## 7. Run inference

```sh
niyah run \
  --tokenizer tok.bin \
  --checkpoint model-0200.ckpt \
  --prompt "Hello" \
  --max-new-tokens 32 \
  --temperature 0 \
  --seed 42 \
  --backend cpu
```

`temperature=0` selects deterministic greedy sampling. Positive temperatures use the seeded stochastic sampler.

When the project is built with CUDA support, `--backend cuda` is also available.

## 8. Inspect data or a checkpoint with `niyah_probe`

`niyah_probe` always requires a tokenizer path, but a tokenizer-only invocation currently emits no report. Use it with a shard and/or with a checkpoint plus prompt.

Tokenizer + shard statistics:

```sh
niyah_probe --tokenizer tok.bin --shard shard.bin
```

Checkpoint prompt/logit inspection:

```sh
niyah_probe \
  --tokenizer tok.bin \
  --checkpoint model-0200.ckpt \
  --prompt "A sequence is" \
  --topk 10 \
  --trace-steps 8
```

The trace repeatedly samples through the same decode/sampler path used by generation and reports cache movement, selected token, logit/probability, and decoded bytes.

## 9. What a successful quickstart proves

A successful end-to-end run establishes that the selected repository revision can:

```text
corpus
  → tokenizer + shard
  → train
  → checkpoint + cursor
  → resume
  → held-out evaluation + baseline
  → load
  → generate
```

It does not establish broad language quality, reasoning ability, factual reliability, safety, or production readiness.

# Niyah.Engine

A clean-room, local-first C11 language-model foundation. This repository intentionally starts with a small deterministic baseline that can actually be built, trained, evaluated, saved, loaded, and tested without external model weights or cloud services.

## Current implemented baseline

- Pure C11 static library and CLI.
- Byte-level 256-symbol bigram language model.
- UTF-8-safe at the byte level: Arabic text is accepted as raw UTF-8 bytes without an external tokenizer.
- Add-one-smoothed evaluation with bits-per-byte and perplexity.
- Seeded deterministic generation.
- Versioned binary model format (`NIYAHBG1`).
- Linux and Windows CI.
- Unit and CLI smoke tests.

This is a verified bootstrap engine, not a claim of transformer-scale capability. It provides a stable executable training/evaluation contract on which tokenizer, tensor, checkpoint, optimizer, and GPU work can be added deliberately.

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

## Train

```sh
./build/niyah train --input corpus.txt --model model.nyh
```

Windows multi-config generators usually place the executable under `build/Release/niyah.exe`.

## Evaluate

```sh
./build/niyah eval --input heldout.txt --model model.nyh
```

## Generate

```sh
./build/niyah generate --model model.nyh --prompt "الرياضيات " --tokens 128 --seed 42
```

## Inspect

```sh
./build/niyah inspect --model model.nyh
```

## Evidence contract

A claim is accepted only when the corresponding command exits successfully and its output is preserved. Documentation, filenames, or prior runs are not substitutes for current execution evidence.

## Scope boundary

No external pretrained model is embedded or required. No Qwen, Llama, cloud API, telemetry SDK, or network dependency is part of the runtime.

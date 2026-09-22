# Niyah.Engine

Clean-room native C11 baseline for a deterministic, local-first language-model runtime.

This repository intentionally has **no dependency on Qwen, llama.cpp, cloud inference, Python training frameworks, or external model weights**. It is a new baseline, not a byte-for-byte recovery of any deleted repository.

## What is implemented

- portable SHA-256 with a known-answer test;
- deterministic UTF-8 byte tokenizer (256-token vocabulary);
- checksummed binary dataset shard format (`NIYDS1`);
- deterministic add-1 byte-bigram model;
- dataset-bound, checksummed checkpoint format (`NIYCK1`);
- new/resume training with byte-equivalent state evolution;
- evaluation with NLL / average NLL / perplexity;
- greedy local generation;
- CLI tools (`niyah`, `niyah-train`);
- Linux and Windows CMake/CTest CI;
- seven regression tests, including resume equivalence and dataset-binding rejection.

## Explicit non-claims

This baseline does **not** claim to be a transformer, neural-network replacement, CUDA trainer, GGUF runtime, or reconstruction of historical Niyah.Engine behavior. It is a small deterministic engine that establishes file formats, identities, training state, CLI contracts, and tests from a clean source tree.

## Build

### Linux / WSL

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### Windows PowerShell + Visual Studio

```powershell
cmake -S . -B build -A x64
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## Minimal end-to-end run

```powershell
Set-Content .\sample.txt 'مرحبا من نياه. Niyah is local.' -Encoding utf8NoBOM

.\build\Release\niyah.exe prepare `
  --input .\sample.txt `
  --output .\sample.srd

.\build\Release\niyah-train.exe new `
  --shard .\sample.srd `
  --checkpoint-out .\model.ckpt `
  --updates 10000

.\build\Release\niyah.exe eval `
  --shard .\sample.srd `
  --checkpoint .\model.ckpt

.\build\Release\niyah.exe run `
  --checkpoint .\model.ckpt `
  --prompt 'Niyah' `
  --max-tokens 64
```

## Repository layout

```text
include/niyah/   public C11 API
src/             implementation
tools/           command-line programs
tests/           regression tests
docs/            format and architecture notes
.github/          CI
```

## Evidence rule

A successful build or documentation statement is not runtime evidence by itself. Use CTest results, exact command outputs, hashes, and reproduced artifacts when making claims about a specific build.

# Niyah.Engine

Local-first Arabic technical language-model engine and training workspace.

The repository contains two deliberately separate layers:

1. **Native C11 runtime baseline** — buildable CLI/library with deterministic byte-level training/evaluation/generation contracts.
2. **Reference Transformer training path** — a small PyTorch causal Transformer used to prove end-to-end dataset, checkpoint, resume, evaluation, and generation behavior without depending on external pretrained weights.

No Qwen, Llama, hosted API, telemetry SDK, or downloaded model weight is required by either path.

## Native build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Native baseline commands:

```sh
./build/niyah train --input corpus.txt --model model.nyh
./build/niyah eval --input heldout.txt --model model.nyh
./build/niyah generate --model model.nyh --prompt "الرياضيات " --tokens 128 --seed 42
./build/niyah inspect --model model.nyh
```

Windows multi-config generators usually place the executable under `build/Release/niyah.exe`.

## Verified Arabic math dataset

Generate 10,000 deterministic records:

```sh
python tools/generate_verified_math.py \
  --out data/generated/verified_math_v1.jsonl \
  --records 10000 \
  --seed 1448

python tools/validate_verified_math.py \
  data/generated/verified_math_v1.jsonl \
  --expect-records 10000
```

Create a stable record-level train/validation split:

```sh
python tools/split_jsonl.py data/generated/verified_math_v1.jsonl \
  --train data/generated/train.jsonl \
  --validation data/generated/validation.jsonl \
  --validation-percent 5 \
  --seed 1448
```

The generator currently covers exact integer arithmetic, rational arithmetic, linear equations, GCD, modular inverses, and combinations. Answers are recomputed by an independent verifier. Generated corpora are reproducible and intentionally excluded from Git; the generator, seed corpus, validation code, and CI gate are committed.

## Reference Transformer

Install the training dependency:

```sh
python -m pip install -r requirements-train.txt
```

Train from scratch:

```sh
python python/niyah_ref.py train \
  --data data/generated/verified_math_v1.jsonl \
  --out artifacts/verified-math-v1/niyah-ref.pt \
  --steps 2000 \
  --batch 16 \
  --context 256 \
  --seed 1448
```

Generate from the resulting checkpoint:

```sh
python python/niyah_ref.py generate \
  --model artifacts/verified-math-v1/niyah-ref.pt \
  --prompt "حل المعادلة" \
  --tokens 128 \
  --temperature 0.8 \
  --top-k 40
```

Windows end-to-end entrypoint:

```powershell
pwsh -File scripts/train_verified_math.ps1 -Records 10000 -Steps 2000
```

That script records dataset SHA256, checkpoint SHA256, Git HEAD, Python/Torch version, and detected device after a successful run.

## Data evidence contract

A dataset claim is accepted only when provenance and verification are explicit. `SYNTHETIC_UNVERIFIED` is not silently promoted to training-approved data. The deterministic math generator emits `MECHANICALLY_VERIFIABLE` records, and CI regenerates, validates, splits, and compares them for reproducibility.

## Runtime evidence contract

A runtime claim is accepted only when the corresponding command exits successfully and its output is preserved. Documentation, filenames, previous chat output, or a prior run are not substitutes for current execution evidence.

## Current boundary

The reference Transformer is a training implementation, not a claim that the native C11 runtime already executes those Transformer checkpoints. Native Transformer export/import is a separate compatibility gate and must be proven before that claim is made.

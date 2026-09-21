# Contributing to Niyah.Engine

Niyah.Engine accepts changes that are narrow, testable, and supported by repository evidence.

## Before changing code

1. Identify the concrete requirement or demonstrated defect.
2. Inspect the relevant public interface, implementation, tests, and build wiring.
3. Reuse existing contracts before proposing a new API or dependency.
4. Keep the patch limited to the proven requirement.

Do not add infrastructure, frameworks, databases, services, or model features merely to make the project appear more complete.

## Build and test

CPU reference build:

```sh
cmake -S . -B build -DNIYAH_BUILD_TESTS=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

Sanitizer build on supported non-MSVC configurations:

```sh
cmake -S . -B build-sanitize \
  -DNIYAH_BUILD_TESTS=ON \
  -DNIYAH_ENABLE_SANITIZERS=ON \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-sanitize
ctest --test-dir build-sanitize --output-on-failure
```

CUDA changes should include CUDA-specific build/test evidence when the change affects the CUDA path. CPU success alone must not be presented as CUDA verification.

## Regression gate

A defect fix is complete when:

- the proven defect is addressed;
- relevant existing tests still pass;
- a focused regression test covers the defect when feasible;
- unrelated behavior is not changed merely to make the suite pass.

## Claim discipline

Pull requests should separate:

- **implemented behavior** — visible in source;
- **verified behavior** — demonstrated by a test/build/runtime result;
- **model-quality evidence** — measured on a specified dataset/evaluation method;
- **unestablished claims** — not yet supported by adequate evidence.

A successful build is not evidence of language quality. A falling training loss is not, by itself, held-out generalization. Passing a fixed validation set is not broad capability.

## Pull-request evidence

For material runtime/training changes, include the smallest relevant record:

```text
base commit SHA
changed files
build command + exit status
targeted test command + result
full regression result when relevant
sanitizer result when relevant
CUDA result when relevant
known unverified platforms/paths
```

For model experiments, additionally record:

```text
tokenizer/dataset identity or provenance
model configuration
seeds
optimizer settings
checkpoint identity
evaluation-set identity
metric output
```

## Documentation changes

Documentation must not promote local observations into repository-wide guarantees. If an experimental result does not include the exact repository SHA, state that limitation instead of assigning one retroactively.

Keep [docs/VERIFICATION.md](docs/VERIFICATION.md), [docs/EVALUATION.md](docs/EVALUATION.md), and [docs/MODEL_CARD.md](docs/MODEL_CARD.md) consistent when capability/evidence status materially changes.

## Style

The C core targets C11. Existing project warning/error policy and local file style take precedence over unrelated formatting churn.

Prefer small patches with explicit behavior over broad refactors without a proven requirement.

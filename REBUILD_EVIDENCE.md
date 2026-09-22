# Rebuild evidence

This file records only evidence reproduced in the clean-room workspace that produced this repository snapshot.

## Toolchain

- CMake 3.31.6
- GCC 14.2.0
- C standard: C11

## Release build

Configuration and compilation completed successfully with the repository warning flags enabled. The captured build contained no `warning:` diagnostics.

## CTest

Seven tests passed:

1. `sha256` — SHA-256 `abc` known-answer test.
2. `tokenizer` — UTF-8 byte round trip.
3. `dataset` — shard save/load and identity equality.
4. `checkpoint` — checkpoint save/load state equality.
5. `eval` — finite add-1 baseline perplexity.
6. `train_resume` — 7 + 10 resumed updates equal 17 uninterrupted updates.
7. `binding` — resume rejects a checkpoint bound to a different dataset.

Result reproduced in the build environment: `7/7 PASS`.

## Sanitizers

A Debug build with GCC AddressSanitizer and UndefinedBehaviorSanitizer completed and CTest reproduced `7/7 PASS`.

## End-to-end probe

A UTF-8 Arabic/English sample was prepared into a shard, trained for 1000 transition updates, evaluated, inspected, and passed through local generation. Reproduced values for that ephemeral sample:

```text
bytes=42
tokens=42
dataset_sha256=9160cc6b7a6fe907165f6f58bf71af50c1997a352ede69a6e3ad8856feceb933
updates_applied=1000
cursor_after=1000
checkpoint_eval_perplexity=11.305389651076
run_exit=0
```

The sample shard/checkpoint were removed after the probe.

## Not yet reproduced

- MSVC/Windows build in this workspace.
- CUDA compilation or GPU runtime.
- Compatibility with any historical Niyah.Engine binary/file format.
- Transformer training/inference.

# Architecture

## v0.1 execution path

`UTF-8 corpus -> byte transitions -> NIYAHBG1 model -> eval/generate`

The baseline deliberately uses bytes so the repository begins with a deterministic, dependency-free implementation. It is not intended to be the final tokenizer or final model architecture.

## Stable boundaries introduced now

- `include/niyah/niyah.h`: public C API.
- `src/`: implementation only.
- `tools/niyah.c`: CLI translation layer.
- `tests/`: executable evidence.
- Model files carry magic + format version + alphabet metadata.

Future tensor or tokenizer work should preserve testable boundaries rather than silently changing model semantics.

# R3/D3 evidence summary

R3/D3 is an external PEFT/QLoRA symbolic-logic adapter over `humain-ai/ALLaM-7B-Instruct-preview`; it is separate from the native NiyahModel checkpoint format.

## Artifact

- Adapter SHA-256: `c6a90800e51c3a8f8568cda42edbebf7334f9d81fbafb9a11e10a72b436d0fe7`
- Trainable parameters: `39,976,960`
- Train rows: `960`; dev rows: `240`
- Train SHA-256: `ed67fe365a29ebc7c3b1e7dda771fb44490c4a5a7bbbca4a0fe8bafe434e15c9`
- Dev SHA-256: `62c0ab46fe0c827b4a0d551635f4b022c4b0cd8536d353c5183547842dc86c06`
- Frozen V6 SHA-256: `a2d27bc8a249c0d32fdd33c7abc89323d50b1000aba9a18b111bc9334d4c6a63`
- Manifest records: `v6_used_for_training=false`, `v6_used_for_scoring=false`, `v6_errors_inspected=false`.

## Frozen V6 result أ¢â‚¬â€‌ 600 rows

| System | Correct | Accuracy |
|---|---:|---:|
| ALLaM-7B base (B0) | 357/600 | 59.50% |
| Qwen3-8B | 430/600 | 71.67% |
| Mistral-7B-Instruct-v0.3 | 412/600 | 68.67% |
| **R3/D3** | **600/600** | **100.00%** |

Four-model comparison receipt SHA-256: `0c1277134fea387a67ae71a4f60863d1dc9cb828bd3d1218079f3c58e81b96f2`.

## Release

Hugging Face: https://huggingface.co/sulaimanalshammari/R3-D3-ALLaM-7B-Logic-Adapter

## Scope

These are task-specific symbolic-entailment measurements under the recorded binary scoring protocols. They are not claims of general model superiority.

**NO CLAIM BEYOND THE HASH.**

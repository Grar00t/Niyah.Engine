# llm64 — syscall-only Niyah inference target

This directory is a separate Linux x86-64 ELF target. It does not reuse the DOS execution model from `opcode-orchestra`, and it does not link libc, libm, BLAS, llama.cpp, Python, or another model runtime.

Current stage: **validated loader + scalar math primitives**.

Implemented now:

- raw Linux `openat`, `fstat`, `mmap`, `munmap`, `close`, `write`, and `exit` syscalls;
- current 72-byte Niyah checkpoint V2 header parsing;
- exact canonical FP32 weight-count recomputation from Niyah geometry;
- V2 section-set validation: duplicate known sections rejected, required known sections enforced, unknown required sections rejected, optional LR-schedule shape checked;
- model/AdamW tensor section size checks;
- IEEE CRC32 verification over checkpoint/tokenizer payloads;
- NaN/Inf rejection for model weights;
- Niyah tokenizer v1 header and merge-order validation;
- scalar FP32 row-major matvec;
- RMSNorm with FP64 square accumulation followed by the C reference's FP32 normalization path;
- stable greedy argmax primitive;
- no dynamic runtime dependency.

Not implemented yet:

- checkpoint tokenizer-identity SHA-256 comparison against the supplied `tok.bin`;
- persistent mapped model/tokenizer state;
- tokenizer encode/decode buffers;
- RoPE and exp/softmax;
- grouped-query attention and persistent KV cache;
- SwiGLU / complete Transformer block wiring;
- autoregressive decode and generation.

The intended endpoint is genuine autoregressive inference using existing Niyah weights, not a conversion of the DOS music engine.

## Build

```sh
cd llm64
make
```

The executable is linked directly with `ld`:

```text
./niyah-asm model.ckpt tok.bin
```

At this stage, a successful invocation establishes that the V2 checkpoint and tokenizer file each pass the implemented structural and CRC checks. It does **not** yet establish that the checkpoint's stored tokenizer SHA-256 identity matches the supplied tokenizer file.

On success it prints:

```text
NIYAH_ASM64_LOADER=OK
```

It does **not** claim Transformer inference is complete yet.

## Verification

```sh
make clean all check
make test-formats
```

`test-formats` uses Python only as a CI/test-fixture generator. The produced `niyah-asm` runtime remains NASM + `ld` with no Python or C runtime dependency.

The black-box format regression covers a valid V2 file, the optional LR-schedule section, unknown required sections, missing required tokenizer identity, a known section missing its required flag, a zero LR schedule, and non-finite model weights.

## Runtime boundary

`niyah-asm` depends only on the Linux x86-64 syscall ABI. Build and CI tools are outside that runtime boundary.

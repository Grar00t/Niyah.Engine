# llm64 — syscall-only Niyah inference target

This directory is a separate Linux x86-64 ELF target. It does not reuse the DOS execution model from `opcode-orchestra`, and it does not link libc, libm, BLAS, llama.cpp, Python, or another model runtime.

Current stage: **loader foundation**.

Implemented now:

- raw Linux `openat`, `fstat`, `mmap`, `munmap`, `close`, `write`, and `exit` syscalls;
- current 72-byte Niyah checkpoint V2 header parsing;
- exact canonical FP32 weight-count recomputation from Niyah geometry;
- section scanning with model-weight section size checks;
- IEEE CRC32 verification over checkpoint/tokenizer payloads;
- NaN/Inf rejection for model weights;
- Niyah tokenizer v1 header and merge-order validation;
- no dynamic runtime dependency.

Not implemented yet:

- persistent mapped model/tokenizer state;
- tokenizer encode/decode buffers;
- FP32 matvec/RMSNorm/RoPE/softmax/SwiGLU;
- KV cache and grouped-query attention;
- autoregressive decode and greedy argmax.

Those are the next vertical slices. The intended endpoint is genuine autoregressive inference using existing Niyah weights, not a conversion of the DOS music engine.

## Build

```sh
cd llm64
make
```

The executable is linked directly with `ld`:

```text
./niyah-asm model.ckpt tok.bin
```

At this stage, a successful invocation validates the checkpoint/tokenizer pair structurally and prints:

```text
NIYAH_ASM64_LOADER=OK
```

It does **not** claim Transformer inference is complete yet.

## Runtime boundary

`niyah-asm` is intended to depend only on the Linux x86-64 syscall ABI. Build and CI tools are outside that runtime boundary.

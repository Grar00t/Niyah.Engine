# niyah-kernel

Low-level SIMD inference primitives and GGUF model loader for the NIYAH engine.

Written in C11. Zero external dependencies beyond libc and libm.
Targets: ARM64 (NEON), x86-64 (AVX2+FMA), scalar fallback.

---

## Contents

| Path | Description |
|---|---|
| `niyah/src/niyah_simd.c` | Q4_0/Q8_0 dequant dot products, SGEMV, softmax, RMSNorm |
| `niyah/src/niyah_gguf.c` | GGUF v1/v2/v3 zero-copy loader (mmap) |
| `niyah/src/test_simd.c` | Unit tests for all SIMD primitives |
| `niyah/include/niyah_simd.h` | Public API for SIMD primitives |
| `niyah/include/niyah_gguf.h` | Public API for GGUF loader |
| `kernel/configs/khawrizm_defconfig` | ARM64 Linux kernel defconfig |

---

## Build

### Linux / macOS

```bash
# Release (auto-detects AVX2 or NEON)
make

# Debug + sanitizers
make debug

# Run unit tests
make test
```

### CMake

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

---

## SIMD Primitives

- **`niyah_dot_q4_0_fp32`** — Q4_0 quantized dot product (dispatch: NEON / AVX2 / scalar)
- **`niyah_sgemv_f32`** — FP32 matrix-vector multiply
- **`niyah_softmax_f32`** — numerically stable in-place softmax
- **`niyah_rmsnorm_f32`** — RMS normalization with weight vector

## GGUF Loader

- Supports GGUF v1, v2, v3
- Zero-copy: tensor data pointers map directly into the mmap region
- Handles F32, F16, Q4_0, Q8_0 tensor types

---

## Requirements

- C11 compiler: GCC 7+, Clang 6+
- Linux or macOS (GGUF loader uses `mmap`)
- No external libraries

---

## License

Public Domain

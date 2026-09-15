# Niyah.Engine — Native LLM Inference Core

Sovereign C11/CUDA implementation of transformer inference.
No external LLM runtime. No cloud dependencies. Deterministic execution.

## Architecture
- GGUF weight loader (zero-copy mmap)
- SIMD-optimized kernels (AVX2/NEON)
- CUDA backend (RTX 3060, Compute 8.6)
- SHA-256 proof receipts

## Build
\`\`\`bash
cmake -S . -B build -G Ninja
cmake --build build
./build/niyah_test_simd
\`\`\`

## License
KhZ License — Sovereign Non-Commercial 1.0

# Niyah native verification - 2026-10-27

This report records a local, isolated verification. It is not evidence of model quality or a release.

## Windows
- LLVM Clang 23.1.4 and Ninja 1.13.2
- CMake configure: exit 0
- Ninja build: exit 0
- CTest: 56 discovered, 54 passed, 2 skipped (external Casper), 0 failed
- Logs retained locally under D:/SYSTEM_AUDIT/cleanup-20261027/clang23-build.log

## Linux
- Kali WSL2 POSIX release verification: 56 discovered, 53 passed, 3 skipped, 0 failed

## Training smoke test
- Local CPU-only 1-layer miniature model; 36 optimizer steps on clean10-final-2582
- Held-out probe: 786 tokens, 4 samples; loss 8.29880577 before vs 8.29765682 after
- Output generation remains incoherent. Not suitable for deployment.
- No private dataset, model weights, or checkpoint files published.

## Limitations
- The held-out probe is too small to establish quality or generalization.
- Data leakage and overlap not independently ruled out.
- CUDA training not verified by this CPU run.
- No claim of production readiness.

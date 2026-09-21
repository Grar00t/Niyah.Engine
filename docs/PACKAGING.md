# Packaging

Niyah.Engine `v0.1.0` uses CMake install rules and CPack. The release package is intentionally CPU-focused unless a release explicitly says otherwise.

## Install from an existing Release build

POSIX:

```sh
cmake --install build/verify-posix-release --prefix build/stage
```

Windows / Visual Studio:

```powershell
cmake --install build/verify-windows-release --config Release --prefix build\stage
```

The staged tree contains:

```text
bin/
  niyah
  niyah-train
  niyah_probe
lib/
  libniyah.a or niyah.lib
include/
  niyah/*.h
share/doc/NiyahEngine/
  LICENSE
  README.md
```

Platform-specific executable and library suffixes are determined by the active toolchain.

## Build a package with CPack

Linux example:

```sh
cd build/verify-posix-release
cpack -G TGZ
```

Windows example:

```powershell
Set-Location build\verify-windows-release
cpack -C Release -G ZIP
```

Package names use the project version and CMake platform identity:

```text
Niyah.Engine-0.1.0-<system>-<processor>
```

## Release boundary

The first release package includes the native static library, public headers, and the three primary CPU CLI tools. Tests, development fixtures, diagnostic build products, datasets, checkpoints, and pretrained model weights are not release payloads.

CUDA support remains optional source-build functionality and is not implied by a CPU release artifact unless the artifact is explicitly labeled and verified as CUDA-enabled.

A package existing is not evidence of model quality or production readiness. Release claims remain limited to the behavior reproduced by the release gate and documented verification evidence.

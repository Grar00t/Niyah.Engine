# AGENTS.md

This file defines the working contract for coding agents and automated contributors operating on Niyah.Engine.

## Repository boundary

Work only from the state visible in this repository and the explicitly named base revision. Do not import assumptions, APIs, architecture, or claims from other projects.

Niyah.Engine is a native C11 language-model runtime and training engine. Optional CUDA support is additive; CPU FP32 remains the reference path unless a task explicitly says otherwise.

## Required workflow

For a code change:

1. inspect the relevant public header, implementation, tests, and CMake wiring;
2. identify the smallest demonstrated defect or requirement;
3. reuse existing contracts before adding a new one;
4. implement the narrowest patch;
5. add or adjust a focused regression test;
6. run the relevant targeted test and the repository verification gate;
7. report exact evidence and any remaining limitation.

Do not replace evidence with README claims, comments, or intended behavior.

## Canonical verification

POSIX / Linux release:

```sh
bash scripts/verify.sh release
```

Windows PowerShell release:

```powershell
pwsh -NoProfile -File scripts/verify.ps1 -Configuration Release -Jobs 2
```

Sanitizers on supported non-MSVC hosts:

```sh
bash scripts/verify.sh sanitize
```

CUDA changes require CUDA-specific evidence. CPU success is not CUDA verification.

## Change discipline

Do not:

- invent public APIs or silently rename existing contracts;
- expand scope because a neighboring subsystem could be improved;
- add frameworks, services, databases, network layers, or dependencies without a concrete requirement;
- change checkpoint/dataset formats as a side effect of an unrelated task;
- weaken validation merely to make a failing test pass;
- present a local model output as broad capability evidence;
- treat validation-set improvement as a benchmark or production claim;
- claim CPU/CUDA equivalence without a test proving the specified equivalence.

Prefer fail-closed behavior when the current API cannot represent correct semantics.

## Public contracts

Public headers under `include/niyah/` are contracts. Before adding a symbol, prove that existing symbols cannot express the required behavior cleanly.

Persistence changes must preserve explicit versioning, identity binding, corruption detection, and cleanup-on-failure behavior.

Evaluation changes must state which target tokens are scored and ensure any baseline uses the same evaluation geometry.

## Tests

Regression tests should prove the defect rather than merely increase coverage. Include invalid-input and cleanup behavior when the changed contract can fail.

Warnings are errors in supported C builds. Keep changes C11-compatible.

## Documentation and claims

Separate:

- implemented behavior;
- verified behavior;
- measured model-quality evidence;
- unestablished claims.

When evidence is incomplete, write the limitation explicitly. Do not assign an old local experiment to a repository SHA unless that SHA was actually recorded with the experiment.

## Pull requests

Keep PRs reviewable. A PR should normally have one primary objective, a stated scope boundary, a regression gate, and no unrelated formatting churn.

If review finds a real but separable issue, either fix it within the objective or track it explicitly as a follow-up; do not mark it solved by documentation alone.

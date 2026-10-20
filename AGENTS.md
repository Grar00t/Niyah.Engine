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

## Base44 sandbox environment

This section describes the sandbox preview harness only. It does not change the project build, public contracts, or the canonical verification gate; CI and releases continue to use `scripts/verify.sh`.

- Start with `docker compose -f docker-compose.base44.yml up -d --build`. The service builds the tools from the bind-mounted working tree inside `niyah-base44-toolchain:local` (Debian + gcc + cmake; no source is baked into the image) and then serves the console on host port 3000.
- Niyah.Engine is a CLI toolchain and serves no HTTP surface of its own, so the preview needs something on port 3000: `sandbox/` is a standard-library-only console that shells out to the compiled `niyah`, `niyah-train` and `niyah_probe` binaries and streams their real stdout. It performs no engine work itself and adds no dependency to the CMake build.
- No external credentials are required: the model core needs no hosted API or external LLM runtime.
- The console writes nothing inside the repository. Run artifacts (tokenizer, shard, checkpoints, cursors) go to `/tmp/niyah-console/runs/` inside the container, so a container recreate discards them.
- `sandbox/corpus.txt` and `sandbox/heldout.txt` are small synthetic fixtures for the demo console only. They are not training data for any claim in `docs/`.
- There is no hot reload for C. After editing `src/`, `include/` or `tools/`, press **Rebuild tools** in the console or run `docker compose -f docker-compose.base44.yml exec engine cmake --build build/base44 --parallel 2` before the next engine run.
- Two constraints that bite when shaping a demo corpus:
  - supervised shards require each whole record to fit: `prompt_tokens + response_tokens + 1 <= --sequence-length`, otherwise `niyah prepare` exits with `shard_build NIYAH_ERR_INVALID_CONFIG` (the demo uses `--target-vocab 512` so records compress under 64 tokens);
  - `niyah run` requires `1 + prompt_tokens + max_new_tokens <= context_length`, otherwise it exits with `context_capacity` (the demo presets use a 128-token context to leave room for 48 generated tokens).
- Verification in the sandbox: the console's **Run regression suite** button runs configure + build + `ctest --output-on-failure` (39/39 passing at the revision recorded in `.base44/environment.json`), or run `docker compose -f docker-compose.base44.yml exec engine bash scripts/verify.sh release`.
- `BASE44_PREVIEW_MODE` is passed through to the container, but no application behavior is gated on it: the harness is confined to `sandbox/` and is started only by `docker-compose.base44.yml`.

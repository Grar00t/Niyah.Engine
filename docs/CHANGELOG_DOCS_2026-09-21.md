# Documentation Refresh — 2026-09-21

This note records the scope of the documentation-only refresh branch.

## Goals

- provide a polished repository landing page;
- make the documentation navigable from a single index;
- separate architecture, training, data, evaluation, model status, and verification concerns;
- visualize the native lifecycle and measured held-out trajectory;
- document current CLI surfaces from source;
- preserve evidence boundaries and explicit non-claims.

## Evidence incorporated

Repository evidence:

- base `main`: `732f84fc34b2b5cad2ac40b4195e293fc2fad9aa`;
- successful `core-ci` push workflow for that commit;
- successful CodeQL push analysis;
- source-backed `niyah`, `niyah-train`, and `niyah_probe` command surfaces.

Local diagnostic evidence:

- completed resume stage to optimizer step 1905;
- mean stage training loss `3.74314785`;
- four-checkpoint validation trajectory from loss `5.581078354` to `3.643954027`;
- perplexity trajectory from `265.357600904` to `38.242751050`;
- evaluation exit `0`.

## Important boundary

The supplied local model-training transcript does not include the exact repository SHA used to build its binary. The refreshed documentation therefore records those model results as local diagnostic evidence and does not claim they were reproduced specifically at the documentation base SHA.

## Code scope

No model/runtime source, CMake behavior, tests, CI workflow, or public C API is changed by this documentation refresh.

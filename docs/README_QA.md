# Documentation QA

Documentation refresh acceptance checklist:

- [x] README links only to repository paths created or already present on the branch.
- [x] Current CLI syntax is derived from `tools/niyah.c`, `tools/niyah_train.c`, and `tools/niyah_probe.c`.
- [x] `niyah eval` is not documented as a merged command.
- [x] Held-out values match the supplied evaluation transcript.
- [x] The repeatedly inspected 30-record set is labeled validation-like.
- [x] Local model results are not retroactively assigned to repository base `732f84f`.
- [x] PR #56 Evidence V1 functionality is explicitly identified as unmerged.
- [x] No source/runtime/CMake/test/CI behavior is changed by the documentation branch.
- [x] Visual assets are repository-hosted SVGs.
- [x] Broad capability and production-readiness claims remain unestablished.

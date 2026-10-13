# Shared-guard evidence follow-up: 2026-10-13

The corrected shared-guard candidate has saved evidence of four selected CPU
tests and two selected CUDA tests passing in an isolated source copy. It remains
unapplied to the review checkout at dirty base
`6ec58869fca15cb6034bd5a689e9a04483651d4c`. The original checkout remains at dirty
base `2a0ec79faf28c42cd4fac49c47259df8bc95a9a5`. No publication is implied.

This follows the [initial audit](LOCAL_AUDIT_2026-10-13.md) and updates only the
candidate's artifact identity and execution evidence. The initial audit,
documentation patch and check receipts are preserved as byte copies under
`D:\NiyahChecks\documentation-followup-20261013-shared-guard\initial-receipt`.
The original delivery folder `D:\NiyahChecks\documentation-audit-20261013`
is unchanged. The follow-up has its own documentation patch and check receipt.

## Locations and identity

- P: `D:\NiyahChecks\guard-review-99ace539e8be4e0bb8358766c26ef71f\patch-output`
- I: `D:\NiyahChecks\guard-cpu-test-56f4ba5fd8`
- R: `D:\Niyah.Engine-review-14480502-115310`

The current `P/manifest.json` lists 11 candidate files. All 11 current candidate
files and all 11 corresponding `I/source` files match its after-hashes. All nine
replacement preimages match both `P/baseline` and R; the two new target files
remain absent from R. The corrected patch passes an independent
`git apply --check` against R. This check does not apply the patch.

| Artifact | Raw SHA-256 |
|---|---|
| P `Niyah.SharedGuard.fixed.patch` (58,974 bytes) | `cd040652b20069cfbea6d9b2f1d2462c5ed0405c1470702996e10b4f181305a4` |
| P retained `Niyah.SharedGuard.v1.patch` | `c8e37a153c2e473d170003eb458e8277284e99e4ef3b944d1e5e076ca412296d` |
| P current `manifest.json` | `b207ba52e299caf267d1aff0623acf363f7ba6571d779951a6d467324beed77f` |
| Initial manifest, recorded in the initial audit | `d83d96651f35624753e44a2809790c236344ca1730a41b516153d235867518d7` |
| Corrected candidate `src/niyah_cuda_matvec.h` | `f8650f50a773ba59bb7f0120238b713e957873d93e798fc8f8dfdc7f1478893c` |

The patch correction removes one unconditional `extern "C"` from the added
prototype in `src/niyah_cuda_matvec.h`; the header already has conditional C++
linkage around its declarations. `I/CPU_BUILD.log` records the preceding C2059
syntax error at that header's line 250. `I/CPU_REBUILD.log` records the subsequent
successful build of the selected CPU test targets and CLI. The v1 patch is
retained; its manifest was overwritten by the
corrected candidate's manifest. The initial manifest hash must not be assigned
to the current manifest or corrected patch.

Both CMake caches point to `I/source`. CPU has CUDA disabled; the CUDA build has
CUDA enabled and architecture 86. Configure logs identify MSVC 19.51.36257.0,
NVIDIA 13.4.92 and Visual Studio 18 2026. These logs and the current 11-file
identity check do not establish a frozen whole-source snapshot at build time.

## Saved execution evidence

| Receipt under I | Selected tests | Passed / failed / skipped | Recorded duration |
|---|---|---|---|
| `CPU_TESTS.log` | `niyah_generation_test`, `niyah_repetition_control_test`, `niyah_generation_repetition_test`, `niyah_shared_guard_test` | 4 / 0 / 0 | 0.39 seconds |
| `CUDA_TESTS.log` | `niyah_cuda_generation_test`, `niyah_cuda_generation_repetition_test` | 2 / 0 / 0 | 0.43 seconds |

The generated CTest files register 57 CPU and 72 CUDA tests; the saved summary
logs above report only the named selections. The control test uses decoder test
doubles, while the generation repetition tests use actual tiny models.
`build-cuda/Testing/Temporary/LastTest.log` also records the two CUDA selections
and real tiny CPU/CUDA repetition and CUDA generation backend pass markers.
The CPU temporary `LastTest.log` was overwritten by a later partial selection
during this inspection; it is not the durable receipt for the four-test run.
The separate `CPU_TESTS.log` remains unchanged. Component test doubles remain
distinct from the actual tiny-model CPU/CUDA executions.

These results verify the selected paths in the isolated candidate. They are
neither complete 57/72-test suite results nor the canonical repository release
gate. The two external Casper skips in the earlier 71-test CUDA receipt belong
to that earlier suite; there are no skipped tests in these six selections.

| Current local artifact under I | Raw SHA-256 |
|---|---|
| `CPU_TESTS.log` | `02aafff4fd78e726d24e2f60feaf44ad24035e4862856eeec85b6db5512cdf52` |
| `CUDA_TESTS.log` | `cb82758ea691f679928180486beb2a2aa96cab76bbcd6863aa9c6162bdb260c6` |
| `build/Release/niyah.exe` | `56c5b5abe909f2501c56f96c07b3c53469cfed091ae7ccad994c86f9ec86fa75` |
| `build-cuda/Release/niyah.exe` | `ea32bab8d02aea2e15df8ed308e62c940c854c27100269076852cda74d432315` |

Binary hashes are audit-time identities, not contemporaneous metadata for every
reported command. P's unchanged `STATUS.txt` still says `CUDA_NOT_RUN`; that is
the original generation-stage receipt. The later logs establish isolated CUDA
execution without establishing application to either repository checkout.

## Behavioral and quality boundaries

The candidate adds shared token-cycle stops and `_ex` stop reports to ordinary
and repetition CPU/CUDA generation. Legacy wrappers also call the guarded
implementation, so default behavior changes: periods two through eight stop
after four repetitions. The CLI reports a stop reason and uses exit 20 for
degraded entropy/loop stops. Deadline/cancellation, Arabic-text guards and
session-store fencing helpers remain outside the wired generation/session path.
The header linkage correction does not expand that scope or repair the separate
source findings in the initial audit.

The selected real-model tests exercise candidate generation paths, but no
dedicated real-model assertion of the new cycle stop report or CLI regression
for degraded exit 20 was established. Comparing two changed candidate APIs
does not establish equivalence to pre-patch default behavior.

The supplied later terminal transcript reports a CUDA probe using
`--max-new-tokens 1`, penalty 1, `NIYAH_OK`, `TOKEN_BUDGET`, cycle period 0,
`context_commit_allowed=0`, `quality_status=not_validated`, and exit 0. No
separate saved raw receipt for that final probe was located locally; it is
transcript evidence. This probes the disabled-penalty/token-budget branch, not
loop detection, degraded exit 20, language quality or training provenance.
The named `D:\NiyahChecks\weighted-1fcba38ff6\mixed-update-0100.ckpt` currently
hashes to `44092057a90deaebd2bfc235d53b2ba637ca830e277ffd78457fff99211155e9`.
That identity alone establishes none of those broader claims.

This follow-up ran read-only artifact and documentation consistency checks.
It did not apply production patches or rerun builds, tests, generation,
evaluation or training. All source files and user work in both checkouts were
preserved; only the two cross-reference documents and this new document changed.

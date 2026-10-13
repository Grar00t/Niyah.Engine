# Local and GitHub evidence audit: 2026-10-13

This audit compares published revisions, both local working trees, source
contracts, and saved build/model receipts. It is a documentation and evidence
audit, not an exhaustive correctness certification. No training or evaluation
was repeated, and no checkpoint, corpus, source code, branch, or commit was
changed. Documentation edits apply only to the review checkout.

## Revision and working-tree boundary

| Location | Revision observed | Meaning |
|---|---|---|
| GitHub `main` | `dd8a621223e59d3cd3d8d920a48653ce94192993` | Published baseline |
| GitHub `feature/oracle-engine-three-lobe-20261002` | `efa0a5cb09d998aa2263ff6239cb2774f89d42c4` | Five commits ahead of main, zero behind |
| `D:\Niyah.Engine` | `2a0ec79faf28c42cd4fac49c47259df8bc95a9a5` | Dirty original checkout; six local-only and two remote-only commits versus the current feature tip |
| `D:\Niyah.Engine-review-14480502-115310` | `6ec58869fca15cb6034bd5a689e9a04483651d4c` | Dirty review checkout; one commit behind the feature tip |

The remote feature's one additional commit adds a draft document. It does not
publish the local persistence, tool, numerical-validation or repetition repairs.
`git cherry` identifies original commit `2a0ec79` as patch-equivalent to remote
work; the other five local commits are distinct. Commit absence alone therefore
does not show that a fix is missing. Remote refs were checked with `git
ls-remote`; the exact feature object was fetched without changing branches.

The pre-edit inventory hashes source, headers, tools, tests, scripts, docs,
evaluation fixtures, workflow/configuration files, and root text files. It
contains 199 files in the original checkout and 201 in the review checkout:
189 common files are byte-identical, seven common files differ, three are
original-only, and five are review-only. This inventory excludes build outputs,
model runs, corpus contents and Git internals; it is not a full disk digest.

The seven differences are `CMakeLists.txt`, `include/niyah/generate.h`,
`include/niyah/sampler.h`, `src/niyah_generate.c`, `src/niyah_sampler.c`,
`tests/check_niyah_run.cmake`, and `tools/niyah.c`. Review-only files are the
repetition document, CUDA repetition source/header, and two repetition tests.
The original alone contains `niyah-engine-run.py`, `tools/audit_exact_duplicates.py`
and the local Kaggle recovery script. All are preserved. Existing persistence
and corpus-tool docs are present in both trees, superseding earlier receipts
that described them as missing.

The review checkout is the later tested implementation for this audit because
its repetition code, saved build source directory and generation binary agree.
Neither clean HEAD represents its dirty source. A contemporaneous full-source
manifest was not found for the later CUDA build; this audit's new inventory
cannot retroactively supply that missing binding.

## Original test receipts

Local path aliases:

```text
O = D:\Niyah.Engine
R = D:\Niyah.Engine-review-14480502-115310
H = O\build\repair-consolidated-20261012T191010Z-cc4bf402
C = R\build\repetition-20261013T104406Z-83e38e3920fe4243b1f3edf780558fca
B = R\build\cuda-repetition-verified-a2ecbdc9a75b4e49a8ca1da805c6fd8f
G = B\generation-comparison-fc2da5b604244836a982e6bd3c724f19
W = O\runs\wiki-ar-14480501T202204Z-66bce91917834d73af22930c50c5047f
```

| Saved gate | Registered | Passed | Skipped | Failed | Scope |
|---|---:|---:|---:|---:|---|
| H: CPU Release | 54 | 52 | 2 | 0 | Older original repair snapshot |
| H: CUDA Release | 68 | 66 | 2 | 0 | Older original repair snapshot |
| H: sanitizer | 54 | 51 | 3 | 0 | Older sanitizer snapshot; profiler also unavailable |
| C: `cpu-full.log` | 56 | 54 | 2 | 0 | Review repetition integration, 14.39 s |
| B: `tests.log` and `tests.xml` | 71 | 69 | 2 | 0 | Later CUDA architecture 86, 17.02 s |

The two recurring skips are `niyah_casper_external_mmap_test` and
`niyah_casper_external_identity_test`: the required external model path and
SHA-256 were not configured. They are not passed tests and are not CUDA skips.
H's third sanitizer skip is `niyah_profile_nsys_test`. When that wrapper test
does pass, it uses an Nsight stub; it is not a GPU performance measurement.

H's `verification-summary.json` records source-manifest digest
`6e6f5b4c4ef41de62ed8875dcc8971a94c5268911602ab096e1f187ebd837394`
and no source mismatches. Two separate H compute-sanitizer logs record zero
errors for CUDA generation and accumulated training. Those receipts concern
that older snapshot and cannot be combined into a final-tree aggregate pass.

C's receipt reports its CUDA configure attempt as blocked. B separately records
the later successful configure/build/test with `NIYAH_ENABLE_CUDA=ON`, architecture
`86`, NVIDIA compiler `13.4.92`, MSVC `19.51.36257.0`, and R as source directory.
The earlier configuration failure is superseded for that later build only.

The canonical Windows gate sets `NIYAH_BUILD_TESTS=ON`; a fresh default build
does not enable CUDA. CUDA-specific evidence requires an explicit CUDA build.
The repository's `cuda-compile` workflow builds and runs `ctest -N` to enumerate
tests; it does not execute GPU tests. Workflow definitions do not establish that
any current GitHub run passed. Historical CI claims remain dated in
[Verification](VERIFICATION.md).

## CPU/CUDA parity and generation quality

The source tests establish bounded FP32 comparisons: full-backward loss and
gradients within absolute `2e-4` on tiny fixtures; accumulation compares two
updates with unequal supervised lengths, weights within `1e-5`, moments within
`2e-5`, and equal step counts. The accumulation comparison uses CPU AdamW on
both paths after CUDA backward/device accumulation. It is not an all-GPU
optimizer or bitwise-equivalence claim. Generation tests compare selected
greedy/seeded tiny-model output IDs; the repetition test adds enabled greedy
CPU/CUDA comparison and disabled legacy equivalence.

G contains 18 runs: three Arabic prompts, penalties `1.0`, `1.1`, `1.2`, and
CPU/CUDA backends. Temperature is `0`, seed `42`, maximum new tokens `48`,
window `64`. All runs exit zero and have valid strict UTF-8; all nine paired
raw stdout outputs match. Rehashing saved stdout confirmed 18/18 receipt hashes.
This proves equality for those greedy samples, not arbitrary prompts or hardware.

The saved metric is repeated **whitespace-word trigrams**, not tokenizer-token
trigrams. Counts at penalty `1.0` are 11, 7 and 8; all are zero at `1.1` and
`1.2`. Saved output still contains irrelevant city/date fragments, malformed
numbers, and repeated characters. Reduced repetition and valid UTF-8 do not
establish coherent Arabic answers, factuality, reasoning, or broad chat quality.

Both local ordinary and repetition-enabled generation paths retain stops for
raw entropy below `0.05` and eight identical emitted tokens. The entropy guard
runs before sampling and may suppress even a confident legitimate first token.
The public result does not expose a distinct guard stop reason. At published
feature `efa0a5c`, ordinary CUDA lacks these CPU guards and the CLI lacks the
repetition options. Local matching samples do not establish remote parity.
The isolated C/C++ shared-guard component tests are separate work; the current
production paths still contain their existing guards. No integration into either
repository checkout is claimed; isolated candidate evidence follows.

| Source state | CPU entropy / same-token guards | Ordinary CUDA guards | Repetition CLI |
|---|---|---|---|
| Published main `dd8a621` | Absent | Absent | Absent |
| Published feature `efa0a5c` | Present | Absent | Absent |
| Original local dirty tree | Present | Present | Absent |
| Review local dirty tree | Present | Present | Present |

Claims about these guards must name the source state. In particular, the
feature branch's CPU/CUDA stop mismatch must not be assigned to main.

### Initial shared-guard proposal inspection

This subsection preserves the initial proposal inspection. A later corrected
patch was built and tested in an isolated copy; its current identities and
selected CPU/CUDA results are recorded in the
[shared-guard follow-up](SHARED_GUARD_EVIDENCE_2026-10-13.md).
Neither proposal has been applied to R. The manifest hash below identifies the
initial manifest, which was subsequently replaced during the correction.

A later proposal exists at
`D:\NiyahChecks\guard-review-99ace539e8be4e0bb8358766c26ef71f\patch-output\Niyah.SharedGuard.v1.patch`,
SHA-256 `c8e37a153c2e473d170003eb458e8277284e99e4ef3b944d1e5e076ca412296d`.
Its initial manifest SHA-256 was
`d83d96651f35624753e44a2809790c236344ca1730a41b516153d235867518d7`.
At that inspection, all nine existing-file preimages matched R, both new target
paths were absent, and all 11 candidate hashes matched the manifest. An
independent `git apply --check` returned zero against R. This verified target
applicability, not compilation,
integration correctness, or CUDA execution. The patch was not applied.

The candidate adds stop-report `_ex` APIs and shared token-cycle detection to
ordinary and repetition CPU/CUDA paths. Legacy wrappers call the new guarded
implementation too: multi-token cycle stopping changes default behavior,
including penalty `1`: cycles of periods two through eight stop after four
repetitions. The candidate CLI writes raw output, reports a stop
reason and returns exit `20` for degraded entropy/loop stops. Consumers would
need to handle that exit explicitly. Deadline/cancellation, Arabic-text checks,
and session-store fencing helpers are not integrated into generation/session
state, as its `STATUS.txt` also states. This proposal does not repair the three
open source findings below. At this initial stage, its standalone component
results did not establish full-engine or CUDA integration.
The added tests cover the header and repetition control doubles; no new CLI
regression for stop diagnostics/exit `20` or real CPU/CUDA `_ex` integration
test was found. Comparing two changed candidate APIs does not establish
equivalence with the pre-patch default behavior.

## Persistence and failure behavior

The local serializer and CLI publication repairs are present in both checkouts.
Public serializers permit replacement through a private staged file; CLI
publication uses exclusive no-replace links. A failed CLI operation must not
remove an existing final destination. The detailed contract and regression
coverage are in [Persistence protocol](PERSISTENCE_PROTOCOL.md).

The trainer retains an eight-byte `.pending` reservation (`NIYAHPI1`) and writes
a 72-byte `.complete` last (`NIYAHPR1` plus both raw file hashes). Resume rejects
pending without completion. Valid completion is accepted with or without
pending; absence of both selects legacy binding checks. Retained pending does
not mean a completed pair is corrupt. This is not atomic two-file publication,
cryptographic authentication, or a power-loss durability guarantee. Single
checkpoint inference/evaluation/fine-tuning do not assert pair completion.

Published `efa0a5c` has an early-error data-loss path in `niyah shard`: a missing
tokenizer jumps to cleanup before checking the output, and cleanup removes the
requested shard path. The local repair instead cleans only owned staging files;
`tests/test_publication.py:test_cli_early_failures` covers this case. An additional
isolated check during this audit invoked the saved B binary with a missing
tokenizer and a pre-existing 49-byte sentinel. It returned exit `1`,
`error_stage=tokenizer_path`, and preserved the sentinel SHA-256
`526b8890f8151f9945bea372f69921f65b52a772aae8fb843b10f356a5e99780`.
The receipt is
`D:\NiyahChecks\shard-preservation-78fccdaa8014403c8e1fd7bce27473ef\receipt.json`.
No real corpus/shard was used, and no production fix was made by this audit.

For two output spellings that alias an initially absent destination, local
prepare and trainer still use literal string checks. Exclusive publication
provides the actual protection: after the first artifact claims the name, the
second link fails without overwriting it. Partial output may remain, without
overall success or pair completion. Existing tests cover publication aliases,
but no dedicated regression for this exact initially-absent two-output case
was found. That case is a static conclusion, not a new execution result.

## Open source findings

These findings remain in both local trees. Documentation records them; it does
not repair them. No hostile file, large allocation, or mapped-model CUDA
inference was executed to demonstrate them.

| Finding | Current source evidence | Narrow production follow-up |
|---|---|---|
| Casper mapped-file byte-size overflow | `src/niyah_casper.c:154` computes `64 + total_floats * 4` without checking byte arithmetic. Valid element-count geometry can overflow the byte conversion and pass an undersized-file check. | Check payload length using safe arithmetic before publishing a mapped model; test a tiny rejection fixture and mapping cleanup in `tests/test_casper_mmap.c`. |
| CUDA mapped-storage layout mismatch | `src/niyah_cuda_matvec.cu:166`, `:217` and `:286` do not reject mapped storage; create/sync upload raw weights while kernels use canonical offsets. Casper's CPU adapter uses a different stored order. | Fail closed for non-owned storage in create and sync matching, unless an explicit conversion is implemented. Add CUDA rejection and unchanged-state tests; CPU success alone is insufficient. |
| Tokenizer expansion before integrity/resource validation | `src/niyah_tokenizer.c:721` allocates from header counts; `:756` expands merges, using allocation/copy at `:156`, before footer/CRC validation at `:769`. Doubling merges amplify memory requirements even without integer overflow. | Preflight structure, integrity and checked expanded-memory accounting before materialization. A valid CRC alone does not bound expansion; any resource policy needs explicit compatibility review and safe allocation-budget regressions. |

The CUDA finding concerns callable internal model-state APIs; a vulnerable CLI
route was not established. The tokenizer ordering is also present in published
main and feature; the Casper findings concern the feature/local adapter.
External Casper test skips and successful ordinary-model CUDA tests do not
settle these cases.

The published training header's equal-sample averaging wording is a separate
resolved local documentation contract: both local headers specify weighting by
supervised-target count for unequal lengths, retaining the equal-length path.
The implementation and CPU analytical regressions agree with that local wording.

## Corpus provenance and adaptation

W's `prepared/manifest.json` records 17,448 training documents (99,665,711 bytes)
and 350 validation documents (2,095,011 bytes). The split is cleaned-document
SHA-256 modulo 50; exact duplicates share a split. Near-duplicate checking was
not performed. The preflight receipt under
`O\build\wiki-preflight-receipts\wiki-preflight-20261013T091514Z-8cabb0a5b55d.json`
names Kaggle `z3rocool/arabic-wikipedia-dump-2021`, but explicitly records
`UNVERIFIED_KAGGLE_LICENSE_UNKNOWN` and an expected source from a pasted terminal
log that was not currently re-attested. Local hashes do not resolve those gaps.
The inspected `O\corpus\kaggle_ar.txt` is zero bytes; an all-zero duplicate
report on that file cannot characterize the actual training corpus.

Five 100-update chunks end at optimizer step 500. Fine-tuning creates fresh
optimizer/cursor state, so `wiki-update-0500` is not evidence of global step
2600 or a continuation of step2100 optimizer state.

| Evaluation distribution | Before loss / perplexity | Wiki update-0500 loss / perplexity | Scored targets |
|---|---|---|---:|
| Wikipedia validation | 6.14290959 / 465.40574124 | 5.55822240 / 259.36138652 | 457,403 |
| Original mixed validation | 5.49607625 / 243.73370458 (secondary receipt) | 5.87614984 / 356.43426610 | 426,943 |

Raw logs are W's `before-validation-0001.log`, `after-0500-validation-0001.log`
and `original-heldout-after.log`. All use `objective=all_tokens`, sequence length
256, per-sample context reset, and a bigram baseline fitted/scored on the same
included transitions. Wikipedia has 1,965 samples; the original set has 1,668.
Do not compare their absolute perplexities as though they were one distribution.

The original-distribution before value was found locally in the secondary
`O\build\readonly-diagnostics\inspect-20261013T090546Z-65e4f6732b\report.json`,
which identifies supplied completed logs rather than a new execution. That
baseline's primary log was not independently recovered in this local audit.
Its recorded comparison indicates worsening on the original distribution
while Wikipedia improves, not generic quality improvement. No untouched final
test or controlled broad generation-quality improvement is established.

## Artifact identities

These are raw SHA-256 values unless explicitly marked structured:

| Artifact | SHA-256 |
|---|---|
| B `Release/niyah.exe`, matching G metadata | `3bfa56e8d56f4f03f86499192822065746a919036810954a6657f08fc9281850` |
| W `wiki-update-0500.ckpt` | `2b26aff95eaa41643737e00549a32b6f34ddb8c9c5902fc22ca71a1eb699715b` |
| W `wiki-update-0500.cur` | `4f9a30d08aaea4ffa0f4bc3396bafd31f36c0b39f23de7c66d5a77f77e673510` |
| Tokenizer file used by G | `b21c9264d80ae2b2f941a28a3f9f3d039631f5fadc29803e44d20a58c31e530e` |
| Structured tokenizer identity in evaluation | `ee02feffa15cbb6c277c937fc9b3a2cc636a6c35d83c59f58ff58bca4356416a` |
| B `tests.log` | `d220a4e80ba6d8c8f6c5b0f6972a2987092f14ed328b0558caea89fbe065bc17` |
| B `tests.xml` | `fd942fea89629517f7d24e8f72fb0e64f1b7ade12a44bc8f3d939a57c28751bd` |
| G `results.json` | `554ca691e2aba2a6f7d8b28479eac1822890683736d32b4cbbfb4629c5af3636` |

Structured tokenizer/shard identities and raw file hashes are different
contracts. Their differing values do not by themselves indicate corruption.

## Coverage and remaining boundaries

| Area inspected | Evidence and boundary |
|---|---|
| Repository/history | Live remote heads, exact main/feature objects, both local branches and dirty inventories; no branch switching or publication |
| Native lifecycle | Headers, implementation and test/CMake wiring for model/math, tokenizer/dataset, forward/backward, optimizer/training, eval/baseline, decode/sampler/generation |
| Persistence/CLI | Publication/pair/prompt contracts, cleanup tests, UTF-8 paths, command dispatch, shard/eval/finetune and repetition options |
| Casper/ORACLE/receipts | Mapped read-only model views and deterministic coordination/receipt contracts; absent external fixtures remain unverified |
| CUDA | Actual saved architecture-86 test outcomes and assertion scope; no general mixed-precision, bitwise or performance claim |
| Corpus/tools | Preparation/provenance manifests, corpus utility contracts and offline tests; no new download, corpus rewrite or licensing determination |
| Model quality | Existing checkpoint, evaluation logs and 18 generation artifacts; weak output and adaptation tradeoff remain explicit |
| Build/release | Canonical scripts, five workflows, CMake install/package rules; configured CI behavior distinguished from observed execution |
| Documentation | Durable guides, dated reports, model/evaluation cards and human-evaluation inventory; historical measurements retain their dates |

No old linked task, raw Codex session, or memory summary was used as evidence.
The audit's source inventories, documentation preimages, scoped patch and check
receipt are retained locally under
`D:\NiyahChecks\documentation-audit-20261013`.
The exported patch is relative to the pre-edit dirty review tree, including
its existing untracked documents; it is not a patch against clean HEAD or main.
Full engine gates were not repeated
for documentation-only edits; existing gates retain the source-binding limits
above. Publication, absent external Casper models, a frozen full-source CUDA
receipt, upstream corpus provenance/rights and broad model quality remain
outside the verified result. The source defects above need separate production
patches and focused regression gates; none is marked solved by this audit.

Documentation validation checks relative file links, balanced code fences,
CLI modes/options against source, Python test prerequisites, the pinned remote
guard matrix, saved CUDA counts/skips, 18 output hashes and nine output pairs,
and binary/tokenizer/checkpoint hashes. The task-only patch passes whitespace
and reverse-application checks without applying or reverting it. Pre-edit hash
comparison preserves all 199 inventoried original-checkout files and all review
files outside the 13 intentionally edited documents; one new audit document is
added. These checks are recorded by `verify_audit.py` in the local audit folder.

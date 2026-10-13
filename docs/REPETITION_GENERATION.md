# Generation repetition penalty

## Local status: 2026-10-13

The patch is present in the review working tree at base
`6ec58869fca15cb6034bd5a689e9a04483651d4c` with uncommitted changes. Saved local
receipts establish a 54-pass/2-skip CPU suite and a later architecture-86 CUDA
suite with 69 passed, 2 skipped, 0 failed, including real generation integration
tests. Both skips require external Casper fixtures. Eighteen saved runs provide
nine matching CPU/CUDA output pairs on three prompts, with reduced repetition
but weak language output. See the [dated audit](LOCAL_AUDIT_2026-10-13.md) for
artifact identities and the missing full-source binding limitation.

A shared-guard proposal remains unapplied to this dirty review tree. A corrected
version subsequently passed four selected CPU tests and two selected CUDA tests
in an isolated source copy. These are selected tests, not full verification
gates. Its added stop report and token-cycle behavior have not replaced the
production paths described below. The
[shared-guard follow-up](SHARED_GUARD_EVIDENCE_2026-10-13.md) records exact hashes,
the initial C header failure and correction, and remaining test limitations.

## Scope and contracts

This additive patch targets the 12 files in the manifest of
`Niyah_Generation_Input_5bc26f7d64d54ee3ba01bbdfc9ed396f.zip`.
Input archive SHA-256:
`dd8d9730cca77f607d798e660cb41db4081cd7df601abc473f2ba73cd829dc9b`.
It is not a complete snapshot of the Windows repository. The original archive
is unchanged. No checkpoint, tokenizer, corpus, training or eval implementation
is edited. The existing sampler already implements the sign-aware penalty;
this change wires that API into generation and CLI.

The additive repetition patch retains `NiyahGenerationConfig` and the legacy
CPU/CUDA function signatures and implementations from its input snapshot.
That is not a claim that the local legacy paths match published GitHub source.
New `NiyahRepetitionConfig` has
`penalty` and `window`. Penalty must be finite and at least 1; window must be
nonzero. `penalty == 1.0f` delegates directly to the corresponding legacy API.
The generated-token buffer supplies history. Only previously emitted tokens
are included, never prompt tokens. Each distinct token is penalized once per
step even if it occurs several times in the active trailing window.

Positive logits are divided by the penalty, negative logits are multiplied,
and zero remains zero. Sampling operates on scratch; raw decoder logits are
not changed by the penalty. This is an inference-time decoding control, not a
repair of model weights and not evidence of improved language quality.

## Memory and stopping

CPU enabled workspace: `[decode workspace][raw logits V][scratch V]`.
CUDA enabled host workspace: `[raw logits V][scratch V]`; the already-created
`NiyahCudaDecodeState` owns persistent device decode workspace and KV storage.
The new query functions check element-count addition and byte-size overflow
before pointer arithmetic. At penalty 1 the legacy count is retained.
There are no new allocation/free calls inside either generation loop.
The caller must supply valid, nonoverlapping arrays and model/cache storage.

The enabled paths use the current CPU stop rules: EOS when configured, maximum
new tokens, raw entropy below 0.05, or eight identical consecutive emitted
tokens. Disabled CUDA delegates to legacy CUDA. The currently inspected local
legacy CUDA implementation also has both stop guards; the GitHub feature
revision `efa0a5cb09d998aa2263ff6239cb2774f89d42c4` does not. A separate
shared-guard component test is not production integration. Failed enabled
generation clears result and resets cache. If CUDA
reset itself fails, NIYAH_ERR_IO is returned and the device state must be
destroyed rather than reused. Output tokens after a failure are unspecified.

## CLI

`niyah run` accepts `--repetition-penalty 1.1 --repetition-window 64`.
Defaults are penalty 1.0 (disabled) and window 64. Window accepts only a nonzero
unsigned decimal integer with checked range; signs, spaces, fractions and
trailing junk are rejected. Missing values and nonfinite penalties are
rejected. Both backend branches explicitly pass the config and queried
workspace. `niyah eval` rejects these generation-only options.

## Verification levels

- `niyah_repetition_control_test`: deterministic decoder and CUDA TEST DOUBLES.
  Exercises real generation/sampler/control C code, not transformer math/GPU.
- `niyah_generation_repetition_test`: actual tiny CPU model and manual decode
  comparison, legacy equivalence, workspace rejection, and failure cleanup.
- `niyah_cuda_generation_repetition_test`: actual CUDA state/decode on a tiny
  model, legacy equivalence and greedy CPU/CUDA comparison. Needs a working GPU.
- Existing per-backend CLI fixture adds parser negatives and disabled equivalence.
  That fixture trains only its own temporary tiny test checkpoint.

Historical patch-authoring evidence (before Windows application): control 19/19 passed under GCC and Clang
ASan/UBSan; 73/73 exact-extracted parser cases passed; original sampler contract
passed. Real CPU integration and full CLI tests were added but only syntax
checked, not executed against the full engine. CUDA compiler/hardware and the
Windows device were unavailable. A successful control test must never be
reported as real CPU transformer verification or GPU verification.

At that authoring stage the patch had not been applied on Windows. The local
receipts above supersede that application/test limitation, without converting
the original control-test results into GPU evidence. No publication is implied.

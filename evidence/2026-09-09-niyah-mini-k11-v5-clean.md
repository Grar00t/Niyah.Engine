# Evidence — niyah-mini-k11-v5-clean held-out BPB

```
STATUS             = FRESH_REPRODUCED
RECORDED_UTC       = 2026-09-10T02:25:00Z
MEASURED_UTC       = 2026-09-09T12:38:03Z (original evaluation)
REPRODUCED_UTC     = 2026-09-10T02:40:00Z (rebuilt from source at cb5691d0)
REPRODUCED_ON_MAIN = 2026-09-10T02:50:00Z (rebuilt from origin/main, EVAL_RC=0)
ARTIFACT_ACCESS    = LOCAL_ARTIFACT_REQUIRED — weights and data streams are not committed here
```

Every value below was printed by a command. Nothing was re-derived from prose.
What was not measured is marked `UNMEASURED`, never `ABSENT`.

---

## 1. Model under test

| field | value |
|---|---|
| `MODEL` | `niyah-mini-k11-v5-clean` |
| `SOURCE_RUN` | `k11-lr3e-4-20260909-121720` |
| `WEIGHTS_BYTES` | `4074496` |
| `WEIGHTS_SHA256` | `fbf91f0e5c187a1310e9c476d980fe4ddd5ee29d7ef38418c84993c68e9f1ebf` |
| `CONFIG_SHA256` | `d8992a9b9ef78ea77c9b0bdbcfb46397afecd7ea780e4de0a0febb35a8cd0ce3` |

The released weight file and the training-run weight file are byte-identical:
both hash to `fbf91f0e…`. The release is the trained artifact, not a copy of
something else.

`config.json`:

```json
{
  "n_layers": 4,
  "n_dim": 128,
  "n_heads": 4,
  "n_kv_heads": 2,
  "n_ff": 512,
  "n_vocab": 269,
  "n_ctx": 64,
  "rope_theta": 10000,
  "norm_eps": 9.99999975e-06,
  "tie_word_embeddings": true
}
```

### 1.1 Structural check — the config accounts for the file exactly

| tensor group | shape | float32 params |
|---|---|---|
| token embedding (tied with output) | 269 × 128 | 34,432 |
| attention `q` × 4 layers | 128 × 128 | 65,536 |
| attention `k` × 4 layers (GQA, 2 kv heads, head_dim 32) | 128 × 64 | 32,768 |
| attention `v` × 4 layers | 128 × 64 | 32,768 |
| attention `o` × 4 layers | 128 × 128 | 65,536 |
| feed-forward `gate` + `up` + `down` × 4 layers | 128 ↔ 512 | 786,432 |
| RMSNorm × 8 + final norm | 128 | 1,152 |
| **total** | | **1,018,624** |

`1,018,624 × 4 bytes = 4,074,496 bytes` — exactly the size of `weights.f32.bin`.
No header, no padding, no undeclared tensor. `PARAM_ACCOUNTING=PASS`.

---

## 2. Held-out evaluation

```
EVAL_STREAM        = v5-clean-c64-20260909-042702/val-windows-clean.bin
EVAL_STREAM_SHA256 = 7aed1728413b3ceb4536a87550a26f005f55a7c48ea20330288b5977a331d23d
EVAL_STREAM_BYTES  = 468520
WINDOW_TOKENS      = 65   (64 shifted next-token predictions per window)
VAL_WINDOWS        = 1802
VAL_PREDICTIONS    = 115328
OBJECTIVE          = shifted_next_token
VOCAB              = 269

MEAN_NLL_NATS      = 2.165499
VAL_BPB            = 3.124155
```

`BPB = mean_nll_nats / ln(2)`. Check: `2.165499 / 0.693147 = 3.124155`.
Check: `1802 × 64 = 115,328`.

### 2.1 Fresh reproduction — 2026-09-10

The evaluator was rebuilt from committed source and re-executed, twice:
once at `cb5691d0`, once at `origin/main`. Both agree with the 2026-09-09
record to all six printed decimals.

```
PRODUCER = tools/niyah_eval.c  (blob e91b1bb4fc8395d33e24a4b884a7ba256b5e1502)
COMMIT   = dab2cf9  "tools: add K8-compatible niyah mini evaluator"

make -C native/niyah_mini lib          -> LIB_RC=0    libniyah_mini.a  84546 bytes
cc -std=c11 -O2 -Wall -Wextra \
   -Inative/niyah_mini -o /tmp/niyah_eval \
   tools/niyah_eval.c native/niyah_mini/libniyah_mini.a -lm
                                       -> BUILD_RC=0  /tmp/niyah_eval  65272 bytes

/tmp/niyah_eval CONFIG WEIGHTS EVAL_BIN
EVAL_COMPLETE
windows=1802
avg_loss=2.165499
perplexity=8.718951
min_loss=0.613356
max_loss=4.041076
EVAL_RC=0

FRESH_MEAN_NLL_NATS  = 2.165499
FRESH_BITS_PER_BYTE  = 3.124155
GATE_BIGRAM_ADD1_BPB = 3.768400
VERDICT              = PASS
```

Same command against a worktree of `origin/main` (`be73ca26`):

```
MAIN_LIB_RC=0
MAIN_BUILD_RC=0
windows=1802  avg_loss=2.165499  perplexity=8.718951
min_loss=0.613356  max_loss=4.041076
MAIN_EVAL_RC=0
```

`HISTORICAL -> FRESH`, on both refs.

Two facts recorded here for the first time:

- Per-window dispersion. `min_loss=0.613356` nats, `max_loss=4.041076` nats —
  a 6.6× spread across windows. Derived: about `0.885` and `5.830` bits/byte.
  The reported figure is a mean over a wide distribution, not a uniform result.
- The evaluator's own vocabulary. `tools/niyah_eval.c` prints only
  `avg_loss`, `perplexity`, `min_loss`, `max_loss`. The labels
  `MEAN_NLL_NATS`, `BITS_PER_BYTE` and `SEMANTIC_EQUIVALENCE` in the
  2026-09-09 log were produced by a wrapper above this binary, not by the
  binary. The values agree; the label provenance is now stated correctly.

### 2.2 Byte identity — measured, and it does not hold exactly

`BPB = nats / ln(2)` is only exact if every predicted token is one byte.
The 269-symbol vocabulary is 13 special ids plus 256 byte ids. Counted:

```
TOTAL_TOKENS   = 117130          (= 1802 × 65)
SPECIAL_TOKENS = 4
BY_ID          = {1: 2, 2: 2}    (BOS × 2, EOS × 2)
BYTE_IDENTITY  = BROKEN
```

Four non-byte symbols in 117,130. Worst case, all four fall on predicted
positions, so the true byte count is 115,324 rather than 115,328:

```
bpb_byte_exact <= 3.124155 × 115328 / 115324 = 3.124263
correction      = +0.000108 bits/byte
```

The identity is broken and the correction is `1.7e-4` of the margin against
the bigram gate. Stated because it is true, not because it changes anything.

Side observation, `UNMEASURED`: only two document boundaries fall inside a
1802-window validation stream. Whether the two leaked windows of §4 sit at
those boundaries has not been checked.

---

## 3. Baselines on the same split

| baseline | BPB |
|---|---|
| uniform over 269 symbols | `8.071462` |
| unigram + add-1 | `4.796900` |
| **bigram + add-1 (the gate)** | **`3.768400`** |
| model | **`3.124155`** |

```
OFFICIAL_BIGRAM_GATE_BPB = 3.768400
FRESH_BIGRAM_ADD1_BPB    = 3.768422516
DELTA_VS_OFFICIAL_GATE   = -0.644245
```

The bigram baseline was recomputed at release time and landed `2.2516e-05`
from the frozen constant. The gate value was re-derived, not copied. The
frozen constant is marginally the stricter of the two.

The model beats the bigram baseline by `0.644245` bits/byte, a 17.1% reduction.
A model that does not beat the bigram has learned nothing; this one does.

---

## 4. The defect this measurement exposed

Re-run 2026-09-10, same gate, same inputs:

```
check_window_disjoint.py train-windows-clean.bin val-windows-clean.bin

WINDOW_TOKENS=65
RECORD_BYTES=260
TRAIN_BYTES=6132100 TRAIN_WINDOWS=23585 TRAIN_UNIQUE=23585
EVAL_BYTES=468520   EVAL_WINDOWS=1802   EVAL_UNIQUE=1802
OVERLAP_WINDOWS=2
LEAKAGE_PCT=0.110988
HELDOUT_DISJOINT=FAIL
GATE_RC=1
```

Two of the 1802 evaluation windows are byte-identical to windows in the
training stream. The split is therefore not strictly held out. The gate
fails loudly, as designed.

**Bounded impact.** Those 2 windows carry 128 of the 115,328 predictions.
Assigning them the most favourable possible value (zero loss, i.e. assuming
they were perfectly memorised) and removing them:

```
clean_mean_nll <= 2.165499 * 115328 / 115200 = 2.167905
clean_bpb      <= 3.127626
margin vs 3.768400 = 0.640774
```

Both corrections applied together (leak + byte identity) stay below `3.1278`.
Neither can flip the baseline comparison. The leak is still a real pipeline
defect and the split must be rebuilt with the gate enforced.
Status: `TICKET3 = PASS_WITH_DISCLOSED_LEAK (0.111%)`.

Tracked as issue #34.

---

## 5. Training run

```
steps           = 23585      (= TRAIN_WINDOWS, exactly one pass)
micro_batch     = 8
optimizer_steps = 2949
first_loss      = 3.281901
last_loss       = 2.437673
avg_loss        = 2.446508
banner          = ONLINE_TRAINING_COMPLETE
```

Open question, `UNMEASURED`: validation NLL (`2.165499`) is **lower** than the
final training loss (`2.437673`) by `0.272` nats. A single-epoch schedule
explains part of this, but the validation split may also be easier than the
training split. Not resolved here.

---

## 6. Provenance of the dual-evaluation run (k8, earlier)

```
created_utc        = 2026-09-08T21:51:37Z
repo_head          = 5edd898584a6d39909cf0b2930a8921e98b42598
repo_branch        = test/niyah-current-main   (LOCAL_ONLY — not present on the remote)
metrics_sha256     = ed124833f79f7e4d7552c980efc492af20387f089d0756e8545bdc7bac6dde72  (verified)
v4_heldout_sha256  = a355bf4e22e253171a1d5e6e04422da2c00bafd00046cae3e9fd37d56a7b934f
ind_heldout_sha256 = 22e931252b41acfc114197ada7226451e2dd0216cd19ba753a57ca934706bda8
```

The file named `gptoss-heldout-v1.bin` (150,020 bytes) is an **independent
evaluation stream label**, not a weight lineage. `150,020 / 260 = 577` exactly:
same 65-token record format, 577 windows. Every snapshot carrying the
`gptoss` prefix is `4,074,496` bytes — the niyah-mini architecture above —
and hashes to values distinct from each other and from k11:

```
1775c920…  snapshots/gptoss-1387.weights.f32.bin
303177e2…  snapshots/gptoss-rescue-2349.weights.f32.bin
3a4ca32d…  models/niyah-mini-gptoss-online/weights.f32.bin
fbf91f0e…  k11-lr3e-4-20260909-121720/weights.f32.bin
```

The contents of `gptoss-heldout-v1.bin` are `UNMEASURED`. The `3.124155`
result does not use that stream.

---

## 7. What this evidence does NOT prove

- Sanitizer cleanliness. `NIYAH_SANITIZE` defaults to `OFF` and no run in this
  record enabled it. `ASAN/UBSAN = UNMEASURED`.
- CI. No workflow run is tied to these numbers. `CI_FOR_THESE_NUMBERS = NOT_RUN`.
  Both reproductions were run by hand on one host.
- Disjointness. The split leaks; §4.
- Exact byte identity. It is broken by four symbols; §2.2.
- Generalisation. 1802 windows over a 269-symbol byte vocabulary at `n_ctx=64`
  measures this split and nothing else.
- Any claim about the training corpus licence.

---

## 8. Reproduction

Works at `origin/main` and at `cb5691d0`, with the local artifacts present:

```sh
make -C native/niyah_mini lib

cc -std=c11 -O2 -Wall -Wextra -Inative/niyah_mini \
   -o /tmp/niyah_eval tools/niyah_eval.c \
   native/niyah_mini/libniyah_mini.a -lm

# argv contract is CONFIG WEIGHTS EVAL_BIN — three arguments, not two
/tmp/niyah_eval <model>/config.json <model>/weights.f32.bin <val-windows-clean.bin>
# expect: avg_loss=2.165499 ; BPB = avg_loss / ln(2) = 3.124155

python3 tools/check_window_disjoint.py \
  <train-windows-clean.bin> <val-windows-clean.bin>
# exit 0 = disjoint, 1 = leak measured, 2 = malformed input

sha256sum -c SHA256SUMS.txt   # inside the release directory; RC=0 observed
```

`tools/niyah_eval.c` is not committed on `main`; it lives on
`salvage/local-evolution-20260909` at `cb5691d0`. Copy it in, or build from
that ref.

---

## 9. How `main` and the salvage branch differ

An earlier revision of this file inferred from file sizes that `main` might
not reproduce the number, and recorded `REPRODUCIBLE_ON_MAIN = UNVERIFIED`.
That was replaced by measurement: `main` reproduces it exactly, including
`min_loss` and `max_loss`. Size is not semantics.

```
git diff --stat origin/main cb5691d0 -- native/niyah_mini/niyah_mini_model.{c,h}
  niyah_mini_model.c | 294 +++++----
  niyah_mini_model.h |  30 ++++
  2 files changed, 279 insertions(+), 45 deletions(-)
```

The difference is additive and does not touch the forward result. Five
functions exist at `cb5691d0` and are absent from `main`:

```
niyah_mini_arena_init
niyah_mini_arena_alloc
niyah_mini_forward_state_bind
niyah_mini_runtime_memory_size
niyah_mini_model_bind_runtime
```

`native/niyah_mini/test_niyah_mini_heapfree.c` is likewise present at
`cb5691d0` and absent from `main`.

So the arena / caller-owned-memory work for `niyah_mini`, and the test that
guards it, exist only on the salvage branch. `main` builds and evaluates
correctly without them, but has no heap-discipline guarantee for this module.

```
REPRODUCIBLE_AT      = cb5691d0
REPRODUCIBLE_ON_MAIN = YES (be73ca26, MAIN_EVAL_RC=0)
ARENA_API_ON_MAIN    = ABSENT
```

Author: Suliman Nazal Alshammari · سليمان نزال الشمري

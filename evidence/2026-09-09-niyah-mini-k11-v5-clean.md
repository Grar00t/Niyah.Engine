# Evidence — niyah-mini-k11-v5-clean held-out BPB

```
STATUS             = FRESH_REPRODUCED_AND_DISJOINT_AND_STRATIFIED
HEADLINE           = VAL_BPB = 3.123631 over 1800 disjoint windows
PROSE_ONLY         = VAL_BPB = 3.180956 over 1722 windows, 15.91% under the bigram
GATE               = BIGRAM_ADD1_BPB 3.768400   VERDICT = PASS
RECORDED_UTC       = 2026-09-10T03:20:00Z
MEASURED_UTC       = 2026-09-09T12:38:03Z (original evaluation, leaked split)
REPRODUCED_UTC     = 2026-09-10T02:40:00Z (rebuilt from source at cb5691d0)
REPRODUCED_ON_MAIN = 2026-09-10T02:50:00Z (rebuilt from origin/main, EVAL_RC=0)
DISJOINT_UTC       = 2026-09-10T03:00:00Z (leak removed, re-evaluated)
STRATIFIED_UTC     = 2026-09-10T03:15:00Z (filler/prose split, advantage survives)
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

The released weight file and the training-run weight file are byte-identical.
The release is the trained artifact, not a copy of something else.

```json
{
  "n_layers": 4, "n_dim": 128, "n_heads": 4, "n_kv_heads": 2,
  "n_ff": 512, "n_vocab": 269, "n_ctx": 64,
  "rope_theta": 10000, "norm_eps": 9.99999975e-06,
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

`1,018,624 × 4 bytes = 4,074,496` — exactly the file size. No header, no
padding, no undeclared tensor. `PARAM_ACCOUNTING=PASS`.

---

## 2. Held-out evaluation

Two streams were evaluated. The second is the one that counts: it is the
first with the disjointness gate passing.

| | leaked stream | **disjoint stream** |
|---|---|---|
| file | `val-windows-clean.bin` | **`val-windows-disjoint.bin`** |
| sha256 | `7aed1728…a331d23d` | **`9e6989b3335ab7e2c80fc879f9a77ff2a391e77e950f9e0d1750a436b49b6cfd`** |
| bytes | `468520` | `468000` |
| windows | `1802` | **`1800`** |
| predictions | `115328` | **`115200`** |
| `HELDOUT_DISJOINT` | `FAIL` (`OVERLAP_WINDOWS=2`) | **`PASS` (`OVERLAP_WINDOWS=0`)** |
| `mean_nll_nats` | `2.165499` | **`2.165136`** |
| `perplexity` | `8.718951` | **`8.715783`** |
| `min_loss` | `0.613356` | `0.613356` |
| `max_loss` | `4.041076` | `4.041076` |
| **`BPB`** | `3.124155` | **`3.123631`** |

```
WINDOW_TOKENS = 65   (64 shifted next-token predictions per window)
OBJECTIVE     = shifted_next_token
VOCAB         = 269
BPB           = mean_nll_nats / ln(2)
```

Check: `2.165136 / 0.693147 = 3.123631`. Check: `1800 × 64 = 115,200`.

### 2.1 Fresh reproduction — 2026-09-10

The evaluator was rebuilt from committed source and re-executed, at
`cb5691d0` and at `origin/main`. Both agree with the 2026-09-09 record to
all six printed decimals.

```
PRODUCER = tools/niyah_eval.c  (blob e91b1bb4fc8395d33e24a4b884a7ba256b5e1502)
COMMIT   = dab2cf9  "tools: add K8-compatible niyah mini evaluator"

make -C native/niyah_mini lib   -> LIB_RC=0        libniyah_mini.a  84546 bytes
cc ... -o /tmp/niyah_eval       -> BUILD_RC=0      /tmp/niyah_eval  65272 bytes
/tmp/niyah_eval CONFIG WEIGHTS EVAL_BIN -> EVAL_RC=0

MAIN_LIB_RC=0  MAIN_BUILD_RC=0  MAIN_EVAL_RC=0   (identical output)
```

`HISTORICAL -> FRESH`, on both refs.

Two facts recorded here for the first time:

- Per-window dispersion. `min_loss=0.613356`, `max_loss=4.041076` nats — a
  6.6× spread, about `0.885` to `5.830` bits/byte. The headline is a mean
  over a wide distribution.
- The evaluator's own vocabulary. `tools/niyah_eval.c` prints only
  `avg_loss`, `perplexity`, `min_loss`, `max_loss`. The labels
  `MEAN_NLL_NATS`, `BITS_PER_BYTE`, `SEMANTIC_EQUIVALENCE` in the 2026-09-09
  log came from a wrapper above the binary, not the binary. Values agree;
  label provenance is now stated correctly.

### 2.2 Byte identity — measured, and it does not hold exactly

`BPB = nats / ln(2)` is exact only if every predicted symbol is one byte.
The 269-symbol vocabulary is 13 control ids plus 256 byte ids. Counted on
the leaked stream:

```
TOTAL_TOKENS   = 117130          (= 1802 × 65)
SPECIAL_TOKENS = 4
BY_ID          = {1: 2, 2: 2}    (BOS × 2, EOS × 2)
BYTE_IDENTITY  = BROKEN
```

One of those `BOS` symbols sat in a window later dropped as a leak, so the
disjoint stream carries three. Worst case, all three fall on predicted
positions:

```
bpb_byte_exact <= 3.123631 × 115200 / 115197 = 3.123712
correction      = +0.000081 bits/byte
```

The identity is broken, and the correction is `1.3e-4` of the margin against
the bigram gate. Stated because it is true, not because it changes anything.

---

## 3. Baselines on the same split

### 3.0 The gate now has a committed producer

Until 2026-09-10 none of the three baseline constants had code in this
repository that produced them. Every `PASS` was measured against a number
that could be quoted but not re-derived. `tools/bigram_baseline.py` closes
that gap: it fits unigram and bigram add-1 models on the training window
stream and scores any evaluation stream with the same prediction convention
as `tools/niyah_eval.c` — 65-token windows, 64 shifted predictions, nats
divided by `ln 2`.

| baseline | frozen constant | producer, **disjoint** stream | delta |
|---|---|---|---|
| uniform over 269 symbols | `8.071462` | `8.071462` | `0` |
| unigram + add-1 | `4.796900` | `4.796600` | `-0.000300` |
| bigram + add-1 (the gate) | `3.768400` | `3.768088895` | `-0.000311` |
| **model** | | **`3.123631`** | |

`PRODUCER_VALIDATION = PENDING`. The columns are not comparable: the frozen
constants were computed on the 1802-window stream, the producer ran on the
1800-window stream. The validating run is the producer against
`val-windows-clean.bin`.

Falsifiable prediction. If the producer is faithful it must print
`BIGRAM_ADD1_BPB = 3.7684…` on 1802 windows, and the 128 extra predictions
must carry exactly this much loss:

```
producer, 1800 windows : 2.611840 nats × 115200 = 300,883.97 nats
frozen,   1802 windows : 2.612071 nats × 115328 = 301,244.97 nats
difference             :    361.00 nats over 128 predictions
                       =      2.820 nats each   =   4.069 bits/byte
```

If the producer disagrees on the 1802-window stream, either the producer or
the frozen constant is wrong, and no `PASS` in this file stands until that
is resolved.

Fit statistics, `TRAIN_PREDICTIONS = 1,509,440`:

```
TRAIN_CONTEXTS_SEEN   = 168 of 269      (101 symbols never occur as a context)
EVAL_UNSEEN_CONTEXTS  = 0
EVAL_UNSEEN_PAIRS     = 192 of 115200 = 0.167%
```

The effective alphabet is about 155 distinct byte values, not 256. The
uniform-over-269 baseline is therefore a strawman, reported only for
completeness; `log2(155) ~= 7.28`.

The model beats the bigram by `0.644458` bits/byte on the same stream — a
17.10% reduction. §3.1 shows where that reduction comes from.

### 3.1 Stratified — the advantage survives on prose

RFC text is heavily templated: tables of contents built from dot leaders,
horizontal rules, page footers. A byte model beats a bigram trivially on
`. . . . . .`, so an unstratified headline can be carried by filler.

Each held-out window was classified `FILLER` if at least 20% of its 65 bytes
fall in `.-_=*`, or if it contains a run of 8 or more identical bytes.
Otherwise `PROSE`. The threshold is a judgement call, not a measurement.

| stratum | windows | share | model BPB | bigram BPB | advantage | advantage % |
|---|---|---|---|---|---|---|
| filler | `78` | `4.33%` | `1.858041` | `3.444950` | `1.586909` | **`46.06%`** |
| prose | `1722` | `95.67%` | `3.180956` | `3.782726` | `0.601769` | **`15.91%`** |
| all | `1800` | `100%` | `3.123631` | `3.768089` | `0.644458` | `17.10%` |

Check: `(4992 × 1.858041 + 110208 × 3.180956) / 115200 = 3.123631`. The
strata reconstruct the headline exactly.

**The claim survives.** Filler inflates the headline by `1.19` percentage
points — 17.10% overall against 15.91% on the 95.67% of the stream that is
ordinary prose. The advantage over the bigram is not an artifact of dot
leaders.

Two secondary facts. The global `min_loss = 0.613356` belongs to the filler
stratum; prose bottoms out at `1.500355`. And the 46% advantage on filler is
real learning of document structure — merely cheap learning.

---

## 4. The defect this measurement exposed, and its repair

### 4.1 Measured

```
check_window_disjoint.py train-windows-clean.bin val-windows-clean.bin
OVERLAP_WINDOWS=2  LEAKAGE_PCT=0.110988  HELDOUT_DISJOINT=FAIL  GATE_RC=1
```

### 4.2 Identified

```
VAL_IDX=1585  TRAIN_IDX=11833  SPECIALS=[1]
  'Internet Engineering Task Force (IETF)  M. Lepinski  Request for C'
VAL_IDX=1770  TRAIN_IDX=4900   SPECIALS=[]
  'Infrastructure Certificate and Certificate Revocation List (CRL) '
```

Both are IETF RFC boilerplate: a masthead, and a fragment of a cited document
title. Neither is unique prose.

### 4.3 Diagnosed — not a splitting bug

`tools/build_heldout_v3.py` was suspected of mis-partitioning documents.
The content refutes that. These spans occur in *different* source documents
that legitimately share templated text. A correct document-level split still
produces them.

The defect is one level up: **the pipeline had no window-level dedup across
the split.** For a byte-window corpus, document-level separation is necessary
but not sufficient. Any span of `WINDOW_TOKENS` symbols that is templated
across documents will collide however the documents are partitioned.

### 4.4 Repaired

`tools/dedup_heldout_windows.py`, committed on `main`:

```
DROPPED_WINDOWS=2  KEPT_WINDOWS=1800  DROPPED_PCT=0.110988
PREDICTIONS_BEFORE=115328  PREDICTIONS_AFTER=115200
OUT_BYTES=468000
OUT_SHA256=9e6989b3335ab7e2c80fc879f9a77ff2a391e77e950f9e0d1750a436b49b6cfd

check_window_disjoint.py train-windows-clean.bin val-windows-disjoint.bin
OVERLAP_WINDOWS=0  LEAKAGE_PCT=0.000000  HELDOUT_DISJOINT=PASS  GATE_RC=0
```

`TICKET3 = PASS` on a stream that passes the disjointness gate. The earlier
status `PASS_WITH_DISCLOSED_LEAK (0.111%)` is superseded.

### 4.5 The leak was not inflating the score — it was depressing it

Removing the two leaked windows made the result **better**, not worse:
`3.124155 -> 3.123631`, a change of `-0.000524` bits/byte.

The loss carried by the two dropped windows, derived from the printed means
and therefore accurate to about `±0.001`:

```
dropped_mean_nll   ~= 2.4922 nats   (= ~3.595 bits/byte)
corpus_mean_nll     = 2.1651 nats   (=  3.1236 bits/byte)
ratio              ~= 1.151
```

The model scored the windows it had already seen **15% worse** than its own
average. They were not memorised.

Three independent scorers agree, each reconstructed the same way from its
own 1800-window and 1802-window totals:

| scorer | corpus mean | the two dropped windows | ratio |
|---|---|---|---|
| niyah-mini | `3.1236` bpb | `~3.595` bpb | `1.15` |
| bigram + add-1 | `3.7681` bpb | `~4.069` bpb | `1.08` |
| unigram + add-1 | `4.7966` bpb | `~5.066` bpb | `1.06` |

A bigram cannot memorise a 65-token window. That all three scorers find
these two windows harder than average is evidence that the **text** is hard —
mixed-case masthead with heavy punctuation — not that the model had seen it.

This contradicts the usual assumption that a train/val overlap flatters the
held-out number. Here it did the opposite. The prior conservative bound
(`clean_bpb <= 3.127626`, computed by assuming the leaked windows were
perfectly memorised) held: measured `3.123631` sits below it. The bound was
right and its reasoning was wrong.

The leak was still a real pipeline defect and had to be fixed. `UNMEASURED`:
whether one epoch over 23,585 windows is enough for a 1.02M-parameter model
to memorise anything at all.

Tracked as issue #34, which remains open until `build_heldout_v3.py` refuses
to emit a split that fails the gate.

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

Open question, `UNMEASURED`: validation NLL (`2.165136`) is **lower** than the
final training loss (`2.437673`) by `0.273` nats. A single-epoch schedule
explains part of it. §4.5 makes the alternative explanation — that the
validation split is simply easier than the training split — more likely, since
even the *seen* windows scored worse than the held-out mean. Not resolved.

---

## 6. Provenance

```
created_utc        = 2026-09-08T21:51:37Z
repo_head          = 5edd898584a6d39909cf0b2930a8921e98b42598
repo_branch        = test/niyah-current-main   (LOCAL_ONLY — not on the remote)
metrics_sha256     = ed124833f79f7e4d7552c980efc492af20387f089d0756e8545bdc7bac6dde72  (verified)
```

`gptoss-heldout-v1.bin` (150,020 bytes) is an **independent evaluation stream
label**, not a weight lineage: `150,020 / 260 = 577` windows in the same
record format. Snapshots carrying the `gptoss` prefix are all `4,074,496`
bytes and hash distinctly:

```
1775c920…  snapshots/gptoss-1387.weights.f32.bin
303177e2…  snapshots/gptoss-rescue-2349.weights.f32.bin
3a4ca32d…  models/niyah-mini-gptoss-online/weights.f32.bin
fbf91f0e…  k11-lr3e-4-20260909-121720/weights.f32.bin   (this release)
```

Contents of `gptoss-heldout-v1.bin` are `UNMEASURED`. This result does not
use that stream.

### 6.1 Corpus composition — measured 2026-09-10

```
train-windows-clean.bin   WINDOWS=23585  BOS=21  EOS=23  DOCS_LOWER_BOUND=23
val-windows-disjoint.bin  WINDOWS=1800   BOS=1   EOS=2   DOCS_LOWER_BOUND=2
```

About 1,025 windows per document, roughly 65 KB each — consistent with RFC
lengths. Two documents are identified from window heads. `INFERRED` from
running headers, not measured against a source manifest:

```
train: "]\n\nRFC 4271 BGP-4 Januar" ×6  ->  RFC 4271 (BGP-4)
val:   "la & Korver Best Current"  ×2  ->  "Rescorla & Korver, Best Current
                                             Practice" = RFC 3552 / BCP 72
```

Most frequent 24-byte window heads, showing how templated the corpus is:

```
train: '........................' ×91   '. . . . . . . . . . . . ' ×43
       ' . . . . . . . . . . . .' ×38   '------------------------' ×11
val:   '. . . . . . . . . . . . ' ×5    '........................' ×4
```

**Licence.** §4.2 and this section establish that the corpus is IETF RFC
text. No manifest in this repository states that. RFC text is governed by
the IETF Trust Legal Provisions, not by "public domain". Whether a derived
byte-window stream may be redistributed is `UNRESOLVED`. This repository has
no `LICENSE` file.

---

## 7. What this evidence does NOT prove

- Corpus coverage. `HELDOUT_DOCUMENT_COUNT` has a measured lower bound of
  **2** (§6.1). The held-out set spans about two documents out of about 23
  in the corpus. `115,200` predictions is therefore not `115,200`
  independent samples; no confidence interval has been computed, and one
  computed under an i.i.d. assumption would be wrong by orders of magnitude.
  **This is a larger threat to the result than the leak was. It is now
  measured, not fixed.**
- Baseline agreement. `PRODUCER_VALIDATION = PENDING` (§3.0). The committed
  baseline producer has not yet been run on the same stream as the frozen
  gate constant.
- Sanitizer cleanliness. `NIYAH_SANITIZE` defaults to `OFF`; no run here
  enabled it. `ASAN/UBSAN = UNMEASURED`.
- CI. No workflow run is tied to these numbers.
  `CI_FOR_THESE_NUMBERS = NOT_RUN`. Every run was by hand on one host.
- Exact byte identity. Broken by three symbols; §2.2.
- Generalisation. 1800 windows over a 269-symbol byte vocabulary at
  `n_ctx=64` measures this split and nothing else.

---

## 8. Reproduction

Works at `origin/main` and at `cb5691d0`, with the local artifacts present:

```sh
make -C native/niyah_mini lib

cc -std=c11 -O2 -Wall -Wextra -Inative/niyah_mini \
   -o /tmp/niyah_eval tools/niyah_eval.c \
   native/niyah_mini/libniyah_mini.a -lm

# 1. make the held-out stream disjoint
python3 tools/dedup_heldout_windows.py \
  <train-windows-clean.bin> <val-windows-clean.bin> <val-windows-disjoint.bin>

# 2. prove it
python3 tools/check_window_disjoint.py \
  <train-windows-clean.bin> <val-windows-disjoint.bin>   # expect exit 0

# 3. evaluate; argv is CONFIG WEIGHTS EVAL_BIN -- three arguments, not two
/tmp/niyah_eval <model>/config.json <model>/weights.f32.bin <val-windows-disjoint.bin>
# expect: windows=1800  avg_loss=2.165136 ; BPB = avg_loss / ln(2) = 3.123631

# 4. re-derive the baselines instead of quoting them
python3 tools/bigram_baseline.py \
  <train-windows-clean.bin> <val-windows-disjoint.bin>
# expect: BIGRAM_ADD1_BPB = 3.768088895
```

`tools/niyah_eval.c` is not committed on `main`; it lives on
`salvage/local-evolution-20260909` at `cb5691d0`. Copy it in, or build from
that ref.

---

## 9. How `main` and the salvage branch differ

An earlier revision of this file inferred from file sizes that `main` might
not reproduce the number, and recorded `REPRODUCIBLE_ON_MAIN = UNVERIFIED`.
Measurement replaced the inference: `main` reproduces it exactly, including
`min_loss` and `max_loss`. Size is not semantics.

```
git diff --stat origin/main cb5691d0 -- native/niyah_mini/niyah_mini_model.{c,h}
  2 files changed, 279 insertions(+), 45 deletions(-)
```

The difference is additive and does not change the forward result. Five
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

The arena / caller-owned-memory work for `niyah_mini`, and the test guarding
it, exist only on the salvage branch. `main` builds and evaluates correctly
without them, but carries no heap-discipline guarantee for this module.

```
REPRODUCIBLE_AT      = cb5691d0
REPRODUCIBLE_ON_MAIN = YES (be73ca26, MAIN_EVAL_RC=0)
ARENA_API_ON_MAIN    = ABSENT
```

---

## 10. Downstream artifacts now stale

The published release manifest still carries the leaked-stream figures and
must be corrected:

```
VAL_WINDOWS=1802      -> 1800
VAL_PREDICTIONS=115328 -> 115200
MEAN_NLL_NATS=2.165499 -> 2.165136
VAL_BPB=3.124155       -> 3.123631
DELTA_VS_OFFICIAL_GATE=-0.644245 -> -0.644769
HELDOUT_DISJOINT       -> PASS (new field)
EVAL_STREAM_SHA256     -> 9e6989b3335ab7e2c80fc879f9a77ff2a391e77e950f9e0d1750a436b49b6cfd
PROSE_ONLY_BPB         -> 3.180956 over 1722 windows (new field)
```

`RELEASE_MANIFEST = STALE` until updated.

Author: Suliman Nazal Alshammari · سليمان نزال الشمري

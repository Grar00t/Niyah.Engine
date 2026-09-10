# Evidence — niyah-mini-k11-v5-clean held-out BPB

```
STATUS          = HISTORICAL_MEASUREMENT
RECORDED_UTC    = 2026-09-10T02:25:00Z
MEASURED_UTC    = 2026-09-09T12:38:03Z (evaluation) / 2026-09-10T02:21:00Z (gates 3-6 below)
ARTIFACT_ACCESS = LOCAL_ARTIFACT_REQUIRED — weights and data streams are not committed here
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

Two independent code paths in the same evaluator agreed:
`LEGACY_AVG_LOSS = CANONICAL_NLL = 2.165499`, `DERIVED_BPB = CANONICAL_BPB = 3.124155`,
`SEMANTIC_EQUIVALENCE=PASS`. Agreement between two paths is not proof of
correctness — see §4.

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

```
check_window_disjoint.py train-windows-clean.bin val-windows-clean.bin

TRAIN_BYTES=6132100 TRAIN_WINDOWS=23585 TRAIN_UNIQUE=23585
EVAL_BYTES=468520   EVAL_WINDOWS=1802   EVAL_UNIQUE=1802
OVERLAP_WINDOWS=2
LEAKAGE_PCT=0.111
HELDOUT_DISJOINT=FAIL
```

Two of the 1802 evaluation windows are byte-identical to windows in the
training stream. The split is therefore not strictly held out.

**Bounded impact.** Those 2 windows carry 128 of the 115,328 predictions.
Assigning them the most favourable possible value (zero loss, i.e. assuming
they were perfectly memorised) and removing them:

```
clean_mean_nll <= 2.165499 * 115328 / 115200 = 2.167905
clean_bpb      <= 3.127626
margin vs 3.768400 = 0.640774
```

The leak cannot flip the baseline comparison. It is still a real pipeline
defect and the split must be rebuilt with the gate enforced.
Status: `TICKET3 = PASS_WITH_DISCLOSED_LEAK (0.111%)`.

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
evaluation stream label**, not a weight lineage. Every snapshot carrying the
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

- `FRESH_RUN` — every number above is `HISTORICAL`. The evaluator has not been
  re-executed since this file was written.
- Sanitizer cleanliness. `NIYAH_SANITIZE` defaults to `OFF` and no run in this
  record enabled it. `ASAN/UBSAN = UNMEASURED`.
- CI. No workflow run is tied to these numbers. `CI_FOR_THESE_NUMBERS = NOT_RUN`.
- Generalisation. 1802 windows over a 269-symbol byte vocabulary at `n_ctx=64`
  measures this split and nothing else.
- Any claim about the training corpus licence.

---

## 8. Reproduction

```sh
python3 tools/check_window_disjoint.py \
  <train-windows-clean.bin> <val-windows-clean.bin>
# exit 0 = disjoint, 1 = leak measured, 2 = malformed input

sha256sum -c SHA256SUMS.txt   # inside the release directory; RC=0 observed
```

Author: Suliman Nazal Alshammari · سليمان نزال الشمري

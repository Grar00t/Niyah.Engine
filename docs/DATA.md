# Data, Provenance, and Quality Boundary

Niyah.Engine treats model code and training data as separate evidence domains. A correct training engine can faithfully learn undesirable patterns, factual errors, boilerplate, duplication, or low-quality dialogue if those patterns are present in the corpus.

## Native data contract

The repository provides a native tokenizer and persisted dataset-shard format.

The preparation pipeline can produce:

```text
raw corpus
  → native byte-level BPE tokenizer
  → tokenizer identity
  → NIYAHSRD shard
  → explicit sample geometry
  → deterministic dataset cursor during training
```

Current preparation modes include:

- continuous stream records;
- blank-line record boundaries;
- supervised prompt/response records using explicit response delimiters;
- response-only objective masking while retaining the prompt as causal context.

## Tokenizer identity

The tokenizer has a stable SHA-256 identity. Dataset/checkpoint compatibility code uses tokenizer identity so that numeric token IDs are not silently reinterpreted under an unrelated vocabulary.

A historical token file with IDs inside the current vocabulary range is **not** automatically compatible. Numeric range is not semantic identity.

## Current diagnostic corpus status

The current v6 pilot training run discussed in project diagnostics was assembled from Hugging Face/web-sourced material. The run record also identifies known quality defects, including factual errors in at least part of the collected material.

Therefore:

- the corpus is suitable for exercising the training/evaluation pipeline;
- the corpus is **not** currently documented as a gold-quality conversational or factual dataset;
- falling loss must not be interpreted as proof that the corpus content is correct or desirable;
- weak generated answers can remain compatible with successful optimization when the target distribution itself is weak.

This statement applies to the specific diagnostic corpus. It is not a blanket statement about Hugging Face-hosted datasets or web data in general.

## Known data-risk classes

For future corpus acceptance, measure rather than assume the following:

| Risk class | Why it matters |
|---|---|
| Exact duplicates | Overweights repeated examples and can inflate apparent fit |
| Near-duplicates | Reduces effective diversity and contaminates evaluation more subtly |
| Boilerplate/navigation text | Teaches non-semantic web formatting patterns |
| Truncated or malformed records | Corrupts prompt/response or language structure |
| Wrong-language records | Changes the intended language distribution |
| Template-heavy/synthetic text | Can dominate style and produce repetitive generations |
| Factual errors | Optimizes the model toward incorrect targets |
| Low-quality dialogue | Produces superficially fluent but poor assistant behavior |
| Train/validation overlap | Invalidates a clean generalization interpretation |

## Minimum corpus acceptance record

A future dataset promoted beyond diagnostic use should record at minimum:

```text
source names and versions
retrieval date
license/usage terms for each source where applicable
raw file identities
filtering/deduplication procedure
record counts before and after filtering
language distribution
train/validation/test split procedure
split identities
contamination checks
known exclusions and unresolved defects
```

## Recommended quality audit

Before changing model architecture to fix answer quality, sample and classify the corpus.

A minimal audit should report, on a reproducibly selected random sample:

```text
malformed_rate
duplicate_rate
near_duplicate_rate
boilerplate_rate
wrong_language_rate
low_quality_dialogue_rate
synthetic_or_template_rate
factual_error_review_rate   # only where a reliable truth source exists
```

The exact acceptance thresholds should be chosen for the target application. Niyah.Engine does not currently define universal thresholds.

## Validation and test separation

The current 30-record held-out set has been used to guide continuation decisions. It should therefore be considered validation-like.

For future runs:

```text
training   → gradient updates
validation → checkpoint/training decisions
final test → untouched until decisions are frozen
```

Do not repeatedly inspect the final test set and continue calling it held-out test evidence.

## Data-quality decision rule

If a clean corpus run uses the same model shape and training mechanics and materially improves fixed unseen generation/evaluation results, that provides direct evidence that data quality was a bottleneck.

Until such a controlled comparison exists:

**CAUSALITY: NOT ESTABLISHED** for any claim that corpus quality is the sole cause of weak generations.

## Scope

This document establishes dataset-engineering requirements and records the current diagnostic limitation. It does not make a legal determination about dataset licensing, privacy, copyright, or regulatory compliance.

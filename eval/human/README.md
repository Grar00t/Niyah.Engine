# Blind Human Evaluation v1

This directory defines a small external human-evaluation protocol for comparing Niyah.Engine output with another model or checkpoint without telling raters which system produced which answer.

This is **not RLHF by itself**. It becomes preference-training data only if the collected preferences are later incorporated into a defined training objective. Until then, it is external human evaluation.

## Goal

Measure whether independent readers prefer one output over another on fixed unseen prompts while reducing brand/model-name bias.

The protocol is deliberately small enough to hand to external reviewers. It is not a benchmark-leadership claim.

## Files

- `prompts-v1.tsv` — frozen prompt set and metadata.
- `rating-template-v1.tsv` — one blank row per prompt for a single rater.
- `references-v1.tsv` — objective reference answers only where an exact answer is appropriate; keep this away from generation systems and preferably away from raters until after they submit preferences.

## Generation protocol

For each prompt:

1. run the two candidate systems independently;
2. freeze decoding settings before seeing the results;
3. record exact model/checkpoint identity, tokenizer identity, repository revision, decoding settings, and exit status;
4. do not edit, trim, correct, translate, or normalize either generated answer;
5. randomly assign the two raw outputs to `A` and `B` independently per prompt;
6. keep the A/B identity key private until ratings are frozen.

When comparing checkpoints from the same model family, use the same decoding policy for both. When comparing different runtimes, record any unavoidable decoding differences rather than describing them as equivalent.

## Rater instructions

Raters should not be told which output is Niyah or which system is expected to win.

For every prompt, record:

- `preference`: `A`, `B`, or `TIE`;
- `coherence_a` / `coherence_b`: `0`, `1`, or `2`;
- `instruction_a` / `instruction_b`: `0`, `1`, or `2`;
- `language_a` / `language_b`: `0`, `1`, or `2`;
- `factuality_a` / `factuality_b`: `0`, `1`, `2`, or `NA`;
- optional free-text note.

Score meanings:

```text
0 = clearly fails the criterion
1 = partially satisfies / mixed
2 = clearly satisfies the criterion
NA = criterion is not applicable or cannot be verified
```

`preference` is the primary human-preference signal. The criterion scores help explain why.

## Arabic handling

Do not penalize valid Arabic merely because it differs from one preferred regional style. Judge whether the answer is understandable, appropriate for the prompt, and linguistically coherent. A dialect-specific prompt may be rated for dialect appropriateness, but this pack does not claim comprehensive Saudi-dialect coverage.

## Factuality

The reference file contains answers only for prompts with an objective compact answer. For open-ended prompts, factuality may be `NA` unless the answer introduces a checkable factual claim.

Do not use the reference file to rewrite model answers.

## Minimum useful external run

A useful first run is:

```text
20 frozen prompts
2 candidate outputs per prompt
>= 5 independent raters
blind A/B assignment
no post-editing
```

Five raters is a practical pilot target, not a statistical guarantee.

## Reporting

After ratings are frozen and the identity key is revealed, report at minimum:

```text
prompt-pack revision / SHA
candidate identities
checkpoint/tokenizer identities when available
decoding settings
number of raters
A/B randomization method
preference counts: candidate-1 / candidate-2 / tie
criterion-score aggregates
missing/invalid ratings
raw anonymized ratings or a clear reason they cannot be published
```

Do not convert this small human study into claims about broad intelligence, safety, reasoning, or benchmark leadership.

## Data/privacy boundary

Do not collect phone numbers, account identifiers, private chat history, or other unnecessary personal data from raters. A random rater ID is sufficient for aggregation.

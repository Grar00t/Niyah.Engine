# Evaluation Packs

Repository evaluation assets live here when they are intended to be frozen, reviewable inputs rather than implementation code.

## Human evaluation

[`human/`](human/) contains the first blind A/B human-evaluation protocol and frozen prompt pack.

The pack is designed for external preference measurement. It does not modify Niyah.Engine training and must not be described as RLHF unless the collected preferences are later incorporated into a defined preference-training objective.

Published prompt packs are immutable by version. If a prompt materially changes after ratings have been collected, create a new version rather than silently rewriting the old input set.

## Evidence rules

Evaluation packs should:

- use stable IDs;
- record language/category metadata where useful;
- avoid including model identity in blind rater material;
- separate objective references from generation inputs;
- preserve raw outputs without post-editing;
- record candidate/checkpoint identities and decoding settings with the final report;
- state the limits of what the evaluation establishes.

Do not tune a candidate against a final test/evaluation pack after observing its result and then continue calling that pack untouched.

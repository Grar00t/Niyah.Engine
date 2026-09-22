# NIYAH data contract

Datasets in this repository are not accepted because they are large or model-generated. They are accepted only when provenance and verification rules are explicit.

## Verified math v1

Generate a deterministic Arabic math corpus:

```bash
python tools/generate_verified_math.py --out data/generated/verified_math_v1.jsonl --records 10000 --seed 1448
python tools/validate_verified_math.py data/generated/verified_math_v1.jsonl --expect-records 10000
```

The generator currently covers exact integer arithmetic, rational arithmetic, linear equations, GCD, modular inverses, and combinations. Every answer is derived from exact Python integer/Fraction arithmetic and carries a machine-checkable verification payload.

Generated corpora are excluded from Git by default; regenerate them from the committed generator and seed. For release/training runs, preserve the emitted SHA256 alongside the exact generator commit SHA.

## Status labels

- `MECHANICALLY_VERIFIABLE`: deterministic verifier recomputes the answer.
- `SYNTHETIC_UNVERIFIED`: generated candidate, not training-approved.
- `HUMAN_VERIFIED`: requires an external review receipt; this repository does not infer it automatically.

Do not mix evaluation-only material into train splits without explicitly changing its provenance and documenting the reason.

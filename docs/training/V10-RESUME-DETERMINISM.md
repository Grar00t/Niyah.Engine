# Niyah V10 Resume Determinism Gate

Status: **PASS**

Two independent CUDA resume runs were executed from the exact same
V10 checkpoint and cursor before extending training beyond update 100.

## Input lineage

- Checkpoint SHA-256: `4127fbdef90b5aa36ad1098e3439c8dd4ef2e779bec21e783ac4506fb2679a90`
- Cursor SHA-256: `43de8ce5fa076104adf84c83f8b3777e0087c86cbb2761f527fb29777241668f`
- Tokenizer SHA-256: `d47a69a1ec180f577aeebe3905d6aed22d4b8ea3408b66fe4b8571fccc2fa841`
- Shards: `33`
- Training binary SHA-256: `614d46d5de3c9e2672eda5bc25d55e652a6ee31ec136d763369728af5042ae7a`

## Resume contract

- Backend: `cuda`
- Runs: `2`
- Updates per run: `2`
- Optimizer step: `100` -> `102`
- Cursor position: `400` -> `408`

## Loss trace

- Update 101: `8.40049553`
- Update 102: `8.38345718`
- Mean loss: `8.39197636`
- Loss trace identical: **YES**

## Byte identity

- Checkpoint SHA-256:
  `21b7eb01693b97ab6f94936088f42ef1fac14c78a1e4f0bac415d6a9bb337328`
- Cursor SHA-256:
  `70a372640fb2c82da8256dc103e4de83fb7e41090f5ef1dce01bcdee71bcd021`
- Checkpoint byte-identical: **YES**
- Cursor byte-identical: **YES**

## Immutability

- Input checkpoint changed: NO
- Input cursor changed: NO
- Tokenizer changed: NO
- Shards changed: NO
- Source code changed: NO

## Result

`NIYAH_V10_RESUME_DETERMINISM=PASS`

The 0102 artifacts are verification artifacts only.

Canonical extended training continues from:

`model-0100.ckpt`

rather than selecting either deterministic verification output as the
training parent.

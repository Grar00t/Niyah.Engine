# NIYAH V10 — STEP0200 Training Receipt

## Repository

- Branch: `integration/humain-final-20260927`
- Source commit before receipt: `ba5aadf7fe7c71c939c3a4c6a3fbf53892e09879`

## Training Result

- Mode: resume
- Backend: CUDA
- Shards: 33
- Updates in this run: 100
- Final optimizer step: 200
- Final cursor epoch: 0
- Final cursor position: 800
- Final loss: 8.23955345
- Mean loss: 8.42103672
- Batch size: 4
- Accumulation steps: 1
- Exit status: 0
- Wall time: 2:20:25

## Artifacts

### model-0200.ckpt

- Path: `/mnt/d/training-data/v10-31m-step0200-20260928/model-0200.ckpt`
- Bytes: 377592032
- SHA-256: `112d5e380d1cbd1bbcfcfffb839d07efa8d5658b91145eaa8c02095cc42a0e8f`

### cursor-0200.bin

- Path: `/mnt/d/training-data/v10-31m-step0200-20260928/cursor-0200.bin`
- Bytes: 120
- SHA-256: `717e3f1f8cf0dcbb5e0763098a9fbdc05b57d00ffe2f48653fb1c8cfb1d91b4e`

### niyah-train

- Path: `/home/a/niyah/builds/humain-final-cuda/niyah-train`
- SHA-256: `614d46d5de3c9e2672eda5bc25d55e652a6ee31ec136d763369728af5042ae7a`

### tokenizer

- Path: `/mnt/d/training-data/v10-clean-20260927/tokenizer-final/tok.bin`
- Bytes: 144400
- SHA-256: `d47a69a1ec180f577aeebe3905d6aed22d4b8ea3408b66fe4b8571fccc2fa841`

## Gate

```text
V10_STEP0200_COMPLETE=YES
OPTIMIZER_STEP=200
CURSOR_POSITION=800
CHECKPOINT_PRESENT=YES
CURSOR_PRESENT=YES
HASH_GATE=PASS
EXIT_STATUS=0
TRAINING_RUNNING=NO
```

## Next Gate

Do not extend training solely from completion status.

Next required step:

1. establish the repository-supported checkpoint loading/generation path;
2. load `model-0200.ckpt` without mutation;
3. run deterministic generation/health evaluation;
4. compare STEP0100 and STEP0200 behavior;
5. decide whether STEP0200 -> STEP0500 is justified.

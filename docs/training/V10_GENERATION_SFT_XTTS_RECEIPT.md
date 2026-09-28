# V10 generation and Arabic voice execution receipt — 2026-09-28

SYSTEM_STATE: integration/humain-final-20260927, source `977167246e51f94b28ff8f59d579e630d4edb3be`. Original artifacts preserved. No native engine/sampler change was justified.

FIXED: a separate V10 response-masked SFT derivative learns four controlled Arabic fixtures. XTTS reference cloning now produces valid Arabic audio through the installed Python API, without retraining or package edits.

ROOT_CAUSES_PROVEN: baseline greedy repetition originates in the checkpoint logits before selection. All 16 CPU/CUDA baseline cases reproduce it. The V10 objective is stream pretraining; exact first800 samples contain204800 targets and no BOS input or EOS target. All33 shard hashes/CRCs and ordered cursor binding pass. XTTS used the wrong checkpoint argument; installed eval returns None, and TorchCodec DLL loading also failed.

ROOT_CAUSES_NOT_ESTABLISHED: sole underlying cause of weak prompt conditioning (`ROOT_CAUSE=NOT_ESTABLISHED`), independent effects of lower learning rate vs full batch, and broad language competence. No source defect was demonstrated in tested paths.

NIYAH_RESULT: controlled fixture result `PASS` across 12 CPU/CUDA/repeated cases. The native gate also passes 12/12 exact response token sequences plus EOS, generated counts [5,8,4,6], and stopped_on_eos=true. Original checkpoints remain unhealthy. Two of three unseen formatted controls are wrong: “كيف حالك؟” produces “الرياض عاصمة السعودية.”; “حدثني عن الرياض” produces “ما اسمك؟”. All four raw prompts fail without the trained prefix/suffix, including repeated “ك؟” for raw “وش علومك؟”. This is controlled learning evidence, not a conversational release.

Training format is prefix `سؤال: ` and suffix `\nجواب: `. The V3 shard has4 samples,19 response targets plus4 EOS, correct +1 shift, and no truncation. Early canaries scored0/4 at8updates,1/4 at40,2/4 at100,2/4 at200. Calibration uses all4 examples per batch at learning rate0.0001 for40updates after resetting optimizer state. Total new updates240; training command wall time 350.23s. Final persisted optimizer_step40, cursor_epoch39, position4. STEP0200→STEP0500 was not run because the generalization gate fails.

Final candidate checkpoint: `/mnt/d/training-data/niyah-execution-20260928/sft-canary/calibrated-0040.ckpt`; SHA256 `10729970e6a3e25ca569e90629178d9abaa9c82c5fba97066a603b2dfed0fc7a`. Cursor SHA256 `ffa266cdf8ecc41dad6c2301380add71186d4854dcd7a3179980d23edd931fd9`. Both canonical checkpoint/cursor pairs and tokenizer rehash unchanged. V9 SFT is rejected because its tokenizer identity differs.

QWEN_TEACHER_RESULT: actual local base+adapter load and two offline CPU greedy replays pass (“نعم”+EOS). Adapter SHA256 `7b4781c63fbbe98c1a44680e49196ba44e36adee172e6f14f6efca25f57796ff`; base SHA256 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`. All392 tensor dimensions match, all values finite. Candidate only: dataset revision/hash, metrics, and original-to-frozen provenance receipt are missing. No teacher output was used for SFT.

XTTS_RESULT: VOICE_CLONE=PASS for valid reference-conditioned audio. `C:\Users\A\Documents\XTTS-Arabic\output\niyah-ar-reference-20260928.wav`, 5.749333s, mono PCM16 24000Hz, 276012bytes, SHA256 `24b3747756f4003c8e22483d9e312f65ec4937746ec377100ed6595c851b6a39`. Full FFmpeg decode and finite/nonzero samples pass. Reference SHA256 `91e8e62fad5e71a9680786756d57ec226a4719fbeb7d421145ee017146b84bdc`. Perceptual identity and transcript were not independently certified.

TESTS: release39/39, sanitizer39/39, CUDA53/53; includes prompt prefix/suffix, checkpoint binding and resume equivalence. Actual31M checkpoint: CPU training/full-forward logits identical, CUDA max difference1.43e-6, all-weight gradient max difference2.38e-7;12 sampled finite differences pass. Full/KV logits are identical for both original checkpoints across all four prompts and eight generated steps each. These are scoped checks, not universal proof.

ARTIFACTS: `/mnt/d/training-data/niyah-execution-20260928/FINAL-RECEIPT.json` and `ARTIFACT-SHA256.json` bind commands, scripts, logs, artifacts and limitations. Reproduce original behavior with `python3 /mnt/d/training-data/niyah-execution-20260928/reproduce_baseline.py`; final text gate with `python3 /mnt/d/training-data/niyah-execution-20260928/evaluate_canary.py calibrated-0040 --full`; scope controls with `evaluate_scope.py calibrated-0040`. Native probes have their source and compile/run recipes in the same evidence directory. Existing checkpoint outputs are never overwritten.

GIT_HEAD: source before this documentation receipt `977167246e51f94b28ff8f59d579e630d4edb3be`. Only this receipt and its JSON are intended for Git; no checkpoints, datasets, binaries, or secrets.

REMAINING_BLOCKERS: the unchanged original models still loop, and the calibrated four-fixture derivative fails two of three unseen paraphrases and all four raw prompts, one with a repeated sequence. Thus NIYAH general conversation health is not complete. Qwen provenance is insufficient to qualify a judge. No longer training run was launched in response to these failures.

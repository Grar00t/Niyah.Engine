# 2026-10-13 execution receipt

This consolidates the existing local repair checkout; it is not a claim that all changes originated in one session. The recorded source base was `6ec58869fca15cb6034bd5a689e9a04483651d4c` plus the source manifest in the accompanying JSON.

100 additional CUDA resume updates completed (100 to 200) on the unchanged old/Wikipedia mixture. Original inputs were preserved. Both held-out evaluations were rerun with matching tokenizer, sample geometry and held-out identity. Old perplexity changed from 234.45281605 to 231.95511051; Wikipedia from 421.94060712 to 401.38575107. These are observations of this run, not causal isolation or general language benchmarks.

All 24 tested generation calls (before/after, CPU/CUDA, penalties 1 and 1.1, three fixed prompts) stopped with degraded exit 20. The guard behavior is observable; Arabic conversation quality is not acceptable in these examples. No model-quality release is approved.

The official Tanzil Simple Clean 1.1 source was downloaded and retained unchanged, and native shards were built with the existing tokenizer. The 6236 verses are partitioned into 5950 training and 286 validation verses. Five exact verse strings overlap across those partitions; this is disclosed rather than treating the split as a contamination-free benchmark. The new data was not silently inserted into the resume run. Keep its source notice with all data artifacts.

Local full tests are rerun before this consolidation commit. Two external Casper tests are skipped because their external fixtures are not configured. Synthetic fixture training inside tests is distinct from the model resume experiment. Local Windows CUDA verification is not Linux/GitHub CI verification.

Remaining: live deadline/cancellation, incremental character guard, session commit/rollback integration, and model-quality improvement. This is a research runtime snapshot, not a finished conversational model. No checkpoints, training corpora, or credentials are included in the source commit.

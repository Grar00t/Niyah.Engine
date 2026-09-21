# Eval CLI Verification Gate

This file defines the acceptance gate for the `niyah eval` + add-1 bigram baseline patch. It records no pass result until CI completes on the patch head.

Required checks:

- baseline unit test passes;
- text-mode `niyah eval` returns finite model and baseline metrics;
- repeated evaluation leaves checkpoint bytes unchanged;
- missing tokenizer fails closed;
- raw text is rejected as `--format shard`;
- valid shard evaluation succeeds;
- shard output omits bits-per-byte metrics;
- shard mode rejects explicit `--sequence-length`;
- Ubuntu build/test passes;
- Windows build/test passes;
- sanitizer build/test passes.

The historical 30-record pilot metrics remain diagnostic evidence from the earlier helper and are not retroactively attributed to this CLI until rerun with the CLI on the same artifacts.

# Receipt-chain regression count provenance

The return value and text output of a C test executable do not establish a universal model capability. This document records exactly what the receipt-chain test measured.

## Source versions

- main at dd8a621223e59d3cd3d8d920a48653ce94192993 had **10** test invocations and executed **54** CHECK assertions. Its committed test blob was Git SHA-1 3b4db15d97a8a688a92d053913fb074515b1bc8f (file SHA-256 0afdab0e468885373462deb9869b661d5898d64b969d869f61627e1b86f0f6a8).
- An uncommitted local extension included test_independent_chain_hashes(): **one** new_chain() call (containing one executed CHECK) and **four** direct CHECK calls. The local source file SHA-256 was 77b7edb5da214a9648180da162406d0dd9b0b60992a00354f544b3cfbb543bab.
- This change includes that independent fixed-vector regression and an **actual runtime counter** within the CHECK macro. Its **11** test cases execute **59** assertions in the successful control flow.

## Reproduce

Run niyah_receipt_chain_test (registered by CMakeLists.txt). Inspect both exit status and RECEIPT_EXECUTED_CHECKS=59. Unlike grepping the file, incrementing the counter inside the macro counts assertions in helper functions once per invocation. The counter is printed even when a check fails.

The new fixed-vector regression pins known digest bytes for two accepted sequential receipts. Its scope is only the C11 receipt-chain implementation under the tested configuration; it does not establish MMR integration, ALLaM token-path coupling, cryptographic authorship, or factual/logical correctness.

## Provenance / scope

The previously reported **54** and **59** were both reproduced by compiling instrumented *isolated copies* of the respective inputs against the Niyah source implementation, using GCC with -std=c11 -O2 -Wall -Wextra -Werror -pedantic. This repository addition brings the extra locally tested scenario into review, rather than claiming 59 were already present in the previous public commit.

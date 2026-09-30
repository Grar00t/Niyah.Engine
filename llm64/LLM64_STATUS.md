# llm64 status

## Verified
- ASM/C stdout parity: PASS for tested boundary matrix.
- ASM/C exit-code parity: PASS for tested boundary matrix.
- CLI accepts max-new-tokens: PASS.
- Context preflight: PASS.
- Greedy EOS loop on fixture: PASS.

## Not verified
- Semantic Arabic generation quality.
- Arithmetic reasoning quality.
- General LM competence.

## Current interpretation
llm64 is a syscall-only inference runtime proof.
It is not an Arabic LLM product until checkpoint/tokenizer/training evaluation passes.

## Known bad behavior
Real checkpoint outputs show degenerate repetition:
- "كم ناتج 17 + 25؟" -> repeated bullet/2 pattern
- "1+1=" -> repeated 1
- "ABC" -> repeated AI/and/a
- "x" -> cure/and repetition

This is present in C reference too, so it is not proven to be an ASM divergence.

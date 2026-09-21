# FAQ

## Is Niyah.Engine a language model or only a runtime?

Both concepts exist in the project. The repository implements the runtime/training system, while checkpoints contain learned model parameters produced by training that runtime.

The current evidence establishes a working native training/inference lifecycle and measurable held-out learning on a diagnostic pilot. It does not establish a mature general-purpose assistant.

## Is Niyah.Engine based on Qwen, Llama, or another pretrained model?

No external pretrained model runtime is required by the core implementation. Niyah.Engine implements its own tokenizer/model/training/inference path.

This architectural independence does not imply equal capability or scale.

## Is the model implemented in C11?

The engine that defines and executes the model lifecycle is primarily native C11. Optional CUDA code exists for accelerator paths. A trained model itself is better understood as architecture + learned tensor values + tokenizer semantics, rather than as a programming language.

## Does Niyah.Engine use PyTorch or Hugging Face Transformers?

They are not required dependencies for the native model core.

Hugging Face may still be relevant as a data/model ecosystem outside the engine, but Niyah.Engine does not require Transformers to execute its native training/inference path.

## Is it production-ready?

No current evidence establishes production readiness.

The project is documented as a research runtime with verified implementation paths and bounded diagnostic model evidence.

## Does falling training loss prove that the model is good?

No. Falling training loss shows that optimization is reducing the training objective.

The current pilot also shows falling held-out/validation loss across four checkpoints, which is stronger evidence of learning on that distribution. Neither measurement alone proves broad conversational quality, reasoning, factual reliability, or safety.

## Why can the loss improve while generated answers remain poor?

Loss measures how well the model predicts the distribution it is trained/evaluated against. If the corpus contains poor dialogue, repetition, boilerplate, factual errors, or weak task structure, the model can become better at modeling that distribution without becoming a high-quality assistant.

The current diagnostic corpus is known to have quality concerns, but data quality has not been established as the sole cause of every weak generation.

## What is the current held-out result?

The recorded pilot trajectory is:

```text
step 0200: loss 5.581078354, ppl 265.357600904
step 0635: loss 4.589545587, ppl  98.449683132
step 1270: loss 3.942194185, ppl  51.531547081
step 1905: loss 3.643954027, ppl  38.242751050
```

See [EVALUATION.md](EVALUATION.md) for methodology and limitations.

## Is that a final test score?

No. The same 30-record set has been used to guide continuation decisions, so it should now be treated as validation-like. A separate untouched test set is required for a clean final characterization.

## Why is CPU FP32 the reference path?

It gives the project one explicit correctness baseline. CUDA is optional and can be verified independently without making accelerator behavior the definition of core semantics.

## Does segment/role embedding prevent prompt injection?

No. Segment IDs provide a learned modeling signal. They are not a security boundary and should not be represented as one.

## Does the repository currently have `niyah eval`?

Not on the documented `main` revision. The current primary CLI exposes `prepare` and `run`; evaluation exists as a native API and has been exercised with diagnostic helpers.

See [CLI.md](CLI.md).

## Is the open Evidence V1 PR part of main?

No. PR #56 remains an open draft at the current documentation snapshot. Its IR/receipt/evidence additions should not be described as merged main functionality until merged.

## Where should I start?

Use [QUICKSTART.md](QUICKSTART.md), then read [MODEL_CARD.md](MODEL_CARD.md) and [VERIFICATION.md](VERIFICATION.md) before interpreting model-quality claims.

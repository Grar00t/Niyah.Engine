# Niyah.Engine

Local C11 model runtime, retrieval layer, evidence/reasoning utilities, graph tooling, and an optional Hugging Face training path.

The native runtime fails explicitly when model weights are unavailable: `niyah_llm_generate` returns `NIYAH_ERR_NO_WEIGHTS` with no generated text. That behavior is covered by native tests.

Model output is treated as untrusted data. The repository does not claim that prompt wording, fine-tuning, or a model policy creates a security boundary.

## Repository layout

| Path | Contents |
| --- | --- |
| `native/` | C11 runtime, kernels, tokenizer, model loader, generation, evidence subsystem, constraint solver, deterministic capability policy, CLI, C ABI, native tests |
| `native/niyah_mini/` | Separate compact model implementation and tests |
| `search/` | BM25 index, URL handling, HTTP fetch, HTML extraction, search front end, retrieval tests |
| `storage/` | Local C storage plus SQL/PostgreSQL material |
| `tools/` | GGUF converter, build/check scripts, converter fixtures/tests |
| `scripts/` | Canonical graph audit, validation, chunk/rebuild, export, and embedding-space configuration |
| `src/inference/` | Deterministic `part_of` transitive inference over a graph |
| `tests/` | Python standard-library regression tests for graph tooling, inference, and model profiles |
| `knowledge/` | Canonical knowledge data and taxonomy |
| `normalized/` | Normalized graph exports |
| `rag/` | Retrieval source policy data |
| `neutral/` | Optional Hugging Face corpus, LoRA domain adaptation, and direct inference utilities; defaults to a GPT-OSS 20B weights-first path |
| `ui/Niyah.App/` | C# desktop front end using the native shared library through P/Invoke |

## Build and test

Native runtime:

```sh
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

Retrieval layer:

```sh
cmake -S search -B build/search -DCMAKE_BUILD_TYPE=Release
cmake --build build/search
ctest --test-dir build/search --output-on-failure
```

Full local checks:

```sh
sh tools/ci.sh
```

Individual stages are `native`, `make`, `search`, and `python`.

Hosted checks are defined in `.github/workflows/native.yml` and run on pushes to `main`, pull requests, and manual dispatch. The workflow covers native CMake/ctest, ASan+UBSan native tests, search CMake/ctest, and Python tooling/tests.

The Python stage compiles `tools/`, `neutral/`, `scripts/`, `src/`, and `tests/`; runs `unittest` graph/model-profile tests; then runs the GGUF converter and K-quant fixtures.

## Verified implementation areas

The repository contains assertion-backed implementations for:

- Kleene three-valued truth helpers.
- Matrix operations, normalization, activations, RoPE, causal attention, transformer blocks, sampling, tokenizer, and arena runtime.
- Flat float32 model loading and local autoregressive generation.
- Evidence envelopes/graph/reasoner and the constraint solver.
- Deterministic fixed-capacity capability authorization in `native/niyah_control.c`: exact capability + exact opaque resource id, default deny, no prefix widening.
- C ABI document bridge.
- BM25 retrieval with real per-document term frequency.
- Owned retrieval document text: `niyah_index_add_document` copies caller text, so caller buffers may be released or reused after insertion.
- Deterministic graph `part_of` transitive inference with thresholded path confidence.
- Canonical graph validation, audit, chunk/rebuild deduplication, and GraphML/Cytoscape export.
- Verified package execution accepts a required base artifact and an optional adapter artifact; artifact hashes are checked before launch.
- Run proofs bind the prompt, emitted bytes, package/model hashes, and execution-contract fields. The contract explicitly records that network isolation is not enforced and that model output is untrusted.

No performance, accuracy, safety, factuality, or throughput claim should be inferred from this list. `evidence/` does not currently contain benchmark results.

## Authority boundary

`native/niyah_control.*` is deterministic authorization code, not an LLM classifier. A request is allowed only when its single capability bit and exact resource id match an explicit grant. Unknown capability bits, combined capability bits, invalid resources, missing grants, and invalid requests are denied.

The current `niyah run` path does not execute model-generated tool calls. It launches a verified model package and streams text. The proof contract therefore records `model_output_trust=untrusted` and `privileged_actions=disabled`.

This is a software boundary, not a claim of hardware memory separation. Filesystem, process, GPU, and network isolation for the model process remain separate implementation work.

## Known limits

- No GPU execution backend is implemented in the native runtime.
- Native generation is exercised at batch size 1; batched inference is not established by the current tests.
- `search/niyah_index.c` still uses linear term and document lookup. This is a performance limitation, not a correctness claim.
- `search/niyah_index.h` and `native/niyah_document.h` define different structures named `NiyahDocument`; the headers deliberately reject inclusion together in one translation unit.
- The optional `neutral/` path is not part of the C11 runtime and does not implement factuality scoring, reinforcement learning, LVU, peer prediction, or a Merkle audit log.
- `niyah run` does not currently enforce a network namespace, filesystem sandbox, seccomp profile, or equivalent OS-level confinement. The execution proof says so instead of claiming `local_only=true`.

## GPT-OSS 20B path

The optional `neutral/` runner defaults to `openai/gpt-oss-20b` for weights-first inference:

```sh
python3 -m pip install -r neutral/requirements.txt
./neutral/run.sh infer 'Explain BM25 briefly.'
```

`neutral/model_profiles.py` requires the model's chat template for GPT-OSS rather than synthesizing a replacement format. The GPT-OSS training profile keeps the checkpoint's native MXFP4 path and limits LoRA targets to the attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`).

If `OUTPUT_DIR/adapter_config.json` exists, `neutral/run.sh infer` uses that adapter automatically; otherwise it uses the base weights. See `neutral/README.md` for exact commands and limitations.

Training here means LoRA domain adaptation. It does not train a foundation model from scratch and does not claim to remove instruction/data conflation inside the transformer.

## GGUF conversion

`tools/convert_gguf_to_niyah.py` converts a GGUF checkpoint into a configuration JSON and flat float32 weights blob:

```sh
python3 tools/convert_gguf_to_niyah.py model.gguf weights.bin \
  --config config.json --progress
```

The converter currently decodes:

- F32
- F16
- Q4_0
- Q4_1
- Q4_K
- Q6_K

Q2_K, Q3_K, Q5_K, Q8_K, Q5_0, Q5_1, Q8_0, and Q8_1 are not decoded by the current converter. Requantise unsupported input before conversion.

`tools/download_and_convert.sh` uses Qwen2.5-0.5B-Instruct as a conversion example and defaults to a Q4_K_M checkpoint.

The emitted weight order is:

1. token embedding
2. per layer: attention norm, Q/K/V/O projections, FFN norm, gate/up/down projections
3. final norm
4. LM head when embeddings are not tied

The converter emits both configuration key schemes required by `native/niyah_model.c` and `native/niyah_mini/niyah_mini_model.c`.

## Graph utilities

The committed canonical graph is `knowledge/canonical_knowledge_v2.json`. Audit and validation use it by default:

```sh
python scripts/validate_graph.py
python scripts/audit_graph.py
```

Chunk and rebuild:

```sh
python scripts/chunk_graph.py --output-dir ./chunks
python scripts/build_graph.py --chunks-dir ./chunks --output ./data/rebuilt_graph.json
```

`build_graph.py` deduplicates records repeated across chunk boundaries.

Embedding-space configuration records a model/dimension binding only; it does not train embeddings:

```sh
python scripts/configure_embedding_space.py \
  --model-name <model-or-local-path> \
  --dimensions <dimensions>
```

## Optional Hugging Face path

`neutral/` is an independent optional path for preparing a local corpus, model inference, and LoRA domain adaptation. See `neutral/README.md` for its exact scope and dependencies.

## License

See repository metadata.

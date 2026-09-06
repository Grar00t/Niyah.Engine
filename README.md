# Niyah.Engine

Local C11 model runtime, retrieval layer, evidence/reasoning utilities, graph tooling, and an optional model-hub training path.

The native runtime fails explicitly when model weights are unavailable: `niyah_llm_generate` returns `NIYAH_ERR_NO_WEIGHTS` with no generated text. That behavior is covered by native tests.

## Repository layout

| Path | Contents |
| --- | --- |
| `native/` | C11 runtime, kernels, tokenizer, model loader, generation, evidence subsystem, constraint solver, CLI, C ABI, native tests |
| `native/niyah_mini/` | Separate compact model implementation and tests |
| `search/` | BM25 index, URL handling, HTTP fetch, HTML extraction, search front end, retrieval tests |
| `storage/` | Local C storage plus SQL/PostgreSQL material |
| `tools/` | GGUF converter, build/check scripts, converter fixtures/tests |
| `scripts/` | Canonical graph audit, validation, chunk/rebuild, export, and embedding-space configuration |
| `src/inference/` | Deterministic `part_of` transitive inference over a graph |
| `tests/` | Python standard-library regression tests for graph tooling/inference |
| `knowledge/` | Canonical knowledge data and taxonomy |
| `normalized/` | Normalized graph exports |
| `rag/` | Retrieval source policy data |
| `neutral/` | Optional model-hub corpus, QLoRA domain-adaptation, and direct inference utilities |
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

The Python stage compiles `tools/`, `neutral/`, `scripts/`, `src/`, and `tests/`; runs `unittest` graph tests; then runs the GGUF converter and K-quant fixtures.

## Verified implementation areas

The repository contains assertion-backed implementations for:

- Kleene three-valued truth helpers.
- Matrix operations, normalization, activations, RoPE, causal attention, transformer blocks, sampling, tokenizer, and arena runtime.
- Flat float32 model loading and local autoregressive generation.
- Evidence envelopes/graph/reasoner and the constraint solver.
- C ABI document bridge.
- BM25 retrieval with real per-document term frequency.
- Owned retrieval document text: `niyah_index_add_document` copies caller text, so caller buffers may be released or reused after insertion.
- Deterministic graph `part_of` transitive inference with thresholded path confidence.
- Canonical graph validation, audit, chunk/rebuild deduplication, and GraphML/Cytoscape export.

No performance, accuracy, or throughput claim should be inferred from this list. `evidence/` does not currently contain benchmark results.

## Known limits

- No GPU execution backend is implemented in the native runtime.
- Native generation is exercised at batch size 1; batched inference is not established by the current tests.
- `search/niyah_index.c` still uses linear term and document lookup. This is a performance limitation, not a correctness claim.
- `search/niyah_index.h` and `native/niyah_document.h` define different structures named `NiyahDocument`; the headers deliberately reject inclusion together in one translation unit.
- The optional `neutral/` path is not part of the C11 runtime and does not implement factuality scoring, reinforcement learning, LVU, peer prediction, or a Merkle audit log.

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

`tools/download_and_convert.sh` uses legacy-model-2.5-0.5B-Instruct as a conversion example and defaults to a Q4_K_M checkpoint.

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

## Optional model-hub path

`neutral/` is an independent optional path for preparing a local corpus, QLoRA causal-LM domain adaptation, and direct generation. See `neutral/README.md` for its exact scope and dependencies.

## License

See repository metadata.

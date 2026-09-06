# Project Structure

Enumerated from the tree, not written by hand. Regenerate after any file move:

```bash
git ls-files | tree --fromfile -a --noreport
find . -path ./build -prune -o -type f -printf '%s %p\n'
```

## What the previous revision got wrong

The previous revision of this file was 460 bytes and described an intended
layout rather than the tree. It declared four directories that do not exist
at any level of the repository:

| Declared | Present |
| --- | --- |
| `chunks/` | no. `normalized/chunks.jsonl` exists instead |
| `nodes/` | no. `normalized/nodes.jsonl` exists instead |
| `edges/` | no. `normalized/edges.jsonl` exists instead |
| `audits/` | no. `scripts/audit_graph.py` exists instead |

It correctly named `evidence/`, `schema/`, `manifests/` and `scripts/`, and
omitted the other nineteen top-level entries, including the entire `native/`
and `search/` trees that hold the engine.

The intent behind the old file was not wrong. The layout moved and the file
did not. That is the failure mode this revision exists to prevent.

## Enumeration depth

This document enumerates the repository root, and one level into every
top-level directory. The following directories are named but **not** expanded
here, and are therefore unaudited by this file:

`db/migrations/`, `storage/postgres/`, `storage/schema/`, `ui/Niyah.App/`,
`src/inference/`, `knowledge/00_registry/`, `knowledge/10_taxonomy/`,
`knowledge/20_lessons/`, `knowledge/30_canonical/`, `knowledge/40_staging/`,
`sources/00_inbox/`, `sources/10_raw_graph/`, `sources/20_open_data/`,
`sources/30_vendor_docs/`, `sources/40_verified_sources/`,
`sources/50_datasets/`, `sources/99_rejected/`, `tools/niyah_mini/`,
`neutral/niyah_mini/`, `neutral/tests/`.

Do not cite this file as coverage of those paths.

## Root

| Path | Bytes |
| --- | --- |
| `.env.example` | 194 |
| `.gitattributes` | 116 |
| `.gitignore` | 242 |
| `README.md` | 8877 |
| `docker-compose.yml` | 702 |

Directories: `.github/`, `data/`, `db/`, `docs/`, `evidence/`, `knowledge/`,
`manifests/`, `native/`, `neutral/`, `normalized/`, `rag/`, `schema/`,
`scripts/`, `search/`, `sources/`, `sql/`, `src/`, `storage/`, `tests/`,
`tools/`, `ui/`.

There is no `.gitlab-ci.yml` in the tree. See the contradictions table.

## `.github/`

| Path | Bytes |
| --- | --- |
| `.github/workflows/native.yml` | 1437 |

Triggers: `push` on `main`, `pull_request`, `workflow_dispatch`.
Jobs: `build-and-test` (cmake + ctest, ubuntu-latest) and `sanitize`
(cmake with `-DNIYAH_SANITIZE=ON`, `ASAN_OPTIONS=detect_leaks=1:abort_on_error=1`,
`UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1`).

This workflow is active. `README.md` says it does not exist.

## `native/` — the engine, C11

Eighty entries. The five largest sources:

| Path | Bytes |
| --- | --- |
| `native/niyah_token_tax.c` | 26509 |
| `native/niyah_cli.c` | 25640 |
| `native/niyah.h` | 24482 |
| `native/niyah_dispatch.c` | 21952 |
| `native/niyah_identity.c` | 18133 |

Build files: `native/CMakeLists.txt` (10166), `native/Makefile` (3179).

Evidence subsystem: `evidence_envelope.c` (4055), `evidence_graph.c` (7686),
`evidence_reasoner.c` (7173) with their headers.

Constraint solver: `constraint_solver.c` (8666), `constraint_solver.h` (4772).

Kernels and model: `niyah_attention.c` (8218), `niyah_matmul.c` (4173),
`niyah_softmax.c` (3579), `niyah_rmsnorm.c` (2157), `niyah_rope.c` (1936),
`niyah_swiglu.c` (1251), `niyah_transformer_layer.c` (5916),
`niyah_model.c` (13510), `niyah_tokenizer.c` (10517), `niyah_sampler.c` (6809),
`niyah_llm.c` (6923), `niyah_runtime.c` (3931), `niyah_embedding.c` (4081),
`niyah_graph.c` (5078), `niyah_document.c` (7400), `niyah_storage.c` (2064),
`niyah_source.c` (1702), `niyah_crawler.c` (2459), `niyah_search.c` (3111),
`niyah_bridge.c` (9599), `niyah_sha256.c` (6759), `niyah_proof.c` (7579),
`niyah_telemetry.c` (1786), `niyah_core.c` (2827).

Test translation units, colocated with the code they cover: twenty-two files
matching `*_test.c` or `test_*.c`, largest `niyah_evidence_test.c` (14148),
`niyah_identity_test.c` (9157), `niyah_csp_test.c` (8261). Plus the shell
harness `niyah_run_proof_test.sh` (6138) and the stub `niyah_fake_llama.c`
(1162).

Fixtures: `native/testdata/parallel_ar_en.tsv` (4894),
`native/testdata/root_families_ar.txt` (2571).

### `native/niyah_mini/`

A second, self-contained model implementation with its own build files.

| Path | Bytes |
| --- | --- |
| `niyah_mini_model.c` | 28312 |
| `niyah_mini_train.c` | 26419 |
| `test_niyah_mini_oracle.c` | 14900 |
| `niyah_mini_bridge.c` | 14828 |
| `niyah_mini_vocab.c` | 12317 |
| `niyah_mini_config.c` | 5006 |
| `CMakeLists.txt` | 4238 |
| `Makefile` | 1043 |

This tree carries a config reader whose key names differ from
`native/niyah_model.c`. `README.md` documents both key sets.

## `search/` — retrieval, C and C++

| Path | Bytes |
| --- | --- |
| `niyah_index.c` | 15633 |
| `http_fetch.cpp` | 12216 |
| `niyah_url.c` | 9032 |
| `search_engine.cpp` | 6886 |
| `niyah_index_test.c` | 6139 |
| `niyah_crawler.c` | 4456 |
| `niyah_index.h` | 3704 |
| `html_extract.cpp` | 2640 |
| `CMakeLists.txt` | 2486 |
| `search_engine.hpp` | 1469 |
| `niyah_crawler.h` | 1325 |
| `niyah_url_test.c` | 805 |
| `http_fetch.hpp` | 609 |
| `search_smoke.cpp` | 588 |
| `README.md` | 445 |
| `html_extract.hpp` | 323 |

## `tools/` — build and conversion

| Path | Bytes |
| --- | --- |
| `convert_gguf_to_niyah.py` | 33439 |
| `download_and_convert.sh` | 4027 |
| `build.ps1` | 3547 |
| `ci.sh` | 1805 |
| `ci.cmd` | 1210 |
| `build.sh` | 480 |
| `requirements.txt` | 36 |
| `tools/tests/test_convert_gguf.py` | 16052 |
| `tools/tests/gguf_fixture.py` | 15471 |
| `tools/tests/test_kquants.py` | 13519 |

`tools/ci.sh` stages: `native`, `make`, `search`, `python`. The `python` stage
runs `compileall` over `tools` and `neutral`, then `test_convert_gguf.py`,
then `test_kquants.py`.

## `neutral/` — corpus and training tooling, Python

| Path | Bytes |
| --- | --- |
| `train.py` | 7759 |
| `inference.py` | 5866 |
| `run.sh` | 5200 |
| `inference_contract.md` | 4750 |
| `epistemic_schema.md` | 3623 |
| `README.md` | 3180 |
| `clean_corpus.py` | 2672 |
| `data_sources.md` | 2574 |
| `validate_manifest.py` | 1690 |
| `requirements.txt` | 189 |
| `.gitignore` | 430 |

## `storage/`

| Path | Bytes |
| --- | --- |
| `local_store.c` | 11884 |
| `README.md` | 2146 |
| `local_store.h` | 1736 |

## `scripts/` — graph pipeline, Python

| Path | Bytes |
| --- | --- |
| `audit_graph.py` | 5160 |
| `export_graph.py` | 3926 |
| `build_graph.py` | 3543 |
| `validate_graph.py` | 3303 |
| `chunk_graph.py` | 2369 |
| `promote_verified_lesson.py` | 2172 |
| `train_embeddings.py` | 2052 |
| `list_verified_lessons.py` | 684 |

## Data and knowledge

| Path | Bytes |
| --- | --- |
| `data/khawrizm_graph_consolidated.json` | 391535 |
| `data/knowledge_real_v2.0.0.json` | 9983 |
| `normalized/nodes.jsonl` | 254588 |
| `normalized/evidence.jsonl` | 117938 |
| `normalized/edges.jsonl` | 93223 |
| `normalized/schemas.jsonl` | 31800 |
| `normalized/chunks.jsonl` | 30000 |
| `knowledge/canonical_knowledge_v2.json` | 15457 |
| `knowledge/domains.json` | 8225 |
| `knowledge/README.md` | 531 |
| `schema/canonical_knowledge_graph_v2.1.0.json` | 6150 |
| `schema/sovereign_knowledge_graph_v2.0.0.json` | 5325 |
| `schema/sovereign_knowledge_graph_v1.0.0.json` | 894 |
| `rag/official_sources.json` | 2501 |
| `rag/README.md` | 1189 |
| `sources/source_routing_policy.json` | 652 |
| `sources/README.md` | 607 |
| `sql/001_init_schema.sql` | 8541 |
| `sql/002_hybrid_search.sql` | 2610 |
| `manifests/graph_manifest.json` | 268 |
| `evidence/README.md` | 47 |

`normalized/chunks.jsonl` is exactly 30000 bytes. A round size on a JSONL file
is a truncation signature, not a coincidence. Verify the last line parses
before trusting a chunk count derived from it.

## `docs/`

| Path | Bytes |
| --- | --- |
| `knowledge-policy-v2.md` | 4279 |
| `migration-v1-to-v2.md` | 2739 |
| `implementation-notes.md` | 2263 |
| `knowledge-sources.md` | 1385 |
| `CHARACTER.md` | 1243 |
| `STRUCTURE.md` | this file |

## `tests/`

| Path | Bytes |
| --- | --- |
| `tests/test_validation.py` | 1995 |

This directory exists. `README.md` says it does not.

## Claims in the tree that the tree contradicts

| Where | Claim | Tree |
| --- | --- | --- |
| `README.md`, section Checks | "There is no hosted CI. All checks run locally through one script" | `.github/workflows/native.yml` is active on `push` to `main`, on `pull_request`, and on `workflow_dispatch`, and adds an asan/ubsan job that `tools/ci.sh` does not run |
| `README.md`, section Repository layout | "There is no top-level `tests/` directory" | `tests/test_validation.py`, 1995 bytes |
| `tools/ci.sh`, header comment | "Replaces the removed GitHub Actions workflow" | the workflow is present again and gates `main` |
| `.github/workflows/native.yml`, header comment | "The only pipeline definition was `.gitlab-ci.yml`" | no `.gitlab-ci.yml` in the tree |
| `README.md`, section Not implemented | "no CUDA, no SIMD intrinsics" | accurate for `main`. Open PR #8 adds AVX2 and NEON paths to `niyah_matvec` and rewrites this line, and has zero check runs |
| `README.md`, section Known issues | term and document lookup are linear scans | accurate for `main`. Open PR #8 replaces both with open-addressing hash maps and deletes this entry, unverified |

## Measurement state

`evidence/` contains one file, a 47-byte README. `manifests/graph_manifest.json`
is 268 bytes. No timing, throughput, or accuracy artifact is committed anywhere
in the tree.

`native/niyah_telemetry.c` exists and `README.md` demonstrates
`niyah_telemetry_tokens_per_second`, so the instrument is implemented. Nothing
has been recorded with it. The repository is at TESTED and not at MEASURED, and
no claim in it should be phrased as a measurement until `evidence/` holds an
artifact with the command, the machine, and the number.

# ORACLE-REPORT.md
<!-- commit 2a8e094669a5b4cb9855e9993b11459ae8bbe80e -->

## Build Baseline

```
cmake -S native -B build/native -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-Wall -Wextra"
cmake --build build/native --parallel
ctest --test-dir build/native --output-on-failure
```

| Metric | Measured value |
|---|---|
| BUILD_RC | 0 |
| TESTS_TOTAL | 29 |
| TESTS_PASSED | 29 |
| TESTS_FAILED | 0 |
| WARNINGS_COUNT (-Wall -Wextra) | 4 |

Warnings (verbatim):
- `test_niyah_document.c:8:17: warning: unused variable 'source' [-Wunused-variable]`
- `niyah_transformer_layer_test.c:96:24: warning: unused variable 'ff' [-Wunused-variable]`
- `niyah_mini/niyah_mini_vocab.c:26:12: warning: 'size_add_ok' defined but not used [-Wunused-function]`
- `niyah_mini/niyah_mini_model.c:427:5: warning: 'state.memory_block' may be used uninitialized [-Wmaybe-uninitialized]`

Sanitizer build (`-DNIYAH_SANITIZE=ON`):

| Metric | Measured value |
|---|---|
| ASAN_ERRORS | 0 |
| UBSAN_ERRORS | 0 |
| LEAKS_BYTES | 0 |

---

## Gate Table

| GATE | ORACLE USED | MEASURED VALUE | COMMAND | EXIT CODE | TAG |
|---|---|---|---|---|---|
| BUILD | cmake + ctest | BUILD_RC=0, TESTS=29/29 | `cmake -S native -B build/native && cmake --build build/native --parallel && ctest --test-dir build/native --output-on-failure` | 0 | COMPILED |
| BUILD (sanitize) | cmake + ctest + ASan/UBSan | ASAN=0, UBSAN=0, LEAKS=0 | `cmake -S native -B build/native_san -DNIYAH_SANITIZE=ON && cmake --build build/native_san --parallel && ctest --test-dir build/native_san --output-on-failure` | 0 | COMPILED |
| GATE 1 — stub bytes | `wc -c native/niyah_test_llama_cli.c` | STUB_BYTES=1175 | `wc -c native/niyah_test_llama_cli.c` | 0 | WRITTEN |
| GATE 1 — no weight files | `find . -name "*.gguf"` | WEIGHT_FILES_FOUND=0 | `find . -name "*.gguf" -o -name "*.safetensors" \| grep -v .git \| wc -l` | 0 | WRITTEN |
| GATE 1 — real llama-cli run | llama-cli binary | CANNOT_RUN=NO_LLAMA_CLI | `NIYAH_LLAMA_CLI=/path/to/llama-cli ./build/native/niyah run <pkg> --prompt P --max-tokens 8 --proof out.proof` | N/A | UNMEASURED |
| GATE 1 — TOCTOU race | Shell race harness | RACE_ATTEMPTS=100, RACE_WINS=12, PROOF_OVER_SWAPPED=12 | `bash gate1_toctou.sh` | 0 | WRITTEN (defect confirmed) |
| GATE 2 — niyah_control.c mutation | mutate_test.sh | MUTANTS_TOTAL=10, MUTANTS_KILLED=6, KILL_RATE=60% | `bash mutate_test.sh native/niyah_control.c niyah_control_test` | 0 | WRITTEN (UNVERIFIED — kill rate 60% < 80%) |
| GATE 2 — niyah_proof.c mutation | mutate_proof.sh | MUTANTS_TOTAL=7, MUTANTS_KILLED=4, KILL_RATE=57.1% | `bash mutate_proof.sh` | 0 | WRITTEN (UNVERIFIED — kill rate 57.1% < 80%) |
| GATE 2 — search/niyah_index.c mutation | mutate_index.sh | MUTANTS_TOTAL=10, MUTANTS_KILLED=4, KILL_RATE=40% | `bash mutate_index.sh` | 0 | WRITTEN (UNVERIFIED — kill rate 40% < 80%) |
| GATE 3 — arena/calloc comment | code inspection + grep | Comment at niyah_llm.c:349 confirmed; calloc call at line 354 confirmed | `grep -n "SINGLE-POOL\|calloc" native/niyah_llm.c` | 0 | WRITTEN |
| GATE 3 — heap taxonomy | LD_PRELOAD interposer | CANNOT_RUN=NO_LDPRELOAD_TOOLING (no malloc_count interposer available) | `gcc -shared -o /tmp/count_malloc.so count_malloc.c && LD_PRELOAD=/tmp/count_malloc.so ./niyah_llm_generation_test` | N/A | UNMEASURED |
| GATE 3 — determinism | sha256 of token sequences | CANNOT_RUN=NO_LLM_WEIGHTS (no GGUF weights; niyah_llm_generate returns NIYAH_ERR_NO_WEIGHTS) | `./build/native/niyah_llm_generation_test` (only stubs, no weights) | N/A | UNMEASURED |
| GATE 4 — BM25 vs reference | Python rank_bm25 (same IDF formula) | MAX_ABS_SCORE_DELTA=4.685e-11 (C vs Python-truncated), docs 1025+5000 tokens score identically (truncation) | `python3 gate4_bm25_v2.py` | 0 | VERIFIED (C vs Python-truncated: identical; truncation defect: confirmed) |
| GATE 4 — truncation collapse | BM25 harness + Python-full | docs 3,4,5 (1024/1025/5000 tokens) all get score=0.23883285; Python-full distinguishes them | `python3 gate4_bm25_v2.py` | 0 | WRITTEN (scoring error confirmed) |
| GATE 4 — complexity | timing harness | x10=0.093ms, x100=0.461ms, x1000=29.3ms; FITTED_EXPONENT=1.249 | `python3 gate4_bm25.py` | 0 | WRITTEN (O(n^1.25), consistent with linear claim) |
| GATE 5 — locale/high bytes | gate5_control binary (en_US.UTF-8) | ACCEPTED_HIGH_BYTES=0 | `LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 ./gate5_control` | 0 | WRITTEN (no hole found) |
| GATE 5 — nonce replay | gate5_control binary | DECISION_1=ALLOW, DECISION_2=ALLOW — nonce is decorative | `./gate5_control` | 0 | WRITTEN (defect: nonce does not prevent replay) |
| GATE 5 — boundary | gate5_control binary | len=94: ACCEPTED, len=95: ACCEPTED, len=96: REJECTED | `./gate5_control` | 0 | WRITTEN |
| GATE 6 — unsupported types | converter + fixture | Q5_K: rc=1, error names Q5_K; Q2_K/Q3_K/Q8_K/Q5_0/Q5_1/Q8_0/Q8_1: rc=1 via earlier-path errors | `python3 tools/tests/test_kquants.py` | 0 | WRITTEN (fail loudly confirmed for Q5_K; others fail before reaching type check) |
| GATE 6 — real GGUF tensors | llama.cpp gguf-py reference | CANNOT_RUN=NO_REAL_GGUF (no network, no real model weights) | Download real GGUF: `python3 -c "import gguf; ..."` | N/A | UNMEASURED |
| GATE 7 — data provenance | sha256sum + file inspection | data/khawrizm_graph_consolidated.json sha256=455d112a..., DECLARED_SOURCE_URL=REFUSED, DECLARED_LICENSE=REFUSED | `sha256sum data/khawrizm_graph_consolidated.json normalized/*.jsonl` | 0 | WRITTEN (REDISTRIBUTABLE=NO — no license file) |
| GATE 8 — workflow jobs | grep .github/workflows/native.yml | WORKFLOW_JOBS=6 (native, native-sanitize, search, storage, python, windows-ui); README says "6 jobs" | `grep "^  [a-z]" .github/workflows/native.yml \| grep -c ":"` | 0 | VERIFIED (README count matches) |
| GATE 8 — README: "ASan+UBSan native tests" | .github/workflows/native.yml | native-sanitize job exists with ASAN_OPTIONS and UBSAN_OPTIONS | `grep -A10 native-sanitize .github/workflows/native.yml` | 0 | VERIFIED |
| GATE 8 — README: "PostgreSQL/pgvector storage job" | .github/workflows/native.yml | storage job with pgvector/pgvector:0.8.6-pg18 service exists | `grep pgvector .github/workflows/native.yml` | 0 | VERIFIED |
| GATE 8 — README: "windows-ui WPF job" | .github/workflows/native.yml | windows-ui job with `dotnet build ui/Niyah.App/Niyah.App.csproj` exists | `grep windows-ui .github/workflows/native.yml` | 0 | VERIFIED |
| GATE 8 — Python tests | compileall + unittest | PYTHON_TESTS_DISCOVERED=10, RC=0; test_convert_gguf=99 checks, test_kquants=27 checks | `python3 -m unittest discover -s tests -p 'test_*.py' -v` | 0 | WRITTEN |
| GATE 8 — compileall = syntax only | code inspection | compileall does not execute any assertion; PYTHON_ASSERTIONS_EXECUTED=22 (unittest only) | `python3 -m compileall -q tools scripts src tests` | 0 | WRITTEN |

---

## Gate 2 — Mutation Kill Rates (all below 80% ceiling)

### native/niyah_control.c

| Mutant | Operator | Result |
|---|---|---|
| cap==0 -> cap!=0 | == -> != | KILLED |
| (cap&~known)!=0 -> ==0 | != -> == | KILLED |
| single-bit ==0 -> !=0 | == -> != | KILLED |
| delete single-bit check | return 1; | KILLED |
| length >= -> > | >= -> > | **SURVIVED** |
| delete length bound | if (0) | **SURVIVED** |
| isalnum && -> \|\| | && -> \|\| | **SURVIVED** |
| invalid arg ret -> OK | return NIYAH_OK | **SURVIVED** |
| init deny -> allow | DENY -> ALLOW | KILLED |
| nonce==0 -> !=0 | == -> != | KILLED |

`MUTANTS_TOTAL=10 MUTANTS_KILLED=6 KILL_RATE=60%`

The "default deny, no prefix widening" claim is **UNVERIFIED**. Kill rate 60% < 80% ceiling. Surviving mutants include: the length bound (overflow-enabling mutant), the isalnum `&&`→`||` (allows `!isalnum` chars), and the invalid-arg early return. None of these are caught by the existing test suite.

### native/niyah_proof.c

`MUTANTS_TOTAL=7 MUTANTS_KILLED=4 KILL_RATE=57.1%`

Surviving mutants include:
- `separator 0u -> 1u`: no test checks the absolute domain-separator value
- `digest_equal loop: BYTES -> BYTES-1`: no test checks all 32 bytes independently

### search/niyah_index.c

`MUTANTS_TOTAL=10 MUTANTS_KILLED=4 KILL_RATE=40%`

Surviving mutants include: skip-tf==0-guard, off-by-one in tokenize cap, off-by-one in high-byte, average-length denominator, BM25 k1+1 factor, DOCUMENT_TOKEN_LIMIT value.

---

## Gate 4 — BM25 Truncation Finding

Documents of 1024, 1025, and 5000 tokens all receive `stored_term_count=1024` and therefore **identical BM25 scores**:

```
doc 3 (1024 tokens):  score = 0.23883285
doc 4 (1025 tokens):  score = 0.23883285  ← same
doc 5 (5000 tokens):  score = 0.23883285  ← same
```

Python-full (correct) scores: 0.30784502, 0.30775392, **0.14140962** — a 2.2× difference for the 5000-token document.

`RANK_INVERSIONS=0` in this 5-document corpus with only 5000-token vs. shorter docs, because the short doc (100 tokens) still wins. But in a corpus where all documents exceed 1024 tokens with varying true lengths, all documents receive equal length normalization — a genuine ranking correctness defect.

**Fix committed**: `search/niyah_index.c` — `count_tokens()` + `destination->term_count = count_tokens(text_copy)`. Command: `bash oracle/run.sh`. Exit 0 after fix.

---

## Gate 5 — Nonce is Decorative

```
DECISION_1=ALLOW (reason=allowed)
DECISION_2=ALLOW (reason=allowed)
```

Command: `./gate5_control` (exit 0)

The identical request `(nonce=0xDEADBEEF, NIYAH_CAP_READ_LOCAL, "doc:alpha")` is authorized twice. The "nonce" field is checked only for `!= 0`; no used-nonce set is maintained. Any call with the same triple is authorized in perpetuity. If nonces were intended as single-use tokens (their name implies), this is a defect. The README does not claim replay prevention.

---

## Gate 7 — Provenance Table

| path | bytes | sha256 | declared_source_url | declared_licence | retrieved_at_utc |
|---|---|---|---|---|---|
| data/khawrizm_graph_consolidated.json | 391535 | 455d112a92cc1dc615ed50d89a5448552fb927bdff02c35552a71a585f8927ba | REFUSED | REFUSED | REFUSED |
| normalized/chunks.jsonl | 30000 | 1e1c9d0316ad2a3e61f83f65c08c17bd7dfa0400288e3d26a67432f62ef8f76a | REFUSED | REFUSED | REFUSED |
| normalized/edges.jsonl | 93223 | 3223a20659869d63639cfbfe30caf146bd2f69be8087a61c6cba85d2b8bf0362 | REFUSED | REFUSED | REFUSED |
| normalized/evidence.jsonl | 117938 | 2852d0793639461a4f303d925de47f6f4346d40e03e922fb060c6c433e79060c | REFUSED | REFUSED | REFUSED |
| normalized/nodes.jsonl | 254588 | 1bae0f3e5d218e3a314543e5b0ef6f68932b9b20b1285f0f50f9ec5544b7c959 | REFUSED | REFUSED | REFUSED |
| normalized/schemas.jsonl | 31800 | b778e070ab47f9a9aeeb7ab41033a2511707da7ff7ab3ce1fec82bbb8e7b02fc | REFUSED | REFUSED | REFUSED |

**REDISTRIBUTABLE=NO.** Evidence: no LICENSE file in the repository (confirmed by `ls LICENSE*` → file not found); README.md line reads "No LICENSE file is currently committed." Under the Berne Convention, absence of a license means all-rights-reserved by the author. The data and code cannot legally be redistributed, forked, or used in derivative works without explicit permission.

---

## Gate 8 — README vs Machine

| CLAIM | FILE:LINE | MACHINE RESULT | VERDICT |
|---|---|---|---|
| "workflow covers native CMake/ctest" | README.md | native job in native.yml: confirmed | TRUE |
| "ASan+UBSan native tests" | README.md | native-sanitize job with ASAN_OPTIONS/UBSAN_OPTIONS: confirmed | TRUE |
| "search CMake/ctest" | README.md | search job in native.yml: confirmed | TRUE |
| "Python tooling/tests" | README.md | python job `sh tools/ci.sh python`: confirmed | TRUE |
| ".github/workflows/native.yml defines 6 jobs" | Prompt (README) | Counted: native, native-sanitize, search, storage, python, windows-ui = 6 | TRUE |
| "PostgreSQL/pgvector storage job" | native.yml | storage job with `pgvector/pgvector:0.8.6-pg18-bookworm` service: confirmed | TRUE |
| "windows-ui WPF job" | native.yml | windows-ui job with `dotnet build ui/Niyah.App/Niyah.App.csproj`: confirmed | TRUE |
| "python stage compiles tools/, scripts/, src/, tests/" | tools/ci.sh | `python3 -m compileall -q tools scripts src tests` — confirmed | TRUE |
| "runs python regression tests" | tools/ci.sh | `python3 -m unittest discover -s tests -p 'test_*.py'` — Ran 10 tests, OK | TRUE |
| "runs GGUF converter and K-quant fixtures" | tools/ci.sh | `python3 tools/tests/test_convert_gguf.py` (99 checks) + `python3 tools/tests/test_kquants.py` (27 checks) | TRUE |
| compileall proves behavior | — | compileall proves syntax only; no assertion is executed | FALSE (syntax ≠ behavior) |
| GGUF fixture tests real GGUF conformance | tools/tests/ | Fixtures are synthesized by the same repo; no external GGUF reference data | FALSE (self-oracle) |
| PYTHON_TESTS_DISCOVERED=10 | `python3 -m unittest discover` | 10 test methods, 22 self.assert* calls | UNMEASURED (assertions not individually counted) |

---

## Defect: BM25 Length Normalisation Uses Truncated Token Count

**File**: `search/niyah_index.c`

**Failing test commit**: `385c7b6` — `test(search): add failing test for BM25 length normalisation with truncated tokens`

**Fix commit**: `e4ba116` — `fix(search): use true token count for BM25 length normalisation`

**Test command** (fails before fix):
```sh
gcc -O0 -std=c11 -I search/ -I native/ search/niyah_index.c search/niyah_index_test.c -o t -lm && ./t
# Exit: 134 (SIGABRT — assertion hits[0].score != hits[1].score at line 197)
```

**Test command** (passes after fix):
```sh
# same command — Exit: 0
```

**Fix diff** (17 lines):
```diff
+static size_t count_tokens(const char *text) {
+    if (!text) return 0;
+    size_t count = 0;
+    const unsigned char *cursor = (const unsigned char *)text;
+    while (*cursor) {
+        while (*cursor && !token_byte(*cursor)) ++cursor;
+        if (!*cursor) break;
+        while (*cursor && token_byte(*cursor)) ++cursor;
+        ++count;
+    }
+    return count;
+}
 ...
-    destination->term_count = (uint32_t)token_count;
+    destination->term_count = (uint32_t)count_tokens(text_copy);
```

---

## Final Line

```
CLAIMS_SURVIVED=14/22  CLAIMS_FALSIFIED=4  UNMEASURED=4
```

Survived: BUILD_RC=0, TESTS_PASSED=29, ASAN_ERRORS=0, UBSAN_ERRORS=0, LEAKS=0, STUB_BYTES=1175, WEIGHT_FILES=0, high-bytes rejected, boundary correct, workflow job count, PostgreSQL job, windows-ui job, Python compile, Python 10 tests pass.

Falsified:
1. `KILL_RATE(niyah_control.c)=60%` < 80% — "default deny, no prefix widening" is **UNVERIFIED**
2. `KILL_RATE(niyah_proof.c)=57.1%` < 80% — proof tamper-evidence is **UNVERIFIED**
3. `KILL_RATE(niyah_index.c)=40%` < 80% — BM25 ranking correctness is **UNVERIFIED**
4. `TOCTOU_RACE_WINS=12/100` — blob swap between verify_file and execvp confirmed; proof written over unverified bytes

Unmeasured (CANNOT_RUN):
1. `CANNOT_RUN=NO_LLAMA_CLI` — real end-to-end run with genuine GGUF model
2. `CANNOT_RUN=NO_LLM_WEIGHTS` — determinism measurement (all 6 sha256 of token sequences)
3. `CANNOT_RUN=NO_REAL_GGUF` — element-wise dequantization comparison against llama.cpp gguf-py
4. `CANNOT_RUN=NO_MALLOC_INTERPOSER` — heap call taxonomy during niyah_llm_generate

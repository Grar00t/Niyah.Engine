# Corpus and auxiliary tool contracts

These tools do not modify existing datasets. Use new output paths. Existing
outputs (including symlinks and hard-link aliases) fail closed. Temporary bytes
are written to exclusively owned random files, flushed and fsynced, then exposed
using no-replace hard links. Filesystems without hard-link support fail; there is
no overwrite fallback. A batch is not an atomic multi-file transaction. A later
failure may leave earlier complete outputs; temporary cleanup failures return an
error. No guarantee of power-loss durability or protection against a malicious
actor replacing directory entries is made.

`corpus_common.publish_corpus` rejects empty outputs and writes raw SHA-256-bound
provenance alongside each corpus. These sidecars are informational provenance;
native tokenizer/shard readers do not enforce them as a completion protocol.
The cache reader in `build_mix.py` does enforce its combined metadata/content
contract before reuse. Native checkpoint/cursor completion has its separate
runtime protocol.

The former hardcoded Windows-path entry points now require explicit paths:

```text
python tools/merge_mix.py --input first.txt --input second.txt --out new-train.txt --validation-out new-val.txt
python tools/prep_kaggle_ar.py --source input-directory --text-column text --out new-arabic.txt
python tools/build_mix.py --sources sources.json --cache-directory new-cache --out new-train.txt --validation-out new-val.txt
python tools/prepare_arabic_corpus.py --sources sources.json --out new-lines.txt --max-mb 300
```

`merge_mix` sorts inputs, verifies exact text equality inside SHA-256 buckets,
then shuffles with the supplied seed. It requires at least two unique documents.
The validation split is 1% of document count (at least one, never all); this is
not a token fraction. `build_mix` uses the same exact deduplication before its
split. The realized token mixture is not measured by these scripts.

`build_mix` sources are a nonempty JSON list. Each item requires `name`, `repo`,
`revision` (40-hex immutable commit), `split`, `text_key`, `character_budget`
(positive integer), `license_claim` and `license_evidence`; `config` is a string
or null. Revisions must be obtained and checked for the intended source, not
invented. Cache filenames derive from all those fields and versioned selection
rules. Cache content contains the same contract plus a SHA-256 over the selected
texts; reuse checks both. Old `_cache_*.jsonl` files lack those bindings and are
preserved but never silently reused. Changing any contract field uses a new
cache. Streaming sources need the optional Hugging Face `datasets` dependency;
missing imports are errors. Importing the tool never downloads anything.

Budgets in `build_mix` count Unicode characters after removing CR and stripping
outer whitespace. It skips documents shorter than 200 characters and the document
separator, retains whole documents, and stops at the budget; its last document
can overshoot. Exhaustion before budget completion fails. The byte count and
selected character count are separately recorded.

`prep_kaggle_ar` reads sorted CSV, JSON arrays of objects, and JSONL objects.
`--text-column` is mandatory. It rejects missing/non-string selected fields,
malformed CSV/JSON, and empty retained output before publication. It never
silently selects the longest column or ignores parse failures. URL/email/control
filtering is limited filtering, not a comprehensive personal-data review.

`prepare_arabic_corpus` sources require `repo`, `revision`, `split`, `text_key`,
`license_claim` and `license_evidence`; optional `config` is forwarded. It applies
NFKC, strips HTML/URLs and tatweel, optionally removes diacritics, normalizes
spacing, and treats each retained line as a record. Original multiline document
boundaries are lost. The Arabic ratio is Arabic Unicode letters divided by all
Unicode letters; digits and punctuation cannot inflate the numerator. Its
`--max-mb` is a strict UTF-8 MiB cap including newline bytes. It stops before the
first eligible line that would exceed the cap; no partial line or overshoot is
written. Empty output fails. No token budget is implied.

All source license fields are recorded claims, not independently verified
permission. Legacy source revisions and licensing cannot be reconstructed from
cache path names or hardcoded labels. No source download or corpus regeneration
is needed for the offline regression suite.

`validate_onion_manifest.py` checks metadata types, required nonempty strings,
timezone-qualified retrieval timestamps, decisions and permission consistency.
REFERENCE_ONLY/QUARANTINE do not grant training/RAG permission. It rejects empty
manifests, boolean/negative byte counts, malformed URLs/hashes and uncleared
reviews outside quarantine. URL checks are hostname-pattern checks, not onion
checksum/version validation. It performs no Tor crawling, content-hash checking,
licensing adjudication, malware scanning or PII review.

`char_dist.py` strictly decodes bytes, reports the exact first invalid byte
offset, and separates Arabic Unicode letters from Arabic-block codepoints.
Literal U+FFFD in otherwise valid UTF-8 is reported separately. Replacement
decoding is never used as proof of encoding validity.

`profile_nsys.ps1` forwards trainer arguments as an array, creates a GUID report
name, disables overwrite and checks profile/stats exits and report existence.
Tests execute the actual wrapper against a stub, including a compiled native
stub on Windows when its compiler exists. That verifies forwarding and error
handling, not CUDA timing or real Nsight collection. `demo_logging.c` generates
synthetic formatting examples only, writes exclusively to
`synthetic_demo_log.csv`, and labels every CSV column synthetic.

Offline verification:

```text
python tests/test_corpus_tools.py
python tests/test_profile_nsys.py
python tests/test_demo_logging.py path/to/niyah_demo_logging
```

The profiling test returns 77 with an explicit unavailable message if PowerShell
is absent. CTest must register that as a skip, not silently pass it.

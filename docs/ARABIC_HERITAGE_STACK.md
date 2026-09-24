# Arabic Heritage Stack — License and Training Audit

This document separates **runtime/reference use** from **training-data use** for Arabic morphology and language assets considered around Niyah.Engine.

The rule is intentionally conservative: a tool being open source does **not** automatically make every bundled database, dictionary, model, or corpus suitable for incorporation into model-training data.

## Decision labels

| Label | Meaning |
|---|---|
| `SAFE_TO_USE_CODE` | Code license is clear enough for normal use subject to its stated terms. This does not automatically approve bundled data for training. |
| `SAFE_WITH_ATTRIBUTION` | Data license permits reuse with attribution; provenance and license notices must remain in the dataset manifest. |
| `REVIEW_BEFORE_TRAIN` | Runtime/reference use may be possible, but direct ingestion into training data needs a specific license/provenance decision first. |
| `DO_NOT_TRAIN` | Do not ingest into a Niyah training corpus unless the missing or restricted license condition is resolved. |
| `QUARANTINE` | Exact origin/license of the local artifact is not yet established. Hash and identify it before any use. |

## Current verified matrix

| Asset / family | Upstream license evidence | Niyah decision | Notes |
|---|---|---|---|
| CAMeL Tools source code | MIT | `SAFE_TO_USE_CODE` | The Python toolkit itself is MIT-licensed. Its separately downloaded data packages have independent licenses. |
| CAMeL `calima-glf-01` morphology DB | CC BY 4.0 | `SAFE_WITH_ATTRIBUTION` | Gulf Arabic morphology data. Keep source, version, license, and attribution in manifests. |
| CAMeL `calima-msa-r13` morphology DB | GPL v2 | `REVIEW_BEFORE_TRAIN` | May be used as an external analyzer under GPL obligations; do not silently merge its database contents into model-training text. |
| CAMeL `calima-egy-r13` morphology DB | GPL v2 | `REVIEW_BEFORE_TRAIN` | Same boundary as the MSA r13 database. |
| CAMeL `calima-msa-s31` | Requires a licensed copy of LDC SAMA 3.1 | `DO_NOT_TRAIN` | Do not use unless the required SAMA license is independently established for the intended use. |
| CAMeL MLE disambiguation r13 packages | GPL v2 | `REVIEW_BEFORE_TRAIN` | Treat as separately licensed model/data artifacts, not as MIT merely because CAMeL Tools code is MIT. |
| Arramooz / Arramooz SQLite | GPL | `REVIEW_BEFORE_TRAIN` | Open-source morphology dictionary, but GPL data should stay external/reference-only until a deliberate training-distribution policy is made. |
| `alsaydi/sarf` current GitHub repository | Repository presents MIT license | `REVIEW_BEFORE_TRAIN` | The repository documents that it preserves historical Sarf code originally found elsewhere. Verify the provenance/license chain of the exact local copy before using its linguistic tables as training material. |
| Unknown local `ArabicDictionary.sql` | Not established | `QUARANTINE` | File name alone is not provenance. Do not train from it until exact upstream source and license are identified. |
| Unknown local verb/root `.dic` files | Not established | `QUARANTINE` | Hash, identify source, then classify. |

## Important CAMeL distinction

The CAMeL Tools **code** is MIT, while the morphology and model packages use different licenses. The official CAMeL documentation identifies, among others:

```text
calima-msa-r13     GPL v2
calima-egy-r13     GPL v2
calima-glf-01      CC BY 4.0
calima-msa-s31     requires licensed LDC SAMA 3.1
```

Therefore Niyah must classify the exact data artifact, not infer permission from the `camel_tools` repository license.

Known public package identities useful for local hash matching:

```text
CAMeL morphology-db-msa-r13 / morphology.db
sha256 = 195bc25a333237a2126470da888d7936b59ed3729f9210e0a4194ba43497dd70
license = GPL v2

CAMeL morphology-db-glf-01 / morphology.db
sha256 = 0b88b55d09eda8edc2f0009cb3f46d4b2ad8176cf5dc7a17a7b65439f7aaae7d
license = CC BY 4.0
```

A local file that does not match these hashes must not be assumed to be the same package/version.

## Recommended Niyah architecture

Arabic heritage resources should remain in three explicit layers:

```text
Layer A — native Niyah model data
  only approved training text/shards
  exact source + license + SHA-256 recorded

Layer B — deterministic Arabic linguistic tooling
  morphology / roots / patterns / normalization
  may run externally without becoming LM training data

Layer C — quarantined heritage/reference assets
  unknown, restrictive, or unresolved provenance
  never included in tokenizer/model training
```

This keeps useful Arabic linguistic knowledge available without contaminating the training lineage.

## Preferred use of morphology resources

For approved resources, prefer **structured, auditable generation** rather than dumping entire databases into the language-model corpus. Example record families:

```text
root -> derived form
surface form -> root candidates
lemma -> morphological features
undiacritized -> valid diacritized forms
verb -> tense/person/number/gender inflections
pattern -> generated examples
```

Generated records should carry out-of-band metadata such as:

```json
{
  "source": "camel-calima-glf-01",
  "source_version": "0.1.0",
  "source_license": "CC BY 4.0",
  "generator": "niyah-arabic-heritage-v1",
  "derived": true,
  "source_sha256": "...",
  "record_sha256": "..."
}
```

Do not train the provenance fields themselves unless intentionally part of the model-visible text.

## Hard exclusions

Until separately approved, do not:

- copy proprietary Sakhr software, ROMs, dictionaries, corpora, or model assets into Niyah training data;
- treat emulator support for a historical system as permission to copy the original ROM/software;
- assume a repository-level license overrides a dataset/package-specific license;
- ingest SQL/SQLite/dictionary files whose source cannot be proven;
- remove or falsify source attribution to make a dataset appear first-party;
- turn GPL or restricted morphology databases into unlabeled plain-text training corpora;
- use `calima-msa-s31` without establishing the required licensed SAMA 3.1 dependency.

## Sources used for this audit

Primary upstream references checked on 2026-09-24:

- CAMeL Tools repository: https://github.com/CAMeL-Lab/camel_tools
- CAMeL morphology DB documentation: https://github.com/CAMeL-Lab/camel_tools/blob/master/docs/source/api/morphology/database.rst
- CAMeL data package catalogue: https://github.com/CAMeL-Lab/camel-tools-data/blob/main/catalogue-1.3.json
- Arramooz repository: https://github.com/linuxscout/arramooz
- Sarf repository: https://github.com/alsaydi/sarf

This document is an engineering/provenance policy, not legal advice. If a dataset will be redistributed commercially or used to publish model weights, the exact license obligations should be reviewed for that distribution path.

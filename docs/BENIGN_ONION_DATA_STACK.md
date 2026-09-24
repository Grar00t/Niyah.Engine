# Benign Onion Data Stack

This document defines a conservative ingestion boundary for useful `.onion` content that is unrelated to illicit activity and may have legitimate research, cultural, linguistic, educational, archival, or technical value.

The rule is simple: **the transport is not the trust decision**. A `.onion` address is only a publication route. Content is accepted or rejected based on source identity, provenance, licensing, safety review, privacy impact, and reproducibility.

## Intended uses

Potentially useful categories include:

- public-domain books and historical archives;
- academic papers and research mirrors;
- journalism and censorship-resistant publishing;
- language, dialect, cultural, and historical material;
- open-source software documentation and technical references;
- public-interest archives;
- long-form discussion forums when redistribution/training rights are clear;
- privacy and digital-rights material;
- mirrors of lawful public web resources.

## Hard exclusions

The ingestion pipeline must reject or quarantine material containing or primarily serving:

- stolen credentials, authentication secrets, or access tokens;
- doxxing, private personal records, or non-consensual personal data;
- leaked private databases;
- malware payloads, exploit kits, botnet material, or executable binaries of unknown origin;
- illicit marketplace content;
- sexual abuse material or other illegal media;
- instructions whose primary purpose is facilitating serious wrongdoing;
- content with unclear ownership or redistribution status when training rights cannot be established.

## Training boundary

Useful onion content is **not automatically model-training data**.

A source may be:

- `REFERENCE_ONLY` — readable/retrievable, never copied into LM training;
- `RAG_ALLOWED` — may be indexed externally for retrieval with provenance;
- `TRAIN_ALLOWED` — may be converted into LM text only after explicit license/provenance review;
- `QUARANTINE` — do not ingest until resolved.

No crawler result may enter the tokenizer or model corpus solely because it is publicly reachable.

## Required manifest fields

Every admitted source or captured document should carry out-of-band metadata:

```text
source_url
source_name
retrieved_at
content_type
language
category
license
license_evidence
sha256
bytes
pii_review
malware_review
redistribution_allowed
training_allowed
rag_allowed
reviewer
notes
```

The metadata is provenance. It should not be injected into model-visible training text unless intentionally part of the task.

## Recommended architecture

```text
Tor transport / onion source
        |
        v
source allowlist
        |
        v
fetch text only
        |
        +--> binary / attachment? ----> QUARANTINE
        |
        v
normalize + hash
        |
        v
license / provenance / privacy review
        |
        +--> REFERENCE_ONLY
        +--> RAG_ALLOWED
        +--> TRAIN_ALLOWED
        `--> QUARANTINE
```

## Collection discipline

Prefer an allowlist over broad crawling.

Collectors should:

1. fetch text/HTML only by default;
2. disable script execution;
3. avoid automatic attachment or binary downloads;
4. record the exact `.onion` URL and retrieval timestamp;
5. hash raw captures before normalization;
6. preserve a content-type boundary between raw capture, normalized text, and derived training text;
7. deduplicate by content hash and normalized-text hash;
8. retain takedown/removal capability for every source lineage.

## Identity and source naming

The final model should not inherit a site or network identity merely because data came through Tor. Source names remain provenance metadata. They must not become persona, special-token identity, or system-prompt identity.

## Niyah.Engine boundary

Niyah.Engine remains the model/runtime layer. Onion collection, filtering, licensing review, and external retrieval belong outside the native model core.

```text
Niyah.Engine
  ├─ tokenizer
  ├─ training shards
  ├─ checkpoint/cursor
  └─ inference/evaluation

External data plane
  ├─ onion source allowlist
  ├─ text collector
  ├─ provenance manifest
  ├─ safety/privacy filter
  ├─ license review
  └─ optional RAG index
```

This separation prevents network-origin metadata, dynamic site state, and unreviewed external content from silently becoming part of model semantics.

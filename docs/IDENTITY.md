# Project Identity and Artifact Mark

<p align="center">
  <img src="assets/niyah-project-mark.jpg" alt="Niyah.Engine project mark" width="300" />
</p>

Niyah.Engine uses a distinct project mark as a **human-recognizable identity layer** for documentation, releases, model cards, evidence receipts, and other official project artifacts.

The project owner describes the idea behind the mark through a traditional tribal practice: a recognizable *wasm* (وسم) placed on camels makes belonging and ownership immediately recognizable to the people who know the mark. Niyah.Engine carries that same design principle into software: **an artifact should be recognizable by sight, and verifiable by evidence.**

The visual mark is therefore cultural identity and project provenance. It is not a substitute for cryptographic verification.

## Identity rule

Niyah.Engine separates two layers of identity:

| Layer | Purpose | Authority |
|---|---|---|
| **Project mark** | Fast human recognition | Visual / documentary |
| **Cryptographic receipt** | Exact artifact verification | SHA-256, commit identity, format metadata |

A mark can be copied. A hash binds verification to exact bytes. Official project documentation should use both where practical.

## Where the mark belongs

The mark may be used on:

- repository documentation and diagrams;
- release notes and release packages;
- model cards and evaluation reports;
- tokenizer and dataset receipts;
- checkpoint manifests;
- executable/build receipts;
- container and package metadata;
- project web/UI surfaces;
- evidence bundles.

The preferred pattern is:

```text
visual identity
    +
artifact type
    +
repository commit
    +
SHA-256 identity
    +
reproducible verification command
```

## Where the mark does not belong

The project mark must not be injected into training text merely to make a model repeat the identity. It must not silently alter tokenizer vocabulary, model weights, dataset semantics, evaluation samples, or runtime behavior.

If a future Niyah file format carries project identity metadata, that metadata should be explicit, versioned, and outside the learned token stream.

For example, a future receipt or sidecar manifest could contain:

```text
project       = Niyah.Engine
project_mark  = niyah-mark-v1
artifact      = tokenizer
sha256        = <64 hex characters>
repo_commit   = <40 hex characters>
```

This example defines a documentation convention only. It does **not** claim that current binary formats already contain these fields.

## Authenticity boundary

The project mark by itself does **not** prove that an artifact is authentic. Verification should rely on concrete evidence such as:

1. the expected repository or release origin;
2. the exact commit used to build or produce the artifact;
3. SHA-256 of the artifact bytes;
4. format/version metadata where available;
5. reproducible build, test, evaluation, or training receipts appropriate to that artifact.

This keeps the cultural analogy honest: the mark tells a human **what this is meant to belong to**; the receipt tells a verifier **whether these exact bytes are the artifact that was recorded**.

## Ownership and licensing boundary

The mark identifies the Niyah.Engine project and its own work. It does not create ownership over third-party datasets, code, research, or other material incorporated under separate licenses or provenance records. Third-party provenance and licensing remain independent requirements.

Likewise, displaying the project mark is not itself a software license grant. Repository licensing remains governed by the project `LICENSE` and any artifact-specific terms.

## Design handling

For official use:

- preserve the geometry and proportions of the mark;
- prefer monochrome white-on-black or black-on-white treatment;
- do not stretch, skew, or decorate the core geometry;
- maintain clear space around the mark;
- pair the mark with `Niyah.Engine` when context is ambiguous;
- keep cryptographic identities in machine-readable text rather than baking hashes into decorative artwork.

The repository copy at [`docs/assets/niyah-project-mark.jpg`](assets/niyah-project-mark.jpg) is the current reference image for documentation use.

## Principle

> **Recognizable by people. Verifiable by bytes.**

That is the software equivalent of the idea behind the mark: identity should be obvious, while authenticity should remain independently provable.

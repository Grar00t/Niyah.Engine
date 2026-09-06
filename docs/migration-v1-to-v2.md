# v1 -> v2 Migration

## Canonical authority

`schema/canonical_knowledge_graph_v2.1.0.json` is the current portable schema contract.
`knowledge/canonical_knowledge_v2.json` is the committed canonical graph used by the graph utilities.
PostgreSQL is an indexed projection; derived indexes and embeddings are not evidence.

## 1. Preserve source material

Keep original source records and hashes when migrating. Do not promote prompt, configuration, or generated text to asserted facts solely because it exists in a repository.

## 2. Reclassify claims

Treat unsupported claims as candidate or inferred until implementation evidence exists. Preserve source identity and content hashes in provenance records.

## 3. Generate canonical IDs

Node ID:

`n_` + SHA-256(`normalize(label)` + `|` + `normalize(type)` + `|` + `normalize(scope)`)

Edge ID:

`e_` + SHA-256(`source` + `|` + `edge_type` + `|` + `target`)

Normalization must be stable across ingestion and re-ingestion.

## 4. Import evidence first

Insert evidence before nodes and edges. Bind asserted claims to existing evidence records and reject unresolved provenance.

## 5. Node import

Import `id`, `type`, `label`, `status`, `scope`, description, and canonical properties. Keep variable payloads in JSON/JSONB rather than inventing relational columns for every source field.

## 6. Edge import

Newly inferred edges start as `candidate`. Promotion requires valid endpoints, an allowed relation type, required evidence, and relation-specific validation.

Evidence is mandatory for:

- `contradicts`
- `conflicts_with`
- `supersedes`
- `causes`
- `mitigates`

## 7. Retrieval projection

Embeddings, full-text vectors, and indexes are derived runtime state. One embedding model/dimension pair is allowed per embedding space. Changing either requires a separate space or complete reindexing.

Configure that binding with:

```bash
python scripts/configure_embedding_space.py \
  --model-name <model-name-or-local-path> \
  --dimensions <model-dimension>
```

This command records the embedding-space contract; it does not train an embedding model.

## 8. PostgreSQL deployment

```bash
docker compose up -d
```

Apply the repository SQL migrations in their documented order for the selected storage path.

## 9. Verification

```bash
python scripts/validate_graph.py
python scripts/audit_graph.py
python -m unittest discover -s tests -p 'test_*.py'
sh tools/ci.sh python
```

`validate_graph.py` and `audit_graph.py` default to `knowledge/canonical_knowledge_v2.json`; pass `--graph` explicitly to audit another graph.

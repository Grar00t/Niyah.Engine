# Storage

Niyah.Engine uses PostgreSQL as its runtime database.

The database may run locally on the same machine, in a local container, or on an explicitly selected remote PostgreSQL service. The storage API and schema contract do not change based on location.

## Authority

There is one database migration authority:

`db/migrations/`

No runtime code embeds schema DDL.

No secondary SQLite schema or PostgreSQL schema copy is authoritative.

## Schemas

- `niyah` — sessions, messages, sources, documents, chunks, claims, fetch records, lexical retrieval metadata.
- `skg` — canonical knowledge graph nodes, edges, and evidence relationships.
- `skg_audit` — graph audit events.
- `skg_search` — reserved for search-specific database objects.

## Runtime

The C storage backend uses libpq.

Connection selection is explicit through PostgreSQL connection parameters. Niyah.Engine does not silently select a cloud database and does not contain a hidden network database dependency.

## Retrieval

`niyah.document_chunks.search_vector` is a stored generated `tsvector`.

A GIN index supports lexical retrieval.

`niyah.search_chunks()` provides deterministic PostgreSQL full-text retrieval and stable tie-breaking by chunk id.

The SKG lexical/vector search contract remains separate from the runtime document retrieval contract.

## Migrations

Developer / host runner:

`python3 tools/apply_postgres_migrations.py --dbname "dbname=niyah"`

Container runner:

`tools/apply_postgres_migrations.sh`

Both use `public.niyah_schema_migrations` with SHA-256 drift detection.

Applied migration files are immutable.

# Niyah Knowledge Layout

- `00_registry/`: registries and tree-layout metadata.
- `10_taxonomy/`: taxonomy and curriculum seeds only; entries here are not verified lessons.
- `20_lessons/`: source-backed atomic lessons that satisfy the verification rule below.
- `canonical_knowledge_v2.json`: committed canonical knowledge graph.
- `domains.json`: domain metadata.

A lesson is verified knowledge only when its `index.json` is under `20_lessons/`, has `status = VERIFIED_ATOMIC_LESSON`, and contains non-empty `atomic_facts` with explicit `source_title` and `source_url` values.

Taxonomy records may carry source links and learning levels, but they remain taxonomy metadata until promoted into `20_lessons/` with source-backed atomic facts.

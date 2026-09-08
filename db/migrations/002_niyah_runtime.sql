-- Canonical Niyah runtime storage schema.
-- Graph/evidence authority remains in skg/skg_audit from 001_skg_core.sql.
-- Runtime documents, sessions, retrieval state, and extracted claims live here.

create schema if not exists niyah;

create table if not exists niyah.sessions (
    id text primary key,
    created_at timestamptz not null,
    language text not null,
    title text
);

create table if not exists niyah.messages (
    id bigint generated always as identity primary key,
    session_id text not null references niyah.sessions(id) on delete cascade,
    role text not null check (role in ('user','assistant','system')),
    content text not null,
    created_at timestamptz not null
);

create index if not exists idx_niyah_messages_session
    on niyah.messages(session_id, created_at);

create table if not exists niyah.sources (
    id text primary key,
    canonical_uri text not null,
    title text,
    media_type text,
    language text,
    content_sha256 text not null unique,
    retrieved_at timestamptz,
    source_kind text not null
        check (source_kind in ('local','web','user','system','unknown')),
    trust_tier smallint not null default 3
        check (trust_tier between 0 and 4)
);

create index if not exists idx_niyah_sources_hash
    on niyah.sources(content_sha256);

create table if not exists niyah.documents (
    id text primary key,
    source_id text not null references niyah.sources(id) on delete cascade,
    canonical_uri text not null,
    title text,
    media_type text not null default 'text/plain',
    language text,
    content_sha256 text not null unique,
    content_bytes bigint not null check (content_bytes >= 0),
    retrieved_at timestamptz not null,
    published_at timestamptz,
    modified_at timestamptz,
    etag text,
    last_modified text,
    parser_version text not null,
    status text not null check (status in ('pending','indexed','rejected','failed'))
);

create index if not exists idx_niyah_documents_source
    on niyah.documents(source_id);
create index if not exists idx_niyah_documents_uri
    on niyah.documents(canonical_uri);
create index if not exists idx_niyah_documents_status
    on niyah.documents(status);

create table if not exists niyah.document_chunks (
    id text primary key,
    document_id text not null references niyah.documents(id) on delete cascade,
    ordinal integer not null check (ordinal >= 0),
    start_offset bigint not null check (start_offset >= 0),
    end_offset bigint not null check (end_offset >= start_offset),
    heading text,
    text text not null,
    text_sha256 text not null unique,
    token_count integer not null check (token_count >= 0),
    search_vector tsvector generated always as (
        to_tsvector(
            'simple'::regconfig,
            coalesce(heading, '') || ' ' || text
        )
    ) stored,
    unique(document_id, ordinal)
);

create index if not exists idx_niyah_chunks_document_ordinal
    on niyah.document_chunks(document_id, ordinal);
create index if not exists idx_niyah_chunks_hash
    on niyah.document_chunks(text_sha256);
create index if not exists idx_niyah_chunks_search_vector
    on niyah.document_chunks using gin(search_vector);

create table if not exists niyah.chunk_keywords (
    chunk_id text not null references niyah.document_chunks(id) on delete cascade,
    keyword text not null,
    weight real not null default 1.0 check (weight >= 0.0),
    primary key (chunk_id, keyword)
);

create index if not exists idx_niyah_chunk_keywords_keyword
    on niyah.chunk_keywords(keyword);

create table if not exists niyah.source_fetches (
    id bigint generated always as identity primary key,
    source_id text not null references niyah.sources(id) on delete cascade,
    requested_uri text not null,
    effective_uri text,
    http_status integer,
    content_type text,
    content_length bigint,
    fetched_at timestamptz not null,
    duration_ms bigint,
    error_code text,
    robots_result text
        check (robots_result in ('allow','deny','unavailable','unknown'))
);

create index if not exists idx_niyah_source_fetches_source_time
    on niyah.source_fetches(source_id, fetched_at desc);

create table if not exists niyah.claims (
    id text primary key,
    chunk_id text not null references niyah.document_chunks(id) on delete cascade,
    claim_text text not null,
    claim_sha256 text not null unique,
    classification text not null
        check (classification in ('FACT','INFERENCE','UNCERTAIN','UNKNOWN','CONFLICTED')),
    extractor_version text not null,
    created_at timestamptz not null
);

create index if not exists idx_niyah_claims_chunk
    on niyah.claims(chunk_id);
create index if not exists idx_niyah_claims_classification
    on niyah.claims(classification);

create table if not exists niyah.claim_keywords (
    claim_id text not null references niyah.claims(id) on delete cascade,
    keyword text not null,
    weight real not null default 1.0 check (weight >= 0.0),
    primary key (claim_id, keyword)
);

create index if not exists idx_niyah_claim_keywords_keyword
    on niyah.claim_keywords(keyword);

create or replace function niyah.search_chunks(
    p_query text,
    p_limit integer default 20
)
returns table (
    chunk_id text,
    document_id text,
    heading text,
    chunk_text text,
    rank real
)
language sql
stable
as $$
    select
        c.id,
        c.document_id,
        c.heading,
        c.text,
        ts_rank_cd(
            c.search_vector,
            websearch_to_tsquery('simple'::regconfig, p_query)
        ) as rank
    from niyah.document_chunks c
    where nullif(btrim(p_query), '') is not null
      and c.search_vector @@ websearch_to_tsquery('simple'::regconfig, p_query)
    order by rank desc, c.id
    limit greatest(1, least(coalesce(p_limit, 20), 200));
$$;

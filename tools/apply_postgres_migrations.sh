#!/bin/sh
set -eu

: "${PGDATABASE:?PGDATABASE must be set}"

MIGRATION_DIR="${NIYAH_MIGRATION_DIR:-db/migrations}"

command -v psql >/dev/null 2>&1
command -v sha256sum >/dev/null 2>&1

psql \
    -X \
    -v ON_ERROR_STOP=1 <<'SQL'
CREATE TABLE IF NOT EXISTS public.niyah_schema_migrations (
    version      text PRIMARY KEY,
    filename     text NOT NULL,
    sha256       char(64) NOT NULL,
    applied_at   timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT niyah_schema_migrations_sha_ck
        CHECK (sha256 ~ '^[a-f0-9]{64}$')
);
SQL

found=0

for file in "$MIGRATION_DIR"/*.sql
do
    [ -f "$file" ] || continue
    found=1

    base="$(basename "$file")"
    version="${base%.sql}"

    case "$version" in
        ''|*[!A-Za-z0-9_]*)
            echo "INVALID_MIGRATION_NAME=$base" >&2
            exit 20
            ;;
    esac

    sha="$(
        sha256sum "$file" |
        awk '{print $1}'
    )"

    recorded="$(
        psql \
            -X \
            -v ON_ERROR_STOP=1 \
            -Atqc "
                SELECT trim(sha256)
                FROM public.niyah_schema_migrations
                WHERE version='$version';
            "
    )"

    if [ -n "$recorded" ]; then
        if [ "$recorded" != "$sha" ]; then
            echo "MIGRATION_CHECKSUM_DRIFT=$version" >&2
            echo "RECORDED=$recorded" >&2
            echo "CURRENT=$sha" >&2
            exit 30
        fi

        echo "MIGRATION_OK=$version SHA256=$sha"
        continue
    fi

    psql \
        -X \
        -v ON_ERROR_STOP=1 \
        -1 \
        -f "$file" \
        -c "
            INSERT INTO public.niyah_schema_migrations(
                version,
                filename,
                sha256
            )
            VALUES(
                '$version',
                'db/migrations/$base',
                '$sha'
            );
        "

    echo "MIGRATION_APPLIED=$version SHA256=$sha"
done

test "$found" = "1"

echo "POSTGRES_MIGRATIONS=PASS"

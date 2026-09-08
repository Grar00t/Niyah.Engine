#!/usr/bin/env python3
"""Apply canonical PostgreSQL migrations with immutable SHA-256 history.

Uses the psql client instead of a Python database package so the same runner
works on Linux/WSL and Windows wherever PostgreSQL client tools are installed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

LEDGER_SQL = r"""
CREATE TABLE IF NOT EXISTS public.niyah_schema_migrations (
    version text PRIMARY KEY,
    filename text NOT NULL,
    sha256 char(64) NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT niyah_schema_migrations_sha_ck
        CHECK (sha256 ~ '^[a-f0-9]{64}$')
);
"""


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def run_psql(db: str, sql: str, *, tuples: bool = False) -> str:
    command = ["psql", "-X", "-v", "ON_ERROR_STOP=1", "--dbname", db]
    if tuples:
        command.extend(["-A", "-t"])
    completed = subprocess.run(
        command,
        input=sql,
        text=True,
        stdout=subprocess.PIPE,
        stderr=None,
        check=True,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dbname",
        default=os.environ.get("NIYAH_DATABASE_URL", "dbname=niyah"),
        help="libpq connection string/URI (default: NIYAH_DATABASE_URL or dbname=niyah)",
    )
    parser.add_argument(
        "--migrations",
        default=str(Path(__file__).resolve().parents[1] / "db" / "migrations"),
    )
    args = parser.parse_args()

    migration_dir = Path(args.migrations).resolve()
    files = sorted(migration_dir.glob("*.sql"))
    if not files:
        print(f"NO_MIGRATIONS_FOUND={migration_dir}", file=sys.stderr)
        return 2

    run_psql(args.dbname, LEDGER_SQL)

    for path in files:
        version = path.stem
        content = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

        query = (
            "SELECT trim(sha256) FROM public.niyah_schema_migrations "
            f"WHERE version={sql_literal(version)};"
        )
        recorded = run_psql(args.dbname, query, tuples=True)

        if recorded:
            if recorded != digest:
                print(f"MIGRATION_CHECKSUM_DRIFT={version}", file=sys.stderr)
                print(f"RECORDED_SHA={recorded}", file=sys.stderr)
                print(f"CURRENT_SHA={digest}", file=sys.stderr)
                return 3
            print(f"MIGRATION_OK={version} SHA256={digest}")
            continue

        ledger_insert = (
            "INSERT INTO public.niyah_schema_migrations(version,filename,sha256) VALUES("
            f"{sql_literal(version)},"
            f"{sql_literal(path.relative_to(migration_dir.parent.parent).as_posix())},"
            f"{sql_literal(digest)}"
            ");"
        )

        transactional = "BEGIN;\n" + content.rstrip() + "\n" + ledger_insert + "\nCOMMIT;\n"
        run_psql(args.dbname, transactional)
        print(f"MIGRATION_APPLIED={version} SHA256={digest}")

    print("POSTGRES_MIGRATIONS=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

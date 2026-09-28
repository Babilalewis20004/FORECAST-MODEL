"""Restores Postgres from a pg_dump custom-format backup produced by
backup_db.py. Destructive: --clean drops existing objects before
recreating them from the dump, so anything written to the database after
that backup was taken is lost. Requires --yes to actually run.

Usage:
    python scripts/restore_db.py [DUMP_FILE] --yes
    (DUMP_FILE defaults to the most recent file in backend/backups/)"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy.engine import make_url

from app.config import settings

DEFAULT_BACKUP_DIR = Path(__file__).resolve().parents[1] / "backups"


def _latest_backup(db_name: str) -> Path | None:
    if not DEFAULT_BACKUP_DIR.exists():
        return None
    dumps = sorted(DEFAULT_BACKUP_DIR.glob(f"{db_name}_*.dump"), key=lambda p: p.stat().st_mtime, reverse=True)
    return dumps[0] if dumps else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_file", nargs="?", type=Path, help="Backup file to restore (default: newest in backend/backups/)")
    parser.add_argument("--yes", action="store_true", help="Actually run the restore (required -- this overwrites the database)")
    args = parser.parse_args()

    if shutil.which("pg_restore") is None:
        sys.exit("pg_restore not found on PATH -- see backup_db.py's error message for where it lives.")

    url = make_url(settings.database_url)
    dump_file = args.dump_file or _latest_backup(url.database)
    if dump_file is None or not dump_file.exists():
        sys.exit(f"No backup found. Pass a dump file explicitly, or run backup_db.py first (looked in {DEFAULT_BACKUP_DIR}).")

    print(f"About to restore {dump_file} into database '{url.database}' on {url.host}:{url.port}.")
    print("This DROPS and recreates existing tables first -- anything written since that backup was taken will be lost.")
    if not args.yes:
        sys.exit("Re-run with --yes to proceed.")

    env = os.environ.copy()
    if url.password:
        env["PGPASSWORD"] = url.password

    cmd = [
        "pg_restore",
        "-h", url.host or "localhost",
        "-p", str(url.port or 5432),
        "-U", url.username or "postgres",
        "-d", url.database,
        "--clean", "--if-exists", "--no-owner",
        str(dump_file),
    ]
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        sys.exit(f"pg_restore failed (exit {result.returncode})")
    print("Restore complete.")


if __name__ == "__main__":
    main()

"""Shared pg_dump backup logic -- used by both scripts/backup_db.py (CLI,
for manual/scheduled use) and POST /api/refresh/backup (job-manager-driven,
for the frontend's "Backup database" button). Keeping this in one place
means the two callers can't drift on flags or retention behavior."""

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from sqlalchemy.engine import make_url

from app.config import settings
from app.services.job_manager import Job

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "backups"
DEFAULT_KEEP = 14


class BackupError(Exception):
    pass


def run_backup(keep: int = DEFAULT_KEEP, out_dir: Path = DEFAULT_OUT_DIR, job: Job | None = None) -> dict:
    def log(message: str) -> None:
        if job is not None:
            job.log_line(message)

    if shutil.which("pg_dump") is None:
        raise BackupError(
            "pg_dump not found on PATH. It ships with the Postgres install "
            "(e.g. C:\\Program Files\\PostgreSQL\\<version>\\bin on Windows)."
        )

    url = make_url(settings.database_url)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{url.database}_{stamp}.dump"

    env = os.environ.copy()
    if url.password:
        env["PGPASSWORD"] = url.password

    log(f"Backing up '{url.database}'...")
    cmd = [
        "pg_dump",
        "-h", url.host or "localhost",
        "-p", str(url.port or 5432),
        "-U", url.username or "postgres",
        "-d", url.database,
        "-Fc",  # custom format: compressed, supports selective pg_restore
        "-f", str(out_path),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        out_path.unlink(missing_ok=True)
        raise BackupError(f"pg_dump failed (exit {result.returncode}): {result.stderr.strip()}")

    size_mb = out_path.stat().st_size / 1_048_576
    log(f"Wrote {out_path.name} ({size_mb:.1f} MB)")

    pruned = _prune_old_backups(out_dir, url.database, keep)
    for name in pruned:
        log(f"Pruned old backup: {name}")

    return {"file": out_path.name, "size_mb": round(size_mb, 1), "pruned": pruned}


def _prune_old_backups(out_dir: Path, db_name: str, keep: int) -> list[str]:
    dumps = sorted(out_dir.glob(f"{db_name}_*.dump"), key=lambda p: p.stat().st_mtime, reverse=True)
    pruned = []
    for stale in dumps[keep:]:
        stale.unlink()
        pruned.append(stale.name)
    return pruned

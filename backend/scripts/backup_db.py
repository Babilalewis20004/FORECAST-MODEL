"""CLI wrapper around app.services.backup_service.run_backup -- see there
for the actual pg_dump logic (shared with POST /api/refresh/backup, the
frontend's "Backup database" button). Dumps to backend/backups/
(gitignored) in pg_dump's custom format and keeps the most recent N
(default 14), pruning older ones on each run.

Usage:
    python scripts/backup_db.py [--keep N] [--out-dir PATH]"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.backup_service import DEFAULT_KEEP, DEFAULT_OUT_DIR, BackupError, run_backup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP, help=f"Number of most recent backups to retain (default: {DEFAULT_KEEP})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory to write the dump into")
    args = parser.parse_args()

    try:
        result = run_backup(keep=args.keep, out_dir=args.out_dir)
    except BackupError as e:
        sys.exit(str(e))

    print(f"Done: {result['file']} ({result['size_mb']} MB)")
    for name in result["pruned"]:
        print(f"Pruned old backup: {name}")


if __name__ == "__main__":
    main()

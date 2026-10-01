"""Pulls TheStatsAPI's full per-match stats (shot locations, big chances, ...)
for our matches into the response cache -- the raw material for the
proxy-xG experiment (scripts/experiment_proxy_xg.py). Writes nothing to the
database; every response lands in downloaded_files/thestatsapi_cache and
reruns cost no requests.

  THESTATSAPI_KEY=... python scripts/pull_thestatsapi_stats.py "Ligue 2" "Serie B"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from import_thestatsapi_xg import FIRST_DATE, Client, join_matches, season_matches  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.orm_models import Match  # noqa: E402

COMPETITIONS = {
    "Ligue 2": "comp_9777",
    "Segunda División": "comp_0976",
    "Serie B": "comp_5450",
    "J1 League": "comp_6240",
    "Liga Profesional Argentina": "comp_4540",
}


def main() -> None:
    key = os.environ.get("THESTATSAPI_KEY")
    if not key:
        raise SystemExit("set THESTATSAPI_KEY")
    c = Client(key)
    db = SessionLocal()
    try:
        for league in sys.argv[1:]:
            comp_id = COMPETITIONS[league]
            ours = db.query(Match).filter(Match.league == league, Match.date >= FIRST_DATE).all()
            seasons = c.get(f"/football/competitions/{comp_id}/seasons", f"seasons_{comp_id}.json")["data"]
            theirs = []
            for s in seasons:
                if s.get("end_year") is not None and s["end_year"] < FIRST_DATE.year:
                    continue
                theirs += season_matches(c, comp_id, s["id"])
            pairs = join_matches(ours, theirs)
            before = c.requests
            for i, (_, t) in enumerate(pairs, 1):
                c.get(f"/football/matches/{t['id']}/stats", f"stats_{t['id']}.json")
                if i % 250 == 0:
                    print(f"  {league.encode('ascii', 'replace').decode()}: {i}/{len(pairs)}", flush=True)
            print(f"{league.encode('ascii', 'replace').decode()}: {len(pairs)} matches cached, {c.requests - before} new requests", flush=True)
    finally:
        db.close()


if __name__ == "__main__":
    main()

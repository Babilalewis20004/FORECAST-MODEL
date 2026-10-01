"""Fills missing match xG from TheStatsAPI for leagues where API-Football's xG
coverage is thin (J1 League, Serie B, Liga Profesional Argentina).

  THESTATSAPI_KEY=... python scripts/import_thestatsapi_xg.py --check   # scale check only, writes nothing
  THESTATSAPI_KEY=... python scripts/import_thestatsapi_xg.py           # fill NULL xG

Only NULL home_xg/away_xg are filled; API-Football values are never
overwritten, so a league keeps whichever source it already had per match.
Raw responses are cached under downloaded_files/thestatsapi_cache, so reruns
cost no requests. TheStatsAPI matches are joined to ours on league + date
(+/-1 day) + regulation score, then team-name similarity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import requests

from app.db import SessionLocal
from app.orm_models import Match
from app.services.odds_history import MIN_NAME_SIMILARITY, _similarity

BASE = "https://api.thestatsapi.com/api"
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "downloaded_files" / "thestatsapi_cache"

# our league name -> TheStatsAPI competition id
COMPETITIONS = {
    "J1 League": "comp_6240",
    "Serie B": "comp_5450",
    "Liga Profesional Argentina": "comp_4540",
}
FIRST_DATE = date(2022, 7, 1)
CHECK_SAMPLE = 80


class Client:
    def __init__(self, key: str) -> None:
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {key}"
        self.requests = 0

    def get(self, path: str, cache_name: str | None = None, **params) -> dict:
        cache = CACHE / cache_name if cache_name else None
        if cache is not None and cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        for _ in range(8):
            try:
                r = self.s.get(f"{BASE}{path}", params=params, timeout=30)
            except requests.ConnectionError:
                time.sleep(5)  # dropped connection: back off and retry
                continue
            self.requests += 1
            if r.status_code == 429:
                if r.json().get("error", {}).get("type") == "USAGE_LIMIT_EXCEEDED":
                    raise SystemExit("monthly TheStatsAPI quota exhausted")
                time.sleep(15)
                continue
            # A match with no stats page is just a match without xG.
            data = {} if r.status_code == 404 else r.json()
            if r.status_code != 404:
                r.raise_for_status()
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(data), encoding="utf-8")
            time.sleep(0.15)
            return data
        raise RuntimeError(f"rate limited: {path}")


def season_matches(c: Client, comp_id: str, season_id: str) -> list[dict]:
    out, page = [], 1
    while True:
        j = c.get(
            "/football/matches", f"matches_{season_id}_p{page}.json",
            competition_id=comp_id, season_id=season_id, status="finished", per_page=100, page=page,
        )
        out += j["data"]
        if page >= j["meta"]["total_pages"]:
            return out
        page += 1


def xg_of(stats: dict) -> tuple[float, float] | None:
    xg = ((stats.get("data") or {}).get("overview") or {}).get("expected_goals") or {}
    both = xg.get("all")
    if not both or both.get("home") is None or both.get("away") is None:
        return None
    return float(both["home"]), float(both["away"])


def join_matches(ours: list[Match], theirs: list[dict]) -> list[tuple[Match, dict]]:
    """Pair each of their matches with ours: same date +/-1 day and
    regulation score, best team-name similarity above the odds-join threshold."""
    by_score: dict[tuple[int, int], list[Match]] = {}
    for m in ours:
        by_score.setdefault((m.home_goals, m.away_goals), []).append(m)
    used, pairs = set(), []
    for t in theirs:
        sc = t["score"].get("regulation") or t["score"]
        if sc.get("home") is None or sc.get("away") is None:
            continue
        d = datetime.fromisoformat(t["utc_date"].replace("Z", "+00:00")).date()
        best, best_score = None, MIN_NAME_SIMILARITY
        for m in by_score.get((sc["home"], sc["away"]), []):
            if m.id in used or abs((m.date - d).days) > 1:
                continue
            s = min(_similarity(t["home_team"]["name"], m.home_team), _similarity(t["away_team"]["name"], m.away_team))
            if s > best_score:
                best, best_score = m, s
        if best is not None:
            used.add(best.id)
            pairs.append((best, t))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare scales against API-Football xG; write nothing")
    args = ap.parse_args()
    key = os.environ.get("THESTATSAPI_KEY")
    if not key:
        raise SystemExit("set THESTATSAPI_KEY")
    c = Client(key)
    db = SessionLocal()
    try:
        for league, comp_id in COMPETITIONS.items():
            ours = db.query(Match).filter(Match.league == league, Match.date >= FIRST_DATE).all()
            seasons = c.get(f"/football/competitions/{comp_id}/seasons", f"seasons_{comp_id}.json")["data"]
            theirs = []
            for s in seasons:
                if s.get("end_year") is not None and s["end_year"] < FIRST_DATE.year:
                    continue
                theirs += season_matches(c, comp_id, s["id"])
            pairs = join_matches(ours, theirs)
            print(f"\n{league}: ours {len(ours)}, theirs {len(theirs)}, joined {len(pairs)}")

            if args.check:
                both = [(m, t) for m, t in pairs if m.home_xg is not None and m.away_xg is not None][:CHECK_SAMPLE]
                a, b = [], []
                for m, t in both:
                    x = xg_of(c.get(f"/football/matches/{t['id']}/stats", f"stats_{t['id']}.json"))
                    if x:
                        a += [m.home_xg, m.away_xg]
                        b += list(x)
                if len(a) >= 20:
                    a, b = np.array(a), np.array(b)
                    print(f"  scale check on {len(a)//2} matches with both sources: "
                          f"API-Football mean {a.mean():.3f}, TheStatsAPI mean {b.mean():.3f}, "
                          f"corr {np.corrcoef(a, b)[0, 1]:.3f}, mean abs diff {np.abs(a - b).mean():.3f}")
                else:
                    print(f"  only {len(a)//2} overlapping matches -- no scale check possible")
                continue

            filled = no_xg = 0
            for m, t in pairs:
                if m.home_xg is not None and m.away_xg is not None:
                    continue
                x = xg_of(c.get(f"/football/matches/{t['id']}/stats", f"stats_{t['id']}.json"))
                if x is None:
                    no_xg += 1
                    continue
                m.home_xg, m.away_xg = x
                filled += 1
                if filled % 200 == 0:
                    db.commit()
            db.commit()
            total = len(ours)
            have = sum(1 for m in ours if m.home_xg is not None)
            print(f"  filled {filled}, no xG in TheStatsAPI {no_xg}; xG coverage now {have}/{total} ({have / total:.0%})")
        print(f"\nrequests made: {c.requests}")
    finally:
        db.close()


if __name__ == "__main__":
    main()

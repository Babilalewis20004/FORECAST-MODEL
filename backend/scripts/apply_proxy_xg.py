"""EXPERIMENT: fill NULL match xG with a proxy predicted from TheStatsAPI's
shot-location stats (ridge fitted on every cached match that has real xG).

  python scripts/apply_proxy_xg.py "Serie B"            # fill NULL xG, log touched match ids
  python scripts/apply_proxy_xg.py "Serie B" --revert   # set those rows back to NULL

Real xG is never overwritten. The touched ids are logged to
downloaded_files/experiments/proxy_filled_<league>.json so the fill can be undone.
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import Ridge

from experiment_proxy_xg import LOCATION, load
from import_thestatsapi_xg import FIRST_DATE, Client, join_matches, season_matches
from pull_thestatsapi_stats import COMPETITIONS

from app.db import SessionLocal
from app.orm_models import Match

LOG_DIR = Path(__file__).resolve().parents[2] / "downloaded_files" / "experiments"


def main() -> None:
    league = sys.argv[1]
    log = LOG_DIR / f"proxy_filled_{re.sub(r'[^a-z0-9]+', '_', league.lower())}.json"
    db = SessionLocal()
    try:
        if "--revert" in sys.argv:
            ids = json.loads(log.read_text())
            db.query(Match).filter(Match.id.in_(ids)).update({Match.home_xg: None, Match.away_xg: None}, synchronize_session=False)
            db.commit()
            print(f"reverted {len(ids)} rows")
            return

        df = load()
        lab = df[df.xg.notna()]
        med = lab[LOCATION].median()
        model = Ridge(1.0).fit(lab[LOCATION].fillna(med), lab.xg)
        df["proxy"] = np.clip(model.predict(df[LOCATION].fillna(med)), 0.0, None)
        proxy = {(r.match, r.side): r.proxy for r in df.itertuples()}
        have_stats = set(df.match)

        c = Client(os.environ["THESTATSAPI_KEY"])
        comp_id = COMPETITIONS[league]
        ours = db.query(Match).filter(Match.league == league, Match.date >= FIRST_DATE).all()
        theirs = []
        for s in c.get(f"/football/competitions/{comp_id}/seasons", f"seasons_{comp_id}.json")["data"]:
            if s.get("end_year") is not None and s["end_year"] < FIRST_DATE.year:
                continue
            theirs += season_matches(c, comp_id, s["id"])
        touched = []
        for m, t in join_matches(ours, theirs):
            if m.home_xg is None and m.away_xg is None and t["id"] in have_stats:
                m.home_xg, m.away_xg = float(proxy[(t["id"], "home")]), float(proxy[(t["id"], "away")])
                touched.append(m.id)
        db.commit()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps(touched))
        have = db.query(Match).filter(Match.league == league, Match.date >= FIRST_DATE, Match.home_xg.isnot(None)).count()
        print(f"{league}: filled {len(touched)} proxy rows; xG coverage now {have}/{len(ours)} ({have / len(ours):.0%})")
    finally:
        db.close()


if __name__ == "__main__":
    main()

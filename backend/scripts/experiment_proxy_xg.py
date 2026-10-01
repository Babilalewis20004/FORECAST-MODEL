"""Can TheStatsAPI's detailed shot/chance stats reproduce its own xG?

Reads the cached per-match stats (scripts/pull_thestatsapi_stats.py), builds
one row per team-match from the "all" period, and fits xG ~ shot-location
stats on matches that have xG. Held out by *match* (not row) so a game's two
sides never straddle train and test. Reports R2/MAE against a baseline that
only has the stats we already store (shots, SoT, corners, possession).
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit

CACHE = Path(__file__).resolve().parents[2] / "downloaded_files" / "thestatsapi_cache"

STORED = ["total_shots", "shots_on_target", "corner_kicks", "ball_possession"]
LOCATION = STORED + [
    "shots_inside_box", "shots_outside_box", "blocked_shots", "big_chances", "big_chances_missed",
    "touches_in_penalty_area", "hit_woodwork", "accurate_crosses", "final_third_entries",
]


def _val(d: dict, cat: str, key: str, side: str):
    blk = (d.get(cat) or {}).get(key)
    if not isinstance(blk, dict):
        return None
    blk = blk.get("all", blk)
    return blk.get(side) if isinstance(blk, dict) else None


def load() -> pd.DataFrame:
    rows = []
    for p in glob.glob(str(CACHE / "stats_mt_*.json")):
        d = json.load(open(p)).get("data")
        if not d:
            continue
        xg = ((d.get("overview") or {}).get("expected_goals") or {}).get("all")
        for side, opp in (("home", "away"), ("away", "home")):
            r = {"match": d["match_id"], "side": side, "xg": xg.get(side) if xg else None}
            for k in LOCATION:
                v = None
                for cat in ("overview", "shots", "attack", "passes"):
                    v = _val(d, cat, k, side)
                    if v is not None:
                        break
                r[k] = v
            r["opp_shots_inside_box"] = _val(d, "shots", "shots_inside_box", opp)
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    print(f"team-match rows: {len(df)}; with xG: {df.xg.notna().sum()}")
    lab = df[df.xg.notna()].copy()
    for c in LOCATION:
        print(f"  {c:<24} present {lab[c].notna().mean():.0%}")
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(lab, groups=lab.match))
    tr, te = lab.iloc[train_idx], lab.iloc[test_idx]
    print(f"train {len(tr)} rows / test {len(te)} rows ({te.match.nunique()} matches)\n")

    def fit_eval(name, cols, model):
        a, b = tr[cols].fillna(tr[cols].median()), te[cols].fillna(tr[cols].median())
        model.fit(a, tr.xg)
        p = model.predict(b)
        print(f"{name:<44} R2 {r2_score(te.xg, p):.3f}   MAE {mean_absolute_error(te.xg, p):.3f}   "
              f"mean pred {p.mean():.3f} vs actual {te.xg.mean():.3f}")
        return model

    fit_eval("stored stats only (ridge)", STORED, Ridge(1.0))
    fit_eval("stored stats only (gbm)", STORED, HistGradientBoostingRegressor(random_state=0))
    fit_eval("+ shot location & chances (ridge)", LOCATION, Ridge(1.0))
    fit_eval("+ shot location & chances (gbm)", LOCATION, HistGradientBoostingRegressor(random_state=0))
    box = ["total_shots", "shots_on_target", "shots_inside_box", "shots_outside_box", "big_chances"]
    fit_eval("minimal: shots in/out of box, SoT, big chances", box, HistGradientBoostingRegressor(random_state=0))


if __name__ == "__main__":
    main()

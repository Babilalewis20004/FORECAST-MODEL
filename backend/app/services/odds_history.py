"""Historical bookmaker odds from football-data.co.uk season CSVs, joined to
our matches. API-Football only keeps odds ~7 days after a fixture, so past
seasons (model features, shrinkage fitting, backtests) come from here.

The two sources spell teams differently ("Man City" / "Manchester City"),
so rows are joined on league + date (+/-1 day) + final score first, then
team-name similarity. Only the six domestic leagues are covered.
"""

from __future__ import annotations

from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.orm_models import Match, MatchOdds

ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = ROOT / "downloaded_files" / "football_data"
FD_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"

FD_LEAGUES = {
    "Premier League": "E0",
    "EFL Championship": "E1",
    "League One": "E2",
    "League Two": "E3",
    "La Liga": "SP1",
    "Scottish Premiership": "SC0",
}

# (price, bookmaker) -> football-data columns for home/draw/away, over/under
# 2.5. "opening" is football-data's pre-closing snapshot (taken ~2 days
# before kickoff), "closing" the last price before kickoff.
FD_ODDS = {
    ("closing", "pinnacle"): (("PSCH", "PSCD", "PSCA"), ("PC>2.5", "PC<2.5")),
    ("opening", "pinnacle"): (("PSH", "PSD", "PSA"), ("P>2.5", "P<2.5")),
    ("closing", "bet365"): (("B365CH", "B365CD", "B365CA"), ("B365C>2.5", "B365C<2.5")),
    ("opening", "bet365"): (("B365H", "B365D", "B365A"), ("B365>2.5", "B365<2.5")),
    ("closing", "avg"): (("AvgCH", "AvgCD", "AvgCA"), ("AvgC>2.5", "AvgC<2.5")),
    ("opening", "avg"): (("AvgH", "AvgD", "AvgA"), ("Avg>2.5", "Avg<2.5")),
    ("closing", "max"): (("MaxCH", "MaxCD", "MaxCA"), ("MaxC>2.5", "MaxC<2.5")),
    ("opening", "max"): (("MaxH", "MaxD", "MaxA"), ("Max>2.5", "Max<2.5")),
}
FLAT_KEYS = ("home_win", "draw", "away_win", "over_2.5", "under_2.5")
MIN_NAME_SIMILARITY = 0.45


def season_code(start_year: int) -> str:
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def current_season_start(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def load_fd_odds(season: str, leagues, refresh: bool = False) -> pd.DataFrame:
    """football-data rows for one season (e.g. "2526"), with `date` and our
    `league` name added. Downloads once, then reads the cache; refresh=True
    re-downloads (for a season still in progress)."""
    frames = []
    for league in leagues:
        code = FD_LEAGUES[league]
        path = CACHE_DIR / f"fd_{season}_{code}.csv"
        if refresh or not path.exists():
            resp = requests.get(FD_URL.format(season=season, code=code), timeout=60)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(resp.content)
        df = pd.read_csv(path, encoding="utf-8-sig").dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
        df["league"] = league
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# football-data -> API-Football spellings too different for string
# similarity to bridge (it handles "Man City" / "Manchester City" etc.).
TEAM_ALIASES = {
    "Ath Bilbao": "Athletic Club",
    "Hearts": "Heart Of Midlothian",
}


def _norm(name: str) -> str:
    name = TEAM_ALIASES.get(str(name), str(name))
    return "".join(ch for ch in name.lower() if ch.isalnum() or ch == " ").replace(" fc", "").strip()


def _similarity(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if a in b or b in a:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def join_odds(matches: pd.DataFrame, odds: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Attach each football-data row to the row of `matches` (league, date,
    home_team, away_team, home_goals, away_goals) for the same game. The
    result carries `matches`' index as `match_id` plus every odds column.
    Returns (joined rows, number of odds rows left unmatched)."""
    if odds.empty:
        return pd.DataFrame(), 0
    matches = matches.copy()
    matches["date"] = pd.to_datetime(matches["date"])
    by_key = {k: g for k, g in matches.groupby(["league", "home_goals", "away_goals"])}
    pairs, used = [], set()
    for i, (league, d, home, away, hg, ag) in enumerate(
        zip(odds["league"], odds["date"], odds["HomeTeam"], odds["AwayTeam"], odds["FTHG"], odds["FTAG"])
    ):
        cands = by_key.get((league, int(hg), int(ag)))
        if cands is None:
            continue
        cands = cands[(cands["date"] - d).abs() <= pd.Timedelta(days=1)]
        best, best_score = None, MIN_NAME_SIMILARITY
        for mid, c_home, c_away in zip(cands.index, cands["home_team"], cands["away_team"]):
            score = min(_similarity(home, c_home), _similarity(away, c_away))
            if score > best_score and mid not in used:
                best, best_score = mid, score
        if best is not None:
            used.add(best)
            pairs.append((i, best))
    if not pairs:
        return pd.DataFrame(), len(odds)
    odds_idx, match_idx = map(list, zip(*pairs))
    joined = pd.concat(
        [
            matches.loc[match_idx].rename_axis("match_id").reset_index(),
            odds.iloc[odds_idx].drop(columns=["league", "date"]).reset_index(drop=True),
        ],
        axis=1,
    )
    return joined, len(odds) - len(pairs)


def sync_match_odds(db, leagues=None, seasons: list[int] | None = None, bookmaker: str = "bet365") -> dict:
    """Upsert football-data opening/closing odds into match_odds for our
    matches. seasons: start years; default = every season from 2022 (our
    earliest scraped data) to the current one. The current season is always
    re-downloaded, completed ones read from cache."""
    leagues = [lg for lg in (leagues or FD_LEAGUES) if lg in FD_LEAGUES]
    current = current_season_start()
    seasons = seasons or list(range(2022, current + 1))
    summary = {}
    for league in leagues:
        rows = db.execute(
            select(Match.id, Match.league, Match.date, Match.home_team, Match.away_team, Match.home_goals, Match.away_goals)
            .where(Match.league == league)
        ).all()
        if not rows:
            continue
        matches = pd.DataFrame(rows, columns=["id", "league", "date", "home_team", "away_team", "home_goals", "away_goals"]).set_index("id")
        n_joined = n_unmatched = 0
        for start in seasons:
            odds = load_fd_odds(season_code(start), [league], refresh=start == current)
            joined, unmatched = join_odds(matches, odds)
            n_unmatched += unmatched
            if joined.empty:
                continue
            opening = flat_odds(joined, "opening", bookmaker)
            closing = flat_odds(joined, "closing", bookmaker)
            for match_id, o, c in zip(joined["match_id"], opening, closing):
                if o is None and c is None:
                    continue
                stmt = pg_insert(MatchOdds).values(
                    match_id=int(match_id), bookmaker=bookmaker, source="football-data.co.uk", opening=o, closing=c
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=[MatchOdds.match_id, MatchOdds.bookmaker],
                    set_={"opening": stmt.excluded.opening, "closing": stmt.excluded.closing},
                )
                db.execute(stmt)
                n_joined += 1
        db.commit()
        summary[league] = {"matches": len(matches), "with_odds": n_joined, "odds_rows_unmatched": n_unmatched}
    return summary


def numeric_col(df: pd.DataFrame, col: str) -> pd.Series:
    # Older seasons lack some bookmakers' columns: missing -> NaN.
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def flat_odds(joined: pd.DataFrame, price: str, bookmaker: str) -> list[dict | None]:
    """Per joined row, {home_win, draw, away_win, over_2.5, under_2.5} odds
    (only the complete, valid books), or None when neither book is priced."""
    mr_cols, ou_cols = FD_ODDS[(price, bookmaker)]
    cols = dict(zip(FLAT_KEYS, (*mr_cols, *ou_cols)))
    values = {k: numeric_col(joined, c).to_numpy() for k, c in cols.items()}
    out = []
    for i in range(len(joined)):
        row = {}
        for book in (("home_win", "draw", "away_win"), ("over_2.5", "under_2.5")):
            v = [values[k][i] for k in book]
            if all(np.isfinite(x) and x > 1.0 for x in v):
                row.update({k: float(x) for k, x in zip(book, v)})
        out.append(row or None)
    return out

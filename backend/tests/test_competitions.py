import json

from app.services import scrape_service
from app.services.model_service import ALL_LEAGUES, ALL_MODEL_GROUPS, group_competitions, model_group


def test_every_competition_is_scraped_and_served_by_a_trained_group():
    assert set(ALL_LEAGUES) == set(scrape_service.LEAGUES)
    for league in ALL_LEAGUES:
        assert model_group(league) in ALL_MODEL_GROUPS
        assert league in group_competitions(model_group(league))


def test_knockout_results_use_the_regular_time_score(tmp_path, monkeypatch):
    def item(fixture_id, status, goals, fulltime):
        return {
            "fixture": {"id": fixture_id, "date": "2025-06-28T19:00:00+00:00", "status": {"short": status}},
            "teams": {"home": {"name": "Home FC"}, "away": {"name": "Away FC"}},
            "goals": goals,
            "score": {"fulltime": fulltime},
        }

    payload = {
        "errors": [],
        "response": [
            item(1, "AET", {"home": 3, "away": 2}, {"home": 2, "away": 2}),
            item(2, "FT", {"home": 1, "away": 0}, {"home": 1, "away": 0}),
        ],
    }
    monkeypatch.setattr(scrape_service, "CACHE_DIR", tmp_path)
    (tmp_path / "fixtures_38_2025.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = scrape_service._fetch_league_season(None, scrape_service.Budget(), "Premier League", 38, 2025)

    assert [(r["home_goals"], r["away_goals"]) for r in rows] == [(2, 2), (1, 0)]


def test_cached_season_is_only_reused_once_every_fixture_is_settled(tmp_path):
    def cache(name, *statuses):
        path = tmp_path / name
        payload = {"response": [{"fixture": {"status": {"short": s}}} for s in statuses]}
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    assert scrape_service._season_complete(cache("done.json", "FT", "PEN", "CANC"))
    # Upcoming (NS) or postponed fixtures mean the season must be refetched.
    assert not scrape_service._season_complete(cache("live.json", "FT", "NS"))
    assert not scrape_service._season_complete(cache("pst.json", "FT", "PST"))
    assert not scrape_service._season_complete(tmp_path / "missing.json")

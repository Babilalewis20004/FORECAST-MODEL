import { useEffect, useRef, useState } from "react";
import { getLeagues, getMatches, getUpcomingFixtures } from "../api/client";
import { DateRangeFilter, type DateRange } from "../components/DateRangeFilter";
import { FixtureList } from "../components/FixtureList";
import { LeagueSelector } from "../components/LeagueSelector";
import { RefreshPanel } from "../components/RefreshPanel";
import type { Fixture, League, RecentMatch } from "../types";

function todayISO(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

// Must match the backend's Query(default=...) in routers/fixtures.py -- used
// only as a heuristic ("we got back exactly the cap, so there may be more")
// since the endpoints don't return a total count.
const UPCOMING_LIMIT = 500;
const MATCHES_LIMIT = 200;

function parseInitialLeague(): string | null {
  return new URLSearchParams(window.location.search).get("league");
}

// "Whole dataset" (start: null, end: null) and "no filter chosen yet" would
// otherwise serialize to the same empty query string -- range=all is the
// explicit marker that disambiguates a shared/refreshed "Whole dataset" URL
// from a first-ever visit, which should default to Today.
function parseInitialDateRange(): DateRange {
  const params = new URLSearchParams(window.location.search);
  if (params.get("range") === "all") return { start: null, end: null };
  if (params.has("start") || params.has("end")) return { start: params.get("start"), end: params.get("end") };
  return { start: todayISO(), end: todayISO() };
}

function parseInitialValueOnly(): boolean {
  return new URLSearchParams(window.location.search).get("value") === "1";
}

function matchToFixture(m: RecentMatch): Fixture {
  return {
    fixture_id: m.fixture_id ?? -1,
    date: m.date,
    league: m.league,
    home_team: m.home_team,
    away_team: m.away_team,
    status: "FT",
    prediction: m.prediction ?? null,
    home_goals: m.home_goals,
    away_goals: m.away_goals,
  };
}

export function FixturesPage() {
  const [leagues, setLeagues] = useState<League[]>([]);
  const [league, setLeague] = useState<string | null>(parseInitialLeague);
  const [dateRange, setDateRange] = useState<DateRange>(parseInitialDateRange);
  const [valueOnly, setValueOnly] = useState<boolean>(parseInitialValueOnly);
  const [fixtures, setFixtures] = useState<Fixture[]>([]);
  const [truncated, setTruncated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // loadFixtures is called from two places (the [league, dateRange] effect
  // below and RefreshPanel's onJobFinished) -- if an older call's request
  // resolves after a newer one, it must not be allowed to clobber the newer
  // result with stale data. Each call claims the next id; only the call
  // whose id still matches when its response lands is allowed to commit it.
  const requestIdRef = useRef(0);

  async function loadLeagues() {
    setLeagues(await getLeagues());
  }

  async function loadFixtures() {
    const id = ++requestIdRef.current;
    setLoading(true);
    setError(null);
    try {
      const today = todayISO();
      const yesterday = new Date(Date.now() - 86400000);
      const yesterdayISO = `${yesterday.getFullYear()}-${String(yesterday.getMonth() + 1).padStart(2, "0")}-${String(yesterday.getDate()).padStart(2, "0")}`;

      const wantsUpcoming = dateRange.end === null || dateRange.end >= today;
      const wantsPast = dateRange.start === null || dateRange.start <= yesterdayISO;

      const [upcoming, past] = await Promise.all([
        wantsUpcoming
          ? getUpcomingFixtures({
              league: league ?? undefined,
              start: dateRange.start && dateRange.start > today ? dateRange.start : undefined,
              end: dateRange.end ?? undefined,
            })
          : Promise.resolve([]),
        wantsPast
          ? getMatches({
              league: league ?? undefined,
              start: dateRange.start ?? undefined,
              end: dateRange.end && dateRange.end < today ? dateRange.end : yesterdayISO,
              withPredictions: true,
            })
          : Promise.resolve([]),
      ]);

      if (id !== requestIdRef.current) return; // a newer call has since started
      const merged = [...upcoming, ...past.map(matchToFixture)].sort((a, b) => a.date.localeCompare(b.date));
      setFixtures(merged);
      setTruncated((wantsUpcoming && upcoming.length === UPCOMING_LIMIT) || (wantsPast && past.length === MATCHES_LIMIT));
    } catch (e) {
      if (id !== requestIdRef.current) return;
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (id === requestIdRef.current) setLoading(false);
    }
  }

  useEffect(() => {
    loadLeagues();
  }, []);

  useEffect(() => {
    loadFixtures();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [league, dateRange]);

  // Keeps the URL a live reflection of the current filters -- refreshing or
  // sharing the link reproduces the same view. replaceState (not
  // pushState): filter changes shouldn't spam browser history.
  useEffect(() => {
    const params = new URLSearchParams();
    if (league) params.set("league", league);
    if (dateRange.start === null && dateRange.end === null) {
      params.set("range", "all");
    } else {
      if (dateRange.start) params.set("start", dateRange.start);
      if (dateRange.end) params.set("end", dateRange.end);
    }
    if (valueOnly) params.set("value", "1");
    const qs = params.toString();
    window.history.replaceState(null, "", qs ? `?${qs}` : window.location.pathname);
  }, [league, dateRange, valueOnly]);

  const displayedFixtures = valueOnly ? fixtures.filter((f) => f.prediction?.is_value_bet) : fixtures;

  return (
    <div>
      <h1 style={{ fontSize: 24, marginBottom: 16 }}>Predict Model</h1>

      <RefreshPanel
        leagues={leagues}
        league={league}
        onLeagueChange={setLeague}
        onJobFinished={() => {
          loadLeagues();
          loadFixtures();
        }}
      />

      <div style={{ marginBottom: 16, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
        <LeagueSelector leagues={leagues} value={league} onChange={setLeague} />
        <button
          type="button"
          onClick={() => setValueOnly((v) => !v)}
          style={{
            padding: "6px 14px",
            borderRadius: 999,
            border: `1px solid ${valueOnly ? "var(--accent)" : "var(--border)"}`,
            background: valueOnly ? "color-mix(in srgb, var(--accent) 14%, var(--surface-1))" : "var(--surface-1)",
            color: valueOnly ? "var(--accent)" : "var(--text-primary)",
            fontSize: 13,
            fontWeight: 600,
            fontFamily: "inherit",
            cursor: "pointer",
          }}
        >
          {valueOnly ? "★" : "☆"} Value bets only
        </button>
      </div>

      <DateRangeFilter value={dateRange} onChange={setDateRange} />

      {error && <p style={{ color: "var(--critical)" }}>{error}</p>}
      {!error &&
        (loading && fixtures.length === 0 ? (
          <p>Loading fixtures...</p>
        ) : (
          <>
            <p style={{ fontSize: 13, color: "var(--text-muted)", margin: "0 0 12px" }}>
              {loading
                ? "Updating…"
                : `${displayedFixtures.length} fixture${displayedFixtures.length === 1 ? "" : "s"}${
                    truncated ? " (more may exist past the fetch limit — narrow your filter to see them)" : ""
                  }`}
            </p>
            <div style={{ opacity: loading ? 0.6 : 1, transition: "opacity 150ms ease" }}>
              <FixtureList fixtures={displayedFixtures} />
            </div>
          </>
        ))}
    </div>
  );
}

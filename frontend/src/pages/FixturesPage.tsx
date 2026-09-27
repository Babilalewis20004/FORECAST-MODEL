import { useEffect, useRef, useState } from "react";
import { getLeagues, getUpcomingFixtures } from "../api/client";
import { FixtureList } from "../components/FixtureList";
import { LeagueSelector } from "../components/LeagueSelector";
import { RefreshPanel } from "../components/RefreshPanel";
import type { Fixture, League } from "../types";

export function FixturesPage() {
  const [leagues, setLeagues] = useState<League[]>([]);
  const [league, setLeague] = useState<string | null>(null);
  const [fixtures, setFixtures] = useState<Fixture[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // loadFixtures is called from two places (the [league] effect below and
  // RefreshPanel's onJobFinished) -- if an older call's request resolves
  // after a newer one, it must not be allowed to clobber the newer result
  // with stale data. Each call claims the next id; only the call whose id
  // still matches when its response lands is allowed to commit it.
  const requestIdRef = useRef(0);

  async function loadLeagues() {
    setLeagues(await getLeagues());
  }

  async function loadFixtures() {
    const id = ++requestIdRef.current;
    setLoading(true);
    setError(null);
    try {
      const data = await getUpcomingFixtures({ league: league ?? undefined });
      if (id !== requestIdRef.current) return; // a newer call has since started
      setFixtures(data);
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
  }, [league]);

  return (
    <div>
      <h1 style={{ fontSize: 24, marginBottom: 16 }}>Predict Model</h1>

      <RefreshPanel
        onJobFinished={() => {
          loadLeagues();
          loadFixtures();
        }}
      />

      <div style={{ marginBottom: 16 }}>
        <LeagueSelector leagues={leagues} value={league} onChange={setLeague} />
      </div>

      {loading && <p>Loading fixtures...</p>}
      {error && <p style={{ color: "var(--critical)" }}>{error}</p>}
      {!loading && !error && <FixtureList fixtures={fixtures} />}
    </div>
  );
}

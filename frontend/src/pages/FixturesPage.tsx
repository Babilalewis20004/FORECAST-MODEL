import { useEffect, useState } from "react";
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

  async function loadLeagues() {
    setLeagues(await getLeagues());
  }

  async function loadFixtures() {
    setLoading(true);
    setError(null);
    try {
      const data = await getUpcomingFixtures({ league: league ?? undefined });
      setFixtures(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
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

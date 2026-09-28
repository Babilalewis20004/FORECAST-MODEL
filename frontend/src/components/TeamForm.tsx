import { useEffect, useRef, useState } from "react";
import { getRecentMatches } from "../api/client";
import type { RecentMatch } from "../types";

interface FormRow {
  opponent: string;
  venue: "H" | "A";
  score: string;
  result: "W" | "D" | "L";
}

function toFormRows(matches: RecentMatch[], team: string): FormRow[] {
  return matches.map((m) => {
    const isHome = m.home_team === team;
    const gf = isHome ? m.home_goals : m.away_goals;
    const ga = isHome ? m.away_goals : m.home_goals;
    const result: FormRow["result"] = gf > ga ? "W" : gf < ga ? "L" : "D";
    return { opponent: isHome ? m.away_team : m.home_team, venue: isHome ? "H" : "A", score: `${gf}–${ga}`, result };
  });
}

function TeamFormList({ team }: { team: string }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [visible, setVisible] = useState(false);
  const [matches, setMatches] = useState<RecentMatch[] | null>(null);
  const [error, setError] = useState(false);

  // Fetch only once this list has actually scrolled near the viewport, not
  // on mount -- a fixture list can render a hundred-plus cards at once, and
  // fetching both teams' form for every one of them immediately (2 requests
  // x every card) exhausts the browser's connection pool. rootMargin starts
  // the fetch slightly before it's actually visible, so it's ready by the
  // time the user scrolls to it.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!visible) return;
    let cancelled = false;

    async function load() {
      try {
        const data = await getRecentMatches(team, 5);
        if (!cancelled) setMatches(data);
      } catch {
        if (!cancelled) setError(true);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [team, visible]);

  return (
    <div ref={ref}>
      {error ? (
        <p className="form-team__status">Couldn't load recent form.</p>
      ) : matches === null ? (
        <p className="form-team__status">{visible ? "Loading…" : " "}</p>
      ) : matches.length === 0 ? (
        <p className="form-team__status">No recent matches.</p>
      ) : (
        <div className="match-list">
          {toFormRows(matches, team).map((r, i) => (
            <div className="match-row" key={i}>
              <span className={`pip pip--${r.result}`}>{r.result}</span>
              <span className="match-opp">
                <span className="match-opp__name">{r.opponent}</span>
                <span className="match-opp__venue">{r.venue === "H" ? "Home" : "Away"}</span>
              </span>
              <span className="match-score">{r.score}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

interface Props {
  homeTeam: string;
  awayTeam: string;
}

// Collapsible, starts collapsed. Each team's last 5 (via the existing
// /api/matches?team=... endpoint, not a new one) is fetched lazily by
// TeamFormList once it scrolls near the viewport -- see the IntersectionObserver
// there -- so a long fixture list doesn't fire hundreds of requests at once.
export function TeamForm({ homeTeam, awayTeam }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="form-section">
      <button type="button" className="form-toggle" onClick={() => setOpen((v) => !v)}>
        <span>Last 5 Matches Form</span>
        <span className={`chev${open ? " chev--open" : ""}`}>▾</span>
      </button>
      {open && (
        <div className="form-body">
          <div className="form-team">
            <div className="form-team__label">
              <span>{homeTeam} — last 5</span>
              <span className="form-team__venue-tag">Home form</span>
            </div>
            <TeamFormList team={homeTeam} />
          </div>
          <div className="form-team">
            <div className="form-team__label">
              <span>{awayTeam} — last 5</span>
              <span className="form-team__venue-tag">Away form</span>
            </div>
            <TeamFormList team={awayTeam} />
          </div>
        </div>
      )}
    </div>
  );
}

import { useState } from "react";

interface Props {
  correctScore: Record<string, number> | null | undefined;
  correctScoreOdds: Record<string, number | null> | null | undefined;
  homeTeam: string;
  awayTeam: string;
  // Actual final score for a finished fixture -- undefined/null for one not
  // yet played.
  result?: { home: number; away: number } | null;
}

function parseScoreline(key: string): [number, number] {
  const [h, a] = key.split("-").map(Number);
  return [h, a];
}

// Header stays visible and collapsible; the grid underneath is live -- tapping
// a cell updates the header's result + odds and highlights that cell, same as
// the goal-range chart already does for its own numbers. Unlike the 1X2/BTTS/
// O-U toggles, one scoreline is always selected (defaults to the model's most
// likely), since "no scoreline picked" isn't a meaningful state here.
export function CorrectScoreGrid({ correctScore, correctScoreOdds, homeTeam, awayTeam, result }: Props) {
  const entries = correctScore ? Object.entries(correctScore).sort((a, b) => b[1] - a[1]) : [];
  const [selected, setSelected] = useState<string | null>(null);
  const [open, setOpen] = useState(false);

  if (entries.length === 0) return null;

  const activeKey = selected ?? entries[0][0];
  const [h, a] = parseScoreline(activeKey);
  const activeOdds = correctScoreOdds?.[activeKey];
  const correctKey = result ? `${result.home}-${result.away}` : null;
  const headerCorrect = correctKey != null && activeKey === correctKey;
  // entries is sorted by probability descending, so entries[0] is both the
  // highest-probability and (odds being 1/prob) lowest-odds scoreline --
  // "favoured" either way you'd define it for this market.
  const favouredKey = entries[0][0];
  const headerFavoured = activeKey === favouredKey;

  return (
    <div className="market">
      <button
        type="button"
        className={`market-row${headerCorrect ? " market-row--correct" : ""}`}
        onClick={() => setOpen((v) => !v)}
      >
        {headerFavoured && <span className="favoured-badge">Favoured</span>}
        <span className="market-row__label">Score</span>
        <span className="market-row__pick">
          {homeTeam} {h} – {a} {awayTeam}
        </span>
        <span className="market-row__right">
          <span className="market-row__odds">{activeOdds != null ? activeOdds.toFixed(2) : "—"}</span>
        </span>
        <span className={`chev${open ? " chev--open" : ""}`}>▾</span>
      </button>
      {open && (
        <div className="score-grid">
          {entries.map(([key, prob]) => {
            const [hg, ag] = parseScoreline(key);
            const odds = correctScoreOdds?.[key];
            const isActive = key === activeKey;
            const isCorrect = key === correctKey;
            const isFavoured = key === favouredKey;
            return (
              <button
                key={key}
                type="button"
                className={`score-cell${isActive ? " score-cell--active" : ""}${isCorrect ? " score-cell--correct" : ""}`}
                onClick={() => setSelected(key)}
                title={`${(prob * 100).toFixed(1)}%`}
              >
                {isFavoured && <span className="favoured-badge">Favoured</span>}
                <span className="score-cell__sc">
                  {hg} – {ag}
                </span>
                <span className="score-cell__odds">{odds != null ? odds.toFixed(2) : "—"}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

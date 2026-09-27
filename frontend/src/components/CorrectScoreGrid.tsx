import { useState } from "react";

interface Props {
  correctScore: Record<string, number> | null | undefined;
  correctScoreOdds: Record<string, number | null> | null | undefined;
  homeTeam: string;
  awayTeam: string;
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
export function CorrectScoreGrid({ correctScore, correctScoreOdds, homeTeam, awayTeam }: Props) {
  const entries = correctScore ? Object.entries(correctScore).sort((a, b) => b[1] - a[1]) : [];
  const [selected, setSelected] = useState<string | null>(null);
  const [open, setOpen] = useState(true);

  if (entries.length === 0) return null;

  const activeKey = selected ?? entries[0][0];
  const [h, a] = parseScoreline(activeKey);
  const activeOdds = correctScoreOdds?.[activeKey];

  return (
    <div className="market">
      <div className="market__title">Correct Score</div>
      <button type="button" className="score-toggle" onClick={() => setOpen((v) => !v)}>
        <span className="score-toggle__result">
          {homeTeam} {h} – {a} {awayTeam}
        </span>
        <span className="score-toggle__right">
          <span className="score-toggle__odds">{activeOdds != null ? activeOdds.toFixed(2) : "—"}</span>
          <span className={`chev${open ? " chev--open" : ""}`}>▾</span>
        </span>
      </button>
      {open && (
        <div className="score-grid">
          {entries.map(([key, prob]) => {
            const [hg, ag] = parseScoreline(key);
            const odds = correctScoreOdds?.[key];
            const isActive = key === activeKey;
            return (
              <button
                key={key}
                type="button"
                className={`score-cell${isActive ? " score-cell--active" : ""}`}
                onClick={() => setSelected(key)}
                title={`${(prob * 100).toFixed(1)}%`}
              >
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

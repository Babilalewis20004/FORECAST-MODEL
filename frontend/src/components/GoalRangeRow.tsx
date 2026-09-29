import { useState } from "react";
import { GoalRangeChart, inRange } from "./GoalRangeChart";
import { RANGE_LABELS } from "../types";
import type { RangeLabel, RangeOdds } from "../types";

interface Props {
  probabilities: Record<RangeLabel, number>;
  odds?: RangeOdds;
  // Actual total goals for a finished fixture -- undefined/null for one not
  // yet played.
  totalGoals?: number | null;
}

// Same collapsed-row shape as the other markets (1X2/BTTS/O-U/Score) --
// shows the model's favoured goal range until tapped open, then reveals the
// full bar chart underneath. Unlike those, there's no per-range tap-to-pick
// here (a bar chart, not discrete buttons), so the row always reflects the
// favourite.
export function GoalRangeRow({ probabilities, odds, totalGoals }: Props) {
  const [open, setOpen] = useState(false);

  const favouredKey = RANGE_LABELS.reduce((best, label) =>
    (probabilities[label] ?? 0) > (probabilities[best] ?? 0) ? label : best,
  );
  const favouredProb = probabilities[favouredKey] ?? 0;
  const favouredOdds = odds?.[favouredKey];
  const headerCorrect = totalGoals != null && inRange(favouredKey, totalGoals);

  return (
    <div className="market">
      <button
        type="button"
        className={`market-row${headerCorrect ? " market-row--correct" : ""}`}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="favoured-badge">Favoured</span>
        <span className="market-row__label">Range</span>
        <span className="market-row__pick">{favouredKey} Goals</span>
        <span className="market-row__right">
          <span className="market-row__odds">{favouredOdds != null ? favouredOdds.toFixed(2) : "—"}</span>
          <span className="market-row__pct">{Math.round(favouredProb * 100)}%</span>
        </span>
        <span className={`chev${open ? " chev--open" : ""}`}>▾</span>
      </button>
      {open && (
        <div className="market-expand">
          <GoalRangeChart probabilities={probabilities} odds={odds} totalGoals={totalGoals} />
        </div>
      )}
    </div>
  );
}

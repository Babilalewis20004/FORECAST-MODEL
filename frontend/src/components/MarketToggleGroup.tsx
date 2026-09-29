import { useState } from "react";

export interface ToggleOption {
  key: string;
  label: string;
  prob: number; // 0..1
  odds?: number | null;
}

interface Props {
  // Short chip text for the collapsed row ("1X2", "BTTS", "O/U 1.5") --
  // doubles as the market's only label now that there's no separate heading.
  chipLabel: string;
  options: ToggleOption[];
  selected: string | null;
  onSelect: (key: string) => void;
  // Key of the option the actual final result matched, for a finished
  // fixture -- null/undefined for an unplayed one. Independent of `selected`
  // (the user's own tap), so a correct call stays marked green whether or
  // not the user picked it.
  correctKey?: string | null;
}

// Collapsed by default (row shows the model's favourite, matching
// CorrectScoreGrid's header pattern below) -- tap to expand into the full
// set of tappable picks. Tapping an option again clears it. Pure UI state
// (no bet is placed); it just highlights a pick the way the goal-range
// chart already highlights this fixture's own numbers.
export function MarketToggleGroup({ chipLabel, options, selected, onSelect, correctKey }: Props) {
  const [open, setOpen] = useState(false);

  // The model's own pick for this market -- highest probability among the
  // options actually shown, independent of user selection or the actual
  // result. Always defined since every group has at least one option.
  const favouredKey = options.reduce((best, opt) => (opt.prob > best.prob ? opt : best), options[0]).key;
  const activeKey = selected ?? favouredKey;
  const active = options.find((o) => o.key === activeKey)!;
  const headerCorrect = correctKey != null && activeKey === correctKey;
  const headerFavoured = activeKey === favouredKey;

  return (
    <div className="market">
      <button
        type="button"
        className={`market-row${headerCorrect ? " market-row--correct" : ""}`}
        onClick={() => setOpen((v) => !v)}
      >
        {headerFavoured && <span className="favoured-badge">Favoured</span>}
        <span className="market-row__label">{chipLabel}</span>
        <span className="market-row__pick">{active.label}</span>
        <span className="market-row__right">
          <span className="market-row__odds">{active.odds != null ? active.odds.toFixed(2) : "—"}</span>
          <span className="market-row__pct">{Math.round(active.prob * 100)}%</span>
        </span>
        <span className={`chev${open ? " chev--open" : ""}`}>▾</span>
      </button>
      {open && (
        <div className="market-toggles" style={{ gridTemplateColumns: `repeat(${options.length}, 1fr)` }}>
          {options.map((opt) => {
            const isSelected = selected === opt.key;
            const isCorrect = correctKey === opt.key;
            const isFavoured = opt.key === favouredKey;
            return (
              <button
                key={opt.key}
                type="button"
                className={`market-toggle${isSelected ? " market-toggle--selected" : ""}${isCorrect ? " market-toggle--correct" : ""}`}
                onClick={() => onSelect(opt.key)}
              >
                {isFavoured && <span className="favoured-badge">Favoured</span>}
                <span className="market-toggle__label">{opt.label}</span>
                <span className="market-toggle__odds">{opt.odds != null ? opt.odds.toFixed(2) : "—"}</span>
                <span className="market-toggle__pct">{Math.round(opt.prob * 100)}%</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

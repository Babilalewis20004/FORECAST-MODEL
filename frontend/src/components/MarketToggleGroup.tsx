export interface ToggleOption {
  key: string;
  label: string;
  prob: number; // 0..1
  odds?: number | null;
}

interface Props {
  title: string;
  options: ToggleOption[];
  selected: string | null;
  onSelect: (key: string) => void;
  // Key of the option the actual final result matched, for a finished
  // fixture -- null/undefined for an unplayed one. Independent of `selected`
  // (the user's own tap), so a correct call stays marked green whether or
  // not the user picked it.
  correctKey?: string | null;
}

// A mutually-exclusive set of tappable picks (1X2 / BTTS / one Over-Under
// line) -- tapping the selected option again clears it. Pure UI state (no
// bet is placed); it just highlights a pick the way the goal-range chart
// already highlights this fixture's own numbers.
export function MarketToggleGroup({ title, options, selected, onSelect, correctKey }: Props) {
  // The model's own pick for this market -- highest probability among the
  // options actually shown, independent of user selection or the actual
  // result. Always defined since every group has at least one option.
  const favouredKey = options.reduce((best, opt) => (opt.prob > best.prob ? opt : best), options[0]).key;

  return (
    <div className="market">
      <div className="market__title">{title}</div>
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
    </div>
  );
}

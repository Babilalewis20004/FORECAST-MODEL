import { RANGE_LABELS } from "../types";
import type { RangeLabel, RangeOdds } from "../types";

interface Props {
  probabilities: Record<RangeLabel, number>;
  odds?: RangeOdds;
  // Actual total goals for a finished fixture -- bars whose range contains it
  // turn green instead of the usual blue, same "did this call land" signal
  // as the market toggles and correct-score grid.
  totalGoals?: number | null;
}

function inRange(label: RangeLabel, total: number): boolean {
  const [min, max] = label.split("-").map(Number);
  return total >= min && total <= max;
}

const WIDTH = 320;
const HEIGHT = 132;
const BAR_GAP = 6;
const BAR_WIDTH = (WIDTH - BAR_GAP * (RANGE_LABELS.length - 1)) / RANGE_LABELS.length;
const LABEL_H = 16;
const ODDS_H = 12;
const VALUE_H = 14;
const PLOT_H = HEIGHT - LABEL_H - ODDS_H - VALUE_H;

// Single series (one fixture's probability per goal range) -- sequential
// blue, no legend needed. Value at the cap, per the dataviz mark spec.
// Odds (1/probability) ride along as a second, muted label under the range
// label -- both are already "label every point" exceptions, justified here
// because all 8 ranges are simultaneously the point of this chart, not one
// extreme among many.
export function GoalRangeChart({ probabilities, odds, totalGoals }: Props) {
  return (
    <svg
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      width="100%"
      height={HEIGHT}
      role="img"
      aria-label="Goal-range probabilities and odds for this fixture"
    >
      <line
        x1={0}
        y1={VALUE_H + PLOT_H}
        x2={WIDTH}
        y2={VALUE_H + PLOT_H}
        stroke="var(--baseline)"
        strokeWidth={1}
      />
      {RANGE_LABELS.map((label, i) => {
        const p = probabilities[label] ?? 0;
        const o = odds?.[label];
        const barH = Math.max(2, p * PLOT_H);
        const x = i * (BAR_WIDTH + BAR_GAP);
        const y = VALUE_H + PLOT_H - barH;
        const correct = totalGoals != null && inRange(label, totalGoals);
        return (
          <g key={label}>
            <title>{`${label} goals: ${(p * 100).toFixed(1)}%${o ? ` (odds ${o.toFixed(2)})` : ""}`}</title>
            <text
              x={x + BAR_WIDTH / 2}
              y={VALUE_H - 4}
              textAnchor="middle"
              fontSize={10}
              fill="var(--text-secondary)"
            >
              {Math.round(p * 100)}%
            </text>
            <rect x={x} y={y} width={BAR_WIDTH} height={barH} rx={4} fill={correct ? "var(--success)" : "var(--series-1)"} />
            <text
              x={x + BAR_WIDTH / 2}
              y={HEIGHT - ODDS_H - 2}
              textAnchor="middle"
              fontSize={10}
              fill="var(--text-muted)"
            >
              {label}
            </text>
            {o != null && (
              <text
                x={x + BAR_WIDTH / 2}
                y={HEIGHT - 2}
                textAnchor="middle"
                fontSize={9}
                fill="var(--text-muted)"
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                {o.toFixed(2)}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

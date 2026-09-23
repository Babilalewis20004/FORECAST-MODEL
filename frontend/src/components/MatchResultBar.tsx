import type { MatchResult, MatchResultOdds } from "../types";

interface Props {
  result: MatchResult;
  odds?: MatchResultOdds;
  homeTeam: string;
  awayTeam: string;
}

const SEGMENTS: Array<{ key: keyof MatchResult; label: string; color: string }> = [
  { key: "home_win", label: "Home", color: "var(--series-1)" },
  { key: "draw", label: "Draw", color: "var(--series-2)" },
  { key: "away_win", label: "Away", color: "var(--series-3)" },
];

const WIDTH = 320;
const HEIGHT = 24;
const GAP = 2;

// 1X2: a single stacked bar (3 categorical series, fixed hue order) plus a
// legend row -- legend is mandatory at >=2 series per the dataviz method,
// and doubles as the "value stays visible even if a segment is too
// narrow to label inline" fallback.
export function MatchResultBar({ result, odds, homeTeam, awayTeam }: Props) {
  let x = 0;
  const segments = SEGMENTS.map((seg) => {
    const p = result[seg.key] ?? 0;
    const w = Math.max(0, p * WIDTH - GAP);
    const rect = { ...seg, p, x, w };
    x += p * WIDTH;
    return rect;
  });

  return (
    <div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} width="100%" height={HEIGHT} role="img" aria-label="Match result probabilities">
        {segments.map((seg) => (
          <g key={seg.key}>
            <title>{`${seg.label}: ${(seg.p * 100).toFixed(1)}%${odds?.[seg.key] ? ` (odds ${odds[seg.key]!.toFixed(2)})` : ""}`}</title>
            {/* Each segment is separated from its neighbours by the 2px gap
                baked into its width, so rounding all four corners (rather
                than only the outer bar ends) reads cleanly without needing
                a clip-path per segment. */}
            <rect x={seg.x} y={0} width={seg.w} height={HEIGHT} fill={seg.color} rx={4} ry={4} />
            {seg.w > 28 && (
              <text
                x={seg.x + seg.w / 2}
                y={HEIGHT / 2 + 4}
                textAnchor="middle"
                fontSize={11}
                fill="#fff"
                fontWeight={600}
              >
                {Math.round(seg.p * 100)}%
              </text>
            )}
          </g>
        ))}
      </svg>
      <div style={{ display: "flex", gap: 14, marginTop: 4, fontSize: 11, color: "var(--text-secondary)" }}>
        {segments.map((seg, i) => (
          <span key={seg.key} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
            <span
              style={{ width: 8, height: 8, borderRadius: 2, background: seg.color, display: "inline-block" }}
            />
            {i === 0 ? homeTeam : i === 2 ? awayTeam : seg.label} {(seg.p * 100).toFixed(0)}%
            {odds?.[seg.key] != null && (
              <span style={{ color: "var(--text-muted)" }}>({odds[seg.key]!.toFixed(2)})</span>
            )}
          </span>
        ))}
      </div>
    </div>
  );
}

interface Props {
  label: string;
  value: number; // 0..1
  valueLabel: string;
  oddsLabel?: string;
}

// Meter figure: fill = the value, unfilled track = a lighter step of the
// same ramp (single sequential hue), per the dataviz "meter" spec -- used
// for BTTS and each Over/Under line, where a single probability against
// its complement (not an identity comparison) is the story.
export function MiniMeter({ label, value, valueLabel, oddsLabel }: Props) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
      <span style={{ width: 62, color: "var(--text-secondary)", flexShrink: 0 }}>{label}</span>
      <div
        style={{
          flex: 1,
          height: 10,
          borderRadius: 5,
          background: "color-mix(in srgb, var(--series-1) 18%, var(--surface-1))",
          overflow: "hidden",
        }}
      >
        <div
          style={{
            width: `${Math.round(value * 100)}%`,
            height: "100%",
            background: "var(--series-1)",
            borderRadius: 5,
          }}
        />
      </div>
      <span style={{ width: 40, textAlign: "right", color: "var(--text-primary)", fontWeight: 600, flexShrink: 0 }}>
        {valueLabel}
      </span>
      {oddsLabel && (
        <span style={{ width: 36, textAlign: "right", color: "var(--text-muted)", flexShrink: 0 }}>{oddsLabel}</span>
      )}
    </div>
  );
}

import type { CSSProperties } from "react";
import type { Prediction } from "../types";

const SHOWN_MARKETS: Array<{ key: string; label: string }> = [
  { key: "match_result", label: "1X2" },
  { key: "btts", label: "BTTS" },
  { key: "over_2.5", label: "O2.5" },
];

// Brier skill score: 1 - Brier(model) / Brier(base rate). Positive = the
// calibrated ensemble beat "always predict the league's historical base
// rate" on held-out matches; negative = it didn't. Measured per league on
// the test split -- a statement about the model, not about this fixture.
function skill(brier: number | null, baseline: number | null): number | null {
  if (brier == null || baseline == null || baseline === 0) return null;
  return 1 - brier / baseline;
}

function fmtSkill(s: number): string {
  const pct = (s * 100).toFixed(1);
  return s >= 0 ? `+${pct}%` : `−${pct.replace("-", "")}%`;
}

export function CalibrationTag({ prediction }: { prediction: Prediction }) {
  const scores = prediction.brier_scores;

  if (!scores) {
    return (
      <div style={tagRow}>
        <span style={pill} title="Predicted before the ensemble shipped -- retrain this league for blended, calibrated probabilities">
          Uncalibrated · Model A only
        </span>
      </div>
    );
  }

  const parts = SHOWN_MARKETS.flatMap(({ key, label }) => {
    const b = scores[key];
    const s = b ? skill(b.calibrated, b.baseline) : null;
    return s == null ? [] : [`${label} ${fmtSkill(s)}`];
  });

  const tooltip = SHOWN_MARKETS.map(({ key, label }) => {
    const b = scores[key];
    if (!b) return null;
    const w = prediction.ensemble_weights?.[key];
    return (
      `${label}: Brier ${b.calibrated?.toFixed(4)} calibrated vs ${b.baseline?.toFixed(4)} base rate` +
      ` (Model A ${b.model_a?.toFixed(4)}, Model B ${b.model_b?.toFixed(4)})` +
      (w != null ? ` · weight A ${w.toFixed(2)} / B ${(1 - w).toFixed(2)}` : "")
    );
  })
    .filter(Boolean)
    .join("\n");

  return (
    <div style={tagRow}>
      <span style={pill} title={`Held-out test split for this league. Lower Brier is better.\n${tooltip}`}>
        Calibrated · skill vs base rate: {parts.join(" · ")}
      </span>
    </div>
  );
}

const tagRow: CSSProperties = { marginTop: 10 };
const pill: CSSProperties = {
  display: "inline-block",
  fontSize: 11,
  padding: "2px 8px",
  borderRadius: 999,
  border: "1px solid var(--border)",
  color: "var(--text-secondary)",
  background: "var(--page)",
  cursor: "help",
};

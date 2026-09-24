import type { Prediction } from "../types";

function selectionLabel(key: string, homeTeam: string, awayTeam: string): string {
  if (key === "home_win") return homeTeam;
  if (key === "away_win") return awayTeam;
  if (key === "draw") return "Draw";
  if (key === "btts_yes") return "BTTS Yes";
  if (key === "btts_no") return "BTTS No";
  const [side, line] = key.split("_");
  return `${side === "over" ? "Over" : "Under"} ${line}`;
}

interface Props {
  prediction: Prediction;
  homeTeam: string;
  awayTeam: string;
}

export function ValueBadge({ prediction, homeTeam, awayTeam }: Props) {
  const { is_value_bet, value_selection, expected_value, recommended_stake, bookmaker_odds } = prediction;
  if (!is_value_bet || !value_selection || !expected_value) return null;

  const ev = expected_value[value_selection];
  const odds = bookmaker_odds?.[value_selection];
  const title =
    `Model edge vs. ${prediction.odds_bookmaker ?? "bookmaker"} odds` +
    (prediction.odds_fetched_at ? ` fetched ${new Date(prediction.odds_fetched_at).toLocaleString()}` : "") +
    ". Stake is quarter-Kelly; 1 unit = 1% of bankroll.";

  return (
    <div className="value-badge" title={title}>
      <span className="value-badge__ev">+{(ev * 100).toFixed(1)}% EV</span>
      <span className="value-badge__pick">
        {selectionLabel(value_selection, homeTeam, awayTeam)}
        {odds ? ` @ ${odds.toFixed(2)}` : ""}
      </span>
      {recommended_stake != null && (
        <span className="value-badge__stake">Stake: {recommended_stake.toFixed(2)} units</span>
      )}
    </div>
  );
}

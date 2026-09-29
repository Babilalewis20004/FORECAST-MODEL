import { useState } from "react";
import { OU_LINES } from "../types";
import type { Btts, BttsOdds, MatchResult, MatchResultOdds, OverUnder, OverUnderOdds } from "../types";
import { MarketToggleGroup } from "./MarketToggleGroup";
import type { ToggleOption } from "./MarketToggleGroup";

interface Props {
  matchResult: MatchResult;
  overUnder: OverUnder;
  btts: Btts;
  matchResultOdds?: MatchResultOdds | null;
  overUnderOdds?: OverUnderOdds | null;
  bttsOdds?: BttsOdds | null;
  homeTeam: string;
  awayTeam: string;
  // Actual final score for a finished fixture -- undefined/null for one not
  // yet played. Drives the green "this call landed" highlight, independent
  // of whatever the user has tapped.
  result?: { home: number; away: number } | null;
}

export function MarketsPanel({
  matchResult,
  overUnder,
  btts,
  matchResultOdds,
  overUnderOdds,
  bttsOdds,
  homeTeam,
  awayTeam,
  result,
}: Props) {
  // One selected key per group ("x12", "btts", "ou_1.5", ...); tapping the
  // same key again clears that group (see MarketToggleGroup).
  const [selections, setSelections] = useState<Record<string, string | null>>({});

  const toggle = (group: string, key: string) => {
    setSelections((prev) => ({ ...prev, [group]: prev[group] === key ? null : key }));
  };

  const x12Options: ToggleOption[] = [
    { key: "home_win", label: homeTeam, prob: matchResult.home_win, odds: matchResultOdds?.home_win },
    { key: "draw", label: "Draw", prob: matchResult.draw, odds: matchResultOdds?.draw },
    { key: "away_win", label: awayTeam, prob: matchResult.away_win, odds: matchResultOdds?.away_win },
  ];

  const bttsOptions: ToggleOption[] = [
    { key: "yes", label: "Yes", prob: btts.yes, odds: bttsOdds?.yes },
    { key: "no", label: "No", prob: btts.no, odds: bttsOdds?.no },
  ];

  const totalGoals = result ? result.home + result.away : null;
  const x12Correct = result ? (result.home > result.away ? "home_win" : result.away > result.home ? "away_win" : "draw") : null;
  const bttsCorrect = result ? (result.home > 0 && result.away > 0 ? "yes" : "no") : null;

  return (
    <div className="markets-panel">
      <MarketToggleGroup
        chipLabel="1X2"
        options={x12Options}
        selected={selections.x12 ?? null}
        onSelect={(k) => toggle("x12", k)}
        correctKey={x12Correct}
      />
      <MarketToggleGroup
        chipLabel="BTTS"
        options={bttsOptions}
        selected={selections.btts ?? null}
        onSelect={(k) => toggle("btts", k)}
        correctKey={bttsCorrect}
      />
      {OU_LINES.map((line) => {
        const overKey = `over_${line}`;
        const underKey = `under_${line}`;
        const over = overUnder[overKey];
        const under = overUnder[underKey];
        if (over === undefined || under === undefined) return null;
        const group = `ou_${line}`;
        const options: ToggleOption[] = [
          { key: overKey, label: `Over ${line}`, prob: over, odds: overUnderOdds?.[overKey] },
          { key: underKey, label: `Under ${line}`, prob: under, odds: overUnderOdds?.[underKey] },
        ];
        const ouCorrect = totalGoals != null ? (totalGoals > line ? overKey : underKey) : null;
        return (
          <MarketToggleGroup
            key={line}
            chipLabel={`O/U ${line}`}
            options={options}
            selected={selections[group] ?? null}
            onSelect={(k) => toggle(group, k)}
            correctKey={ouCorrect}
          />
        );
      })}
    </div>
  );
}

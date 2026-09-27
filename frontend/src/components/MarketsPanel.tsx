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

  return (
    <div className="markets-panel">
      <MarketToggleGroup
        title="1X2 Market"
        options={x12Options}
        selected={selections.x12 ?? null}
        onSelect={(k) => toggle("x12", k)}
      />
      <MarketToggleGroup
        title="Both Teams to Score"
        options={bttsOptions}
        selected={selections.btts ?? null}
        onSelect={(k) => toggle("btts", k)}
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
        return (
          <MarketToggleGroup
            key={line}
            title={`Over / Under ${line} Goals`}
            options={options}
            selected={selections[group] ?? null}
            onSelect={(k) => toggle(group, k)}
          />
        );
      })}
    </div>
  );
}

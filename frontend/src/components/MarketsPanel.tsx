import { OU_LINES } from "../types";
import type { Btts, BttsOdds, MatchResult, MatchResultOdds, OverUnder, OverUnderOdds } from "../types";
import { MatchResultBar } from "./MatchResultBar";
import { MiniMeter } from "./MiniMeter";

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

const fmtOdds = (o: number | null | undefined) => (o != null ? o.toFixed(2) : undefined);

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
  return (
    <div style={{ marginTop: 12, paddingTop: 12, borderTop: "1px solid var(--border)" }}>
      <MatchResultBar result={matchResult} odds={matchResultOdds ?? undefined} homeTeam={homeTeam} awayTeam={awayTeam} />

      <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 6 }}>
        {OU_LINES.map((line) => {
          const over = overUnder[`over_${line}`];
          if (over === undefined) return null;
          return (
            <MiniMeter
              key={line}
              label={`Over ${line}`}
              value={over}
              valueLabel={`${Math.round(over * 100)}%`}
              oddsLabel={fmtOdds(overUnderOdds?.[`over_${line}`])}
            />
          );
        })}
        <MiniMeter
          label="BTTS Yes"
          value={btts.yes}
          valueLabel={`${Math.round(btts.yes * 100)}%`}
          oddsLabel={fmtOdds(bttsOdds?.yes)}
        />
      </div>
    </div>
  );
}

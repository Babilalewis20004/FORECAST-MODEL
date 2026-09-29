import { useState } from "react";
import { OU_LINES, TEAM_TOTAL_LINES } from "../types";
import type { Btts, BttsOdds, MatchResult, MatchResultOdds, OverUnder, OverUnderOdds } from "../types";
import { useBetslip } from "../betslip/BetslipContext";
import { MarketToggleGroup } from "./MarketToggleGroup";
import type { ToggleOption } from "./MarketToggleGroup";

interface Props {
  matchResult: MatchResult;
  overUnder: OverUnder;
  btts: Btts;
  matchResultOdds?: MatchResultOdds | null;
  overUnderOdds?: OverUnderOdds | null;
  bttsOdds?: BttsOdds | null;
  // Derived markets (double chance, team totals, BTTS combos, odd/even) --
  // see Prediction.extra_markets. Anything missing (e.g. everything but
  // double chance on a not-yet-retrained prediction) is simply not shown.
  extraMarkets?: Record<string, Record<string, number>> | null;
  extraMarketsOdds?: Record<string, Record<string, number | null>> | null;
  homeTeam: string;
  awayTeam: string;
  // Identify the fixture on the betslip.
  fixtureKey: string;
  league: string;
  date: string;
  // Flat outcome keys (home_win, btts_yes, over_2.5, ...) -- preferred over
  // the model's fair odds when a pick goes on the slip.
  bookmakerOdds?: Record<string, number> | null;
  // Actual final score for a finished fixture -- undefined/null for one not
  // yet played. Drives the green "this call landed" highlight, independent
  // of whatever the user has tapped.
  result?: { home: number; away: number } | null;
}

interface MarketDef {
  group: string;
  chipLabel: string;
  options: ToggleOption[];
  correctKey: string | string[] | null;
}

export function MarketsPanel({
  matchResult,
  overUnder,
  btts,
  matchResultOdds,
  overUnderOdds,
  bttsOdds,
  extraMarkets,
  extraMarketsOdds,
  homeTeam,
  awayTeam,
  fixtureKey,
  league,
  date,
  bookmakerOdds,
  result,
}: Props) {
  const slip = useBetslip();
  const [showMore, setShowMore] = useState(false);
  // A finished fixture can't be bet on, so its taps stay local highlights
  // (one key per group, tapping again clears) instead of going on the slip.
  const [localSelections, setLocalSelections] = useState<Record<string, string | null>>({});

  const selectedIn = (group: string): string | null =>
    result ? (localSelections[group] ?? null) : (slip.find(`${fixtureKey}|${group}`)?.optionKey ?? null);

  const toggle = (def: MarketDef, key: string) => {
    const opt = def.options.find((o) => o.key === key)!;
    if (result) {
      setLocalSelections((prev) => ({ ...prev, [def.group]: prev[def.group] === key ? null : key }));
      return;
    }
    const bookOdds = opt.bookKey ? bookmakerOdds?.[opt.bookKey] : undefined;
    slip.toggle({
      id: `${fixtureKey}|${def.group}`,
      fixtureKey,
      group: def.group,
      optionKey: key,
      homeTeam,
      awayTeam,
      league,
      date,
      market: def.chipLabel,
      pick: opt.label,
      odds: bookOdds ?? opt.odds ?? null,
      oddsSource: bookOdds != null ? "bookmaker" : "fair",
      prob: opt.prob,
    });
  };

  const hg = result?.home;
  const ag = result?.away;
  const totalGoals = result ? result.home + result.away : null;
  const x12Result = result ? (hg! > ag! ? "home_win" : ag! > hg! ? "away_win" : "draw") : null;
  const bothScored = result ? hg! > 0 && ag! > 0 : null;

  // ---- core markets ------------------------------------------------------
  const core: MarketDef[] = [
    {
      group: "x12",
      chipLabel: "1X2",
      options: [
        { key: "home_win", label: homeTeam, prob: matchResult.home_win, odds: matchResultOdds?.home_win, bookKey: "home_win" },
        { key: "draw", label: "Draw", prob: matchResult.draw, odds: matchResultOdds?.draw, bookKey: "draw" },
        { key: "away_win", label: awayTeam, prob: matchResult.away_win, odds: matchResultOdds?.away_win, bookKey: "away_win" },
      ],
      correctKey: x12Result,
    },
    {
      group: "btts",
      chipLabel: "BTTS",
      options: [
        { key: "yes", label: "Yes", prob: btts.yes, odds: bttsOdds?.yes, bookKey: "btts_yes" },
        { key: "no", label: "No", prob: btts.no, odds: bttsOdds?.no, bookKey: "btts_no" },
      ],
      correctKey: bothScored == null ? null : bothScored ? "yes" : "no",
    },
  ];
  for (const line of OU_LINES) {
    const overKey = `over_${line}`;
    const underKey = `under_${line}`;
    if (overUnder[overKey] === undefined || overUnder[underKey] === undefined) continue;
    core.push({
      group: `ou_${line}`,
      chipLabel: `O/U ${line}`,
      options: [
        { key: overKey, label: `Over ${line}`, prob: overUnder[overKey], odds: overUnderOdds?.[overKey], bookKey: overKey },
        { key: underKey, label: `Under ${line}`, prob: overUnder[underKey], odds: overUnderOdds?.[underKey], bookKey: underKey },
      ],
      correctKey: totalGoals != null ? (totalGoals > line ? overKey : underKey) : null,
    });
  }

  // ---- derived markets ---------------------------------------------------
  // Builds a def from a category's keys, skipping any option the prediction
  // doesn't have (older rows) and the whole market if fewer than two remain.
  const extra: MarketDef[] = [];
  const addExtra = (
    category: string,
    group: string,
    chipLabel: string,
    spec: [key: string, label: string][],
    correctKey: string | string[] | null,
  ) => {
    const probs = extraMarkets?.[category];
    if (!probs) return;
    const options = spec
      .filter(([k]) => probs[k] !== undefined)
      .map(([k, label]): ToggleOption => ({ key: k, label, prob: probs[k], odds: extraMarketsOdds?.[category]?.[k] }));
    if (options.length >= 2) extra.push({ group, chipLabel, options, correctKey });
  };

  addExtra(
    "double_chance",
    "double_chance",
    "Double chance",
    [
      ["home_draw", `${homeTeam} or draw`],
      ["draw_away", `${awayTeam} or draw`],
      ["home_away", "No draw"],
    ],
    // Two of the three always land: the two that include the actual result.
    x12Result == null
      ? null
      : x12Result === "home_win"
        ? ["home_draw", "home_away"]
        : x12Result === "away_win"
          ? ["draw_away", "home_away"]
          : ["home_draw", "draw_away"],
  );

  for (const [side, name] of [
    ["home", "Home"],
    ["away", "Away"],
  ] as const) {
    const goals = side === "home" ? hg : ag;
    for (const line of TEAM_TOTAL_LINES) {
      addExtra(
        "team_totals",
        `tt_${side}_${line}`,
        `${name} team O/U ${line}`,
        [
          [`${side}_over_${line}`, `Over ${line}`],
          [`${side}_under_${line}`, `Under ${line}`],
        ],
        goals == null ? null : `${side}_${goals > line ? "over" : "under"}_${line}`,
      );
    }
  }

  for (const [suffix, label] of [
    ["yes", "BTTS Yes"],
    ["no", "BTTS No"],
  ] as const) {
    addExtra(
      "x12_btts",
      `x12_btts_${suffix}`,
      `1X2 & ${label}`,
      [
        [`home_win_btts_${suffix}`, homeTeam],
        [`draw_btts_${suffix}`, "Draw"],
        [`away_win_btts_${suffix}`, awayTeam],
      ],
      x12Result == null || bothScored == null ? null : bothScored === (suffix === "yes") ? `${x12Result}_btts_${suffix}` : null,
    );
  }

  for (const line of OU_LINES) {
    addExtra(
      "ou_btts",
      `ou_btts_${line}`,
      `O/U ${line} & BTTS`,
      [
        [`over_${line}_btts_yes`, "Over & Yes"],
        [`over_${line}_btts_no`, "Over & No"],
        [`under_${line}_btts_yes`, "Under & Yes"],
        [`under_${line}_btts_no`, "Under & No"],
      ],
      totalGoals == null || bothScored == null
        ? null
        : `${totalGoals > line ? "over" : "under"}_${line}_btts_${bothScored ? "yes" : "no"}`,
    );
  }

  addExtra(
    "odd_even",
    "odd_even",
    "Total goals odd/even",
    [
      ["odd", "Odd"],
      ["even", "Even"],
    ],
    totalGoals == null ? null : totalGoals % 2 === 1 ? "odd" : "even",
  );

  const render = (def: MarketDef) => (
    <MarketToggleGroup
      key={def.group}
      chipLabel={def.chipLabel}
      options={def.options}
      selected={selectedIn(def.group)}
      onSelect={(k) => toggle(def, k)}
      correctKey={def.correctKey}
    />
  );

  return (
    <div className="markets-panel">
      {core.map(render)}
      {extra.length > 0 && (
        <>
          <button type="button" className="more-markets" onClick={() => setShowMore((v) => !v)}>
            {showMore ? "Hide" : "More markets"} ({extra.length})
            <span className={`chev${showMore ? " chev--open" : ""}`}>▾</span>
          </button>
          {showMore && extra.map(render)}
        </>
      )}
    </div>
  );
}

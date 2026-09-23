export const RANGE_LABELS = ["1-3", "1-4", "2-3", "2-4", "2-5", "2-6", "3-5", "3-6"] as const;
export type RangeLabel = (typeof RANGE_LABELS)[number];

export const OU_LINES = [1.5, 2.5, 3.5] as const;

export interface League {
  league: string;
  tier: "rich" | "lower";
  last_trained_at: string | null;
  n_matches: number | null;
}

export interface MatchResult {
  home_win: number;
  draw: number;
  away_win: number;
}

// Keyed "over_1.5"/"under_1.5"/"over_2.5"/... -- see OU_LINES.
export type OverUnder = Record<string, number>;

export interface Btts {
  yes: number;
  no: number;
}

// Decimal odds (1/probability, no bookmaker margin), null where the
// probability was exactly 0. Same key shapes as their probability
// counterparts above.
export type MatchResultOdds = Record<keyof MatchResult, number | null>;
export type OverUnderOdds = Record<string, number | null>;
export type BttsOdds = Record<keyof Btts, number | null>;
export type RangeOdds = Record<RangeLabel, number | null>;

export interface Prediction {
  lambda_home: number;
  lambda_away: number;
  predicted_at: string;
  range_probabilities: Record<RangeLabel, number>;
  range_odds: RangeOdds;
  // Null on predictions generated before these markets shipped, until the
  // next retrain.
  match_result: MatchResult | null;
  over_under: OverUnder | null;
  btts: Btts | null;
  match_result_odds: MatchResultOdds | null;
  over_under_odds: OverUnderOdds | null;
  btts_odds: BttsOdds | null;
}

export interface Fixture {
  fixture_id: number;
  date: string;
  league: string;
  home_team: string;
  away_team: string;
  status: string;
  prediction: Prediction | null;
}

export interface Job {
  job_id: string;
  type: "scrape" | "retrain" | "full";
  status: "queued" | "running" | "success" | "failed";
  started_at: string | null;
  finished_at: string | null;
  detail: string | null;
  error: string | null;
}

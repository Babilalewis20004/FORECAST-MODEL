export const RANGE_LABELS = ["1-3", "1-4", "2-3", "2-4", "2-5", "2-6", "3-5", "3-6"] as const;
export type RangeLabel = (typeof RANGE_LABELS)[number];

export const OU_LINES = [1.5, 2.5, 3.5] as const;

export interface League {
  league: string;
  tier: "rich" | "lower";
  // The trained model serving this competition: the league itself for
  // domestic leagues, "International" for every national-team competition.
  model_group: string;
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

export interface MarketSet {
  match_result?: MatchResult;
  over_under?: OverUnder;
  btts?: Btts;
  goal_ranges?: Record<RangeLabel, number>;
}

export interface RawProbabilities {
  model_a: MarketSet;
  model_b?: MarketSet;
  blended: MarketSet;
  // Present when the fixture had odds: calibrated probabilities before
  // shrinkage toward the de-vigged market, and that market.
  calibrated_unshrunk?: MarketSet;
  market?: MarketSet;
}

// Test-split Brier score for a market, per source. Lower is better;
// "baseline" is always predicting the training base rate.
export interface BrierScores {
  model_a: number | null;
  model_b: number | null;
  blended: number | null;
  calibrated: number | null;
  baseline: number | null;
}

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
  // Ensemble provenance -- null on Model-A-only predictions made before the
  // ensemble shipped. brier_scores is per league (keyed by market, e.g.
  // "match_result", "btts", "over_2.5"), not per fixture.
  raw_probabilities: RawProbabilities | null;
  calibrated_probabilities: MarketSet | null;
  ensemble_weights: Record<string, number | null> | null;
  brier_scores: Record<string, BrierScores> | null;
  // Value layer -- null until bookmaker odds have been fetched for the
  // fixture. Flat outcome keys: home_win, draw, away_win, over_2.5,
  // under_2.5, btts_yes, btts_no, ...
  bookmaker_odds: Record<string, number> | null;
  odds_bookmaker: string | null;
  odds_fetched_at: string | null;
  market_probabilities: Record<string, number> | null;
  expected_value: Record<string, number> | null; // 0.052 = +5.2% EV
  is_value_bet: boolean | null;
  value_selection: string | null;
  recommended_stake: number | null; // quarter-Kelly, units (1 unit = 1% of bankroll)
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

import type { Fixture } from "../types";
import { CalibrationTag } from "./CalibrationTag";
import { CorrectScoreGrid } from "./CorrectScoreGrid";
import { GoalRangeChart } from "./GoalRangeChart";
import { MarketsPanel } from "./MarketsPanel";
import { TeamForm } from "./TeamForm";
import { ValueBadge } from "./ValueBadge";
import "./FixtureCard.css";

export function FixtureCard({ fixture }: { fixture: Fixture }) {
  const prediction = fixture.prediction;
  const hasMarkets = prediction?.match_result && prediction?.over_under && prediction?.btts;
  const isFinished = fixture.home_goals != null && fixture.away_goals != null;
  const result = isFinished ? { home: fixture.home_goals!, away: fixture.away_goals! } : null;

  return (
    <article className="fixture-card">
      <header className="fixture-card__header">
        <span className="fixture-card__league">{fixture.league}</span>
        <span className="fixture-card__date-group">
          {isFinished && (
            <span className="fixture-card__final">
              Final: {fixture.home_goals}–{fixture.away_goals}
            </span>
          )}
          <span className="fixture-card__date">{fixture.date}</span>
        </span>
      </header>
      <h3 className="fixture-card__teams">
        {fixture.home_team} <span className="fixture-card__vs">vs</span> {fixture.away_team}
      </h3>
      {prediction ? (
        <>
          <ValueBadge prediction={prediction} homeTeam={fixture.home_team} awayTeam={fixture.away_team} />
          <p className="fixture-card__lambdas">
            Expected goals: <strong>{prediction.lambda_home.toFixed(2)}</strong> :{" "}
            <strong>{prediction.lambda_away.toFixed(2)}</strong>
          </p>
          <GoalRangeChart
            probabilities={prediction.range_probabilities}
            odds={prediction.range_odds}
            totalGoals={result ? result.home + result.away : null}
          />
          {hasMarkets && (
            <MarketsPanel
              matchResult={prediction.match_result!}
              overUnder={prediction.over_under!}
              btts={prediction.btts!}
              matchResultOdds={prediction.match_result_odds}
              overUnderOdds={prediction.over_under_odds}
              bttsOdds={prediction.btts_odds}
              homeTeam={fixture.home_team}
              awayTeam={fixture.away_team}
              result={result}
            />
          )}
          <CorrectScoreGrid
            correctScore={prediction.correct_score}
            correctScoreOdds={prediction.correct_score_odds}
            homeTeam={fixture.home_team}
            awayTeam={fixture.away_team}
            result={result}
          />
          <TeamForm homeTeam={fixture.home_team} awayTeam={fixture.away_team} />
          <CalibrationTag prediction={prediction} />
        </>
      ) : (
        <p className="fixture-card__no-prediction">
          {isFinished ? "No prediction was recorded for this match." : "No prediction yet — trigger a retrain."}
        </p>
      )}
    </article>
  );
}

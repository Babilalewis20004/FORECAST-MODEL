import type { Fixture } from "../types";
import { GoalRangeChart } from "./GoalRangeChart";
import { MarketsPanel } from "./MarketsPanel";
import "./FixtureCard.css";

export function FixtureCard({ fixture }: { fixture: Fixture }) {
  const prediction = fixture.prediction;
  const hasMarkets = prediction?.match_result && prediction?.over_under && prediction?.btts;

  return (
    <article className="fixture-card">
      <header className="fixture-card__header">
        <span className="fixture-card__league">{fixture.league}</span>
        <span className="fixture-card__date">{fixture.date}</span>
      </header>
      <h3 className="fixture-card__teams">
        {fixture.home_team} <span className="fixture-card__vs">vs</span> {fixture.away_team}
      </h3>
      {prediction ? (
        <>
          <p className="fixture-card__lambdas">
            Expected goals: <strong>{prediction.lambda_home.toFixed(2)}</strong> :{" "}
            <strong>{prediction.lambda_away.toFixed(2)}</strong>
          </p>
          <GoalRangeChart probabilities={prediction.range_probabilities} odds={prediction.range_odds} />
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
            />
          )}
        </>
      ) : (
        <p className="fixture-card__no-prediction">No prediction yet — trigger a retrain.</p>
      )}
    </article>
  );
}

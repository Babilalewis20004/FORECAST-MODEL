import { useBetslip } from "./BetslipContext";
import "./Betslip.css";

const pct = (x: number) => `${(x * 100).toFixed(x < 0.1 ? 2 : 1)}%`;

export function BetslipPage() {
  const { selections, stake, setStake, remove, clear } = useBetslip();

  if (selections.length === 0) {
    return (
      <p className="slip__empty">
        Your betslip is empty. Open a fixture on the Fixtures tab, expand a market and tap a pick to add it here.
      </p>
    );
  }

  const missingOdds = selections.filter((s) => s.odds == null).length;
  const combinedOdds = missingOdds === 0 ? selections.reduce((acc, s) => acc * s.odds!, 1) : null;
  // Product of the model's per-leg probabilities -- exact only if the legs
  // are independent, which same-fixture picks aren't.
  const combinedProb = selections.reduce((acc, s) => acc * s.prob, 1);
  const ev = combinedOdds != null ? combinedProb * combinedOdds - 1 : null;
  const potentialReturn = combinedOdds != null ? stake * combinedOdds : null;

  const perFixture = new Map<string, number>();
  for (const s of selections) perFixture.set(s.fixtureKey, (perFixture.get(s.fixtureKey) ?? 0) + 1);
  const correlated = [...perFixture.values()].some((n) => n > 1);
  const anyFair = selections.some((s) => s.oddsSource === "fair");

  return (
    <div className="slip">
      <div className="slip__list">
        {selections.map((s) => (
          <div key={s.id} className="slip-item">
            <span className="slip-item__pick">
              {s.pick}
              <span className="slip-item__market">{s.market}</span>
            </span>
            <span className={`slip-item__odds${s.oddsSource === "fair" ? " slip-item__odds--fair" : ""}`}>
              {s.odds != null ? s.odds.toFixed(2) : "—"}
            </span>
            <span className="slip-item__match">
              {s.homeTeam} vs {s.awayTeam} · {s.date} · {pct(s.prob)} model
            </span>
            <button type="button" className="slip-item__remove" aria-label="Remove pick" onClick={() => remove(s.id)}>
              ×
            </button>
          </div>
        ))}
      </div>

      {correlated && (
        <p className="slip-warn">
          Some picks are from the same match. Most bookmakers refuse those in an accumulator, and the combined
          probability below assumes independent legs, so treat it as optimistic.
        </p>
      )}
      {anyFair && (
        <p className="slip-warn">
          Greyed odds are the model's fair odds (no bookmaker margin), not prices you can bet at. Fetch bookmaker
          odds for the fixture to get real ones.
        </p>
      )}
      {missingOdds > 0 && (
        <p className="slip-warn">
          {missingOdds} pick{missingOdds === 1 ? " has" : "s have"} no odds, so the accumulator price can't be worked
          out.
        </p>
      )}

      <div className="slip-summary">
        <div className="slip-summary__row">
          <span>
            Accumulator ({selections.length} leg{selections.length === 1 ? "" : "s"})
          </span>
          <strong>{combinedOdds != null ? combinedOdds.toFixed(2) : "—"}</strong>
        </div>
        <div className="slip-summary__row">
          <span>Model win probability</span>
          <strong>{pct(combinedProb)}</strong>
        </div>
        <div className="slip-summary__row">
          <span>Expected value</span>
          <strong className={ev == null ? "" : ev >= 0 ? "slip-ev--pos" : "slip-ev--neg"}>
            {ev != null ? `${ev >= 0 ? "+" : ""}${(ev * 100).toFixed(1)}%` : "—"}
          </strong>
        </div>
        <div className="slip-summary__row">
          <span>Stake</span>
          <input
            className="slip-stake"
            type="number"
            min={0}
            step={1}
            value={stake}
            onChange={(e) => setStake(Math.max(0, Number(e.target.value) || 0))}
          />
        </div>
        <div className="slip-summary__row slip-summary__row--total">
          <span>Potential return</span>
          <strong>{potentialReturn != null ? potentialReturn.toFixed(2) : "—"}</strong>
        </div>
      </div>

      <button type="button" className="slip-clear" onClick={clear}>
        Clear slip
      </button>
    </div>
  );
}

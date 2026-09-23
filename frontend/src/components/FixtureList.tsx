import type { Fixture } from "../types";
import { FixtureCard } from "./FixtureCard";

export function FixtureList({ fixtures }: { fixtures: Fixture[] }) {
  if (fixtures.length === 0) {
    return <p style={{ color: "var(--text-muted)" }}>No upcoming fixtures for this filter.</p>;
  }
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 16 }}>
      {fixtures.map((f) => (
        <FixtureCard key={f.fixture_id} fixture={f} />
      ))}
    </div>
  );
}

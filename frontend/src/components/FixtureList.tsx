import type { Fixture } from "../types";
import { FixtureCard } from "./FixtureCard";

function formatDateHeading(iso: string): string {
  const d = new Date(`${iso}T00:00:00`);
  return d.toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long", year: "numeric" });
}

export function FixtureList({ fixtures }: { fixtures: Fixture[] }) {
  if (fixtures.length === 0) {
    return <p style={{ color: "var(--text-muted)" }}>No fixtures or matches for this filter.</p>;
  }

  // fixtures arrive sorted by date ascending (FixturesPage) -- bucket
  // consecutive same-date runs into groups rather than grouping into a map,
  // which would lose that order.
  const groups: { date: string; items: Fixture[] }[] = [];
  for (const f of fixtures) {
    const last = groups[groups.length - 1];
    if (last && last.date === f.date) last.items.push(f);
    else groups.push({ date: f.date, items: [f] });
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
      {groups.map((g) => (
        <div key={g.date}>
          <h2
            style={{
              fontSize: 13,
              fontWeight: 700,
              color: "var(--text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.04em",
              margin: "0 0 10px",
            }}
          >
            {formatDateHeading(g.date)}
          </h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))", gap: 16 }}>
            {g.items.map((f) => (
              <FixtureCard key={f.fixture_id} fixture={f} />
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

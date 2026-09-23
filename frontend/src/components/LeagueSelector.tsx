import type { League } from "../types";

interface Props {
  leagues: League[];
  value: string | null;
  onChange: (league: string | null) => void;
}

export function LeagueSelector({ leagues, value, onChange }: Props) {
  return (
    <select
      value={value ?? ""}
      onChange={(e) => onChange(e.target.value || null)}
      style={{
        padding: "6px 10px",
        borderRadius: 6,
        border: "1px solid var(--border)",
        background: "var(--surface-1)",
        color: "var(--text-primary)",
        fontSize: 14,
      }}
    >
      <option value="">All competitions</option>
      {[
        { label: "Leagues", items: leagues.filter((l) => l.model_group === l.league) },
        { label: "International", items: leagues.filter((l) => l.model_group !== l.league) },
      ]
        .filter((g) => g.items.length > 0)
        .map((g) => (
          <optgroup key={g.label} label={g.label}>
            {g.items.map((l) => (
              <option key={l.league} value={l.league}>
                {l.league} {l.last_trained_at ? "" : "(untrained)"}
              </option>
            ))}
          </optgroup>
        ))}
    </select>
  );
}

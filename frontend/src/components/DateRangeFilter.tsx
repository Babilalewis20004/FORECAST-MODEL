import "./DateRangeFilter.css";

export interface DateRange {
  start: string | null; // YYYY-MM-DD, local
  end: string | null;
}

function toISODate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function addDays(iso: string, days: number): string {
  const d = new Date(`${iso}T00:00:00`);
  d.setDate(d.getDate() + days);
  return toISODate(d);
}

function today(): string {
  return toISODate(new Date());
}

interface Preset {
  label: string;
  range: () => DateRange;
}

const PRESETS: Preset[] = [
  { label: "Today", range: () => ({ start: today(), end: today() }) },
  { label: "Next 7 days", range: () => ({ start: today(), end: addDays(today(), 6) }) },
  { label: "Last 7 days", range: () => ({ start: addDays(today(), -7), end: addDays(today(), -1) }) },
  { label: "Last 30 days", range: () => ({ start: addDays(today(), -30), end: addDays(today(), -1) }) },
  { label: "Whole dataset", range: () => ({ start: null, end: null }) },
];

function sameRange(a: DateRange, b: DateRange): boolean {
  return a.start === b.start && a.end === b.end;
}

interface Props {
  value: DateRange;
  onChange: (range: DateRange) => void;
}

export function DateRangeFilter({ value, onChange }: Props) {
  const activePreset = PRESETS.find((p) => sameRange(p.range(), value));

  return (
    <div className="date-range-filter">
      <span className="date-range-filter__label">Date range</span>
      <div className="date-range-filter__presets">
        {PRESETS.map((p) => (
          <button
            key={p.label}
            type="button"
            className={`date-range-filter__preset${p.label === activePreset?.label ? " date-range-filter__preset--active" : ""}`}
            onClick={() => onChange(p.range())}
          >
            {p.label}
          </button>
        ))}
      </div>
      <div className="date-range-filter__inputs">
        <label className="date-range-filter__field">
          <span>From</span>
          <input
            type="date"
            value={value.start ?? ""}
            onChange={(e) => onChange({ start: e.target.value || null, end: value.end })}
          />
        </label>
        <label className="date-range-filter__field">
          <span>To</span>
          <input
            type="date"
            value={value.end ?? ""}
            onChange={(e) => onChange({ start: value.start, end: e.target.value || null })}
          />
        </label>
      </div>
    </div>
  );
}

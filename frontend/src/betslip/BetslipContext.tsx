import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

export interface SlipSelection {
  // `${fixtureKey}|${group}` -- one pick per market per fixture, so picking a
  // different option in the same market replaces the earlier one.
  id: string;
  fixtureKey: string;
  group: string; // "x12", "btts", "ou_2.5"
  optionKey: string; // "home_win", "yes", "over_2.5", ...
  homeTeam: string;
  awayTeam: string;
  league: string;
  date: string;
  market: string; // chip label: "1X2", "BTTS", "O/U 2.5"
  pick: string; // "Arsenal", "Over 2.5", ...
  odds: number | null;
  // "bookmaker" when real odds were fetched for the fixture, otherwise the
  // model's fair odds (1/probability, no margin) -- not actually bettable.
  oddsSource: "bookmaker" | "fair";
  prob: number; // model probability, 0..1
}

interface BetslipValue {
  selections: SlipSelection[];
  stake: number;
  setStake: (n: number) => void;
  find: (id: string) => SlipSelection | undefined;
  // Adds the selection, replaces the same market's earlier pick, or removes
  // it when the identical pick is already on the slip.
  toggle: (s: SlipSelection) => void;
  remove: (id: string) => void;
  clear: () => void;
}

const STORAGE_KEY = "predict-model-betslip-v1";
const DEFAULT_STAKE = 10;

const BetslipContext = createContext<BetslipValue | null>(null);

function load(): { selections: SlipSelection[]; stake: number } {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed.selections)) {
        return {
          selections: parsed.selections,
          stake: typeof parsed.stake === "number" && parsed.stake >= 0 ? parsed.stake : DEFAULT_STAKE,
        };
      }
    }
  } catch {
    // storage blocked or corrupt -- start with an empty slip
  }
  return { selections: [], stake: DEFAULT_STAKE };
}

export function BetslipProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState(load);
  const { selections, stake } = state;

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // ignore -- slip just won't survive a reload
    }
  }, [state]);

  const setStake = useCallback((n: number) => setState((p) => ({ ...p, stake: n })), []);
  const find = useCallback((id: string) => selections.find((s) => s.id === id), [selections]);
  const toggle = useCallback((s: SlipSelection) => {
    setState((p) => {
      const existing = p.selections.find((x) => x.id === s.id);
      if (existing && existing.optionKey === s.optionKey) {
        return { ...p, selections: p.selections.filter((x) => x.id !== s.id) };
      }
      if (existing) return { ...p, selections: p.selections.map((x) => (x.id === s.id ? s : x)) };
      return { ...p, selections: [...p.selections, s] };
    });
  }, []);
  const remove = useCallback(
    (id: string) => setState((p) => ({ ...p, selections: p.selections.filter((x) => x.id !== id) })),
    [],
  );
  const clear = useCallback(() => setState((p) => ({ ...p, selections: [] })), []);

  const value = useMemo(
    () => ({ selections, stake, setStake, find, toggle, remove, clear }),
    [selections, stake, setStake, find, toggle, remove, clear],
  );
  return <BetslipContext.Provider value={value}>{children}</BetslipContext.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useBetslip(): BetslipValue {
  const ctx = useContext(BetslipContext);
  if (!ctx) throw new Error("useBetslip must be used inside <BetslipProvider>");
  return ctx;
}

import { useEffect, useState } from "react";
import { BetslipPage } from "./betslip/BetslipPage";
import { BetslipProvider, useBetslip } from "./betslip/BetslipContext";
import { FixturesPage } from "./pages/FixturesPage";
import "./betslip/Betslip.css";

type Tab = "fixtures" | "betslip";

function Tabs() {
  const [tab, setTab] = useState<Tab>(() =>
    new URLSearchParams(window.location.search).get("tab") === "betslip" ? "betslip" : "fixtures",
  );
  const { selections } = useBetslip();

  // Only touches `tab` so the fixture filters FixturesPage keeps in the URL
  // survive a tab switch.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (tab === "betslip") params.set("tab", "betslip");
    else params.delete("tab");
    const qs = params.toString();
    window.history.replaceState(null, "", qs ? `?${qs}` : window.location.pathname);
  }, [tab]);

  return (
    <div>
      <h1 style={{ fontSize: 24, marginBottom: 16 }}>Predict Model</h1>
      <nav className="tabs" role="tablist">
        <button
          type="button"
          role="tab"
          aria-selected={tab === "fixtures"}
          className={`tab${tab === "fixtures" ? " tab--active" : ""}`}
          onClick={() => setTab("fixtures")}
        >
          Fixtures
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "betslip"}
          className={`tab${tab === "betslip" ? " tab--active" : ""}`}
          onClick={() => setTab("betslip")}
        >
          Betslip
          {selections.length > 0 && <span className="tab__count">{selections.length}</span>}
        </button>
      </nav>
      {/* Both stay mounted so switching tabs doesn't refetch fixtures or lose
          which markets were expanded. */}
      <div hidden={tab !== "fixtures"}>
        <FixturesPage />
      </div>
      <div hidden={tab !== "betslip"}>
        <BetslipPage />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <BetslipProvider>
      <Tabs />
    </BetslipProvider>
  );
}

import { useEffect, useRef, useState } from "react";
import { getJobStatus, postBackup, postRefresh } from "../api/client";
import type { Job, League } from "../types";
import { ArchiveIcon, ChevronIcon, CpuIcon, DownloadIcon, LayersIcon, RefreshIcon } from "./icons";
import { LeagueSelector } from "./LeagueSelector";
import "./RefreshPanel.css";

const POLL_MS = 1000;

// The five buttons in this panel, keyed independently of the API's job
// "type" -- "Scrape fixtures only" and "Scrape + stats backfill" are both
// type="scrape" server-side (they only differ by include_stats), so they
// need their own last-run tracking here rather than one shared by type.
const ACTION_KEYS = ["full", "scrape", "scrape_stats", "retrain", "backup"] as const;
type ActionKey = (typeof ACTION_KEYS)[number];

const STORAGE_PREFIX = "predict-model:last-run:";

function loadLastRun(key: ActionKey): string | null {
  try {
    return localStorage.getItem(STORAGE_PREFIX + key);
  } catch {
    return null;
  }
}

function saveLastRun(key: ActionKey, iso: string): void {
  try {
    localStorage.setItem(STORAGE_PREFIX + key, iso);
  } catch {
    // private browsing / storage disabled / quota -- last-run display is a
    // convenience, not worth surfacing an error for.
  }
}

function formatLastRun(iso: string): string {
  const d = new Date(iso);
  const date = d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
  const time = d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit", hour12: false });
  return `${date} · ${time}`;
}

function LastRun({ iso }: { iso: string | null }) {
  if (!iso) return null;
  return <p className="refresh-panel__last-run">Last run: {formatLastRun(iso)}</p>;
}

interface Props {
  leagues: League[];
  // The page's currently selected league filter (shared with the
  // fixture-list LeagueSelector below it via onLeagueChange, not a second
  // independent picker) -- also used as the refresh scope, so "Retrain
  // only" etc. don't always retrain every league. null = "All competitions".
  league: string | null;
  onLeagueChange: (league: string | null) => void;
  onJobFinished?: () => void;
}

export function RefreshPanel({ leagues, league, onLeagueChange, onJobFinished }: Props) {
  const [job, setJob] = useState<Job | null>(null);
  const [actionKey, setActionKey] = useState<ActionKey | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRun, setLastRun] = useState<Record<ActionKey, string | null>>(() =>
    Object.fromEntries(ACTION_KEYS.map((k) => [k, loadLastRun(k)])) as Record<ActionKey, string | null>,
  );
  const timerRef = useRef<number | null>(null);
  const consoleRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
    };
  }, []);

  // Auto-scroll the console to the newest line as it streams in.
  useEffect(() => {
    if (consoleRef.current) consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
  }, [job?.log.length]);

  function pollUntilDone(jobId: string, key: ActionKey) {
    timerRef.current = window.setInterval(async () => {
      try {
        const updated = await getJobStatus(jobId);
        setJob(updated);
        if (updated.status === "success" || updated.status === "failed") {
          if (timerRef.current) window.clearInterval(timerRef.current);
          if (updated.status === "success") {
            const now = new Date().toISOString();
            saveLastRun(key, now);
            setLastRun((prev) => ({ ...prev, [key]: now }));
            onJobFinished?.();
          }
        }
      } catch {
        if (timerRef.current) window.clearInterval(timerRef.current);
      }
    }, POLL_MS);
  }

  async function trigger(key: ActionKey, kind: "scrape" | "retrain" | "full", includeStats = false) {
    setError(null);
    try {
      const started = await postRefresh(kind, {
        leagues: league ? [league] : undefined,
        include_stats: includeStats,
      });
      setActionKey(key);
      setJob(started);
      pollUntilDone(started.job_id, key);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  // Not scoped by league (dumps the whole database) and hits a different
  // endpoint than trigger()'s scrape/retrain/full, so it's its own function
  // rather than another `kind` in trigger().
  async function triggerBackup() {
    setError(null);
    try {
      const started = await postBackup();
      setActionKey("backup");
      setJob(started);
      pollUntilDone(started.job_id, "backup");
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const busy = job?.status === "running" || job?.status === "queued";
  const pct = job?.total_steps ? Math.min(100, Math.round((job.log.length / job.total_steps) * 100)) : busy ? 0 : null;

  return (
    <div className="refresh-panel">
      <div className="refresh-panel__scope">
        <span className="refresh-panel__scope-label">Scope</span>
        <LeagueSelector leagues={leagues} value={league} onChange={onLeagueChange} />
      </div>
      <div className="refresh-panel__main-row">
        <button className="refresh-panel__primary-btn" onClick={() => trigger("full", "full")} disabled={busy}>
          <RefreshIcon className={busy ? "refresh-panel__icon--spin" : undefined} />
          {busy && actionKey === "full" ? "Refreshing..." : "Refresh data (scrape + retrain)"}
        </button>
        <button className="refresh-panel__advanced-toggle" onClick={() => setShowAdvanced((s) => !s)}>
          {showAdvanced ? "Hide advanced" : "Advanced"}
          <ChevronIcon className={showAdvanced ? "refresh-panel__icon--flip" : undefined} />
        </button>
      </div>
      <LastRun iso={lastRun.full} />

      {showAdvanced && (
        <div className="refresh-panel__advanced">
          <div className="refresh-panel__section-label">Run a single stage</div>
          <div className="refresh-panel__stage-row">
            <div>
              <button className="refresh-panel__stage-btn" onClick={() => trigger("scrape", "scrape", false)} disabled={busy}>
                <DownloadIcon /> Scrape fixtures only
              </button>
              <LastRun iso={lastRun.scrape} />
            </div>
            <div>
              <button className="refresh-panel__stage-btn" onClick={() => trigger("scrape_stats", "scrape", true)} disabled={busy}>
                <LayersIcon /> Scrape + stats backfill
              </button>
              <LastRun iso={lastRun.scrape_stats} />
            </div>
            <div>
              <button className="refresh-panel__stage-btn" onClick={() => trigger("retrain", "retrain")} disabled={busy}>
                <CpuIcon /> Retrain only
              </button>
              <LastRun iso={lastRun.retrain} />
            </div>
            <div>
              <button className="refresh-panel__stage-btn" onClick={triggerBackup} disabled={busy}>
                <ArchiveIcon /> Backup database
              </button>
              <LastRun iso={lastRun.backup} />
            </div>
          </div>
        </div>
      )}

      {job && (busy || job.log.length > 0 || job.status === "failed") && (
        <div className="refresh-panel__progress">
          <div className="refresh-panel__progress-label">
            <span>{job.status === "failed" ? "Failed" : (job.detail ?? "Working...")}</span>
            {pct != null && <span>{pct}%</span>}
          </div>
          {pct != null && (
            <div className="refresh-panel__progress-track">
              <div className="refresh-panel__progress-fill" style={{ width: `${pct}%` }} />
            </div>
          )}
          {job.log.length > 0 && (
            <div className="refresh-panel__console" ref={consoleRef}>
              {job.log.map((entry, i) => (
                <div className="refresh-panel__console-line" key={i}>
                  <span className="refresh-panel__console-time">
                    {new Date(entry.at).toLocaleTimeString(undefined, { hour12: false })}
                  </span>
                  <span className="refresh-panel__console-arrow">{">"}</span>
                  <span>{entry.message}</span>
                </div>
              ))}
            </div>
          )}
          {job.status === "failed" && job.error && <p className="refresh-panel__error">{job.error}</p>}
        </div>
      )}
      {error && <p className="refresh-panel__error">{error}</p>}
    </div>
  );
}

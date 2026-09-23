import { useEffect, useRef, useState } from "react";
import { getJobStatus, postRefresh } from "../api/client";
import type { Job } from "../types";

const POLL_MS = 2000;

interface Props {
  onJobFinished?: () => void;
}

export function RefreshPanel({ onJobFinished }: Props) {
  const [job, setJob] = useState<Job | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
    };
  }, []);

  function pollUntilDone(jobId: string) {
    timerRef.current = window.setInterval(async () => {
      try {
        const updated = await getJobStatus(jobId);
        setJob(updated);
        if (updated.status === "success" || updated.status === "failed") {
          if (timerRef.current) window.clearInterval(timerRef.current);
          if (updated.status === "success") onJobFinished?.();
        }
      } catch {
        if (timerRef.current) window.clearInterval(timerRef.current);
      }
    }, POLL_MS);
  }

  async function trigger(kind: "scrape" | "retrain" | "full", includeStats = false) {
    setError(null);
    try {
      const started = await postRefresh(kind, { include_stats: includeStats });
      setJob(started);
      pollUntilDone(started.job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const busy = job?.status === "running" || job?.status === "queued";

  return (
    <div
      style={{
        border: "1px solid var(--border)",
        borderRadius: 8,
        padding: 16,
        marginBottom: 24,
        background: "var(--surface-1)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <button onClick={() => trigger("full")} disabled={busy}>
          {busy ? "Refreshing..." : "Refresh data (scrape + retrain)"}
        </button>
        <button onClick={() => setShowAdvanced((s) => !s)} style={{ fontSize: 12 }}>
          {showAdvanced ? "Hide advanced" : "Advanced"}
        </button>
      </div>

      {showAdvanced && (
        <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
          <button onClick={() => trigger("scrape", false)} disabled={busy}>
            Scrape fixtures only
          </button>
          <button onClick={() => trigger("scrape", true)} disabled={busy}>
            Scrape + stats backfill
          </button>
          <button onClick={() => trigger("retrain")} disabled={busy}>
            Retrain only
          </button>
        </div>
      )}

      {job && (
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 12, marginBottom: 0 }}>
          [{job.type}] {job.status}
          {job.detail ? ` — ${job.detail}` : ""}
          {job.status === "failed" && job.error ? ` — ${job.error}` : ""}
        </p>
      )}
      {error && <p style={{ fontSize: 13, color: "var(--critical)" }}>{error}</p>}
    </div>
  );
}

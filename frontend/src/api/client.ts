import type { Fixture, Job, League, RecentMatch } from "../types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export function getLeagues(): Promise<League[]> {
  return request("/api/leagues");
}

export function getUpcomingFixtures(params: { league?: string; start?: string; end?: string }): Promise<Fixture[]> {
  const qs = new URLSearchParams();
  if (params.league) qs.set("league", params.league);
  if (params.start) qs.set("start", params.start);
  if (params.end) qs.set("end", params.end);
  return request(`/api/fixtures/upcoming?${qs.toString()}`);
}

// Last-5-form data: the existing /api/matches endpoint already supports
// filtering by team and ordering by date desc -- no new backend endpoint
// needed. `matches` only ever contains finished fixtures, so this is always
// "results up to now", never a future leak for an upcoming fixture.
export function getRecentMatches(team: string, limit = 5): Promise<RecentMatch[]> {
  const qs = new URLSearchParams({ team, limit: String(limit) });
  return request(`/api/matches?${qs.toString()}`);
}

// Finished matches for the date-range filter's past-looking presets (Last 7/30
// days, Whole dataset). Separate from getRecentMatches so that lightweight
// team-form lookups don't pay for prediction joins they don't use.
export function getMatches(params: { league?: string; start?: string; end?: string; withPredictions?: boolean }): Promise<RecentMatch[]> {
  const qs = new URLSearchParams();
  if (params.league) qs.set("league", params.league);
  if (params.start) qs.set("start", params.start);
  if (params.end) qs.set("end", params.end);
  if (params.withPredictions) qs.set("with_predictions", "true");
  return request(`/api/matches?${qs.toString()}`);
}

export function postRefresh(kind: "scrape" | "retrain" | "full", body: { leagues?: string[]; include_stats?: boolean }): Promise<Job> {
  return request(`/api/refresh/${kind}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// Not scoped by league (it dumps the whole database), so it doesn't take a
// body like postRefresh's scrape/retrain/full.
export function postBackup(): Promise<Job> {
  return request("/api/refresh/backup", { method: "POST" });
}

export function getJobStatus(jobId: string): Promise<Job> {
  return request(`/api/refresh/status/${jobId}`);
}

export interface ShareResult {
  sent: boolean;
  fixtures: number;
  messages: number;
  detail?: string;
}

// Publishes every fixture/prediction in the given range (same filters as
// getUpcomingFixtures/getMatches) as a Telegram digest -- see
// backend/app/routers/share.py.
export function postShareTelegram(params: { league?: string; start?: string; end?: string }): Promise<ShareResult> {
  const qs = new URLSearchParams();
  if (params.league) qs.set("league", params.league);
  if (params.start) qs.set("start", params.start);
  if (params.end) qs.set("end", params.end);
  return request(`/api/share/telegram?${qs.toString()}`, { method: "POST" });
}

export interface ShareCardsResult {
  sent: boolean;
  fixtures: number;
  albums: number;
  detail?: string;
}

// One image card per fixture, sent as Telegram albums -- alongside
// postShareTelegram's text digest, not a replacement for it.
export function postShareTelegramCards(params: { league?: string; start?: string; end?: string }): Promise<ShareCardsResult> {
  const qs = new URLSearchParams();
  if (params.league) qs.set("league", params.league);
  if (params.start) qs.set("start", params.start);
  if (params.end) qs.set("end", params.end);
  return request(`/api/share/telegram/cards?${qs.toString()}`, { method: "POST" });
}

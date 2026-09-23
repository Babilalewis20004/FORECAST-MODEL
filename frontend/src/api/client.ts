import type { Fixture, Job, League } from "../types";

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

export function postRefresh(kind: "scrape" | "retrain" | "full", body: { leagues?: string[]; include_stats?: boolean }): Promise<Job> {
  return request(`/api/refresh/${kind}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function getJobStatus(jobId: string): Promise<Job> {
  return request(`/api/refresh/status/${jobId}`);
}

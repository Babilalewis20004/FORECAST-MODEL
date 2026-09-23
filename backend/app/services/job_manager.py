from __future__ import annotations

import asyncio
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

_MAX_HISTORY = 20


@dataclass
class Job:
    job_id: str
    type: str
    status: str = "queued"  # queued | running | success | failed
    started_at: datetime | None = None
    finished_at: datetime | None = None
    detail: str | None = None
    error: str | None = None


class JobManager:
    """Single-flight in-memory job tracker. One process, one user -- no need
    for a real queue (Celery/RQ) or persistence across restarts."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._history: deque[str] = deque(maxlen=_MAX_HISTORY)
        self._lock = asyncio.Lock()
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[Job]:
        ids = list(self._history)[-limit:][::-1]
        return [self._jobs[i] for i in ids if i in self._jobs]

    async def start(self, job_type: str, fn: Callable[[Job], None]) -> Job:
        """Runs fn(job) in a worker thread if no job is currently running.
        Raises RuntimeError (caller maps to 409) if one already is."""
        async with self._lock:
            if self._running:
                raise RuntimeError("A refresh job is already running")
            self._running = True

        job = Job(job_id=str(uuid.uuid4()), type=job_type, status="running", started_at=_now())
        self._jobs[job.job_id] = job
        self._history.append(job.job_id)

        async def _run() -> None:
            try:
                await asyncio.to_thread(fn, job)
                if job.status == "running":
                    job.status = "success"
            except Exception as exc:  # noqa: BLE001 -- surfaced via job.error, not raised
                job.status = "failed"
                job.error = f"{exc}\n{traceback.format_exc(limit=3)}"
            finally:
                job.finished_at = _now()
                self._running = False

        asyncio.create_task(_run())
        return job


def _now() -> datetime:
    return datetime.now(timezone.utc)


job_manager = JobManager()

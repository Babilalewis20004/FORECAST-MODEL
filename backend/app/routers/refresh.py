from fastapi import APIRouter, HTTPException

from app.schemas import JobOut, LogEntryOut, RefreshRequest
from app.services.job_manager import Job, job_manager
from app.services.model_service import ALL_MODEL_GROUPS, model_group, retrain_leagues
from app.services.scrape_service import LEAGUES, run_scrape, scrape_odds

router = APIRouter(prefix="/api/refresh", tags=["refresh"])


def _job_out(job: Job) -> JobOut:
    return JobOut(
        job_id=job.job_id,
        type=job.type,
        status=job.status,
        started_at=job.started_at,
        finished_at=job.finished_at,
        detail=job.detail,
        error=job.error,
        total_steps=job.total_steps,
        log=[LogEntryOut(at=e.at, message=e.message) for e in job.log],
    )


async def _start(job_type: str, fn) -> JobOut:
    try:
        job = await job_manager.start(job_type, fn)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _job_out(job)


def _n_leagues(leagues: list[str] | None) -> int:
    return len(leagues) if leagues else len(LEAGUES)


def _n_model_groups(leagues: list[str] | None) -> int:
    return len(dict.fromkeys(model_group(lg) for lg in leagues)) if leagues else len(ALL_MODEL_GROUPS)


def _scrape_total_steps(leagues: list[str] | None, include_stats: bool, include_odds: bool) -> int:
    """Rough estimate of how many job.log_line() milestones a scrape will
    produce, for the frontend's progress bar (len(log)/total_steps). Not
    exact -- e.g. stats backfill only logs leagues that actually need
    it -- just close enough that the bar's pace tracks real work rather
    than a fixed timer."""
    steps = _n_leagues(leagues) + 1  # fixtures (1/league) + historical odds sync
    if include_stats:
        steps += _n_leagues(leagues)  # stats backfill (up to 1/league)
    if include_odds:
        steps += 1 + _n_model_groups(leagues)  # odds summary + 1/re-predicted group
    return steps


def _retrain_total_steps(leagues: list[str] | None) -> int:
    return _n_model_groups(leagues)


@router.post("/scrape", response_model=JobOut)
async def refresh_scrape(body: RefreshRequest) -> JobOut:
    def _run(job: Job) -> None:
        job.total_steps = _scrape_total_steps(body.leagues, body.include_stats, body.include_odds)
        result = run_scrape(body.leagues, body.include_stats, job=job, include_odds=body.include_odds)
        job.detail = f"done: {result}"

    return await _start("scrape", _run)


@router.post("/retrain", response_model=JobOut)
async def refresh_retrain(body: RefreshRequest) -> JobOut:
    def _run(job: Job) -> None:
        job.total_steps = _retrain_total_steps(body.leagues)
        job.detail = "Training models..."
        result = retrain_leagues(body.leagues, job=job)
        job.detail = f"done: {result}"

    return await _start("retrain", _run)


@router.post("/full", response_model=JobOut)
async def refresh_full(body: RefreshRequest) -> JobOut:
    def _run(job: Job) -> None:
        job.total_steps = (
            _scrape_total_steps(body.leagues, body.include_stats, include_odds=False)
            + _retrain_total_steps(body.leagues)
            + (1 + _n_model_groups(body.leagues) if body.include_odds else 0)
        )
        job.detail = "Scraping..."
        scrape_result = run_scrape(body.leagues, body.include_stats, job=job)
        job.detail = "Training models..."
        retrain_result = retrain_leagues(body.leagues, job=job)
        # After the retrain, so fixtures new in this scrape have a prediction
        # to hang their odds on.
        odds_result = scrape_odds(body.leagues, job=job) if body.include_odds else None
        job.detail = f"done: scrape={scrape_result} retrain={retrain_result} odds={odds_result}"

    return await _start("full", _run)


@router.get("/status/{job_id}", response_model=JobOut)
def refresh_status(job_id: str) -> JobOut:
    job = job_manager.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job_id")
    return _job_out(job)


@router.get("/jobs", response_model=list[JobOut])
def refresh_jobs(limit: int = 20) -> list[JobOut]:
    return [_job_out(j) for j in job_manager.recent(limit)]

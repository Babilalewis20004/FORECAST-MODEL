from fastapi import APIRouter, HTTPException

from app.schemas import JobOut, RefreshRequest
from app.services.job_manager import Job, job_manager
from app.services.model_service import retrain_leagues
from app.services.scrape_service import run_scrape, scrape_odds

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
    )


async def _start(job_type: str, fn) -> JobOut:
    try:
        job = await job_manager.start(job_type, fn)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    return _job_out(job)


@router.post("/scrape", response_model=JobOut)
async def refresh_scrape(body: RefreshRequest) -> JobOut:
    def _run(job: Job) -> None:
        result = run_scrape(body.leagues, body.include_stats, job=job, include_odds=body.include_odds)
        job.detail = f"done: {result}"

    return await _start("scrape", _run)


@router.post("/retrain", response_model=JobOut)
async def refresh_retrain(body: RefreshRequest) -> JobOut:
    def _run(job: Job) -> None:
        job.detail = "training models..."
        result = retrain_leagues(body.leagues)
        job.detail = f"done: {result}"

    return await _start("retrain", _run)


@router.post("/full", response_model=JobOut)
async def refresh_full(body: RefreshRequest) -> JobOut:
    def _run(job: Job) -> None:
        job.detail = "scraping..."
        scrape_result = run_scrape(body.leagues, body.include_stats, job=job)
        job.detail = "training models..."
        retrain_result = retrain_leagues(body.leagues)
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

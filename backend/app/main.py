from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import fixtures, leagues, predict, refresh
from app.services.model_service import load_cached_models
from app.services.odds_service import CLOSING_POLL_INTERVAL_MINUTES, poll_closing_lines

# In-process scheduler (no Redis/broker) for the closing-line poll -- this is
# a single-process app already (see job_manager.py), so this stays consistent
# with that rather than introducing a task queue for one periodic job.
scheduler = AsyncIOScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_cached_models()
    scheduler.add_job(
        poll_closing_lines, "interval", minutes=CLOSING_POLL_INTERVAL_MINUTES, id="poll_closing_lines"
    )
    scheduler.start()
    yield
    scheduler.shutdown()


app = FastAPI(title="Predict Model API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(leagues.router)
app.include_router(fixtures.router)
app.include_router(predict.router)
app.include_router(refresh.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import fixtures, leagues, predict, refresh
from app.services.model_service import load_cached_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_cached_models()
    yield


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

# Shared image for the `backend` (FastAPI) and `dashboard` (Streamlit) compose
# services -- both need the same Python dependencies and the same `app/` ML
# package; only their `command:` (and, for the dashboard, `working_dir:`)
# differ in docker-compose.yml.
#
# Build context is the REPO ROOT (see docker-compose.yml), not backend/,
# because the dashboard also needs the root app.py and requirements.txt.
FROM python:3.12-slim

WORKDIR /app

# postgresql-client (pg_dump/pg_restore) for the "Backup database" button /
# scripts/backup_db.py + restore_db.py. python:3.12-slim's Debian release
# (trixie) ships postgresql-client 17 by default, matching the native
# Postgres 17 the app connects to -- no need for the external PGDG repo.
# fonts-dejavu-core: the base image has no fonts at all, and the "Share
# Cards to Telegram" button (card_image_service.py) needs a real TrueType
# font for Pillow -- DejaVu Sans/Sans-Bold, a small standard Debian package.
RUN apt-get update && apt-get install -y --no-install-recommends postgresql-client fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Installed before the rest of the source so this layer only rebuilds when a
# dependency actually changes.
COPY requirements.txt ./requirements.txt
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Backend's own imports (`from app.config import settings`, etc.) are
# relative to backend/ being the working directory, matching how it's run
# natively (see README: `cd backend && uvicorn app.main:app ...`). The
# dashboard service overrides this back to /app via `working_dir:`.
WORKDIR /app/backend

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

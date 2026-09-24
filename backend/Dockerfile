# Shared image for the `backend` (FastAPI) and `dashboard` (Streamlit) compose
# services -- both need the same Python dependencies and the same `app/` ML
# package; only their `command:` (and, for the dashboard, `working_dir:`)
# differ in docker-compose.yml.
#
# Build context is the REPO ROOT (see docker-compose.yml), not backend/,
# because the dashboard also needs the root app.py and requirements.txt.
FROM python:3.12-slim

WORKDIR /app

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

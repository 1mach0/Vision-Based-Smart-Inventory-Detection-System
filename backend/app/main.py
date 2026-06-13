"""FastAPI application entry point.

Wires the routers together, creates database tables on startup, and serves the
single-page UI so the whole app runs from one process on http://localhost:8000
(``uvicorn app.main:app``) — no separate frontend server or build step.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from sqlalchemy.exc import OperationalError

from .api import routes_inference, routes_inventory
# Importing the models module registers all tables on Base.metadata so that
# create_all() below knows about them. The `noqa` marks it as an intentional
# import-for-side-effect (it's not referenced directly).
from .persistence import models as _models  # noqa: F401
from .persistence.database import Base, engine

# Path to the static UI, shipped inside the package so Docker (which copies
# only backend/) includes it.
_INDEX_HTML = Path(__file__).parent / "web" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run once at startup (before the first request) and shutdown.

    We create any missing tables here. This is fine for the demo and small
    deployments; a production system would use a migration tool (Alembic)
    instead so schema changes are versioned. Note this runs on real startup but
    NOT under the test client (which doesn't enter the lifespan), so tests
    manage their own schema.

    If the configured database can't be reached (e.g. `run` was used with the
    default PostgreSQL URL but no Postgres is running), we replace the raw
    driver traceback with a short, actionable message pointing at the SQLite
    demo.
    """
    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError as exc:
        raise RuntimeError(
            f"Could not connect to the database at {engine.url!r}.\n"
            "  • For a zero-setup run, use SQLite:  uv run vision-inventory demo\n"
            "  • Or point run at a SQLite file:     "
            "uv run vision-inventory run --database-url sqlite:///./demo.db\n"
            "  • Or start PostgreSQL first:         docker compose up -d db"
        ) from exc
    yield


app = FastAPI(title="Vision Inventory System", lifespan=lifespan)

# REST API. Routers keep endpoints grouped by concern and out of this file.
app.include_router(routes_inference.router)
app.include_router(routes_inventory.router)


@app.get("/health", tags=["meta"])
async def health():
    """Liveness probe — cheap endpoint for uptime checks / container health."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index():
    """Serve the single-page dashboard at the site root."""
    return FileResponse(_INDEX_HTML)

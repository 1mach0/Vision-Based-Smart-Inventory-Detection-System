"""Application settings.

All tunable values live here and are read from environment variables (or a
local ``.env`` file) exactly once at import time. Centralising configuration
keeps secrets and environment-specific values out of the code, and gives the
rest of the app a single, typed ``settings`` object to import.
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # env_file=".env"  -> load a local .env if present (handy in development).
    # extra="ignore"   -> ignore unrelated env vars instead of erroring.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Persistence -----------------------------------------------------
    # SQLAlchemy URL. Defaults to local Postgres; point it at a SQLite file
    # (e.g. sqlite:///./demo.db) for the zero-setup demo.
    database_url: str = "postgresql+psycopg://vision:vision@localhost:5432/vision"

    # --- Vision ----------------------------------------------------------
    # Path to YOLO weights (.pt). Loaded lazily on first inference.
    yolo_model_path: str = "models/yolo.pt"
    # Name/path of the Tesseract binary pytesseract should call.
    tesseract_cmd: str = "tesseract"

    # --- Domain ----------------------------------------------------------
    # Detections whose confidence is at or below this value are routed to
    # human review instead of automatically updating inventory. Raising it
    # makes the system more cautious (more items sent for review).
    review_confidence_threshold: float = 0.5


# Imported everywhere as `from .config import settings`. Instantiated once so
# the .env file and environment are read a single time per process.
settings = Settings()

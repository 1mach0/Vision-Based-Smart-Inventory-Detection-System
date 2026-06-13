"""Test fixtures: in-memory SQLite DB and a fake vision pipeline.

Lets the domain, persistence, and API be tested end-to-end without a running
PostgreSQL, YOLO weights, or Tesseract.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.deps import get_pipeline
from app.main import app
from app.persistence.database import Base, get_session
from app.vision.pipeline import Observation

engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestSession = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture
def db_session():
    Base.metadata.create_all(engine)
    session = TestSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


class FakePipeline:
    """Returns fixed observations: one confident, one low-confidence."""

    def process(self, image_bytes):
        return [
            Observation("widget", "SKU-1", 0.92, 0.88, (0, 0, 10, 10)),
            Observation("gadget", "", 0.30, 0.0, (0, 0, 10, 10)),
        ]


@pytest.fixture
def client(db_session):
    def _get_session_override():
        yield db_session

    app.dependency_overrides[get_session] = _get_session_override
    app.dependency_overrides[get_pipeline] = lambda: FakePipeline()
    yield TestClient(app)
    app.dependency_overrides.clear()

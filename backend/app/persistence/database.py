"""Database engine, session factory, and the declarative base.

This module is the single place where the app talks to SQLAlchemy's engine.
Everything else (ORM models, the repository, FastAPI dependencies) imports
`Base`, `engine`, or `get_session` from here, so swapping databases only
touches this file.

The engine is created lazily by SQLAlchemy: constructing it does NOT open a
connection, so importing this module never fails just because Postgres is
down. The first real query is what opens a connection.
"""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from ..config import settings


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model.

    SQLAlchemy collects table definitions on ``Base.metadata`` as each model
    class is imported. ``Base.metadata.create_all(engine)`` (called on app
    startup) then issues ``CREATE TABLE IF NOT EXISTS`` for all of them.
    """


# SQLite needs a special connect arg when used from a multi-threaded server
# (uvicorn). Postgres and other databases don't, so we only add it for sqlite.
# This lets the same code run against a throwaway SQLite file for the demo and
# against Postgres in production, controlled entirely by DATABASE_URL.
_connect_args = (
    {"check_same_thread": False}
    if settings.database_url.startswith("sqlite")
    else {}
)

# `future=True` opts into SQLAlchemy 2.0-style behaviour (already the default
# on modern versions, kept explicit for clarity).
engine = create_engine(settings.database_url, future=True, connect_args=_connect_args)

# A session factory. Each request gets its own Session (see get_session).
#   autoflush=False       -> we flush explicitly in the repository, so the
#                            timing of INSERTs is predictable.
#   expire_on_commit=False -> ORM objects stay usable after commit() without a
#                            re-query, which keeps the API response code simple.
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session():
    """Yield a database session, closing it when the request finishes.

    Used as a FastAPI dependency: FastAPI runs the generator, injects the
    yielded Session into the endpoint, and resumes it (running the ``finally``)
    once the response is sent. Tests override this dependency to hand back a
    session bound to an in-memory SQLite database instead.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

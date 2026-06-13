"""Command-line entry point for the Vision Inventory System.

One place to launch everything, with all knobs passable as terminal flags.
Exposed as the ``vision-inventory`` command by pyproject's [project.scripts],
so after ``uv sync`` you can run:

    uv run vision-inventory demo                 # seed SQLite + serve it
    uv run vision-inventory run --port 9000      # serve (uses DATABASE_URL)
    uv run vision-inventory seed --reset         # (re)build the demo database

Design note: configuration flags are applied by setting environment variables
*before* uvicorn imports the app, because ``app.config.Settings`` reads the
environment at import time. This keeps a single source of truth (Settings) and
means the CLI, a .env file, and raw env vars all feed the same place.
"""
from __future__ import annotations

import argparse
import os


def _apply_overrides(args: argparse.Namespace) -> None:
    """Translate CLI flags into the env vars that Settings reads."""
    if getattr(args, "database_url", None):
        os.environ["DATABASE_URL"] = args.database_url
    if getattr(args, "model", None):
        os.environ["YOLO_MODEL_PATH"] = args.model
    if getattr(args, "threshold", None) is not None:
        os.environ["REVIEW_CONFIDENCE_THRESHOLD"] = str(args.threshold)


def _serve(host: str, port: int, reload: bool) -> None:
    """Start the ASGI server. Imported lazily so `seed` needs no web deps."""
    import uvicorn

    # Pass the import string (not the app object) so uvicorn's reloader can
    # re-import the app in worker processes.
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)


def cmd_run(args: argparse.Namespace) -> None:
    """Serve the API + dashboard against whatever DATABASE_URL is configured."""
    _apply_overrides(args)
    _serve(args.host, args.port, args.reload)


def cmd_seed(args: argparse.Namespace) -> None:
    """Create/refresh the SQLite demo database."""
    from scripts.seed import seed

    seed(args.path, args.reset)


def cmd_demo(args: argparse.Namespace) -> None:
    """One command to see it working: seed a fresh SQLite DB, then serve it."""
    from scripts.seed import seed

    seed(args.path, reset=True)
    os.environ["DATABASE_URL"] = f"sqlite:///./{args.path}"
    print(f"Serving demo on http://{args.host}:{args.port}  (DB: {args.path})")
    _serve(args.host, args.port, args.reload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vision-inventory",
        description="Run and manage the Vision Inventory System.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Shared server flags, reused by `run` and `demo`.
    def add_server_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
        p.add_argument("--port", type=int, default=8000, help="bind port (default: 8000)")
        p.add_argument("--reload", action="store_true", help="auto-reload on code changes")

    # run -----------------------------------------------------------------
    p_run = sub.add_parser("run", help="serve the API + dashboard")
    add_server_flags(p_run)
    p_run.add_argument("--database-url", help="SQLAlchemy URL (overrides env/.env)")
    p_run.add_argument("--model", help="path to YOLO weights (overrides env)")
    p_run.add_argument("--threshold", type=float, help="review confidence threshold 0-1")
    p_run.set_defaults(func=cmd_run)

    # demo ----------------------------------------------------------------
    p_demo = sub.add_parser("demo", help="seed a SQLite demo DB and serve it")
    add_server_flags(p_demo)
    p_demo.add_argument("--path", default="demo.db", help="demo DB file (default: demo.db)")
    p_demo.set_defaults(func=cmd_demo)

    # seed ----------------------------------------------------------------
    p_seed = sub.add_parser("seed", help="create/refresh the SQLite demo DB")
    p_seed.add_argument("--path", default="demo.db", help="output DB file (default: demo.db)")
    p_seed.add_argument("--reset", action="store_true", help="overwrite if it exists")
    p_seed.set_defaults(func=cmd_seed)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

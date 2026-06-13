# Vision-Based Smart Inventory Detection System

An end-to-end computer-vision system that turns camera images of inventory
racks into structured, auditable inventory records. Objects are located with
**YOLO**, their labels/SKUs are read with **Tesseract OCR**, and every result
is passed through a **confidence-aware reconciliation** step: confident
detections update stock automatically, while uncertain ones are held for human
review instead of silently corrupting the count.

The system runs from a single process — `uvicorn app.main:app` serves both the
JSON API and a small dashboard at **http://localhost:8000**.

> **About this project.** This is a personal learning project — everything runs
> locally on a laptop, no GPU or cloud services required. It's built to
> practice and demonstrate the end-to-end skills (computer vision, a REST API,
> a database, testing, containers) rather than to be a production or commercial
> product. Where it takes shortcuts (e.g. `create_all` instead of migrations,
> no auth), those are called out honestly in [`WALKTHROUGH.md`](./WALKTHROUGH.md).

> **For the full story** — every design decision, the alternatives considered,
> and how the project grew from an empty folder to what's here — read
> [`WALKTHROUGH.md`](./WALKTHROUGH.md).

---

## Quick start (zero-setup demo)

The project is managed with [uv](https://docs.astral.sh/uv/). No Postgres and no
model weights are needed to see it running against a seeded SQLite database:

```bash
cd backend
uv sync                        # create .venv and install everything
uv run vision-inventory demo   # seed a SQLite DB and serve it
```

Open **http://localhost:8000** — the dashboard shows seeded products and a
review queue immediately. A prebuilt `backend/demo.db` is also included.

### The `vision-inventory` command

`uv sync` installs a CLI that launches and manages the app. All parameters are
passable as terminal flags:

```bash
uv run vision-inventory demo                       # seed SQLite + serve (one command)
uv run vision-inventory run                        # serve using DATABASE_URL / .env
uv run vision-inventory run --port 9000 --reload   # custom port, auto-reload
uv run vision-inventory run \
    --database-url postgresql+psycopg://vision:vision@localhost/vision \
    --threshold 0.6                                # override DB + review threshold
uv run vision-inventory seed --reset               # (re)build demo.db only
```

Run `uv run vision-inventory <command> --help` for every flag. To run real
inference from the UI you additionally need a YOLO weights file
(`--model path/to/yolo.pt`) and Tesseract installed; uploading an image then
runs the full detect → OCR → reconcile → persist path.

## Optional: Postgres + Docker

You don't need this to use the project — it's here to practice running the app
against a real database in containers. It starts the API and a PostgreSQL
instance together (torch is pinned to CPU-only wheels, so the image stays
laptop-sized):

```bash
docker compose up --build      # API on :8000, PostgreSQL on :5432
```

---

## Architecture

Each concern is a separate, independently testable module. Data flows one way:

```
   image bytes
        │
        ▼
 vision/pipeline ──uses──►  vision/detector (YOLO)
        │                   vision/ocr      (Tesseract)
        ▼
   [Observation]           label, text, detection_conf, ocr_conf
        │
        ▼
 domain/reconciliation ──►  APPLY  (confident)  or  REVIEW (uncertain)
        │
        ▼
 persistence/repository ─►  products, observations, inventory_changes  (PostgreSQL / SQLite)
        │
        ▼
       api/  ───────────►  REST endpoints + dashboard
```

```
backend/
  pyproject.toml             uv project + dependencies + `vision-inventory` command
  app/
    cli.py                   terminal launcher (run / demo / seed)
    config.py                settings from env / .env
    deps.py                  FastAPI dependency providers (pipeline, repo)
    main.py                  app wiring, table creation, serves the UI
    vision/
      detector.py            YOLO wrapper (lazy load, swappable)
      ocr.py                 Tesseract wrapper
      pipeline.py            bytes → detect → crop → OCR → [Observation]
    domain/
      reconciliation.py      confidence → APPLY / REVIEW decision
    persistence/
      database.py            engine, session, declarative base
      models.py              products, observations, inventory_changes
      repository.py          the only place that reads/writes those tables
    api/
      routes_inference.py    POST /inference/observe   (write path)
      routes_inventory.py    GET  /inventory/products, /inventory/review
    web/
      index.html             single-file dashboard (no build step)
  scripts/
    seed.py                  create/refresh the SQLite demo database
  tests/                     domain, persistence, API, and pipeline tests
docker-compose.yml           API + PostgreSQL
```

Why this shape: the pipeline speaks only in `Observation` value objects, so the
detector or OCR engine can be swapped or mocked without touching the domain or
API. Reconciliation is pure (standard library only), so the most important
rule is the easiest to test. The repository isolates all SQL, so switching
SQLite ↔ Postgres is a one-line config change.

---

## Data model

| Table               | Purpose                                                        |
|---------------------|----------------------------------------------------------------|
| `products`          | Current stock, one row per SKU (`quantity` is the live count). |
| `observations`      | Every raw detection+OCR result, stored verbatim with both confidences. |
| `inventory_changes` | The decision per observation (`apply`/`review`) and its stock `delta`. |

Observations and changes are append-only, so the raw evidence behind every
count is preserved — that's what makes review and auditing possible. Review
changes carry a null `product_id` and never alter `quantity` until a human
confirms them.

---

## API

| Method | Path                   | Description                                        |
|--------|------------------------|----------------------------------------------------|
| `POST` | `/inference/observe`   | Upload an image; returns observation/apply/review counts and the changes. |
| `GET`  | `/inventory/products`  | Current stock (list of `{sku, name, quantity}`).   |
| `GET`  | `/inventory/review`    | Changes awaiting human review.                     |
| `GET`  | `/health`              | Liveness probe.                                    |
| `GET`  | `/`                    | The dashboard UI.                                  |

Interactive API docs are auto-generated at `/docs`.

The **review threshold** is configurable via `REVIEW_CONFIDENCE_THRESHOLD`
(default `0.5`). An observation's score is its detection confidence, or the
*minimum* of detection and OCR confidence when text was read — so a confident
box with an unreadable label is still treated as uncertain.

---

## Testing

```bash
cd backend
uv run pytest
```

Tests run without model weights, Tesseract, or a database server: a fake vision
pipeline supplies fixed observations and an in-memory SQLite database backs the
API and persistence tests. Coverage spans the reconciliation rule, the
repository's apply/review behaviour, the pipeline wiring (with fakes), and the
API end-to-end.

---

## Stack

Python · FastAPI · YOLO (ultralytics) · Tesseract OCR · OpenCV · SQLAlchemy ·
PostgreSQL / SQLite · Docker

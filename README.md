# Vision-Based Smart Inventory Detection System

A computer-vision inventory system that converts images of storage racks into structured inventory records.

The pipeline uses **YOLO** for object detection and **Tesseract OCR** for reading labels and SKUs. Each observation passes through a confidence-based reconciliation step:

* high-confidence observations update inventory automatically;
* low-confidence observations are added to a review queue.

The application runs as a single FastAPI service. `uvicorn app.main:app` serves both the REST API and the dashboard at `http://localhost:8000`.

This is a personal learning project built to explore an end-to-end computer-vision system: inference, API design, persistence, testing, and containerization. It runs locally and does not require a GPU or cloud services.

For a detailed account of the design decisions and implementation process, see [`WALKTHROUGH.md`](./WALKTHROUGH.md).

---

## Quick Start

The project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

A demo can be run without PostgreSQL, model weights, or Tesseract:

```bash
cd backend
uv sync
uv run vision-inventory demo
```

Open `http://localhost:8000` to view the dashboard with seeded products and review items.

A prebuilt `backend/demo.db` is also included.

### CLI

Installing the project with `uv sync` provides the `vision-inventory` command:

```bash
# Seed a SQLite database and start the application
uv run vision-inventory demo

# Run using DATABASE_URL or .env configuration
uv run vision-inventory run

# Run on a custom port with auto-reload
uv run vision-inventory run --port 9000 --reload

# Override the database and review threshold
uv run vision-inventory run \
    --database-url postgresql+psycopg://vision:vision@localhost/vision \
    --threshold 0.6

# Rebuild the demo database
uv run vision-inventory seed --reset
```

Run the following to see the available options for any command:

```bash
uv run vision-inventory <command> --help
```

To run inference on uploaded images, provide a YOLO weights file with `--model path/to/yolo.pt` and install Tesseract. Uploaded images then pass through the complete pipeline:

```text
detect → OCR → reconcile → persist
```

---

## Docker

The application can also run with PostgreSQL using Docker Compose:

```bash
docker compose up --build
```

This starts:

* the API on port `8000`;
* PostgreSQL on port `5432`.

The Docker image uses CPU-only PyTorch wheels.

---

## Architecture

The system is split into vision, domain, persistence, and API layers.

```text
image bytes
    │
    ▼
vision/pipeline
    │
    ├──► vision/detector    YOLO
    └──► vision/ocr         Tesseract
    │
    ▼
Observation
(label, text, detection confidence, OCR confidence)
    │
    ▼
domain/reconciliation
    │
    ├──► APPLY              update inventory
    └──► REVIEW             queue for manual review
    │
    ▼
persistence/repository
    │
    ▼
products
observations
inventory_changes
    │
    ▼
API + dashboard
```

### Project structure

```text
backend/
  pyproject.toml
  app/
    cli.py                   CLI entry point
    config.py                environment and .env configuration
    deps.py                  FastAPI dependency providers
    main.py                  application setup

    vision/
      detector.py            YOLO wrapper
      ocr.py                 Tesseract wrapper
      pipeline.py            image → detection → OCR → observations

    domain/
      reconciliation.py      apply/review decision logic

    persistence/
      database.py            SQLAlchemy engine and session setup
      models.py              database models
      repository.py          database access layer

    api/
      routes_inference.py    inference endpoints
      routes_inventory.py    inventory and review endpoints

    web/
      index.html             dashboard

  scripts/
    seed.py                  demo database setup

  tests/                     domain, persistence, pipeline, and API tests

docker-compose.yml           API + PostgreSQL
```

The vision pipeline returns `Observation` value objects rather than database or API models. This keeps the detector and OCR implementation separate from the rest of the application and makes both easy to replace with test doubles.

Reconciliation is implemented as pure domain logic with no framework or database dependencies. Database access is isolated in the repository layer.

---

## Data Model

| Table               | Purpose                                                     |
| ------------------- | ----------------------------------------------------------- |
| `products`          | Current inventory state, with one row per SKU.              |
| `observations`      | Raw detection and OCR results with their confidence scores. |
| `inventory_changes` | Reconciliation decisions and resulting stock deltas.        |

Observations and inventory changes are append-only. This preserves the evidence and decision associated with each inventory update.

Items sent for review do not modify product quantities until they are resolved.

---

## Reconciliation

Each observation receives a confidence score.

If OCR returns text:

```text
score = min(detection_confidence, ocr_confidence)
```

Otherwise:

```text
score = detection_confidence
```

The score is compared against `REVIEW_CONFIDENCE_THRESHOLD`, which defaults to `0.5`.

```text
score >= threshold  → APPLY
score < threshold   → REVIEW
```

Using the minimum confidence prevents a high-confidence object detection from automatically updating inventory when its label was read poorly.

The threshold can be configured through the environment or the CLI.

---

## API

| Method | Path                  | Description                                                                     |
| ------ | --------------------- | ------------------------------------------------------------------------------- |
| `POST` | `/inference/observe`  | Process an uploaded image and persist the resulting observations and decisions. |
| `GET`  | `/inventory/products` | Return the current inventory.                                                   |
| `GET`  | `/inventory/review`   | Return items awaiting review.                                                   |
| `GET`  | `/health`             | Liveness check.                                                                 |
| `GET`  | `/`                   | Serve the dashboard.                                                            |

Interactive OpenAPI documentation is available at:

```text
http://localhost:8000/docs
```

---

## Testing

Run the test suite with:

```bash
cd backend
uv run pytest
```

The tests do not require model weights, Tesseract, PostgreSQL, or a GPU.

A fake vision pipeline provides deterministic observations, while persistence and API tests use an in-memory SQLite database.

The test suite covers:

* reconciliation logic;
* inventory apply/review behaviour;
* repository operations;
* pipeline integration with mocked detector and OCR components;
* API request flows.

---

## Stack

**Python** · **FastAPI** · **YOLO / Ultralytics** · **Tesseract OCR** · **OpenCV** · **SQLAlchemy** · **PostgreSQL** · **SQLite** · **Docker**

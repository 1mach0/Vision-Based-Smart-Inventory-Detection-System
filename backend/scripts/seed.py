"""Create a ready-to-explore SQLite demo database.

Why standard-library sqlite3 instead of the app's SQLAlchemy models?
So the demo works with **zero setup** — you can generate and inspect the
database without installing torch/ultralytics/fastapi. The schema below mirrors
``app/persistence/models.py`` exactly, so when the API starts against this file
its ``Base.metadata.create_all`` sees the tables already exist and does
nothing. If you change the ORM models, update this schema to match (or just run
the API once against a fresh file to let SQLAlchemy build the schema).

Usage:
    python -m scripts.seed                 # writes ./demo.db
    python -m scripts.seed --path foo.db   # custom path
    python -m scripts.seed --reset         # overwrite an existing file

Then point the API at it:
    DATABASE_URL=sqlite:///./demo.db uvicorn app.main:app --reload
"""
from __future__ import annotations

import argparse
import os
import sqlite3

# Schema kept byte-for-byte compatible with the ORM models. The unique index
# name (ix_products_sku) matches what SQLAlchemy generates for a column marked
# index=True, unique=True, so create_all won't try to recreate it.
SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id       INTEGER PRIMARY KEY,
    sku      VARCHAR NOT NULL,
    name     VARCHAR NOT NULL,
    quantity INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_products_sku ON products (sku);

CREATE TABLE IF NOT EXISTS observations (
    id                   INTEGER PRIMARY KEY,
    label                VARCHAR NOT NULL,
    text                 VARCHAR NOT NULL,
    detection_confidence FLOAT   NOT NULL,
    ocr_confidence       FLOAT   NOT NULL,
    created_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS inventory_changes (
    id             INTEGER PRIMARY KEY,
    observation_id INTEGER NOT NULL REFERENCES observations(id),
    product_id     INTEGER REFERENCES products(id),
    delta          INTEGER NOT NULL,
    disposition    VARCHAR NOT NULL,
    created_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

# Sample stock. Each applied unit below becomes one observation + one applied
# inventory_change, so quantities and the change history stay consistent — the
# same invariant the real /inference/observe path maintains.
APPLIED = [
    # (detector label, OCR'd SKU, product name, units on the rack)
    ("bottle", "SKU-1001", "Cola 330ml", 3),
    ("bottle", "SKU-1002", "Spring Water 500ml", 2),
    ("box", "SKU-1003", "Classic Chips", 1),
]

# Low-confidence detections: recorded as evidence but held for a human. No SKU
# was read (text=""), and confidence sits below the default 0.5 threshold.
REVIEW = [
    ("bottle", 0.34),
    ("can", 0.28),
]


def seed(path: str, reset: bool) -> None:
    if reset and os.path.exists(path):
        os.remove(path)

    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)

        # Applied items: create the product, then N observations + changes.
        for label, sku, name, units in APPLIED:
            cur = conn.execute(
                "INSERT INTO products (sku, name, quantity) VALUES (?, ?, ?)",
                (sku, name, units),
            )
            product_id = cur.lastrowid
            for _ in range(units):
                obs_id = conn.execute(
                    "INSERT INTO observations "
                    "(label, text, detection_confidence, ocr_confidence) "
                    "VALUES (?, ?, ?, ?)",
                    (label, sku, 0.91, 0.86),
                ).lastrowid
                conn.execute(
                    "INSERT INTO inventory_changes "
                    "(observation_id, product_id, delta, disposition) "
                    "VALUES (?, ?, ?, ?)",
                    (obs_id, product_id, 1, "apply"),
                )

        # Review items: observation + change with no product and no stock impact.
        for label, det_conf in REVIEW:
            obs_id = conn.execute(
                "INSERT INTO observations "
                "(label, text, detection_confidence, ocr_confidence) "
                "VALUES (?, ?, ?, ?)",
                (label, "", det_conf, 0.0),
            ).lastrowid
            conn.execute(
                "INSERT INTO inventory_changes "
                "(observation_id, product_id, delta, disposition) "
                "VALUES (?, ?, ?, ?)",
                (obs_id, None, 1, "review"),
            )

        conn.commit()

        products = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        obs = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        review = conn.execute(
            "SELECT COUNT(*) FROM inventory_changes WHERE disposition='review'"
        ).fetchone()[0]
        print(f"Seeded {path}: {products} products, {obs} observations, {review} pending review.")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed a SQLite demo database.")
    parser.add_argument("--path", default="demo.db", help="output DB file (default: demo.db)")
    parser.add_argument("--reset", action="store_true", help="overwrite if it exists")
    args = parser.parse_args()
    seed(args.path, args.reset)


if __name__ == "__main__":
    main()

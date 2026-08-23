"""Build the sqlite-vec KNN vector index for the catalog.

This is a one-shot, idempotent build that runs after `import_catalog.py` (or
`build_chair_demo_db.py`). The catalog is treated as immutable — there is no
reindex path. Run this once per database file when the catalog is finalized.

Embedding input per listing:
  title_en + first 3 bullet_point values + first 5 item_keywords + brand_en
  Joined with ". " / ", ", lowercased, whitespace-trimmed. Empty inputs skipped.

Model: BAAI/bge-small-en-v1.5 via fastembed (ONNX runtime)
  - 384-dimensional float embeddings, normalized to unit length
  - ~30 MB on disk, CPU-friendly, no torch dependency
  - Apache-2.0 license, no auth, deterministic across runs
  - Fast cold-start — important for Streamlit Cloud free tier where a
    750MB torch install would blow the disk quota

Storage: a `vec_items` virtual table in the SAME sqlite file (foreign-key
referenced against `listings.item_id`). A sidecar `vec_index_meta` table
records model name, dimension, and build timestamp so future-you knows
when the index was built and with what.

Usage:
    # Full 145k catalog (default)
    uv run python scripts/build_vector_index.py

    # Chair subset
    uv run python scripts/build_vector_index.py \\
        --database retail_catalog_chair.db

    # Skip the model download step (CI uses a cached model dir)
    HF_HUB_OFFLINE=1 uv run python scripts/build_vector_index.py
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infrastructure.database import (
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    VECTOR_SCHEMA,
    build_embedding_text,
    load_sqlite_vec,
    vec_index_meta_insert,
    vec_items_count,
)


def _load_model(model_name: str):
    """Load the embedding model via fastembed.

    fastembed uses ONNX Runtime so there's no torch dependency — important
    for free-tier deploy targets that don't have a CUDA runtime and where
    a 750MB torch install would blow the disk quota. The BGE-small model
    is ~30MB and CPU-friendly.
    """
    from fastembed import TextEmbedding  # type: ignore

    print(f"[vec] loading model {model_name!r} ...", file=sys.stderr)
    t0 = time.perf_counter()
    model = TextEmbedding(model_name=model_name)
    print(
        f"[vec] model loaded in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )
    return model


def _fetch_catalog_rows(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return (item_id, embedding_text) for every active listing.

    Embedding text is built in SQL using the same logic as
    ``build_embedding_text`` — title + brand + first 3 bullet_point rows +
    first 5 item_keywords rows. Putting this in SQL avoids loading 11M
    text_value rows into Python just to pick the first 3 bullets.
    """
    sql = """
    SELECT
      l.item_id,
      -- Embedding input: title + brand + first 3 bullet_point + first 5 item_keywords
      LOWER(
        TRIM(
          COALESCE(l.title_en, '') ||
          CASE WHEN l.brand_en IS NOT NULL AND l.brand_en <> '' THEN '. ' || l.brand_en ELSE '' END ||
          CASE WHEN bp.combined IS NOT NULL THEN '. ' || bp.combined ELSE '' END ||
          CASE WHEN ik.combined IS NOT NULL THEN ', ' || ik.combined ELSE '' END
        )
      ) AS embedding_text
    FROM listings l
    -- First 3 bullet_point rows for this listing, concatenated
    LEFT JOIN (
      SELECT listing_id, GROUP_CONCAT(value, '. ') AS combined
      FROM (
        SELECT listing_id, value,
               ROW_NUMBER() OVER (PARTITION BY listing_id ORDER BY id) AS rn
        FROM listing_text_values
        WHERE attribute = 'bullet_point'
      )
      WHERE rn <= 3
      GROUP BY listing_id
    ) bp ON bp.listing_id = l.id
    -- First 5 item_keywords rows for this listing, concatenated
    LEFT JOIN (
      SELECT listing_id, GROUP_CONCAT(value, ', ') AS combined
      FROM (
        SELECT listing_id, value,
               ROW_NUMBER() OVER (PARTITION BY listing_id ORDER BY id) AS rn
        FROM listing_text_values
        WHERE attribute = 'item_keywords'
      )
      WHERE rn <= 5
      GROUP BY listing_id
    ) ik ON ik.listing_id = l.id
    WHERE l.url_active = 1
      AND l.title_en <> ''
    ORDER BY l.id
    """
    return [(row[0], row[1] or "") for row in conn.execute(sql).fetchall()]


def _encode_and_insert(
    conn: sqlite3.Connection,
    model,
    rows: list[tuple[str, str]],
    *,
    batch_size: int = 8,
) -> tuple[int, int]:
    """Encode ``rows`` in batches and INSERT into vec_items.

    Returns (inserted_rows, skipped_rows). Skipped = empty embedding text
    (no title, no bullets, no keywords — degenerate listings that have
    nothing meaningful to embed).
    """
    if vec_items_count(conn) > 0:
        raise RuntimeError(
            "vec_items already populated; this is a one-time build. "
            "DROP TABLE vec_items first if you really mean to rebuild."
        )

    insert_sql = "INSERT INTO vec_items(item_id, embedding) VALUES (?, ?)"
    inserted = 0
    skipped = 0
    total = len(rows)
    t_start = time.perf_counter()

    for batch_start in range(0, total, batch_size):
        batch = rows[batch_start : batch_start + batch_size]
        ids = [item_id for item_id, _ in batch]
        texts = [text for _, text in batch]

        # fastembed's TextEmbedding.embed takes a list of strings and
        # returns a generator of numpy arrays. Use the model's native
        # batching — fastembed dispatches through ONNX with internal
        # batching sized to its parallelism setting.
        vectors = list(model.embed(texts))

        batch_rows = []
        for item_id, text, vec in zip(ids, texts, vectors):
            if not text.strip():
                # Degenerate listing (no title + no bullets + no keywords).
                # Skip — embedding an empty string is meaningless and would
                # pollute nearest-neighbor results.
                skipped += 1
                continue
            # sqlite-vec stores float32 little-endian bytes.
            batch_rows.append((item_id, vec.astype("float32").tobytes()))

        if batch_rows:
            conn.executemany(insert_sql, batch_rows)
            conn.commit()
            inserted += len(batch_rows)

        elapsed = time.perf_counter() - t_start
        done = batch_start + len(batch)
        rate = done / elapsed if elapsed > 0 else 0
        print(
            f"[vec] {done:,}/{total:,} rows ({inserted:,} inserted, "
            f"{skipped:,} skipped) — {rate:.0f} rows/s",
            file=sys.stderr,
        )

    return inserted, skipped


def _verify_index(conn: sqlite3.Connection, expected: int) -> None:
    """Sanity-check the index: row count, sample query, sane distances."""
    actual = vec_items_count(conn)
    assert actual == expected, f"vec_items has {actual} rows, expected {expected}"

    # Sample query: pick one random item's vector, search for its nearest
    # neighbor, assert the top hit is itself (distance 0) or a near-duplicate.
    sample = conn.execute(
        "SELECT item_id, embedding FROM vec_items ORDER BY RANDOM() LIMIT 1"
    ).fetchone()
    assert sample is not None, "vec_items is empty after build"
    sample_id, sample_vec = sample[0], sample[1]

    hits = conn.execute(
        """
        SELECT item_id, distance
        FROM vec_items
        WHERE embedding MATCH ?
        AND k = 5
        ORDER BY distance
        """,
        (sample_vec,),
    ).fetchall()
    assert hits, "KNN MATCH returned no rows — sqlite-vec KNN is broken"
    top_id, top_distance = hits[0]
    assert top_id == sample_id, (
        f"top-1 KNN hit is {top_id!r}, expected {sample_id!r} "
        "(self-match should always win)"
    )
    assert top_distance < 0.01, (
        f"self-match distance {top_distance} too high — embeddings not unit-norm?"
    )
    print(
        f"[vec] verify OK: {actual:,} rows, "
        f"sample self-match distance={top_distance:.6f}, "
        f"next-4 distances={[round(h[1], 3) for h in hits[1:]]}",
        file=sys.stderr,
    )


def build_vector_index(database: Path, *, model_name: str = EMBEDDING_MODEL) -> dict:
    """Build the vec_items index in ``database``. Idempotent only via
    vec_items_count check — re-running on a populated index raises.
    """
    database = database.expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError(f"Catalog database not found: {database}")

    # Open in write mode; enable foreign keys so vec_items references listings.
    conn = sqlite3.connect(str(database))
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        # Load sqlite-vec extension before creating any vec0 tables.
        load_sqlite_vec(conn)

        # Create vec_items + meta table. Idempotent (IF NOT EXISTS).
        conn.executescript(VECTOR_SCHEMA)
        conn.commit()

        # If already built, refuse to rebuild (no reindex path; data is immutable).
        if vec_items_count(conn) > 0:
            print(
                f"[vec] vec_items already populated ({vec_items_count(conn):,} rows); "
                f"skipping build. To rebuild, DROP TABLE vec_items first.",
                file=sys.stderr,
            )
            return {"rows": vec_items_count(conn), "skipped": "already_built"}

        # Pull all (item_id, embedding_text) rows from SQL.
        print("[vec] fetching catalog rows from SQL ...", file=sys.stderr)
        t0 = time.perf_counter()
        rows = _fetch_catalog_rows(conn)
        print(
            f"[vec] fetched {len(rows):,} active listings in "
            f"{time.perf_counter() - t0:.1f}s",
            file=sys.stderr,
        )
        if not rows:
            raise RuntimeError("No listings to embed — catalog is empty")

        # Load model + encode + insert.
        model = _load_model(model_name)
        t0 = time.perf_counter()
        inserted, skipped = _encode_and_insert(conn, model, rows)
        encode_seconds = time.perf_counter() - t0
        print(
            f"[vec] embedded {inserted:,} listings in {encode_seconds:.1f}s "
            f"({inserted / encode_seconds:.0f} listings/s)",
            file=sys.stderr,
        )

        # Write metadata + verify.
        vec_index_meta_insert(conn, model_name=model_name, dim=EMBEDDING_DIM)
        conn.commit()
        _verify_index(conn, expected=inserted)

        return {
            "rows": inserted,
            "skipped": skipped,
            "model": model_name,
            "dim": EMBEDDING_DIM,
            "encode_seconds": round(encode_seconds, 1),
        }
    finally:
        conn.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("retail_catalog.db"),
        help="Path to the catalog SQLite database (default: retail_catalog.db)",
    )
    parser.add_argument(
        "--model",
        default=EMBEDDING_MODEL,
        help=f"sentence-transformers model name (default: {EMBEDDING_MODEL})",
    )
    args = parser.parse_args()
    result = build_vector_index(args.database, model_name=args.model)
    if result.get("skipped") == "already_built":
        print(f"[vec] skipped — index already present ({result['rows']:,} rows)")
    else:
        print(
            f"[vec] build complete: {result['rows']:,} rows, "
            f"model={result['model']!r}, dim={result['dim']}, "
            f"{result['encode_seconds']}s encode time"
        )


if __name__ == "__main__":
    cli()

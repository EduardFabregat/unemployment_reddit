"""Step 1 (save_reddit_data_as_db): load Pushshift-style .zst dumps into SQLite."""

import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import zstandard as zstd

SUBMISSIONS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS Submissions (
        Author TEXT,
        Title TEXT,
        Score INTEGER,
        Publish_Date INTEGER,
        Post_ID TEXT,
        Text_Body TEXT
    )
"""
SUBMISSIONS_INSERT = (
    "INSERT INTO Submissions (Author, Title, Score, Publish_Date, Post_ID, Text_Body) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)

COMMENTS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS Comments (
        Author TEXT,
        Body TEXT,
        Score INTEGER,
        Publish_Date INTEGER,
        Comment_ID TEXT,
        Parent_ID TEXT,
        Link_ID TEXT
    )
"""
COMMENTS_INSERT = (
    "INSERT INTO Comments (Author, Body, Score, Publish_Date, Comment_ID, Parent_ID, Link_ID) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)"
)


def _submission_row(d: dict) -> tuple:
    return (
        d.get("author"),
        d.get("title"),
        d.get("score"),
        int(d.get("created_utc", 0)),
        d.get("id"),
        d.get("selftext"),
    )


def _comment_row(d: dict) -> tuple:
    return (
        d.get("author"),
        d.get("body"),
        d.get("score"),
        int(d.get("created_utc", 0)),
        d.get("id"),
        d.get("parent_id"),  # lets you rebuild reply threads
        d.get("link_id"),  # ties the comment back to its submission
    )


_KINDS = {
    "submissions": (SUBMISSIONS_SCHEMA, SUBMISSIONS_INSERT, _submission_row, 10_000),
    "comments": (COMMENTS_SCHEMA, COMMENTS_INSERT, _comment_row, 50_000),
}


def ingest_zst(zst_path: str | Path, db_path: str | Path, kind: str) -> int:
    """Stream a zstd-compressed NDJSON dump into the ``Submissions`` or ``Comments`` table.

    Rows are appended, so running this twice on the same file duplicates them.
    Returns the number of rows inserted.
    """
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {sorted(_KINDS)}, got {kind!r}")
    schema, insert_sql, to_row, log_every = _KINDS[kind]

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(schema)
        conn.commit()

        count = 0
        batch = []
        with open(zst_path, "rb") as fh:
            with zstd.ZstdDecompressor().stream_reader(fh) as reader:
                for line in io.TextIOWrapper(reader, encoding="utf-8"):
                    batch.append(to_row(json.loads(line)))
                    count += 1
                    if len(batch) >= log_every:
                        conn.executemany(insert_sql, batch)
                        batch.clear()
                        print(f"[{datetime.now():%H:%M:%S}] Processed {count} {kind}...")
        if batch:
            conn.executemany(insert_sql, batch)
        conn.commit()

        print("Optimizing database structure...")
        conn.execute("VACUUM;")
    finally:
        conn.close()

    print(f"Success! Total of {count} {kind} imported into '{db_path}'.")
    return count

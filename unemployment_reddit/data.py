"""Load the SQLite corpus and turn it into the flat comment/submission table used everywhere."""

import sqlite3
from pathlib import Path

import pandas as pd

from .config import REMOVED_PLACEHOLDERS, SAMPLE_SIZE, SEED

# Comments and submission titles are stacked into one table. The id columns are deliberately
# asymmetric (see threads.add_clean_ids): for comments ``submission`` holds the comment's own id,
# for submissions ``parent`` holds the post's own id.
CORPUS_QUERY = """
SELECT
    a.Author as author,
    a.Body as comment,
    a.Score as score,
    datetime(a.Publish_Date, 'unixepoch') AS date,
    a.Parent_ID as parent,
    a.Comment_ID as submission,
    a.Link_ID as link,
    'comment' AS type
FROM Comments AS a

UNION ALL

SELECT
    b.Author AS author,
    b.Title AS comment,
    b.Score AS score,
    datetime(b.Publish_Date, 'unixepoch') AS date,
    b.Post_ID AS parent,
    NULL AS submission,
    NULL AS link,
    'submission' AS type
FROM Submissions AS b
"""


def load_corpus(db_path: str | Path) -> pd.DataFrame:
    """Read every comment and submission title from the database into one DataFrame."""
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql_query(CORPUS_QUERY, conn)
    finally:
        conn.close()
    df["simple_date"] = pd.to_datetime(df["date"]).dt.date
    return df


def clean_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case the text, drop deleted/removed placeholders and de-duplicate identical texts."""
    df = df.copy()
    df["comment"] = df["comment"].str.lower()
    df = df[~df["comment"].isin(REMOVED_PLACEHOLDERS)]
    return df.drop_duplicates(subset=["comment"], keep="first")


def draw_sample(
    df: pd.DataFrame, n: int = SAMPLE_SIZE, random_state: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the corpus into a random training sample and the remaining rows.

    The sample fits the topic model; the rest is assigned out-of-sample afterwards.
    """
    df_docs = df.sample(n, random_state=random_state)
    df_rest = df[~df.index.isin(df_docs.index)]
    return df_docs, df_rest


def monthly_share(df: pd.DataFrame) -> pd.Series:
    """Share of rows per calendar month, used to check the sample mirrors the population."""
    months = pd.to_datetime(df["simple_date"]).dt.to_period("M")
    return months.value_counts(normalize=True).sort_index()

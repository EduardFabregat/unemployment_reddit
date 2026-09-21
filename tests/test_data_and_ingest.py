import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
import zstandard as zstd

from unemployment_reddit import data, frames, topics
from unemployment_reddit.ingest import ingest_zst


def write_zst(path, records):
    payload = "\n".join(json.dumps(r) for r in records).encode("utf-8")
    path.write_bytes(zstd.ZstdCompressor().compress(payload))


@pytest.fixture
def db(tmp_path):
    submissions = tmp_path / "subs.zst"
    comments = tmp_path / "comments.zst"
    write_zst(
        submissions,
        [
            {"author": "a", "title": "Help with PUA", "score": 3, "created_utc": 1_600_000_000.0,
             "id": "p1", "selftext": "body"},
            {"author": "b", "title": "[removed]", "score": 1, "created_utc": 1_600_000_100,
             "id": "p2", "selftext": ""},
        ],
    )
    write_zst(
        comments,
        [
            {"author": "c", "body": "Thanks!", "score": 2, "created_utc": 1_600_000_200,
             "id": "c1", "parent_id": "t3_p1", "link_id": "t3_p1"},
            {"author": "d", "body": "thanks!", "score": 1, "created_utc": 1_600_000_300,
             "id": "c2", "parent_id": "t1_c1", "link_id": "t3_p1"},
            {"author": "[deleted]", "body": "[deleted]", "score": 1, "created_utc": 1_600_000_400,
             "id": "c3", "parent_id": "t1_c1", "link_id": "t3_p1"},
        ],
    )
    path = tmp_path / "test.db"
    assert ingest_zst(submissions, path, "submissions") == 2
    assert ingest_zst(comments, path, "comments") == 3
    return path


def test_ingest_rejects_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        ingest_zst(tmp_path / "x.zst", tmp_path / "x.db", "posts")


def test_ingest_writes_expected_rows(db):
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT Post_ID, Publish_Date FROM Submissions ORDER BY Post_ID").fetchall() == [
        ("p1", 1_600_000_000), ("p2", 1_600_000_100)
    ]
    assert conn.execute("SELECT Parent_ID FROM Comments WHERE Comment_ID='c2'").fetchone() == ("t1_c1",)
    conn.close()


def test_load_and_clean_corpus(db):
    df = data.load_corpus(db)
    assert len(df) == 5
    assert set(df["type"]) == {"comment", "submission"}
    # comments keep their own id in `submission`; submissions keep theirs in `parent`
    assert set(df.loc[df["type"] == "submission", "parent"]) == {"p1", "p2"}
    assert df["submission"].notna().sum() == 3

    clean = data.clean_corpus(df)
    # "[removed]" title and "[deleted]" comment dropped; "Thanks!"/"thanks!" collapse after lower-casing
    assert sorted(clean["comment"]) == ["help with pua", "thanks!"]


def test_draw_sample_partitions_the_corpus():
    df = pd.DataFrame({"comment": [f"t{i}" for i in range(100)]})
    docs, rest = data.draw_sample(df, n=30, random_state=1)
    assert len(docs) == 30 and len(rest) == 70
    assert docs.index.intersection(rest.index).empty
    docs_again, _ = data.draw_sample(df, n=30, random_state=1)
    assert docs.index.equals(docs_again.index)


def test_theta_columns_handles_outlier_column():
    labels = ["0_a", "1_b"]
    assert topics._theta_columns(3, labels) == ["-1_outliers", "0_a", "1_b"]
    assert topics._theta_columns(2, labels) == labels
    with pytest.raises(ValueError):
        topics._theta_columns(5, labels)


def test_assemble_theta_table_puts_sample_first_and_drops_missing_text():
    df_docs = pd.DataFrame({"comment": ["a", "b"]}, index=[7, 3])
    theta_docs = pd.DataFrame({"-1_outliers": np.nan, "0_x": [0.9, 0.8]}, index=[7, 3])
    df_rest = pd.DataFrame({"comment": ["c", None]}, index=[0, 1])
    theta_rest = pd.DataFrame({"-1_outliers": [0.1, 0.2], "0_x": [0.3, 0.4]}, index=[0, 1])

    table = topics.assemble_theta_table(df_docs, theta_docs, df_rest, theta_rest)
    assert table["comment"].tolist() == ["a", "b", "c"]
    assert table["0_x"].tolist() == [0.9, 0.8, 0.3]
    assert table["-1_outliers"].isna().tolist() == [True, True, False]


def test_frame_mapping_is_a_partition_of_the_named_topics():
    mapped = [t for cols in frames.FRAME_MAPPING.values() for t in cols]
    assert len(mapped) == len(set(mapped)) == len(frames.TOPIC_NAMES) == 48
    assert set(mapped) == set(frames.TOPIC_NAMES)


def test_add_frame_features_sums_topic_columns():
    cols = {t: [0.1, 0.2] for t in frames.TOPIC_NAMES}
    df = pd.DataFrame(cols)
    df, frame_cols = frames.add_frame_features(df)
    assert frame_cols == list(frames.FRAME_MAPPING)
    n_tax = len(frames.FRAME_MAPPING["frame_taxation_and_financial_reporting"])
    assert df["frame_taxation_and_financial_reporting"].tolist() == pytest.approx([0.1 * n_tax, 0.2 * n_tax])

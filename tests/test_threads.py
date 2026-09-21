import networkx as nx
import numpy as np
import pandas as pd
import pytest

from unemployment_reddit import threads


def make_corpus():
    """One submission (p1) with a reply tree, plus a comment whose parent was removed."""
    rows = [
        # (type, submission [own id for comments], parent)
        ("submission", None, "p1"),
        ("comment", "c1", "t3_p1"),
        ("comment", "c2", "t1_c1"),
        ("comment", "c3", "t1_c2"),
        ("comment", "c4", "t1_c1"),
        ("comment", "c5", "t3_p1"),
        ("comment", "orphan", "t1_gone"),
    ]
    return pd.DataFrame(rows, columns=["type", "submission", "parent"])


def test_add_clean_ids_strips_prefixes_and_roots_submissions():
    df = threads.add_clean_ids(make_corpus())
    assert df["clean_id"].tolist() == ["p1", "c1", "c2", "c3", "c4", "c5", "orphan"]
    assert pd.isna(df.loc[0, "clean_parent_id"])
    assert df["clean_parent_id"].tolist()[1:] == ["p1", "c1", "c2", "c1", "p1", "gone"]


def test_depth_counts_links_including_to_missing_parent():
    df = threads.add_clean_ids(make_corpus())
    depth = threads.compute_depths(df["clean_id"], df["clean_parent_id"])
    assert depth == {"p1": 0, "c1": 1, "c2": 2, "c3": 3, "c4": 2, "c5": 1, "orphan": 1}


def test_descendant_counts():
    df = threads.add_clean_ids(make_corpus())
    counts = threads.descendant_counts(df["clean_id"], df["clean_parent_id"])
    assert counts == {"p1": 5, "c1": 3, "c2": 1, "c3": 0, "c4": 0, "c5": 0, "orphan": 0}


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_cec_matches_bfs_definition(alpha):
    df = threads.add_clean_ids(make_corpus())
    scores = threads.cec_scores(df["clean_id"], df["clean_parent_id"], alpha=alpha)

    G = nx.DiGraph()
    G.add_edges_from(df[["clean_parent_id", "clean_id"]].dropna().itertuples(index=False, name=None))
    # Only ids that are rows matter: scores are mapped back to the table through clean_id.
    for node in df["clean_id"]:
        lengths = nx.single_source_shortest_path_length(G, node)
        expected = sum(alpha ** (d - 1) for d in lengths.values() if d > 0)
        assert scores.get(node, 0.0) == pytest.approx(expected)


def test_alpha_one_equals_descendant_count():
    df = threads.add_clean_ids(make_corpus())
    ids, parents = df["clean_id"], df["clean_parent_id"]
    counts = threads.descendant_counts(ids, parents)
    cec = threads.cec_scores(ids, parents, alpha=1.0)
    assert {k: int(v) for k, v in cec.items()} == counts


def test_cycle_is_reported_not_looped_forever():
    with pytest.raises(ValueError, match="Cycle"):
        threads.compute_depths(["a", "b"], ["b", "a"])


def test_add_thread_features_columns():
    df = threads.add_thread_features(make_corpus())
    top = df.set_index("clean_id")
    assert top.loc["p1", "target_descendant_count"] == 5
    assert top.loc["c3", "depth"] == 3
    # c2 and c4 are direct replies (1 each), c3 is a grandchild (alpha = 0.5)
    assert top.loc["c1", "cec_score"] == pytest.approx(2.5)


def test_weighted_mean_cec_and_one_hot():
    df = pd.DataFrame(
        {
            "predicted_emotion": ["joy", "anger", "joy"],
            "cec_score": [1.0, 4.0, 3.0],
        }
    )
    df, cols = threads.one_hot_emotions(df)
    assert cols == ["anger", "joy"]
    ranking = threads.weighted_mean_cec(df, cols)
    assert ranking.index.tolist() == ["anger", "joy"]
    assert ranking["joy"] == pytest.approx(2.0)

"""Reply-tree features: depth, descendant counts and Content Engagement Capacity (CEC).

Reddit threads are forests (every comment has exactly one parent), so all three quantities can
be computed in a single bottom-up pass instead of a BFS/recursion per node.
"""

import numpy as np
import pandas as pd

_PREFIX = r"^t[13]_"


def add_clean_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``clean_id`` and ``clean_parent_id`` with Reddit's ``t1_``/``t3_`` prefixes removed.

    Comments carry their own id in ``submission`` and their parent in ``parent``; submissions
    carry their own id in ``parent`` and have no parent (they are the tree roots).

    Columns are added to ``df`` in place (the full table is several GB) and ``df`` is returned.
    """
    is_comment = df["type"] == "comment"
    own_id = df["submission"].where(is_comment, df["parent"])
    parent_id = df["parent"].where(is_comment)  # NaN for submissions
    df["clean_id"] = own_id.astype("string").str.replace(_PREFIX, "", regex=True)
    df["clean_parent_id"] = parent_id.astype("string").str.replace(_PREFIX, "", regex=True)
    return df


def _parent_lookup(ids, parents) -> dict:
    """Map each id to its parent id, or None for roots."""
    return {
        str(i): (None if pd.isna(p) else str(p)) for i, p in zip(ids, parents)
    }


def compute_depths(ids, parents) -> dict:
    """Number of parent links above each node.

    A node whose parent is missing from the data still counts that one link, so a top-level
    comment on a removed submission has depth 1.
    """
    parent_of = _parent_lookup(ids, parents)
    depth: dict[str, int] = {}
    for start in parent_of:
        stack = []
        node = start
        while node not in depth:
            parent = parent_of[node]
            if parent is None:
                depth[node] = 0
                break
            if parent not in parent_of:
                depth[node] = 1
                break
            if len(stack) > len(parent_of):
                raise ValueError(f"Cycle detected in reply tree near id {start!r}")
            stack.append(node)
            node = parent
        base = depth[node]
        while stack:
            base += 1
            depth[stack.pop()] = base
    return depth


def _bottom_up(ids, parents, contribution) -> dict:
    """Accumulate ``contribution(child_total)`` from every node into its parent, leaves first."""
    parent_of = _parent_lookup(ids, parents)
    depth = compute_depths(ids, parents)
    total = dict.fromkeys(parent_of, 0.0)
    for node in sorted(parent_of, key=depth.__getitem__, reverse=True):
        parent = parent_of[node]
        if parent in total:
            total[parent] += contribution(total[node])
    return total


def descendant_counts(ids, parents) -> dict:
    """Number of replies below each node, at any depth."""
    totals = _bottom_up(ids, parents, lambda child_total: 1 + child_total)
    return {k: int(v) for k, v in totals.items()}


def cec_scores(ids, parents, alpha: float = 0.5) -> dict:
    """Content Engagement Capacity of each node.

    Each descendant at distance ``d`` contributes ``alpha ** (d - 1)``: ``alpha=0`` credits
    direct replies only, ``alpha=1`` counts every descendant equally.
    """
    return _bottom_up(ids, parents, lambda child_total: 1 + alpha * child_total)


def add_thread_features(df: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
    """Add ``clean_id``, ``clean_parent_id``, ``depth``, ``target_descendant_count`` and ``cec_score``."""
    df = add_clean_ids(df)
    ids = df["clean_id"].astype(str)
    parents = df["clean_parent_id"]

    depth = compute_depths(ids, parents)
    df["depth"] = ids.map(depth).astype(int)
    df["target_descendant_count"] = ids.map(descendant_counts(ids, parents)).astype(int)
    df["cec_score"] = ids.map(cec_scores(ids, parents, alpha=alpha)).fillna(0.0)
    return df


def weighted_mean_cec(df: pd.DataFrame, weight_cols, score_col: str = "cec_score") -> pd.Series:
    """CEC averaged over documents, weighted by each frame/emotion column, best first."""
    result = {
        col: (df[col] * df[score_col]).sum() / df[col].sum()
        for col in weight_cols
        if col in df.columns
    }
    return pd.Series(result).sort_values(ascending=False)


def one_hot_emotions(
    df: pd.DataFrame, column: str = "predicted_emotion"
) -> tuple[pd.DataFrame, list[str]]:
    """Append one 0/1 column per predicted emotion; returns (df, emotion column names)."""
    dummies = df[column].str.get_dummies()
    return df.join(dummies), list(dummies.columns)

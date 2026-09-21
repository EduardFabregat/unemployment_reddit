"""Steps 2-3 (unemployment_1, assign_docs_to_topics): BERTopic on a sample, then the full corpus.

The heavy libraries (bertopic, umap, hdbscan, gensim, sentence-transformers) are imported inside
the functions that need them so ``import unemployment_reddit`` stays cheap.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ASSIGN_CHUNK_SIZE,
    CLUSTER_SIZES,
    EMBEDDING_MODEL,
    SEED,
    UMAP_NEIGHBORS,
)

OUTLIER_LABEL = "-1_outliers"


def load_embedding_model(model_name: str = EMBEDDING_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_documents(docs: list[str], embedding_model) -> np.ndarray:
    """Encode documents once so tuning and the final fit reuse the same vectors."""
    return embedding_model.encode(docs, show_progress_bar=True)


def _build_bertopic(
    embedding_model, min_cluster_size, n_neighbors, prediction_data=False, **bertopic_kwargs
):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=SEED,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=prediction_data,
    )
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        **bertopic_kwargs,
    )


def _cv_coherence(topic_model, tokenized_docs, id2word) -> tuple[int, float]:
    """Gensim C_v coherence of the model's topics; returns (number of topics scored, score)."""
    from gensim.models.coherencemodel import CoherenceModel

    topic_words = []
    for topic_id, words_with_weights in topic_model.get_topics().items():
        if topic_id == -1:
            continue
        # Gensim fails on words missing from its dictionary and on empty strings.
        words = [
            word
            for word, _ in words_with_weights
            if word is not None and word in id2word.token2id and str(word).strip() != ""
        ]
        if len(words) > 1:
            topic_words.append(words)

    if not topic_words:
        print("Warning: no topics with dictionary-matching words were found; scoring 0.")
        return 0, 0.0

    cm = CoherenceModel(
        topics=topic_words, texts=tokenized_docs, dictionary=id2word, coherence="c_v"
    )
    return len(topic_words), cm.get_coherence()


def tune_hyperparameters(
    docs: list[str],
    embeddings: np.ndarray,
    embedding_model,
    cluster_sizes=CLUSTER_SIZES,
    umap_neighbors=UMAP_NEIGHBORS,
) -> pd.DataFrame:
    """Grid-search HDBSCAN ``min_cluster_size`` x UMAP ``n_neighbors`` by C_v coherence."""
    import gensim.corpora as corpora

    tokenized_docs = [doc.split() for doc in docs]
    id2word = corpora.Dictionary(tokenized_docs)

    results = []
    for size in cluster_sizes:
        for neighbors in umap_neighbors:
            print(f"Testing min_cluster_size={size} | n_neighbors={neighbors}")
            model = _build_bertopic(embedding_model, size, neighbors, verbose=False)
            model.fit_transform(docs, embeddings=embeddings)
            num_topics, score = _cv_coherence(model, tokenized_docs, id2word)
            print(f"  -> {num_topics} topics, C_v coherence {score:.4f}")
            results.append(
                {
                    "min_cluster_size": size,
                    "n_neighbors": neighbors,
                    "num_topics": num_topics,
                    "coherence_score": score,
                }
            )
    return pd.DataFrame(results).sort_values("coherence_score", ascending=False)


def best_params(df_results: pd.DataFrame) -> tuple[int, int]:
    """(min_cluster_size, n_neighbors) of the most coherent configuration."""
    top = df_results.sort_values("coherence_score", ascending=False).iloc[0]
    return int(top["min_cluster_size"]), int(top["n_neighbors"])


def fit_topic_model(
    docs: list[str],
    embeddings: np.ndarray,
    embedding_model,
    min_cluster_size: int,
    n_neighbors: int,
):
    """Fit the final BERTopic model; returns (model, probs) with ``probs`` = docs x topics."""
    model = _build_bertopic(
        embedding_model,
        min_cluster_size,
        n_neighbors,
        prediction_data=True,
        calculate_probabilities=True,
        verbose=True,
    )
    _, probs = model.fit_transform(docs, embeddings=embeddings)
    print("Shape of theta matrix:", probs.shape)
    return model, probs


def topic_labels(topic_model) -> list[str]:
    """Names such as ``'3_claim_my_to_it'`` for topics 0..n-1 (the outlier topic excluded)."""
    info = topic_model.get_topic_info()
    return info.loc[info["Topic"] >= 0].sort_values("Topic")["Name"].tolist()


def sample_theta(probs: np.ndarray, labels: list[str], index: pd.Index) -> pd.DataFrame:
    """Theta matrix of the training sample, labelled and aligned with the sample rows.

    The outlier column is NaN: HDBSCAN soft-membership probabilities only cover real topics.
    """
    theta = pd.DataFrame(probs, columns=labels, index=index)
    theta.insert(0, OUTLIER_LABEL, np.nan)
    return theta


def save_model(topic_model, embeddings: np.ndarray, model_path: Path, embeddings_path: Path):
    topic_model.save(model_path, serialization="safetensors")
    np.save(embeddings_path, embeddings)


def load_model(model_path: Path, embeddings_path: Path | None = None):
    """Load a saved BERTopic model (and its sample embeddings, if a path is given)."""
    from bertopic import BERTopic

    model = BERTopic.load(model_path, embedding_model=EMBEDDING_MODEL)
    embeddings = np.load(embeddings_path) if embeddings_path is not None else None
    return model, embeddings


def assign_out_of_sample(
    topic_model, df_rest: pd.DataFrame, chunk_size: int = ASSIGN_CHUNK_SIZE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign topics to documents the model was not fitted on, in chunks to bound memory.

    Returns ``(assignments, theta)`` indexed like ``df_rest``. ``theta`` columns follow
    ``[OUTLIER_LABEL] + topic_labels(topic_model)``.

    Note: ``BERTopic.transform`` scores unseen documents against topic embeddings, so these
    values are cosine similarities (they can be negative) rather than the HDBSCAN membership
    probabilities in ``sample_theta``. The two are not on the same scale.
    """
    labels = topic_labels(topic_model)
    total = len(df_rest)
    print(f"Assigning topics to {total} rows in chunks of {chunk_size}...")

    assignment_chunks, theta_chunks = [], []
    for start in range(0, total, chunk_size):
        chunk = df_rest.iloc[start : start + chunk_size]
        topics, probs = topic_model.transform(chunk["comment"].tolist())
        probs = np.asarray(probs)
        if probs.ndim < 2:
            raise RuntimeError("transform() returned no per-topic scores; cannot build theta")

        assignment_chunks.append(
            pd.DataFrame(
                {
                    "submission": chunk["submission"].to_numpy(),
                    "Assigned_Topic": [int(t) for t in topics],
                    "Assignment_Probability": probs.max(axis=1).astype(float),
                },
                index=chunk.index,
            )
        )
        theta_chunks.append(pd.DataFrame(probs, index=chunk.index))
        print(f"  -> rows {start} to {min(start + chunk_size, total)} done")

    theta = pd.concat(theta_chunks)
    theta.columns = _theta_columns(theta.shape[1], labels)
    return pd.concat(assignment_chunks), theta


def _theta_columns(n_cols: int, labels: list[str]) -> list[str]:
    if n_cols == len(labels) + 1:
        return [OUTLIER_LABEL] + labels
    if n_cols == len(labels):
        return list(labels)
    raise ValueError(f"theta has {n_cols} columns but the model has {len(labels)} topics")


def assemble_theta_table(
    df_docs: pd.DataFrame,
    theta_docs: pd.DataFrame,
    df_rest: pd.DataFrame,
    theta_rest: pd.DataFrame,
) -> pd.DataFrame:
    """One row per document: metadata plus a theta column per topic, sample rows first."""
    docs = df_docs.join(theta_docs)
    rest = df_rest.join(theta_rest)
    table = pd.concat([docs, rest], ignore_index=True)
    return table[table["comment"].notna()]

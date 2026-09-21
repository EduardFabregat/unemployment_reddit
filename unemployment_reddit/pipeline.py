"""The four stages, in the order the notebooks were run.

    1. ingest    save_reddit_data_as_db.ipynb    .zst dumps -> SQLite
    2. topics    unemployment_1.ipynb            clean, sample, tune + fit BERTopic on the sample
    3. assign    assign_docs_to_topics.ipynb     assign the rest of the corpus, then emotions
    4. analyze   cec_analysis.ipynb              thread features, CEC, XGBoost, SHAP, GLM

Each stage reads what the previous one wrote (paths come from ``Config``), so they can be run
one at a time.
"""

import pandas as pd

from . import data, emotions, frames, models, threads, topics
from .config import Config


def run_ingest(cfg: Config, submissions_zst=None, comments_zst=None) -> None:
    from .ingest import ingest_zst

    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
    if submissions_zst:
        ingest_zst(submissions_zst, cfg.db_path, "submissions")
    if comments_zst:
        ingest_zst(comments_zst, cfg.db_path, "comments")


def run_topics(cfg: Config) -> None:
    """Fit the topic model on a random sample and save the model, embeddings and sample theta."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    df = data.clean_corpus(data.load_corpus(cfg.db_path))
    df_docs, _ = data.draw_sample(df)
    docs = df_docs["comment"].tolist()

    embedding_model = topics.load_embedding_model()
    embeddings = topics.embed_documents(docs, embedding_model)

    leaderboard = topics.tune_hyperparameters(docs, embeddings, embedding_model)
    print("\n--- TUNING LEADERBOARD ---")
    print(leaderboard)
    min_cluster_size, n_neighbors = topics.best_params(leaderboard)

    topic_model, probs = topics.fit_topic_model(
        docs, embeddings, embedding_model, min_cluster_size, n_neighbors
    )
    theta = topics.sample_theta(probs, topics.topic_labels(topic_model), df_docs.index)

    df_docs.to_parquet(cfg.sample_path)
    theta.to_parquet(cfg.sample_theta_path)
    topics.save_model(topic_model, embeddings, cfg.model_path, cfg.embeddings_path)


def run_assign(cfg: Config, with_emotions: bool = True, **emotion_kwargs) -> None:
    """Assign every document outside the sample to a topic, then classify emotions."""
    df = data.clean_corpus(data.load_corpus(cfg.db_path))
    df_docs = pd.read_parquet(cfg.sample_path)
    theta_docs = pd.read_parquet(cfg.sample_theta_path)
    df_rest = df[~df.index.isin(df_docs.index)]

    topic_model, _ = topics.load_model(cfg.model_path)
    _, theta_rest = topics.assign_out_of_sample(topic_model, df_rest)

    table = topics.assemble_theta_table(df_docs, theta_docs, df_rest, theta_rest)
    table["simple_date"] = pd.to_datetime(table["simple_date"])
    table.to_parquet(cfg.meta_theta_path, compression="snappy")

    if with_emotions:
        table = emotions.add_emotions(table, **emotion_kwargs)
        table.to_parquet(cfg.emotions_path, compression="snappy")


def run_analysis(cfg: Config, xgb_iter: int = 20) -> dict:
    """Thread features, CEC by frame and emotion, XGBoost + SHAP and the Negative Binomial GLM."""
    df = pd.read_parquet(cfg.emotions_path)

    df = threads.add_thread_features(df, alpha=0.5)
    df, frame_cols = frames.add_frame_features(df)
    df, emotion_cols = threads.one_hot_emotions(df)

    features = models.feature_columns(frame_cols, emotion_cols)
    X, y = df[features], df[models.TARGET]

    out = {
        "cec_by_frame": threads.weighted_mean_cec(df, frame_cols),
        "cec_by_emotion": threads.weighted_mean_cec(df, emotion_cols),
        "xgboost": models.tune_xgboost(X, y, n_iter=xgb_iter),
    }
    out["negative_binomial"], out["irr"] = models.fit_negative_binomial(X, y)

    print("=== CEC by frame ===")
    print(out["cec_by_frame"])
    print("\n=== CEC by emotion ===")
    print(out["cec_by_emotion"])
    print(f"\nHold-out MAE: {out['xgboost']['mae']:.4f}")
    print(f"Spearman rank correlation: {out['xgboost']['spearman']:.4f}")
    print(out["xgboost"]["importance"].head(10))
    print(out["negative_binomial"].summary())
    print("\n=== Incident Rate Ratios (exp(beta)) ===")
    print(out["irr"])
    return out

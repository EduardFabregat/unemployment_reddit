"""Emotion classification of every document (GoEmotions labels).

Mirrors the version that produced ``meta_theta_df_6_8_26_bert_emotion.parquet`` in
``Unemployment_Reddit_Sentiment_Analysis.ipynb``: a single text-classification pipeline that
streams the texts in large batches on the GPU.
"""

import pandas as pd
from tqdm.auto import tqdm

from .config import EMOTION_BATCH_SIZE, EMOTION_MAX_LENGTH, EMOTION_MODEL


class _TextDataset:
    """Minimal dataset of ``{"comment": text}`` rows, the shape ``KeyDataset`` expects."""

    def __init__(self, texts: list[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> dict:
        return {"comment": self.texts[i]}


def _make_pipeline(model_name: str):
    import torch
    from transformers import pipeline

    kwargs = {}
    if torch.cuda.is_available():
        kwargs = {"device": 0, "dtype": torch.float16}
    else:
        print("No CUDA GPU found: classifying on the CPU, which will be much slower.")
    return pipeline("text-classification", model=model_name, **kwargs)


def classify_emotions(
    texts: list[str],
    model_name: str = EMOTION_MODEL,
    batch_size: int = EMOTION_BATCH_SIZE,
    max_length: int = EMOTION_MAX_LENGTH,
) -> pd.DataFrame:
    """Predict the top emotion label and its confidence for each text, in input order."""
    from transformers.pipelines.pt_utils import KeyDataset

    emotion_pipeline = _make_pipeline(model_name)
    dataset = KeyDataset(_TextDataset(texts), "comment")

    results = []
    for out in tqdm(
        emotion_pipeline(
            dataset, batch_size=batch_size, truncation=True, max_length=max_length
        ),
        total=len(texts),
        desc="Classifying emotions",
    ):
        results.append(
            {"predicted_emotion": out["label"], "emotion_confidence": out["score"]}
        )
    return pd.DataFrame(results)


def add_emotions(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Return ``df`` (index reset) with ``predicted_emotion`` and ``emotion_confidence`` added.

    Missing comments are classified as empty strings, since the tokenizer fails on NaN.
    """
    texts = df["comment"].fillna("").astype(str).tolist()
    emotions = classify_emotions(texts, **kwargs)
    return pd.concat([df.reset_index(drop=True), emotions], axis=1)

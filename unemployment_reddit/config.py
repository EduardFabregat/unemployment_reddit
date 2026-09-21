"""Paths, model names and constants shared across the pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path

SEED = 42

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Model, batch size and truncation used in Unemployment_Reddit_Sentiment_Analysis.ipynb, which
# produced meta_theta_df_6_8_26_bert_emotion.parquet (27 GoEmotions labels) for cec_analysis.
EMOTION_MODEL = "SamLowe/roberta-base-go_emotions"
EMOTION_BATCH_SIZE = 256
EMOTION_MAX_LENGTH = 128

# Placeholder bodies Reddit leaves behind for deleted / removed content.
REMOVED_PLACEHOLDERS = (
    "[deleted]",
    "[deleted by user]",
    "[ removed by reddit ]",
    "[removed]",
    "deleted",
    "removed",
)

SAMPLE_SIZE = 50_000
ASSIGN_CHUNK_SIZE = 20_000

# Hyperparameter grid swept in unemployment_1.ipynb.
CLUSTER_SIZES = (30, 50, 100, 150, 200, 300)
UMAP_NEIGHBORS = (10, 15, 20, 25)

# Number of macro-frames chosen in unemployment_1.ipynb (the silhouette peak was overridden).
N_MACRO_FRAMES = 6


@dataclass
class Config:
    """Where inputs and artifacts live. Defaults can be overridden with environment variables.

    ``tag`` is appended to artifact names so different runs never overwrite each other.
    """

    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("REDDIT_DATA_DIR", "data"))
    )
    db_path: Path | None = None
    tag: str = "12_7_26"

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        if self.db_path is None:
            default_db = self.data_dir / "Unemployment_Full_History.db"
            self.db_path = Path(os.environ.get("REDDIT_DB_PATH", default_db))
        self.db_path = Path(self.db_path)

    # --- artifacts written by the pipeline stages ---------------------------------------
    @property
    def sample_path(self) -> Path:
        return self.data_dir / f"sample_docs_{self.tag}.parquet"

    @property
    def sample_theta_path(self) -> Path:
        return self.data_dir / f"sample_theta_{self.tag}.parquet"

    @property
    def model_path(self) -> Path:
        return self.data_dir / f"unemployment_bertopic_model_{self.tag}"

    @property
    def embeddings_path(self) -> Path:
        return self.data_dir / f"unemployment_docs_embeddings_{self.tag}.npy"

    @property
    def meta_theta_path(self) -> Path:
        return self.data_dir / f"meta_theta_df_{self.tag}.parquet"

    @property
    def emotions_path(self) -> Path:
        return self.data_dir / f"meta_theta_df_{self.tag}_emotion.parquet"

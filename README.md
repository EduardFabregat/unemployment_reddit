# unemployment_reddit

Topic frames, emotions and Content Engagement Capacity (CEC) on r/unemployment.
This package is the notebook code, organised so each step can be rerun and tested.

## Pipeline

| Stage | Command | Replaces notebook | Reads | Writes |
|---|---|---|---|---|
| 1 | `ingest` | `save_reddit_data_as_db` | `.zst` dumps | SQLite db |
| 2 | `topics` | `unemployment_1` | db | sample, sample theta, BERTopic model, embeddings |
| 3 | `assign` | `assign_docs_to_topics` | db + stage 2 | theta table `.parquet`, then `_emotion.parquet` |
| 4 | `analyze` | `cec_analysis` | emotion `.parquet` | prints CEC, XGBoost, SHAP inputs, GLM |

```
pip install -e ".[all]"            # or pick extras: topics, emotions, analysis, plots, dev

python -m unemployment_reddit --data-dir data ingest --submissions X_submissions.zst --comments X_comments.zst
python -m unemployment_reddit --data-dir data topics
python -m unemployment_reddit --data-dir data assign
python -m unemployment_reddit --data-dir data analyze
```

`--data-dir`, `--db` and `--tag` can also be set with `REDDIT_DATA_DIR`, `REDDIT_DB_PATH`
(`--tag` only via the flag). Artifact names include the tag (default `12_7_26`).

## Modules

| Module | Contents |
|---|---|
| `config.py` | paths, model names, grids, constants |
| `ingest.py` | `.zst` to SQLite |
| `data.py` | load corpus, clean, sample |
| `topics.py` | embeddings, tuning, BERTopic fit, save/load, out-of-sample assignment, theta table |
| `frames.py` | topic names, macro-frame mapping, frame-count diagnostics, topic map data |
| `emotions.py` | parallel emotion classification |
| `threads.py` | depth, descendant count, CEC (single bottom-up pass), one-hot emotions |
| `models.py` | Poisson XGBoost search, SHAP, Negative Binomial GLM |
| `plots.py` | the figures (comments per day, sample check, frame curves, topic map, SHAP) |
| `pipeline.py` | the four stages; `cli.py` is the command line |

Individual pieces are importable, e.g. in a notebook:

```python
from unemployment_reddit import frames, plots, topics
topic_model, _ = topics.load_model("unemployment_bertopic_model_12_7_26")
plots.topic_map(frames.topic_map_frame(topic_model)); 
```

## Things to know

- Emotions come from `SamLowe/roberta-base-go_emotions` (batch 256, max length 128, GPU in
  float16 when CUDA is available), copied from `Unemployment_Reddit_Sentiment_Analysis.ipynb`. That
  notebook wrote the parquet `cec_analysis` reads, and its 27 labels match this model. The 6-label
  CPU attempts in `assign_docs_to_topics` were not carried over.
- `TOPIC_NAMES` and `FRAME_MAPPING` in `frames.py` belong to the fitted 48-topic model. Refitting
  the topic model means re-deriving them.
- Out-of-sample theta (`transform`) is on a different scale from the sample theta; see the
  docstring of `topics.assign_out_of_sample`.
- The sample is now seeded (`SEED = 42`); the notebook's `df.sample(50000)` was not.

## Tests

```
pip install -e ".[dev]" && pytest
```

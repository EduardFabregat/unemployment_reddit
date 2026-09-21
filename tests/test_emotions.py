import numpy as np
import pandas as pd

from unemployment_reddit import emotions


def test_add_emotions_keeps_order_and_fills_missing_text(monkeypatch):
    seen = {}

    def fake_pipeline(model_name):
        seen["model"] = model_name

        def run(dataset, batch_size, truncation, max_length):
            seen.update(batch_size=batch_size, truncation=truncation, max_length=max_length)
            return ({"label": f"emo_{text}", "score": 0.5} for text in dataset)

        return run

    monkeypatch.setattr(emotions, "_make_pipeline", fake_pipeline)

    df = pd.DataFrame({"comment": ["a", np.nan, "c"]}, index=[10, 20, 30])
    out = emotions.add_emotions(df)

    assert seen == {
        "model": "SamLowe/roberta-base-go_emotions",
        "batch_size": 256,
        "truncation": True,
        "max_length": 128,
    }
    assert out["predicted_emotion"].tolist() == ["emo_a", "emo_", "emo_c"]  # NaN became ""
    assert out["emotion_confidence"].tolist() == [0.5, 0.5, 0.5]
    assert out.index.tolist() == [0, 1, 2]

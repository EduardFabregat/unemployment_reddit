"""Step 4 (cec_analysis): what predicts how many replies a document receives?

Two complementary models on the same feature matrix: a tuned Poisson XGBoost (with SHAP for
interpretation) and a Negative Binomial GLM (for incident rate ratios).
"""

import gc

import numpy as np
import pandas as pd

from .config import SEED

TARGET = "target_descendant_count"
CONTROL_COLS = ["emotion_confidence", "depth", "score"]

PARAM_GRID = {
    "n_estimators": [100, 250, 500],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7, 9],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 0.5, 1.0],
}


def feature_columns(frame_cols, emotion_cols) -> list[str]:
    return list(frame_cols) + list(emotion_cols) + CONTROL_COLS


def tune_xgboost(X: pd.DataFrame, y: pd.Series, n_iter: int = 20, cv: int = 5) -> dict:
    """Randomized-search a Poisson XGBoost and evaluate it on an 80/20 hold-out.

    Returns a dict with the fitted search, ``best_model``, hold-out ``mae`` and Spearman rank
    correlation, ``importance`` (a DataFrame) and the hold-out split.
    """
    import xgboost as xgb
    from scipy.stats import spearmanr
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import RandomizedSearchCV, train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    search = RandomizedSearchCV(
        estimator=xgb.XGBRegressor(
            objective="count:poisson", tree_method="hist", random_state=SEED
        ),
        param_distributions=PARAM_GRID,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        verbose=2,
        random_state=SEED,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Best hyperparameters: {search.best_params_}")

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)
    importance = pd.DataFrame(
        {"Feature": X.columns, "Importance": best_model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    return {
        "search": search,
        "best_model": best_model,
        "mae": mean_absolute_error(y_test, predictions),
        "spearman": spearmanr(y_test, predictions)[0],
        "importance": importance,
        "X_test": X_test,
        "y_test": y_test,
    }


def shap_values(model, X: pd.DataFrame, n: int = 10_000):
    """SHAP values on a random sample of ``n`` rows; returns ``(explanation, sample)``."""
    import shap

    sample = X.sample(n=min(n, len(X)), random_state=SEED)
    return shap.TreeExplainer(model)(sample), sample


def fit_negative_binomial(X: pd.DataFrame, y: pd.Series, sample_size: int = 150_000):
    """Negative Binomial GLM on a random subsample; returns ``(results, incident_rate_ratios)``.

    Fitting on the full ~1.9M-row table is memory-heavy, so the model is estimated on a
    float32 subsample.
    """
    import statsmodels.api as sm

    idx = X.sample(n=min(sample_size, len(X)), random_state=SEED).index
    X_sample = sm.add_constant(X.loc[idx].astype(np.float32), has_constant="add")
    y_sample = y.loc[idx].astype(np.float32)

    results = sm.GLM(y_sample, X_sample, family=sm.families.NegativeBinomial()).fit()
    irr = np.exp(results.params)
    gc.collect()
    return results, irr

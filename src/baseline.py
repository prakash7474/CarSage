"""XGBoost baseline for the used-car price task.

This is the "traditional" reference point: a gradient-boosted tree model that
is actually trained on the 80% split. Kumo Tabular is never trained — it only
reads that split as in-context examples — so comparing the two tells us how
well in-context learning does here.

Run me directly:
    python -m src.baseline
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import data_prep

RESULTS_DIR = Path("results")


def _prepare_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    log_target: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Encode categoricals with a fixed vocabulary shared by train and test."""
    feats = data_prep.FEATURE_COLUMNS
    x_train = train_df[feats].copy()
    x_test = test_df[feats].copy()

    # Give every categorical column the categories seen in training; unseen
    # values in the test set become NaN (XGBoost handles them natively).
    for col in data_prep.CATEGORICAL_COLUMNS:
        train_vals = x_train[col].astype(str)
        test_vals = x_test[col].astype(str)
        categories = pd.Index(sorted(train_vals.unique()))
        x_train[col] = pd.Categorical(train_vals, categories=categories)
        # Replace unseen values with NaN *before* constructing the
        # Categorical (pandas >= 3 deprecates out-of-category values here).
        test_vals = test_vals.where(test_vals.isin(categories), other=None)
        x_test[col] = pd.Categorical(test_vals, categories=categories)

    y_train = train_df[data_prep.TARGET_COL].astype(float)
    y_test = test_df[data_prep.TARGET_COL].astype(float)
    if log_target:
        y_train, y_test = np.log1p(y_train), np.log1p(y_test)
    return x_train, y_train, x_test, y_test


def train_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    log_target: bool = True,
    random_state: int = data_prep.RANDOM_STATE,
) -> dict:
    """Train XGBoost on the context split and score it on the test split.

    Returns a metrics dict: MAE, RMSE, R2 (in rupees) and timing.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from xgboost import XGBRegressor

    x_train, y_train, x_test, y_test = _prepare_features(
        train_df, test_df, log_target
    )

    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        enable_categorical=True,
        random_state=random_state,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    model.fit(x_train, y_train)
    train_seconds = time.perf_counter() - t0

    # Predict, then undo the log transform before computing rupee errors.
    t0 = time.perf_counter()
    pred_log = model.predict(x_test)
    predict_seconds = time.perf_counter() - t0
    pred = np.expm1(pred_log) if log_target else pred_log
    y_true = np.expm1(y_test) if log_target else y_test

    mae = mean_absolute_error(y_true, pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, pred)))
    r2 = r2_score(y_true, pred)

    return {
        "model": "xgboost",
        "log_target": log_target,
        "n_context": int(len(train_df)),
        "n_test": int(len(test_df)),
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "train_seconds": round(train_seconds, 3),
        "predict_seconds": round(predict_seconds, 3),
        "seconds_per_prediction": round(predict_seconds / max(len(test_df), 1), 6),
        "predictions": pred,
        "y_true": np.asarray(y_true),
    }


def main() -> None:
    train_df, test_df = data_prep.load_and_prepare()
    metrics = train_baseline(train_df, test_df)

    print("XGBoost baseline")
    print(f"  context rows : {metrics['n_context']:,}")
    print(f"  test rows    : {metrics['n_test']:,}")
    print(f"  MAE          : {metrics['mae']:,.0f} INR")
    print(f"  RMSE         : {metrics['rmse']:,.0f} INR")
    print(f"  R2           : {metrics['r2']:.4f}")
    print(f"  train time   : {metrics['train_seconds']:.2f}s")
    print(f"  predict time : {metrics['predict_seconds']:.3f}s")

    RESULTS_DIR.mkdir(exist_ok=True)
    # Drop the large arrays before saving the metrics table.
    serialisable = {k: v for k, v in metrics.items()
                    if k not in ("predictions", "y_true")}
    out = RESULTS_DIR / "baseline.json"
    out.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to {out}")


if __name__ == "__main__":
    main()

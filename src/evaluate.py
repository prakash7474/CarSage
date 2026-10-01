"""Evaluate Kumo Tabular and the XGBoost baseline on the held-out test set.

Metrics reported for both models (in rupees):
  - MAE          mean absolute error
  - RMSE         root mean squared error
  - R2           coefficient of determination
  - time         seconds per prediction

For Kumo we additionally report **range coverage**: the fraction of true test
prices that fall inside the model's 10th-90th percentile interval. A
well-calibrated 80% interval should cover about 80% of the test rows.

Results are written to results/comparison.csv and printed.

Run with:
    python -m src.evaluate
    python -m src.evaluate --sizes small --context 2000 5000 all
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import baseline, data_prep, model as kumo

RESULTS_DIR = Path("results")
COMPARISON_CSV = RESULTS_DIR / "comparison.csv"

#: How many query rows to send through the model per forward pass (VRAM cap).
PREDICT_BATCH_SIZE = 512

#: The interval we check coverage for.
LOW_Q, HIGH_Q = 0.1, 0.9


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE / RMSE / R2 in rupees."""
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_kumo(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    size: str = "small",
    context_limit: int | None = None,
    log_target: bool = True,
    batch_size: int = PREDICT_BATCH_SIZE,
) -> dict:
    """Run Kumo Tabular over the whole test set and collect metrics."""
    if not kumo.cuda_available():
        raise RuntimeError("CUDA GPU required to evaluate Kumo Tabular.")

    import torch

    context_df = train_df
    if context_limit is not None:
        context_df = train_df.sample(
            min(context_limit, len(train_df)),
            random_state=data_prep.RANDOM_STATE,
        )

    model = kumo.load_model(size=size)
    context = kumo.build_context(context_df, log_target=log_target)

    # Predict in batches to keep VRAM bounded.
    preds, lows, highs = [], [], []
    t0 = time.perf_counter()
    for start in range(0, len(test_df), batch_size):
        batch = test_df.iloc[start:start + batch_size]
        p, lo, hi = kumo.predict(model, context, batch, LOW_Q, HIGH_Q)
        preds.append(np.asarray(p))
        lows.append(np.asarray(lo))
        highs.append(np.asarray(hi))
    elapsed = time.perf_counter() - t0

    pred = np.concatenate(preds)
    low = np.concatenate(lows)
    high = np.concatenate(highs)
    y_true = test_df[data_prep.TARGET_COL].to_numpy(dtype=float)

    out = _metrics(y_true, pred)
    out.update({
        "model": "kumo_tabular",
        "size": size,
        "log_target": log_target,
        "context_rows": int(len(context_df)),
        "n_test": int(len(test_df)),
        "coverage": float(((y_true >= low) & (y_true <= high)).mean()),
        "seconds_per_prediction": round(elapsed / max(len(test_df), 1), 6),
        "vram_gb": kumo.device_info().get("vram_gb"),
    })
    del model
    torch.cuda.empty_cache()
    return out


def evaluate_xgboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    log_target: bool = True,
) -> dict:
    """Evaluate the trained XGBoost baseline."""
    metrics = baseline.train_baseline(train_df, test_df, log_target=log_target)
    res = _metrics(metrics["y_true"], metrics["predictions"])
    res.update({
        "model": "xgboost",
        "size": "-",
        "log_target": log_target,
        "context_rows": int(len(train_df)),
        "n_test": int(len(test_df)),
        "coverage": None,             # XGBoost gives no quantiles by default
        "seconds_per_prediction": metrics["seconds_per_prediction"],
        "vram_gb": None,
    })
    return res


def run_comparison(
    sizes: list[str],
    context_limits: list[int | None],
) -> pd.DataFrame:
    """Evaluate every requested (size, context) combo plus XGBoost."""
    train_df, test_df = data_prep.load_and_prepare()
    rows = [evaluate_xgboost(train_df, test_df)]

    if kumo.cuda_available():
        for size in sizes:
            for limit in context_limits:
                print(f"  ... Kumo {size}, context={limit or 'all'}")
                try:
                    rows.append(
                        evaluate_kumo(train_df, test_df,
                                      size=size, context_limit=limit)
                    )
                except RuntimeError as exc:      # e.g. out of memory
                    print(f"      skipped: {exc}")
    else:
        print("  [WARN] no CUDA GPU — skipping Kumo, only XGBoost evaluated.")

    return pd.DataFrame(rows)


def _parse_context(values: list[str]) -> list[int | None]:
    """Turn ['2000','all'] into [2000, None]."""
    out: list[int | None] = []
    for v in values:
        out.append(None if str(v).lower() == "all" else int(v))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=["small"],
                        choices=list(kumo.KUMO_SIZES))
    parser.add_argument("--context", nargs="+",
                        default=["2000", "5000", "all"],
                        help="context row counts, or 'all'")
    args = parser.parse_args()

    print("Evaluating models on the 20% test split...")
    table = run_comparison(args.sizes, _parse_context(args.context))

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(COMPARISON_CSV, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n" + table.to_string(index=False))
    print(f"\nSaved results to {COMPARISON_CSV}")


if __name__ == "__main__":
    main()

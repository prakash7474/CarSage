"""Evaluate Kumo Tabular and the XGBoost baseline on the held-out test set.

Both models are scored with the SAME 80/20 split (random_state=42):

  * MAE, RMSE, R2             accuracy in rupees
  * prediction time           total and per row
  * range coverage            fraction of true prices inside [P10, P90]
                              (target ~80%; XGBoost has no quantiles)
  * average range width       how wide the 80% range is, in rupees

Output:
  results/comparison.csv
  docs/images/pred_vs_actual.png     scatter, predicted vs true (+ y=x line)
  docs/images/coverage_plot.png      true price vs the P10-P90 band

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
IMAGES_DIR = Path("docs") / "images"
COMPARISON_CSV = RESULTS_DIR / "comparison.csv"

#: How many query rows to send through the model per forward pass (VRAM cap).
PREDICT_BATCH_SIZE = kumo.DEFAULT_BATCH_SIZE

#: The interval we check coverage for.
LOW_Q, HIGH_Q = kumo.LOW_Q, kumo.HIGH_Q


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


def _coverage(y_true: np.ndarray, low: np.ndarray, high: np.ndarray) -> dict:
    """Fraction of true prices inside [low, high] and the average width."""
    inside = (y_true >= low) & (y_true <= high)
    return {
        "coverage": float(inside.mean()),
        "avg_range_width": float(np.mean(high - low)),
        "n_covered": int(inside.sum()),
        "n_total": int(len(y_true)),
    }


# ---------------------------------------------------------------------------
# One model
# ---------------------------------------------------------------------------

def evaluate_kumo(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    size: str = kumo.DEFAULT_MODEL_SIZE,
    context_limit: int | None = kumo.DEFAULT_CONTEXT_ROWS,
    log_target: bool = True,
    batch_size: int = PREDICT_BATCH_SIZE,
) -> dict:
    """Run Kumo Tabular over the whole test set and collect metrics.

    The returned dict carries the raw predictions under ``predictions``,
    ``low``, ``high`` and ``y_true`` for the plots (strip them before saving).
    """
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
    # Encode the context ONCE, then reuse it for every batch (Phase 3).
    context = kumo.build_context(context_df, log_target=log_target)
    t_fit = time.perf_counter()
    context = kumo.fit_context(model, context)
    fit_seconds = time.perf_counter() - t_fit

    # Predict in batches to keep VRAM bounded on the 6 GB card.
    t0 = time.perf_counter()
    price, low, high = kumo.predict_batch(
        model, context, test_df, batch_size=batch_size,
        low_q=LOW_Q, high_q=HIGH_Q,
    )
    elapsed = time.perf_counter() - t0

    y_true = test_df[data_prep.TARGET_COL].to_numpy(dtype=float)
    out = _metrics(y_true, price)
    out.update(_coverage(y_true, low, high))
    out.update({
        "model": "kumo_tabular",
        "size": size,
        "log_target": log_target,
        "context_rows": int(len(context_df)),
        "n_test": int(len(test_df)),
        "fit_seconds": round(fit_seconds, 3),
        "predict_seconds": round(elapsed, 3),
        "seconds_per_prediction": round(elapsed / max(len(test_df), 1), 6),
        "peak_vram_gb": round(
            torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        "coverage": out["coverage"],          # already set above
        "predictions": np.asarray(price),
        "low": np.asarray(low),
        "high": np.asarray(high),
        "y_true": y_true,
    })
    del model, context
    torch.cuda.empty_cache()
    return out


def evaluate_xgboost(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    log_target: bool = True,
) -> dict:
    """Evaluate the trained XGBoost baseline (point predictions, no range)."""
    metrics = baseline.train_baseline(train_df, test_df, log_target=log_target)
    res = _metrics(metrics["y_true"], metrics["predictions"])
    res.update({
        "model": "xgboost",
        "size": "-",
        "log_target": log_target,
        "context_rows": int(len(train_df)),
        "n_test": int(len(test_df)),
        # XGBoost was trained for a single point estimate: it produces no
        # quantiles, so there is no interval to measure coverage for.
        "coverage": None,
        "avg_range_width": None,
        "n_covered": None,
        "n_total": int(len(test_df)),
        "fit_seconds": metrics["train_seconds"],
        "predict_seconds": metrics["predict_seconds"],
        "seconds_per_prediction": metrics["seconds_per_prediction"],
        "peak_vram_gb": None,
        "predictions": np.asarray(metrics["predictions"]),
        "low": None,
        "high": None,
        "y_true": np.asarray(metrics["y_true"]),
    })
    return res


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_pred_vs_actual(rows: list[dict], path: Path) -> Path:
    """Scatter of predicted vs true price with a y=x reference line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 6))

    plotted = False
    for row in rows:
        if row.get("predictions") is None or row.get("y_true") is None:
            continue
        y_true = row["y_true"]
        y_pred = row["predictions"]
        label = ("Kumo Tabular" if row["model"] == "kumo_tabular"
                 else "XGBoost")
        ax.scatter(y_true, y_pred, s=12, alpha=0.5, label=label)
        plotted = True

    if plotted:
        all_vals = np.concatenate(
            [np.asarray(r["y_true"]) for r in rows
             if r.get("y_true") is not None]
            + [np.asarray(r["predictions"]) for r in rows
               if r.get("predictions") is not None]
        )
        lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x (perfect)")

    ax.set_xlabel("True price (Rs)")
    ax.set_ylabel("Predicted price (Rs)")
    ax.set_title("Predicted vs actual used-car prices (20% test set)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_coverage(kumo_row: dict, path: Path, sample: int = 200) -> Path:
    """A sorted sample of test cars: true price vs the P10-P90 band.

    ``sample`` cars are taken evenly across the price range (not just the
    cheapest ones) and then sorted by true price, so the whole range is seen.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(exist_ok=True)
    y_true = np.asarray(kumo_row["y_true"], dtype=float)
    low = np.asarray(kumo_row["low"], dtype=float)
    high = np.asarray(kumo_row["high"], dtype=float)
    price = np.asarray(kumo_row["predictions"], dtype=float)

    order = np.argsort(y_true)
    n = min(sample, len(order))
    # Evenly spaced cars across the sorted price range, still sorted.
    idx = order[np.linspace(0, len(order) - 1, n).astype(int)]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(n)
    ax.fill_between(x, low[idx], high[idx], color="#90caf9", alpha=0.7,
                    label="P10-P90 range (80%)")
    ax.plot(x, price[idx], color="#2e7d32", lw=1.2, label="median estimate")
    ax.scatter(x, y_true[idx], s=10, color="#c62828", zorder=3,
               label="true price")
    inside = float(((y_true >= low) & (y_true <= high)).mean())
    ax.set_title(f"Coverage of the 80% range: {inside:.1%} of test prices "
                 "inside [P10, P90] (sorted sample)")
    ax.set_xlabel("Test cars, sorted by true price")
    ax.set_ylabel("Price (Rs)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def run_comparison(
    sizes: list[str],
    context_limits: list[int | None],
) -> tuple[pd.DataFrame, list[dict]]:
    """Evaluate every requested (size, context) combo plus XGBoost."""
    train_df, test_df = data_prep.load_and_prepare()
    rows: list[dict] = [evaluate_xgboost(train_df, test_df)]

    if kumo.cuda_available():
        for size in sizes:
            for limit in context_limits:
                print(f"  ... Kumo {size}, context={limit or 'all'}")
                try:
                    rows.append(
                        evaluate_kumo(train_df, test_df,
                                      size=size, context_limit=limit)
                    )
                except Exception as exc:         # e.g. out of memory
                    print(f"      skipped: {type(exc).__name__}: {exc}")
    else:
        print("  [WARN] no CUDA GPU - skipping Kumo, only XGBoost evaluated.")

    return pd.DataFrame([{k: v for k, v in r.items()
                          if k not in ("predictions", "low", "high", "y_true")}
                         for r in rows]), rows


def _parse_context(values: list[str]) -> list[int | None]:
    """Turn ['2000','all'] into [2000, None]."""
    out: list[int | None] = []
    for v in values:
        out.append(None if str(v).lower() == "all" else int(v))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=[kumo.DEFAULT_MODEL_SIZE],
                        choices=list(kumo.KUMO_SIZES))
    parser.add_argument("--context", nargs="+",
                        default=[str(kumo.DEFAULT_CONTEXT_ROWS or "all")],
                        help="context row counts, or 'all'")
    args = parser.parse_args()

    print("Evaluating models on the 20% test split...")
    table, rows = run_comparison(args.sizes, _parse_context(args.context))

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(COMPARISON_CSV, index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\n" + table.to_string(index=False))

    # --- coverage summary (honest, straight from the numbers) -------------
    kumo_rows = [r for r in rows if r["model"] == "kumo_tabular"]
    for r in kumo_rows:
        print(
            f"\nRange coverage ({r['size']}, context={r['context_rows']:,}): "
            f"{r['n_covered']} of {r['n_total']} true prices inside "
            f"[P10, P90] = {r['coverage']:.1%} (target ~80%), "
            f"average range width Rs {r['avg_range_width']:,.0f}."
        )
    if not kumo_rows:
        print("\nNo Kumo result (no CUDA GPU?), so no coverage was measured.")

    # --- plots ------------------------------------------------------------
    imgs = []
    imgs.append(plot_pred_vs_actual(rows, IMAGES_DIR / "pred_vs_actual.png"))
    if kumo_rows:
        imgs.append(plot_coverage(kumo_rows[0],
                                  IMAGES_DIR / "coverage_plot.png"))

    print(f"\nSaved results to {COMPARISON_CSV}")
    for img in imgs:
        print(f"Saved plot    to {img}")


if __name__ == "__main__":
    main()

"""Benchmark Kumo Tabular settings on the 6 GB RTX 3050.

For every (context size, model size) setting we record:

  * MAE, RMSE           accuracy on the 20% test split
  * coverage            fraction of true prices inside the P10-P90 range
  * avg_seconds         average time per predicted row (warm, cached context)
  * peak_vram_gb        torch.cuda.max_memory_allocated() during the setting
  * fit_seconds         one-off cost of encoding the context (done ONCE)
  * status / notes      "ok", or "OOM" when the setting did not fit in VRAM

Settings:
  context rows : 2,000 / 5,000 / all
  model size   : small / medium   (medium is attempted; an OOM is a result)

Output:
  results/benchmark.csv
  results/benchmark.png

Run with:
    python -m src.benchmark
    python -m src.benchmark --sizes small --context 2000 all
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import data_prep, model as kumo

RESULTS_DIR = Path("results")
BENCHMARK_CSV = RESULTS_DIR / "benchmark.csv"
BENCHMARK_PNG = RESULTS_DIR / "benchmark.png"

#: Context row counts to try; None means "use every context row".
CONTEXT_SIZES: tuple = (2000, 5000, None)

#: Model sizes to try (an OOM for "medium" is recorded, not hidden).
MODEL_SIZES: tuple = ("small", "medium")

#: Rows sent through the model per forward pass (VRAM cap).
BATCH_SIZE = 256

#: The 80% range we check coverage for.
LOW_Q, HIGH_Q = 0.1, 0.9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_vram_gb() -> float | None:
    """Peak GPU memory of this process since the last reset (torch only)."""
    if not kumo.cuda_available():
        return None
    import torch
    return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)


def _reset_peak() -> None:
    """Forget earlier peaks so each setting is measured on its own."""
    if kumo.cuda_available():
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def _free_gpu(*objects) -> None:
    """Drop model references and return their VRAM to the driver."""
    for obj in objects:
        del obj
    gc.collect()
    if kumo.cuda_available():
        import torch
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# One benchmark setting
# ---------------------------------------------------------------------------

def run_setting(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    size: str = "small",
    context_limit: int | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Evaluate one (context size, model size) combination.

    ``context_limit=None`` means "all context rows". The context is encoded
    once (fit_seconds) and reused for every batch, exactly like the app.
    """
    context_df = train_df
    if context_limit is not None:
        context_df = train_df.sample(
            min(context_limit, len(train_df)),
            random_state=data_prep.RANDOM_STATE,
        )

    row = {
        "model_size": size,
        "context_rows": int(len(context_df)),
        "context_setting": str(context_limit) if context_limit else "all",
        "n_test": int(len(test_df)),
        "status": "ok",
        "notes": "",
        "mae": None,
        "rmse": None,
        "coverage": None,
        "avg_range_width": None,
        "fit_seconds": None,
        "predict_seconds": None,
        "seconds_per_prediction": None,
        "peak_vram_gb": None,
    }

    _reset_peak()
    model = None
    try:
        model = kumo.load_model(size=size)

        # --- encode the context ONCE (this is what later clicks reuse) ----
        t0 = time.perf_counter()
        context = kumo.build_context(context_df)
        context = kumo.fit_context(model, context)
        row["fit_seconds"] = round(time.perf_counter() - t0, 3)

        # --- predict the whole test set in bounded batches ----------------
        t0 = time.perf_counter()
        price, low, high = kumo.predict_batch(
            model, context, test_df, batch_size=batch_size,
            low_q=LOW_Q, high_q=HIGH_Q,
        )
        elapsed = time.perf_counter() - t0

        y_true = test_df[data_prep.TARGET_COL].to_numpy(dtype=float)
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        row.update({
            "mae": float(mean_absolute_error(y_true, price)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, price))),
            "coverage": float(((y_true >= low) & (y_true <= high)).mean()),
            "avg_range_width": float(np.mean(high - low)),
            "predict_seconds": round(elapsed, 3),
            "seconds_per_prediction": round(
                elapsed / max(len(test_df), 1), 6),
            "peak_vram_gb": _peak_vram_gb(),
        })
    except Exception as exc:                       # OOM or anything else
        is_oom = kumo.is_out_of_memory(exc)
        row["status"] = "OOM" if is_oom else "error"
        row["notes"] = (f"torch.cuda.OutOfMemoryError: {exc}"
                        if is_oom else f"{type(exc).__name__}: {exc}")
        row["peak_vram_gb"] = _peak_vram_gb()
        print(f"      [{row['status']}] {row['notes'][:140]}")

    _free_gpu(model)
    return row


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sizes: tuple = MODEL_SIZES,
    context_sizes: tuple = CONTEXT_SIZES,
) -> pd.DataFrame:
    """Run every requested setting and return the results table."""
    rows = []
    for size in sizes:
        for limit in context_sizes:
            label = f"context={limit if limit else 'all'}"
            print(f"  ... {size:>6} | {label}")
            rows.append(run_setting(train_df, test_df, size=size,
                                    context_limit=limit))
    return pd.DataFrame(rows)


def plot_benchmark(df: pd.DataFrame, path: Path = BENCHMARK_PNG) -> Path:
    """Two panels: accuracy (MAE) and speed (seconds per prediction)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(exist_ok=True)

    def _label(value) -> str:
        """'2000' -> '2,000'; 'all' stays 'all'."""
        text = str(value)
        return f"{int(text):,}" if text.lower() != "all" else "all"

    # Keep the context order 2,000 -> 5,000 -> all on the x axis.
    order = {"2000": 0, "5000": 1, "all": 2}
    plot_df = df.copy()
    plot_df["_x"] = plot_df["context_setting"].map(
        lambda s: order.get(str(s).lower(), 9))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ok = plot_df[plot_df["status"] == "ok"]
    for size, part in ok.groupby("model_size"):
        part = part.sort_values("_x")
        axes[0].plot(part["_x"], part["mae"], marker="o", label=size)
        axes[1].plot(part["_x"], part["seconds_per_prediction"] * 1000,
                     marker="o", label=size)

    # Mark settings that did not fit in VRAM (none on this card, but keep it).
    bad = plot_df[plot_df["status"] != "ok"]
    for _, r in bad.iterrows():
        y_ref = float(ok["mae"].max()) if len(ok) else 0.0
        axes[0].annotate(f"{r['status']} ({r['model_size']})",
                         (r["_x"], y_ref),
                         textcoords="offset points", xytext=(0, -18),
                         ha="center", color="red", fontsize=9)

    axes[0].set_title("Accuracy: MAE on the 20% test set")
    axes[0].set_ylabel("MAE (Rs)")
    axes[1].set_title("Speed: average time per prediction")
    axes[1].set_ylabel("ms per row")

    ticks = sorted(plot_df["_x"].unique())
    tick_labels = [_label(next(v for v in plot_df["context_setting"]
                               if order.get(str(v).lower(), 9) == t))
                   for t in ticks]
    for ax in axes:
        ax.set_xlabel("Context rows")
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
        ax.grid(alpha=0.3)
        ax.legend(title="model size")
    fig.suptitle("CarQuantile benchmark (Kumo Tabular, RTX 3050 6 GB)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=list(MODEL_SIZES),
                        choices=list(kumo.KUMO_SIZES))
    parser.add_argument("--context", nargs="+",
                        default=["2000", "5000", "all"],
                        help="context row counts, or 'all'")
    args = parser.parse_args()

    limits: list[int | None] = [
        None if str(v).lower() == "all" else int(v) for v in args.context
    ]

    if not kumo.cuda_available():
        raise SystemExit(
            "No CUDA GPU: the benchmark needs the NVIDIA GPU "
            "(see python -m src.gpu_check)."
        )

    print("Loading data...")
    train_df, test_df = data_prep.load_and_prepare()
    print(f"Context (train): {len(train_df):,} rows | test: {len(test_df):,} rows")
    print("Benchmark settings "
          f"(model sizes: {', '.join(args.sizes)}; batch size {BATCH_SIZE}):")

    table = run_benchmark(train_df, test_df,
                          sizes=tuple(args.sizes), context_sizes=tuple(limits))

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(BENCHMARK_CSV, index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\n" + table.to_string(index=False))
    print(f"\nSaved results to {BENCHMARK_CSV}")
    print(f"Saved chart    to {plot_benchmark(table)}")

    # Same chart where the README expects it (docs/images/benchmark.png).
    docs_img = Path("docs") / "images"
    docs_img.mkdir(parents=True, exist_ok=True)
    print(f"Saved chart    to {plot_benchmark(table, docs_img / 'benchmark.png')}")


if __name__ == "__main__":
    main()

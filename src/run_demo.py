"""Phase 1 demo: show Kumo Tabular predictions for 5 sample cars.

Run with:
    python -m src.run_demo

It loads and cleans the dataset, builds the Kumo context from the 80% split,
and prints the model's median estimate plus its 10th/90th percentile range
next to each car's true selling price.
"""

from __future__ import annotations

import sys
import time

from src import data_prep, formatting, model as kumo

DEMO_ROWS = 5


def main() -> int:
    print("CarQuantile — Kumo Tabular demo")
    print("=" * 62)

    # 1. Data -------------------------------------------------------------
    try:
        train_df, test_df = data_prep.load_and_prepare()
    except FileNotFoundError as exc:
        print(exc)
        return 1
    print(f"Dataset: {len(train_df) + len(test_df):,} clean rows "
          f"({len(train_df):,} context / {len(test_df):,} test)")

    # 2. GPU --------------------------------------------------------------
    if not kumo.cuda_available():
        print("\n[ERROR] No CUDA GPU visible to PyTorch.")
        print("Kumo Tabular is designed for NVIDIA GPUs; on CPU it is very "
              "slow.")
        print("Reinstall the CUDA build: pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128")
        return 1
    info = kumo.device_info()
    print(f"GPU: {info['name']} ({info['vram_gb']} GiB, "
          f"CUDA {info['cuda_version']})")

    # 3. Model + context --------------------------------------------------
    print("\nLoading Kumo Tabular (small)...")
    t0 = time.perf_counter()
    model = kumo.load_model(size="small")
    context = kumo.build_context(train_df)
    print(f"Context built: {context.n_rows:,} rows  "
          f"({time.perf_counter() - t0:.1f}s)")

    # 4. Predict 5 test cars ---------------------------------------------
    sample = test_df.sample(DEMO_ROWS, random_state=data_prep.RANDOM_STATE)
    t0 = time.perf_counter()
    price, low, high = kumo.predict(model, context, sample)
    elapsed = time.perf_counter() - t0

    print(f"\nPredictions (total {elapsed:.2f}s, "
          f"{elapsed / len(sample):.3f}s per car)\n")
    header = (f"{'car':32} {'true':>12} {'estimate':>12} "
              f"{'likely range':>28}")
    print(header)
    print("-" * len(header))
    for i in range(len(sample)):
        # A short human label from the features we kept
        # (clean() drops the full name column).
        row = sample.iloc[i]
        car = (f"{row['brand']} {int(row['year'])}, "
               f"{int(row['km_driven']):,}km")[:32]
        true_price = float(sample.iloc[i][context.target_col])
        estimate = float(price[i])
        rng = (f"{formatting.format_inr(low[i])} - "
               f"{formatting.format_inr(high[i])}")
        print(f"{car:32} {formatting.format_inr(true_price):>12} "
              f"{formatting.format_inr(estimate):>12} {rng:>28}")

    # 5. Coverage check on a larger slice ---------------------------------
    n = min(200, len(test_df))
    check = test_df.sample(n, random_state=data_prep.RANDOM_STATE)
    p, lo, hi = kumo.predict(model, context, check)
    true = check[context.target_col].to_numpy(dtype=float)
    covered = ((true >= lo) & (true <= hi)).mean()

    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(true, p)
    print(f"\nOn {n} random test cars: MAE = {mae:,.0f} INR, "
          f"10th-90th coverage = {covered:.1%} (target ~80%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

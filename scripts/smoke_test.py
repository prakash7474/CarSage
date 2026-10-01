"""Smoke test: confirm Kumo Tabular downloads and predicts on a tiny sample.

Run me BEFORE the real dataset arrives:
    python -m scripts.smoke_test

I build a small *synthetic* used-car table (200 context rows + 20 query rows,
same columns as the CarDekho dataset), run the Small regressor once, and
print the predicted median plus the 10th-90th percentile range for a few
cars. If this works, the library, the weights download and the quantile
output are all confirmed.

On the real dataset afterwards, run:  python -m src.run_demo
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from src import model as kumo

N_CONTEXT = 200     # tiny on purpose: fast first download/compile on a 6GB GPU
N_QUERY = 20


def make_synthetic(
    n_context: int = N_CONTEXT,
    n_query: int = N_QUERY,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A toy used-car table shaped like the CarDekho dataset."""
    rng = np.random.default_rng(seed)
    brands = ["maruti", "hyundai", "honda", "toyota", "ford"]
    fuels = ["Petrol", "Diesel", "CNG", "LPG"]
    owners = [
        "First Owner", "Second Owner", "Third Owner",
        "Fourth & Above Owner", "Test Drive Car",
    ]

    def rows(n: int, with_target: bool) -> pd.DataFrame:
        year = rng.integers(2005, 2021, n)
        km = rng.integers(5_000, 200_000, n)
        brand = rng.choice(brands, n)
        fuel = rng.choice(fuels, n)
        transmission = rng.choice(["Manual", "Automatic"], n)
        owner = rng.choice(owners, n)
        seller_type = rng.choice(["Individual", "Dealer"], n)
        # A crude fake price so the model has a learnable signal.
        base = {
            "maruti": 450_000, "hyundai": 500_000, "honda": 600_000,
            "toyota": 700_000, "ford": 650_000,
        }[brand[0]] if False else np.vectorize(
            {"maruti": 450_000, "hyundai": 500_000, "honda": 600_000,
             "toyota": 700_000, "ford": 650_000}.get
        )(brand)
        price = (
            base
            * (year - 2004) / 17
            * np.clip(1 - km / 300_000, 0.3, 1)
            * np.where(transmission == "Automatic", 1.15, 1.0)
            * rng.normal(1, 0.08, n)
        ).clip(30_000, 3_000_000)
        df = pd.DataFrame({
            "brand": brand, "year": year.astype(int),
            "km_driven": km.astype(int), "fuel": fuel,
            "seller_type": seller_type, "transmission": transmission,
            "owner": owner,
        })
        if with_target:
            df["selling_price"] = price.round(-2)
        return df

    return rows(n_context, True), rows(n_query, True)


def main() -> int:
    print("=" * 62)
    print("CarQuantile smoke test — Kumo Tabular on a synthetic table")
    print("=" * 62)

    if not kumo.cuda_available():
        print("[FAIL] No CUDA GPU visible; the smoke test wants a GPU.")
        return 1
    info = kumo.device_info()
    print(f"GPU: {info['name']} ({info['vram_gb']} GiB)")

    context_df, query_df = make_synthetic()
    print(f"Synthetic data: {len(context_df)} context / {len(query_df)} query")

    print("\nLoading KumoTabular (small, regression)...")
    print("(first run downloads the weights from the Hugging Face Hub)")
    t0 = time.perf_counter()
    try:
        model = kumo.load_model(size="small")
    except Exception as exc:  # show download / policy errors clearly
        print(f"[FAIL] load_model: {type(exc).__name__}: {exc}")
        return 1
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    context = kumo.build_context(context_df)
    print(f"Context built ({context.n_rows} rows) in "
          f"{time.perf_counter() - t0:.1f}s")

    print("\nPredicting (single forward pass, no training)...")
    t0 = time.perf_counter()
    price, low, high = kumo.predict(model, context, query_df)
    elapsed = time.perf_counter() - t0
    print(f"  predict() took {elapsed:.2f}s for {len(query_df)} rows "
          f"({elapsed / len(query_df):.3f}s per row)")

    print("\nsample predictions (true vs predicted):")
    print(f"{'true':>12} {'estimate':>12} {'range':>24}")
    for i in range(min(5, len(query_df))):
        rng_txt = f"{low[i]:>10,.0f} - {high[i]:>10,.0f}"
        print(f"{query_df['selling_price'].iloc[i]:>12,.0f} "
              f"{price[i]:>12,.0f} {rng_txt:>24}")

    true = query_df["selling_price"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(true - price)))
    cov = float(np.mean((true >= low) & (true <= high)))
    print(f"\nmedian MAE on the tiny sample : {mae:,.0f}")
    print(f"10th-90th range coverage      : {cov:.0%}")

    free_gb = kumo.device_info().get("free_vram_gb")
    print(f"\nVRAM free after run           : {free_gb} GiB")
    print("SMOKE TEST PASSED" if np.isfinite(price).all() and mae < true.mean()
          else "SMOKE TEST FAILED")
    return 0 if np.isfinite(price).all() else 1


if __name__ == "__main__":
    sys.exit(main())

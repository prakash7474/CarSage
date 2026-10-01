"""Basic tests for CarSage.

Fast tests (data cleaning, input validation) always run. Tests that need the
downloaded dataset and a CUDA GPU are skipped automatically when those are not
available, so `pytest` stays usable on a clean checkout.

Run:  pytest -q
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_string_dtype

from src import data_prep


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

def _fake_raw(n: int = 400) -> pd.DataFrame:
    """A tiny CarDekho-shaped frame, including rows that must be cleaned."""
    rng = np.random.default_rng(0)
    brands = ["Maruti", "Hyundai", "Honda", "Toyota"]
    rows = []
    for i in range(n):
        rows.append({
            "name": f"{brands[i % len(brands)]} Model {i}",
            "year": 2005 + (i % 18),
            "selling_price": 100_000 + (i % 50) * 20_000,
            "km_driven": 10_000 + (i % 40) * 5_000,
            "fuel": ["Petrol", "Diesel", "CNG"][i % 3],
            "seller_type": ["Individual", "Dealer"][i % 2],
            "transmission": ["Manual", "Automatic"][i % 2],
            "owner": ["First Owner", "Second Owner"][i % 2],
        })
    df = pd.DataFrame(rows)

    # A row with a missing value (should be dropped).
    df.loc[0, "fuel"] = None
    # An invalid year (should be dropped).
    df.loc[1, "year"] = 1900
    # A negative price (should be dropped).
    df.loc[2, "selling_price"] = -5
    # A duplicate row (should be dropped).
    df = pd.concat([df, df.iloc[[3]]], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def test_clean_returns_expected_columns_and_dtypes():
    clean = data_prep.clean(_fake_raw())
    assert list(clean.columns) == data_prep.FEATURE_COLUMNS + [data_prep.TARGET_COL]
    assert clean[data_prep.TARGET_COL].dtype == float
    assert clean["year"].dtype == int
    assert clean["km_driven"].dtype == int
    for col in data_prep.CATEGORICAL_COLUMNS:
        assert is_string_dtype(clean[col].dtype)   # object or pandas 'str'


def test_clean_drops_invalid_rows_and_nans():
    clean = data_prep.clean(_fake_raw())
    assert clean.isna().sum().sum() == 0
    assert (clean["year"] >= data_prep.MIN_YEAR).all()
    assert (clean[data_prep.TARGET_COL] > 0).all()
    assert (clean["km_driven"] >= 0).all()
    assert not clean.duplicated().any()


def test_brand_is_first_word_lowercased():
    clean = data_prep.clean(_fake_raw())
    assert set(clean["brand"]).issubset({"maruti", "hyundai", "honda", "toyota"})


def test_split_is_80_20_and_reproducible():
    clean = data_prep.clean(_fake_raw())
    train1, test1 = data_prep.split(clean)
    train2, test2 = data_prep.split(clean)
    assert len(train1) + len(test1) == len(clean)
    assert abs(len(test1) / len(clean) - 0.2) < 0.02
    pd.testing.assert_frame_equal(train1, train2)   # fixed random_state


def test_load_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        data_prep.load_raw(tmp_path / "nope.csv")
    assert "kaggle" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# predict() input validation (no GPU needed)
# ---------------------------------------------------------------------------

def test_predict_rejects_missing_columns():
    from src import model as kumo

    fake_context = SimpleNamespace(
        feature_columns=data_prep.FEATURE_COLUMNS,
    )
    bad = pd.DataFrame({"year": [2015]})    # missing brand, fuel, ...
    with pytest.raises(ValueError, match="missing required columns"):
        kumo.predict(model=None, context=fake_context, rows_df=bad)


def test_predict_rejects_empty_input():
    from src import model as kumo

    fake_context = SimpleNamespace(feature_columns=data_prep.FEATURE_COLUMNS)
    empty = pd.DataFrame(columns=data_prep.FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="No rows"):
        kumo.predict(model=None, context=fake_context, rows_df=empty)


# ---------------------------------------------------------------------------
# Full model tests (skipped unless data + CUDA are present)
# ---------------------------------------------------------------------------

def _needs_gpu_and_data() -> bool:
    """True only when both the dataset and a CUDA GPU are available."""
    try:
        from src import model as kumo
        return kumo.cuda_available() and data_prep.DATA_PATH.exists()
    except Exception:            # torch/sdm missing, bad data path, etc.
        return False


_HAS_GPU_AND_DATA = _needs_gpu_and_data()
_NEEDS = pytest.mark.skipif(
    not _HAS_GPU_AND_DATA,
    reason="needs data/cars.csv and a CUDA GPU",
)


@_NEEDS
def test_predict_returns_ordered_quantiles():
    from src import model as kumo

    train_df, test_df = data_prep.load_and_prepare()
    model = kumo.load_model(size="small")
    context = kumo.build_context(train_df.iloc[:1000])
    sample = test_df.head(3)
    price, low, high = kumo.predict(model, context, sample)

    assert len(price) == len(low) == len(high) == 3
    assert np.all(low <= price)
    assert np.all(price <= high)
    assert np.all(price > 0)


@_NEEDS
def test_unseen_brand_does_not_crash():
    from src import model as kumo

    train_df, test_df = data_prep.load_and_prepare()
    model = kumo.load_model(size="small")
    context = kumo.build_context(train_df.iloc[:1000])

    row = test_df.iloc[[0]].copy()
    row["brand"] = "not_a_real_brand_xyz"
    price, low, high = kumo.predict(model, context, row)
    assert np.isfinite(price[0]) and low[0] <= price[0] <= high[0]

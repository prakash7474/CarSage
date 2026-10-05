"""Basic tests for CarQuantile.

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


# ---------------------------------------------------------------------------
# Derived features: city, insurance status, service history
# ---------------------------------------------------------------------------

def test_derived_features_exist_and_are_deterministic():
    """clean() adds the three derived columns, with the same values twice."""
    clean1 = data_prep.clean(_fake_raw())
    clean2 = data_prep.clean(_fake_raw())
    for col in data_prep.DERIVED_COLUMNS:
        assert col in clean1.columns
    pd.testing.assert_frame_equal(clean1, clean2)   # no randomness anywhere


def test_derived_values_stay_inside_the_documented_domains():
    clean = data_prep.clean(_fake_raw())
    assert set(clean["city"]).issubset(set(data_prep.CITY_POOL))
    assert set(clean["insurance_status"]).issubset(
        {"Comprehensive", "Third Party", "Expired"})
    assert set(clean["service_history"]).issubset(
        {"Full", "Partial", "Not Recorded"})


def test_insurance_status_follows_the_year_rule():
    clean = data_prep.clean(_fake_raw())
    assert (clean.loc[clean["year"] >= data_prep.INSURANCE_COMPREHENSIVE_MIN_YEAR,
                      "insurance_status"] == "Comprehensive").all()
    mid = clean[(clean["year"] >= data_prep.INSURANCE_THIRD_PARTY_MIN_YEAR)
                & (clean["year"] < data_prep.INSURANCE_COMPREHENSIVE_MIN_YEAR)]
    assert (mid["insurance_status"] == "Third Party").all()
    assert (clean.loc[clean["year"] < data_prep.INSURANCE_THIRD_PARTY_MIN_YEAR,
                      "insurance_status"] == "Expired").all()


def test_service_history_follows_owner_and_km_rule():
    clean = data_prep.clean(_fake_raw())
    full = ((clean["owner"] == "First Owner")
            & (clean["km_driven"] <= data_prep.SERVICE_HISTORY_FULL_MAX_KM))
    partial = (~full
               & clean["owner"].isin(["First Owner", "Second Owner"])
               & (clean["km_driven"] <= data_prep.SERVICE_HISTORY_PARTIAL_MAX_KM))
    assert (clean.loc[full, "service_history"] == "Full").all()
    assert (clean.loc[partial, "service_history"] == "Partial").all()
    assert (clean.loc[~(full | partial), "service_history"]
            == "Not Recorded").all()


def test_raw_csv_columns_win_over_the_derivation():
    """A richer CSV that already has the columns is used as-is."""
    raw = _fake_raw()
    raw["city"] = "Mumbai"
    raw["insurance_status"] = "Comprehensive"
    raw["service_history"] = "Full"
    clean = data_prep.clean(raw)
    assert clean["city"].eq("Mumbai").all()
    assert clean["insurance_status"].eq("Comprehensive").all()
    assert clean["service_history"].eq("Full").all()


def test_ui_options_include_the_derived_features():
    clean = data_prep.clean(_fake_raw())
    opts = data_prep.get_ui_options(clean)
    for col in data_prep.DERIVED_COLUMNS:
        assert opts[col] == sorted(clean[col].unique().tolist())


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


_NEEDS_DATA = pytest.mark.skipif(
    not data_prep.DATA_PATH.exists(),
    reason="needs data/cars.csv (the dataset is not in place)",
)


@_NEEDS_DATA
def test_real_dataset_has_expected_columns_and_no_missing_values():
    clean = data_prep.clean(data_prep.load_raw())
    assert list(clean.columns) == data_prep.FEATURE_COLUMNS + [data_prep.TARGET_COL]
    assert clean.isna().sum().sum() == 0          # no missing values
    assert (clean["year"] >= data_prep.MIN_YEAR).all()
    assert (clean[data_prep.TARGET_COL] >= data_prep.MIN_PRICE).all()
    train_df, test_df = data_prep.split(clean)
    assert len(train_df) > len(test_df) * 3       # roughly 80/20


# ---------------------------------------------------------------------------
# predict() input validation (no GPU needed)
# ---------------------------------------------------------------------------

def _fake_context(**kwargs):
    """A context that is only good enough for the validation code paths."""
    defaults = dict(
        feature_columns=data_prep.FEATURE_COLUMNS,
        cat_values={},
        num_ranges={},
        median_price=0.0,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _valid_row(**overrides) -> pd.DataFrame:
    """One car in the app's exact schema (no price column)."""
    row = {
        "brand": "maruti",
        "year": 2018,
        "km_driven": 50_000,
        "fuel": "Diesel",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner",
        "city": "Mumbai",
        "insurance_status": "Third Party",
        "service_history": "Full",
    }
    row.update(overrides)
    return pd.DataFrame([row])[data_prep.FEATURE_COLUMNS]


def test_predict_rejects_missing_columns():
    from src import model as kumo

    bad = pd.DataFrame({"year": [2015]})    # missing brand, fuel, ...
    with pytest.raises(ValueError, match="missing required columns"):
        kumo.predict(model=None, context=_fake_context(), rows_df=bad)


def test_predict_rejects_empty_input():
    from src import model as kumo

    empty = pd.DataFrame(columns=data_prep.FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="No rows"):
        kumo.predict(model=None, context=_fake_context(), rows_df=empty)


def test_predict_rejects_negative_km_with_clear_message():
    from src import model as kumo

    row = _valid_row(km_driven=-500)
    with pytest.raises(ValueError, match="cannot be negative"):
        kumo.predict(model=None, context=_fake_context(), rows_df=row)


def test_predict_rejects_missing_value_with_clear_message():
    from src import model as kumo

    row = _valid_row(fuel=None)
    with pytest.raises(ValueError, match="Missing value"):
        kumo.predict(model=None, context=_fake_context(), rows_df=row)


def test_predict_rejects_non_numeric_year():
    from src import model as kumo

    row = _valid_row(year="next year")
    with pytest.raises(ValueError, match="must be a number"):
        kumo.predict(model=None, context=_fake_context(), rows_df=row)


def test_unseen_brand_warns_but_does_not_raise():
    from src import model as kumo

    ctx = _fake_context(cat_values={"brand": {"maruti", "hyundai"}})
    warnings: list[str] = []
    # No exception: an unseen brand only produces a plain-language warning.
    kumo.validate_rows(_valid_row(brand="not_a_real_brand_xyz"), ctx, warnings)
    assert any("not_a_real_brand_xyz" in w for w in warnings)


def test_extreme_year_and_km_warn_but_do_not_raise():
    from src import model as kumo

    ctx = _fake_context(num_ranges={"year": (1991, 2020),
                                    "km_driven": (1000, 200_000)})
    warnings: list[str] = []
    kumo.validate_rows(_valid_row(year=1850, km_driven=900_000), ctx, warnings)
    assert len(warnings) == 2
    assert all("outside the training range" in w for w in warnings)


def test_output_is_always_ordered_and_positive():
    """The sanitiser must guarantee low <= price <= high, even on junk."""
    from src import model as kumo

    ctx = _fake_context(median_price=400_000.0)
    price, low, high = kumo._sanitise(
        price=np.array([500_000.0, -1.0, np.nan]),
        low=np.array([600_000.0, -5.0, -5.0]),
        high=np.array([400_000.0, 10.0, np.nan]),
        context=ctx,
    )
    assert np.all(np.isfinite(price)) and np.all(np.isfinite(low))
    assert np.all(price >= kumo.MIN_PREDICTION_PRICE)
    assert np.all(low <= price) and np.all(price <= high)


# ---------------------------------------------------------------------------
# "Add a new sale" / "Reset context" logic (no GPU needed)
# ---------------------------------------------------------------------------

def _labeled_sale() -> pd.DataFrame:
    """A new sale row: the features plus the known selling price."""
    row = _valid_row(km_driven=12_345)
    row[data_prep.TARGET_COL] = 350_000.0
    return row


def test_add_new_sale_grows_context_by_one_and_resets():
    from src import model as kumo

    original = data_prep.clean(_fake_raw()).head(50).reset_index(drop=True)
    pristine = original.copy()                 # the app keeps this for reset

    grown = kumo.add_sale(original, _labeled_sale())
    assert len(grown) == len(original) + 1     # +1 row after "Add to context"
    assert int(grown.iloc[-1]["km_driven"]) == 12_345
    assert float(grown.iloc[-1][data_prep.TARGET_COL]) == 350_000.0

    # "Reset context" restores the pristine copy, byte for byte.
    reset = pristine.copy()
    assert len(reset) == len(original)
    pd.testing.assert_frame_equal(reset, original)


def test_add_sale_rejects_a_row_without_a_price():
    from src import model as kumo

    original = data_prep.clean(_fake_raw()).head(5).reset_index(drop=True)
    with pytest.raises(ValueError, match="missing columns"):
        kumo.add_sale(original, _valid_row())   # no selling_price column


# ---------------------------------------------------------------------------
# Full model tests (skipped unless data + CUDA are present)
# ---------------------------------------------------------------------------

def _skip_reason() -> str | None:
    """Why the GPU tests cannot run here, or None when they can."""
    if not data_prep.DATA_PATH.exists():
        return "needs data/cars.csv (the dataset is not in place)"
    try:
        from src import model as kumo            # noqa: F401
    except Exception as exc:
        return f"needs the sdm/torch packages ({type(exc).__name__}: {exc})"
    from src import model as kumo
    if not kumo.cuda_available():
        return "needs a CUDA GPU (torch.cuda.is_available() is False)"
    return None


_SKIP_REASON = _skip_reason()
_NEEDS = pytest.mark.skipif(
    _SKIP_REASON is not None,
    reason=_SKIP_REASON or "CUDA and dataset available",
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
    warnings: list[str] = []
    price, low, high = kumo.predict(model, context, row, warnings_out=warnings)
    assert np.isfinite(price[0]) and low[0] <= price[0] <= high[0]
    assert any("not_a_real_brand_xyz" in w for w in warnings)


@_NEEDS
def test_context_size_grows_by_one_on_add_and_resets():
    """The Kumo context itself (not just the DataFrame) follows add/reset."""
    from src import model as kumo

    train_df, _test = data_prep.load_and_prepare()
    base = train_df.iloc[:100].reset_index(drop=True)

    context = kumo.build_context(base)
    grown = kumo.build_context(kumo.add_sale(base, _labeled_sale()))
    reset = kumo.build_context(base.copy())     # the app's "Reset context"

    assert context.n_rows == 100
    assert grown.n_rows == context.n_rows + 1
    assert reset.n_rows == context.n_rows


@_NEEDS
def test_predict_batch_keeps_order_and_range():
    from src import model as kumo

    train_df, test_df = data_prep.load_and_prepare()
    model = kumo.load_model(size="small")
    context = kumo.build_context(train_df.iloc[:1000])
    sample = test_df.head(5)

    price, low, high = kumo.predict_batch(model, context, sample,
                                          batch_size=2)   # 3 batches
    assert len(price) == len(low) == len(high) == 5
    assert np.all(np.isfinite(price))
    assert np.all(low <= price) and np.all(price <= high)

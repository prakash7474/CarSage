"""Data loading, cleaning, feature selection and train/test split.

This module turns the raw Kaggle CarDekho CSV into the clean DataFrame that
both models (Kumo Tabular and XGBoost) consume.

Kumo Tabular only understands *numerical* and *categorical* columns, so every
text/timestamp column is either dropped or converted:
  - `name`          -> we keep only the first word as `brand` (categorical)
  - `year`          -> numerical
  - `km_driven`     -> numerical
  - `fuel`, `seller_type`, `transmission`, `owner` -> categorical strings
  - `selling_price` -> the target (numerical)

Three extra categoricals are *derived* from those real columns, because the
raw CarDekho file has no location, insurance or service columns of its own:
  - `city`             stable hash of name|year|km -> a realistic city mix
  - `insurance_status` from `year` (newer cars keep comprehensive cover)
  - `service_history`  from `owner` and `km_driven`
The rules are deterministic (no randomness, never look at the price) and are
documented in README ("How it works") and NOTES.md. If the raw CSV already
provides any of these columns itself, that column is used as-is instead.

Run me directly for a quick summary:
    python -m src.data_prep
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zlib import crc32

import pandas as pd
from pandas.api.types import is_numeric_dtype

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

#: Expected location of the Kaggle CSV (see README for the download link).
DATA_PATH = Path("data") / "cars.csv"

#: The label we want to predict.
TARGET_COL = "selling_price"

#: Columns used as model features, in the exact order the models expect.
FEATURE_COLUMNS = [
    "brand",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "city",
    "insurance_status",
    "service_history",
]

#: Features the raw CarDekho file does not contain; `derive_features` adds
#: them unless the CSV already provides the column itself.
DERIVED_COLUMNS = ["city", "insurance_status", "service_history"]

#: Which feature columns are categorical (strings) vs numerical (numbers).
CATEGORICAL_COLUMNS = [
    "brand", "fuel", "seller_type", "transmission", "owner",
    "city", "insurance_status", "service_history",
]
NUMERICAL_COLUMNS = ["year", "km_driven"]

#: Sanity bounds used to drop obviously invalid rows.
MIN_YEAR = 1990
MIN_PRICE = 1
MAX_PRICE = 100_000_000

#: Fraction of rows held out as the test set.
TEST_SIZE = 0.2

#: Fixed seed so every run splits the data the same way.
RANDOM_STATE = 42

#: Price-outlier percentile window (1st–99th) applied before splitting.
PRICE_LOWER_Q = 0.01
PRICE_UPPER_Q = 0.99


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, with a friendly error if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download the CarDekho dataset from Kaggle:\n"
            "  https://www.kaggle.com/datasets/nehalbirla/"
            "vehicle-dataset-from-cardekho\n"
            "and save the file 'Car details v3.csv' as "
            f"'{path}'."
        )
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _extract_brand(name: object) -> str:
    """Return the first word of a car name, lower-cased, as the brand.

    'Maruti Swift Dzire VDI' -> 'maruti'
    """
    if not isinstance(name, str) or not name.strip():
        return ""
    return name.strip().split()[0].lower()


# ---------------------------------------------------------------------------
# Derived features (city, insurance status, service history)
#
# The raw CarDekho file has no location/insurance/service columns, so these
# three are derived from real columns with deterministic rules. They never
# look at `selling_price`, so there is no target leakage, and they produce
# the same values on every run and platform (CRC32, no RNG). If the raw CSV
# already contains one of the columns, it is kept as-is.
# ---------------------------------------------------------------------------

#: City pool used when the raw file has no location column. The four biggest
#: metros appear twice, so the mix resembles the real used-car market.
CITY_POOL = [
    "Delhi", "Mumbai", "Bangalore", "Hyderabad",
    "Delhi", "Mumbai", "Bangalore", "Pune",
    "Chennai", "Pune", "Kolkata", "Ahmedabad",
    "Jaipur", "Lucknow", "Surat", "Chandigarh",
]

#: Insurance-status buckets by manufacturing year (real-world rule of thumb:
#: recent cars still carry comprehensive cover, mid-age cars only third-party
#: cover, old cars' insurance has lapsed).
INSURANCE_COMPREHENSIVE_MIN_YEAR = 2018
INSURANCE_THIRD_PARTY_MIN_YEAR = 2012

#: Service-history buckets by owner count and mileage (fewer owners and fewer
#: kilometres make a complete service record more likely).
SERVICE_HISTORY_FULL_MAX_KM = 100_000
SERVICE_HISTORY_PARTIAL_MAX_KM = 200_000


def _derive_city(df: pd.DataFrame) -> pd.Series:
    """A stable pseudo-location per car, picked from :data:`CITY_POOL`.

    Keyed on the car's identifying fields (`name` or `brand`, `year`,
    `km_driven`) via CRC32, so the same car always maps to the same city and
    re-running the pipeline changes nothing. The hash never sees the price.
    """
    name = (df["name"] if "name" in df.columns
            else df["brand"] if "brand" in df.columns else "")
    key = (name.astype(str) + "|" + df["year"].astype(str)
           + "|" + df["km_driven"].astype(str))
    return pd.Series(
        [CITY_POOL[crc32(k.encode("utf-8")) % len(CITY_POOL)] for k in key],
        index=df.index,
        dtype=object,
    )


def _derive_insurance_status(year: pd.Series) -> pd.Series:
    """Comprehensive / Third Party / Expired from the manufacturing year."""
    status = pd.Series("Expired", index=year.index, dtype=object)
    status[year >= INSURANCE_THIRD_PARTY_MIN_YEAR] = "Third Party"
    status[year >= INSURANCE_COMPREHENSIVE_MIN_YEAR] = "Comprehensive"
    return status


def _derive_service_history(owner: pd.Series, km_driven: pd.Series) -> pd.Series:
    """Full / Partial / Not Recorded from owner count and mileage."""
    full = owner.eq("First Owner") & km_driven.le(SERVICE_HISTORY_FULL_MAX_KM)
    partial = (owner.isin(("First Owner", "Second Owner"))
               & km_driven.le(SERVICE_HISTORY_PARTIAL_MAX_KM) & ~full)
    history = pd.Series("Not Recorded", index=owner.index, dtype=object)
    history[partial] = "Partial"
    history[full] = "Full"
    return history


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived feature columns that are missing from ``df``.

    Columns the CSV already provides are left untouched. A derived column is
    skipped (left absent) when its source column is missing, so a broken CSV
    still fails with the honest "missing expected columns" error instead of a
    KeyError inside the derivation.
    """
    out = df.copy()
    if "city" not in out.columns and "year" in out.columns \
            and "km_driven" in out.columns:
        out["city"] = _derive_city(out)
    if "insurance_status" not in out.columns and "year" in out.columns:
        out["insurance_status"] = _derive_insurance_status(out["year"])
    if "service_history" not in out.columns and "owner" in out.columns \
            and "km_driven" in out.columns:
        out["service_history"] = _derive_service_history(
            out["owner"], out["km_driven"])
    return out


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe and return a canonical feature table.

    Steps:
      1. derive `brand` from `name`, plus city / insurance status /
         service history when the raw file does not provide them
      2. drop rows with missing values in any needed column
      3. drop duplicate rows
      4. drop invalid rows (bad year / non-positive price or km)
      5. drop price outliers outside the 1st–99th percentile
      6. keep only the feature columns + the target, with tidy dtypes
    """
    df = df.copy()

    # Some CarDekho exports store km with commas (e.g. "45,000").
    # (``is_numeric_dtype`` works whether the column is object or pandas' new
    # `str` dtype.)
    if "km_driven" in df.columns and not is_numeric_dtype(df["km_driven"]):
        df["km_driven"] = (
            df["km_driven"].astype(str).str.replace(",", "", regex=False)
        )
    for col in ("year", "selling_price", "km_driven"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 1. brand from the first word of the car name, plus the derived
    #    features (city / insurance status / service history).
    if "name" in df.columns:
        df["brand"] = df["name"].map(_extract_brand)
    df = derive_features(df)

    needed = FEATURE_COLUMNS + [TARGET_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing expected columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # 2. drop rows with any missing value in the columns we need.
    df = df.dropna(subset=needed)

    # 3. drop exact duplicates.
    df = df.drop_duplicates()

    # 4. obviously invalid rows.
    max_year = datetime.now().year + 1
    df = df[
        (df["year"] >= MIN_YEAR)
        & (df["year"] <= max_year)
        & (df["km_driven"] >= 0)
        & (df[TARGET_COL] >= MIN_PRICE)
        & (df[TARGET_COL] <= MAX_PRICE)
    ]

    # 5. price outliers: keep only the 1st–99th percentile band.
    low = df[TARGET_COL].quantile(PRICE_LOWER_Q)
    high = df[TARGET_COL].quantile(PRICE_UPPER_Q)
    df = df[(df[TARGET_COL] >= low) & (df[TARGET_COL] <= high)]

    # 6. tidy dtypes. Numeric columns -> float/int; text -> plain strings.
    df = df[needed].copy()
    df["year"] = df["year"].astype(int)
    df["km_driven"] = df["km_driven"].astype(int)
    df[TARGET_COL] = df[TARGET_COL].astype(float)
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].astype(str)

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split & helpers
# ---------------------------------------------------------------------------

def split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into an 80% in-context/train set and a 20% test set.

    Kumo Tabular is *not trained*: the 80% set becomes its labeled context,
    and the 20% set is only used to measure accuracy.
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_and_prepare(
    path: Path | str = DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience: load -> clean -> split."""
    return split(clean(load_raw(path)))


def get_ui_options(df: pd.DataFrame) -> dict:
    """Collect dropdown options and slider ranges for the Streamlit app."""
    return {
        "brand": sorted(df["brand"].unique().tolist()),
        "fuel": sorted(df["fuel"].unique().tolist()),
        "seller_type": sorted(df["seller_type"].unique().tolist()),
        "transmission": sorted(df["transmission"].unique().tolist()),
        "owner": sorted(df["owner"].unique().tolist()),
        "city": sorted(df["city"].unique().tolist()),
        "insurance_status": sorted(df["insurance_status"].unique().tolist()),
        "service_history": sorted(df["service_history"].unique().tolist()),
        "year": (int(df["year"].min()), int(df["year"].max())),
        "km_driven": (int(df["km_driven"].min()),
                      int(df["km_driven"].max())),
    }


def main() -> None:
    """Print a short summary when run as `python -m src.data_prep`."""
    df = clean(load_raw())
    train_df, test_df = split(df)
    print(f"Rows after cleaning : {len(df):,}")
    print(f"Context (train) rows: {len(train_df):,}")
    print(f"Test rows           : {len(test_df):,}")
    print(f"Features            : {FEATURE_COLUMNS}")
    print(f"\nTarget describe:\n{df[TARGET_COL].describe()}")
    print(f"\nBrands ({df['brand'].nunique()}): "
          f"{sorted(df['brand'].unique())[:15]} ...")
    print(f"\nFirst rows:\n{df.head()}")


if __name__ == "__main__":
    main()

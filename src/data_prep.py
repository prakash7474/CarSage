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

Run me directly for a quick summary:
    python -m src.data_prep
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

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
]

#: Which feature columns are categorical (strings) vs numerical (numbers).
CATEGORICAL_COLUMNS = ["brand", "fuel", "seller_type", "transmission", "owner"]
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


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe and return a canonical feature table.

    Steps:
      1. derive `brand` from `name`
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

    # 1. brand from the first word of the car name.
    if "name" in df.columns:
        df["brand"] = df["name"].map(_extract_brand)

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

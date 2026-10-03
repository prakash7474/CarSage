"""NVIDIA Kumo Tabular wrapper: load the model, build the context, predict.

Kumo Tabular is an *in-context* foundation model: it is never trained on our
data. Instead we hand it a labeled "context" table (the 80% split) and it
predicts new rows in a single forward pass.

Public API (used by the demo, the app, the benchmark and the tests):

    model                  = load_model(size="small")
    context                = build_context(context_df)   # encode ONCE
    context                = fit_context(model, context)  # cache the KV once
    price, low, high       = predict(model, context, rows_df)
    price, low, high       = predict_batch(model, context, rows_df, 256)

`price` is the model's median (q500); `low`/`high` are the requested
quantiles (10th/90th by default). All three are numpy arrays in **rupees**.

Reliability rules (Phase 3):
  * bad input (missing column, missing value, negative km) -> ValueError with
    a plain-language message the UI can show;
  * unusual but usable input (unseen brand, year/km far outside the training
    range) -> still predicted, plus a warning pushed into ``warnings_out``;
  * CUDA out of memory -> the cache is freed and the call is retried with a
    smaller batch, then a smaller context (the retry is reported as a warning);
  * the output always satisfies ``low <= price <= high`` and
    ``price >= MIN_PREDICTION_PRICE``.

API facts were verified against sdm 0.x (see NOTES.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as _dc_replace
from datetime import datetime

import numpy as np
import pandas as pd

try:  # torch / sdm are optional at import time so data_prep can be used alone.
    import torch
    import sdm
except ImportError:  # pragma: no cover - handled with a friendly message
    torch = None
    sdm = None

from src import data_prep

#: Number of quantiles the regression model emits (q001 .. q999).
N_QUANTILES = 999

#: Model sizes offered by KumoTabular.
KUMO_SIZES = ("small", "medium", "large")

#: A predicted price below this is nonsense for a used car; we clip to it.
MIN_PREDICTION_PRICE = 1000.0

#: Context rows kept when a forward pass still runs out of GPU memory.
SHRINK_CONTEXT_ROWS = 1000

#: Default batch size for :func:`predict_batch` (keeps VRAM bounded).
DEFAULT_BATCH_SIZE = 256

#: Default quantile levels: the 10th..90th percentile = an 80% range.
LOW_Q = 0.1
HIGH_Q = 0.9

#: The setting used everywhere unless something else is requested.
#: Chosen from `python -m src.benchmark` (see README "Key findings").
DEFAULT_MODEL_SIZE = "small"
DEFAULT_CONTEXT_ROWS = None      # None = use every context row


def limit_context(context_df: pd.DataFrame,
                  n_rows: int | None = DEFAULT_CONTEXT_ROWS) -> pd.DataFrame:
    """Take at most ``n_rows`` context rows (reproducible sample).

    ``None`` returns the frame unchanged. Used by the app and the evaluation
    so they share the same default context size.
    """
    if n_rows is None or n_rows >= len(context_df):
        return context_df
    return context_df.sample(n_rows, random_state=data_prep.RANDOM_STATE)


def add_sale(context_df: pd.DataFrame, sale_row: pd.DataFrame) -> pd.DataFrame:
    """Append one labeled sale to the context table.

    This is the data half of the app's "Add a new sale" button: the new row
    must have the same columns as the context (features + selling price).
    The caller re-builds the Kumo context from the returned frame.
    """
    if len(sale_row) != 1:
        raise ValueError(
            f"Expected exactly one new sale row, got {len(sale_row)}."
        )
    missing = [c for c in context_df.columns if c not in sale_row.columns]
    if missing:
        raise ValueError(
            f"The new sale is missing columns: {missing}. "
            f"Expected: {list(context_df.columns)}."
        )
    return pd.concat([context_df, sale_row[context_df.columns]],
                     ignore_index=True)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def cuda_available() -> bool:
    """True if PyTorch can see a CUDA GPU."""
    return bool(torch is not None and torch.cuda.is_available())


def default_device() -> str:
    """Return 'cuda' when available, otherwise 'cpu'."""
    return "cuda" if cuda_available() else "cpu"


def device_info() -> dict:
    """Small info dict for the app sidebar (name, VRAM, cuda build)."""
    if not cuda_available():
        return {"cuda": False, "name": None, "vram_gb": None}
    index = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    return {
        "cuda": True,
        "name": torch.cuda.get_device_name(index),
        "vram_gb": round(total_bytes / 1024 ** 3, 2),
        "free_vram_gb": round(free_bytes / 1024 ** 3, 2),
        "cuda_version": torch.version.cuda,
    }


# ---------------------------------------------------------------------------
# Context container
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """Everything the model needs to know about the labeled examples.

    Built **once** by :func:`build_context` and then reused for every
    prediction (this is what makes the in-context approach cheap).
    """

    x: "sdm.TableTensor"               # feature columns, [R_context, D]
    y: "sdm.TableTensor"               # target column,     [R_context, 1]
    feature_columns: list[str]
    feature_stypes: dict               # {column: sdm.Stype}
    target_col: str
    log_target: bool
    n_rows: int
    fitted: bool = False               # True once model.fit() cached the KV

    # --- Phase 3: input sanity checks need to know what "normal" is -------
    num_ranges: dict = field(default_factory=dict)  # {col: (min, max)} of the context
    cat_values: dict = field(default_factory=dict)  # {col: {seen values}}
    median_price: float = 0.0                      # median label, fallback value


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    size: str = DEFAULT_MODEL_SIZE,
    device: str | None = None,
    num_estimators: int = 1,
):
    """Load a pretrained Kumo Tabular regressor.

    Parameters
    ----------
    size:
        "small" (default, ~28M params), "medium" or "large". The library's
        own default is "large"; we override it to fit a 6 GB laptop GPU.
    device:
        "cuda" or "cpu". Defaults to CUDA when available.
    num_estimators:
        Number of forward passes averaged. 1 keeps VRAM low; higher values
        are more accurate but use more memory. The library's default
        (None) also means a single pass.

    The regressor weights are downloaded automatically from the Hugging Face
    Hub (nvidia/Kumo-Tabular, revision v1.0.0) on the first call.
    """
    if sdm is None:
        raise ImportError(
            "The 'sdm' (structured-data-models) package is not installed.\n"
            "Install it with:\n"
            "  pip install git+https://github.com/"
            "NVIDIA/structured-data-models.git"
        )
    if size not in KUMO_SIZES:
        raise ValueError(f"size must be one of {KUMO_SIZES}, got {size!r}")

    device = device or default_device()

    # task=Task.regression loads ONLY the regressor checkpoint (saves memory
    # and download time versus loading the classifier too).
    model = sdm.models.KumoTabular(
        task=sdm.Task.regression,
        size=size,
        pretrained=True,
        device=device,
    )
    model.eval()
    # Stash the ensemble size so predict() can reuse the same setting.
    model._carquantile_num_estimators = num_estimators
    return model


# ---------------------------------------------------------------------------
# Building the context
# ---------------------------------------------------------------------------

def _to_stypes(df: pd.DataFrame, target_col: str) -> dict:
    """Infer semantic types, forcing the target to be numerical."""
    return sdm.infer_stypes(df, overrides={target_col: "numerical"})


def build_context(
    context_df: pd.DataFrame,
    target_col: str = data_prep.TARGET_COL,
    feature_columns: list[str] | None = None,
    log_target: bool = True,
) -> Context:
    """Turn the labeled dataframe into Kumo context tensors (do this once).

    Parameters
    ----------
    context_df:
        The 80% split: feature columns **plus** the price column.
    target_col:
        Name of the label column.
    feature_columns:
        Feature order to use. Defaults to
        :data:`src.data_prep.FEATURE_COLUMNS`.
    log_target:
        If True (default), model ``log1p(price)`` instead of price. Used-car
        prices are heavy-tailed, so this usually helps; predictions are
        converted back to rupees in :func:`predict`.
    """
    if sdm is None:
        raise ImportError("The 'sdm' package is not installed (see load_model).")

    feature_columns = list(feature_columns or data_prep.FEATURE_COLUMNS)
    df = context_df[feature_columns + [target_col]].copy()

    # Cast features to the dtypes the model expects.
    for col in feature_columns:
        if col in data_prep.NUMERICAL_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(str)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # Optionally work on the log scale (expm1 undoes it at predict time).
    df[target_col] = np.log1p(df[target_col]) if log_target else df[target_col]

    stypes = _to_stypes(df, target_col)

    # --- Phase 3: remember what "normal" looks like, for input warnings ---
    num_ranges = {
        c: (float(df[c].min()), float(df[c].max()))
        for c in data_prep.NUMERICAL_COLUMNS if c in df.columns
    }
    cat_values = {
        c: set(df[c].unique().tolist())
        for c in data_prep.CATEGORICAL_COLUMNS if c in df.columns
    }
    labels = np.expm1(df[target_col].to_numpy(dtype=float)) if log_target \
        else df[target_col].to_numpy(dtype=float)
    median_price = float(np.median(labels)) if len(labels) else MIN_PREDICTION_PRICE

    table = sdm.TableTensor.from_pandas(
        df=df,
        stypes=stypes,
        device=default_device(),
    )

    x = table.drop_columns(target_col)
    y = table[:, target_col]

    feature_stypes = {c: stypes[c] for c in feature_columns}
    return Context(
        x=x,
        y=y,
        feature_columns=feature_columns,
        feature_stypes=feature_stypes,
        target_col=target_col,
        log_target=log_target,
        n_rows=len(df),
        num_ranges=num_ranges,
        cat_values=cat_values,
        median_price=median_price,
    )


def fit_context(model, context: Context, num_estimators: int | None = None) -> Context:
    """Encode the context once and cache it inside the model.

    After this call, :func:`predict` uses ``model.predict`` and reuses the
    cached context keys/values, so every later prediction skips the expensive
    re-encoding step (this is the "encode ONCE" optimisation of Phase 3).

    Returns the context to use — **always use the return value**, because if
    encoding the full context hits a CUDA out-of-memory error we free the
    cache and retry with a smaller context (:data:`SHRINK_CONTEXT_ROWS`).
    """
    n = num_estimators or getattr(model, "_carquantile_num_estimators", 1)
    try:
        model.fit(x=context.x, y=context.y, num_estimators=n)
    except Exception as exc:
        if not _is_oom(exc):
            raise
        torch.cuda.empty_cache()                      # free the failed attempt
        smaller = _shrink_context(context, SHRINK_CONTEXT_ROWS)
        if smaller is None:
            raise RuntimeError(
                "The GPU ran out of memory while encoding the context. "
                "Close other GPU apps and try a smaller context."
            ) from exc
        model.fit(x=smaller.x, y=smaller.y, num_estimators=n)
        smaller.fitted = True
        return smaller
    context.fitted = True
    return context


# ---------------------------------------------------------------------------
# GPU error handling
# ---------------------------------------------------------------------------

def _is_oom(exc: BaseException) -> bool:
    """True when an exception is a CUDA out-of-memory error."""
    if torch is None:
        return False
    oom_types = tuple(
        t for t in (
            getattr(torch.cuda, "OutOfMemoryError", None),
            getattr(torch, "OutOfMemoryError", None),
        ) if t is not None
    )
    if oom_types and isinstance(exc, oom_types):
        return True
    # Older builds raise a plain RuntimeError with this message.
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


#: Public alias so app.py can react to OOM errors without touching a private name.
is_out_of_memory = _is_oom


def _shrink_context(context: Context, keep_rows: int = SHRINK_CONTEXT_ROWS):
    """First ``keep_rows`` rows of the context, or None if it is not smaller."""
    if not isinstance(context, Context):
        return None                                     # fake context in tests
    keep = int(min(keep_rows, context.n_rows))
    if keep >= context.n_rows:
        return None
    return _dc_replace(
        context,
        x=context.x[:keep],
        y=context.y[:keep],
        n_rows=keep,
        fitted=False,                                   # must re-encode
    )


# ---------------------------------------------------------------------------
# Input validation (Phase 3)
# ---------------------------------------------------------------------------

def _warn(warnings_out, message: str) -> None:
    """Record a plain-language warning for the UI (no-op when not collected)."""
    if warnings_out is not None:
        warnings_out.append(message)


def validate_rows(
    rows_df: pd.DataFrame,
    context: Context,
    warnings_out: list | None = None,
) -> pd.DataFrame:
    """Check query rows before they reach the model.

    Two different outcomes:

    * **Unusable** input raises :class:`ValueError` with a message the UI can
      show as-is: missing column, missing value, non-numeric year/km,
      negative km.
    * **Unusual** input only adds a warning to ``warnings_out`` and the
      prediction still runs: an unseen brand/category, or a year/km far
      outside the range the model was trained on.

    Returns the rows re-ordered to the context's feature columns.
    """
    if rows_df is None or len(rows_df) == 0:
        raise ValueError("No rows to predict.")

    missing = [c for c in context.feature_columns if c not in rows_df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns: {missing}. "
            f"Expected columns: {list(context.feature_columns)}."
        )

    query = rows_df[list(context.feature_columns)].copy()

    # --- 1. missing values ------------------------------------------------
    nan_cols = [c for c in query.columns if query[c].isna().any()]
    blank_cols = [
        c for c in query.columns
        if c not in data_prep.NUMERICAL_COLUMNS
        and query[c].astype(str).str.strip().eq("").any()
    ]
    empty = sorted(set(nan_cols) | set(blank_cols))
    if empty:
        raise ValueError(
            f"Missing value for {empty}. "
            "Please choose a value for every field before predicting."
        )

    # --- 2. numbers must really be numbers --------------------------------
    for col in data_prep.NUMERICAL_COLUMNS:
        if col not in query.columns:
            continue
        coerced = pd.to_numeric(query[col], errors="coerce")
        if coerced.isna().any():
            bad = query.loc[coerced.isna(), col].iloc[0]
            raise ValueError(f"'{col}' must be a number (got {bad!r}).")
        query[col] = coerced

    # --- 3. hard reject: negative mileage ---------------------------------
    if (query["km_driven"] < 0).any():
        bad = float(query.loc[query["km_driven"] < 0, "km_driven"].iloc[0])
        raise ValueError(
            f"Km driven cannot be negative (got {int(bad):,}). "
            "Please enter the distance the car has been driven."
        )

    # --- 4. soft warnings: unseen categories ------------------------------
    for col, seen in getattr(context, "cat_values", {}).items():
        if col not in query.columns or not seen:
            continue
        for value in sorted({str(v) for v in query[col].unique()}):
            if value not in seen:
                _warn(warnings_out,
                      f"'{value}' is not a known {col} in the training data, "
                      "so the estimate may be unreliable.")

    # --- 5. soft warnings: numbers far outside the training range ---------
    for col, (lo, hi) in getattr(context, "num_ranges", {}).items():
        if col not in query.columns:
            continue
        for value in sorted(query[col].unique()):
            if value < lo or value > hi:
                _warn(warnings_out,
                      f"{col} = {value:,.0f} is outside the training range "
                      f"({lo:,.0f} to {hi:,.0f}), so the estimate may be "
                      "unreliable.")
    return query


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def _quantile_index(level: float) -> int:
    """Index into the 999 quantiles for a probability level in (0, 1).

    q001 = 0.001 ... q999 = 0.999, so q{k} has index k-1.
    """
    if not 0.0 < level < 1.0:
        raise ValueError(f"quantile level must be in (0, 1), got {level}")
    k = int(round(level * (N_QUANTILES + 1)))  # 0.5 -> 500, 0.1 -> 100
    return min(max(k, 1), N_QUANTILES) - 1


def _tensorize(query: pd.DataFrame, context: Context):
    """Turn validated query rows into a TableTensor with the context schema."""
    return sdm.TableTensor.from_pandas(
        df=query.reset_index(drop=True),
        stypes=context.feature_stypes,       # same stypes => same schema
        device=context.x.device,
    )


def _forward(model, context: Context, x_query):
    """One forward pass. Uses the cached context when the model was fitted."""
    num_estimators = getattr(model, "_carquantile_num_estimators", 1)
    with torch.inference_mode():
        if context.fitted:
            return model.predict(x_query)                # fast path: cached KV
        return model(
            x_context=context.x,
            y_context=context.y,
            x_query=x_query,
            num_estimators=num_estimators,
        )


def _extract(out, context: Context, low_q: float, high_q: float):
    """Pull the median / low / high quantiles out of the model output."""
    q = out.numerical.detach().float().cpu().numpy()
    if q.ndim == 3 and q.shape[0] == 1:      # drop a leftover estimator axis
        q = q[0]
    if q.ndim != 2 or q.shape[1] != N_QUANTILES:
        raise RuntimeError(
            f"Unexpected model output shape {q.shape}; "
            f"expected [n_rows, {N_QUANTILES}]."
        )

    price = q[:, _quantile_index(0.5)]
    low = q[:, _quantile_index(low_q)]
    high = q[:, _quantile_index(high_q)]

    if context.log_target:                    # undo the log transform
        price, low, high = np.expm1(price), np.expm1(low), np.expm1(high)
    return price, low, high


def _sanitise(price, low, high, context: Context, warnings_out=None):
    """Guarantee finite, positive, ordered output: low <= price <= high."""
    price = np.asarray(price, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)

    # Non-finite values (NaN/inf) would poison the UI -> use the training median.
    fallback = float(getattr(context, "median_price", 0.0)) or MIN_PREDICTION_PRICE
    for name, arr in (("price", price), ("low", low), ("high", high)):
        bad = ~np.isfinite(arr)
        if bad.any():
            arr[bad] = fallback
            _warn(warnings_out,
                  f"The model returned {int(bad.sum())} unusable {name} value(s); "
                  "the training median was used instead.")

    # Negative or absurdly small prices -> clip to a sensible minimum.
    floor = MIN_PREDICTION_PRICE
    clipped = int(np.sum(price < floor)) + int(np.sum(low < floor)) + int(np.sum(high < floor))
    if clipped:
        _warn(warnings_out,
              f"{clipped} value(s) below Rs {floor:,.0f} were clipped up; "
              "the model produced an unrealistically low price.")

    price = np.maximum(price, floor)
    low = np.maximum(low, floor)
    high = np.maximum(high, floor)

    # Final ordering guarantee: low <= price <= high.
    low = np.minimum(low, price)
    high = np.maximum(high, price)
    return price, low, high


def _predict_validated(
    model,
    context: Context,
    query: pd.DataFrame,
    low_q: float,
    high_q: float,
    warnings_out: list | None,
    _retry: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward pass for rows that already passed :func:`validate_rows`.

    On a CUDA out-of-memory error we free the cache and retry, first with a
    smaller batch, then with a smaller context. Each retry is reported in
    ``warnings_out`` so the UI can tell the user what changed.
    """
    x_query = _tensorize(query, context)
    try:
        out = _forward(model, context, x_query)
    except Exception as exc:
        if not _is_oom(exc):
            raise
        if not _retry:
            raise RuntimeError(
                "The GPU ran out of memory. Close other GPU apps and reload "
                "the page; a smaller context will be used."
            ) from exc

        # Recovery step 1: free the cache.
        torch.cuda.empty_cache()

        if len(query) > 1:                          # Recovery step 2: smaller batch
            half = max(1, len(query) // 2)
            _warn(warnings_out,
                  f"GPU out of memory: cache freed, prediction retried with "
                  f"a smaller batch ({half} row(s) at a time).")
            first = _predict_validated(
                model, context, query.iloc[:half], low_q, high_q, warnings_out)
            second = _predict_validated(
                model, context, query.iloc[half:], low_q, high_q, warnings_out)
            return (np.concatenate([first[0], second[0]]),
                    np.concatenate([first[1], second[1]]),
                    np.concatenate([first[2], second[2]]))

        # Recovery step 3: smaller context (single row already failed).
        smaller = _shrink_context(context, SHRINK_CONTEXT_ROWS)
        if smaller is None:
            raise RuntimeError(
                "The GPU ran out of memory. Close other GPU apps and reload "
                "the page."
            ) from exc
        _warn(warnings_out,
              f"GPU out of memory: cache freed, prediction retried with a "
              f"smaller context ({smaller.n_rows:,} rows instead of "
              f"{context.n_rows:,}).")
        out = _forward(model, smaller, x_query)      # one retry only

    price, low, high = _extract(out, context, low_q, high_q)
    return _sanitise(price, low, high, context, warnings_out)


def predict(
    model,
    context: Context,
    rows_df: pd.DataFrame,
    low_q: float = LOW_Q,
    high_q: float = HIGH_Q,
    warnings_out: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate prices and a quantile range for one or more rows.

    Parameters
    ----------
    model, context:
        The loaded Kumo Tabular model and its (possibly fitted) context.
    rows_df:
        One or more cars with the context's feature columns.
    low_q, high_q:
        Quantile levels of the range (defaults: 10th and 90th = 80% range).
    warnings_out:
        Optional list. Plain-language warnings (unseen brand, extreme year/km,
        out-of-memory retries, clipped prices) are appended to it so the UI
        can show them without parsing exceptions.

    Returns
    -------
    (price, low, high)
        Three numpy arrays of length ``len(rows_df)``, in rupees, always
        satisfying ``low <= price <= high`` and ``price >= 1000``.

    Raises
    ------
    ValueError
        Unusable input: missing column, missing value, negative km. The
        message is written for a human, not a log file.
    """
    # Input problems are reported before anything GPU/sdm related, so the
    # message the user sees is about *their* input, not about the library.
    query = validate_rows(rows_df, context, warnings_out)
    if sdm is None:
        raise ImportError("The 'sdm' package is not installed (see load_model).")
    return _predict_validated(model, context, query, low_q, high_q, warnings_out)


def predict_batch(
    model,
    context: Context,
    rows_df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    low_q: float = LOW_Q,
    high_q: float = HIGH_Q,
    warnings_out: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict many rows in batches so only ``batch_size`` rows hit the GPU.

    Used by the benchmark and the evaluation over the 1,357-row test set —
    sending every row at once would spike VRAM on the 6 GB card.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    query = validate_rows(rows_df, context, warnings_out)
    if sdm is None:
        raise ImportError("The 'sdm' package is not installed (see load_model).")

    prices, lows, highs = [], [], []
    for start in range(0, len(query), int(batch_size)):
        chunk = query.iloc[start:start + int(batch_size)]
        p, lo, hi = _predict_validated(model, context, chunk, low_q, high_q,
                                       warnings_out)
        prices.append(np.asarray(p))
        lows.append(np.asarray(lo))
        highs.append(np.asarray(hi))

    return (np.concatenate(prices),
            np.concatenate(lows),
            np.concatenate(highs))


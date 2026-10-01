"""NVIDIA Kumo Tabular wrapper: load the model, build the context, predict.

Kumo Tabular is an *in-context* foundation model: it is never trained on our
data. Instead we hand it a labeled "context" table (the 80% split) and it
predicts new rows in a single forward pass.

Public API (used by the demo, the app and the tests):

    model                  = load_model(size="small")
    context                = build_context(context_df)
    price, low, high       = predict(model, context, rows_df)

`price` is the model's median (q500); `low`/`high` are the requested
quantiles (10th/90th by default). All three are numpy arrays in **rupees**.

API facts were verified against sdm 0.x (see NOTES.md).
"""

from __future__ import annotations

from dataclasses import dataclass

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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    size: str = "small",
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
    )


def fit_context(model, context: Context, num_estimators: int | None = None) -> Context:
    """Cache the encoded context inside the model (optional optimisation).

    After this call, :func:`predict` uses ``model.predict`` and reuses the
    context keys/values, so repeated predictions skip re-encoding the
    context. Requires the model to already be in eval mode.
    """
    n = num_estimators or getattr(model, "_carquantile_num_estimators", 1)
    model.fit(x=context.x, y=context.y, num_estimators=n)
    context.fitted = True
    return context


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


def predict(
    model,
    context: Context,
    rows_df: pd.DataFrame,
    low_q: float = 0.1,
    high_q: float = 0.9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate prices and a quantile range for one or more rows.

    Returns
    -------
    (price, low, high)
        Three numpy arrays of length ``len(rows_df)``, in rupees.
        ``low``/``high`` are the ``low_q``/``high_q`` quantiles of the model's
        predicted distribution.

    Raises
    ------
    ValueError
        If a required feature column is missing, with a message the UI can show.
    """
    if sdm is None:
        raise ImportError("The 'sdm' package is not installed (see load_model).")

    # --- validate inputs -------------------------------------------------
    missing = [c for c in context.feature_columns if c not in rows_df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if len(rows_df) == 0:
        raise ValueError("No rows to predict.")

    # --- build the query table with the SAME schema as the context -------
    query = rows_df[context.feature_columns].copy()
    for col in context.feature_columns:
        if col in data_prep.NUMERICAL_COLUMNS:
            query[col] = pd.to_numeric(query[col], errors="coerce")
        else:
            query[col] = query[col].astype(str)
    x_query = sdm.TableTensor.from_pandas(
        df=query.reset_index(drop=True),
        stypes=context.feature_stypes,       # same stypes => same schema
        device=context.x.device,
    )

    # --- run the forward pass -------------------------------------------
    num_estimators = getattr(model, "_carquantile_num_estimators", 1)
    with torch.inference_mode():
        if context.fitted:
            # Reuse the cached context (fast path).
            out = model.predict(x_query)
        else:
            out = model(
                x_context=context.x,
                y_context=context.y,
                x_query=x_query,
                num_estimators=num_estimators,
            )

    # --- extract the quantiles ------------------------------------------
    q = out.numerical.detach().float().cpu().numpy()
    if q.ndim == 3 and q.shape[0] == 1:   # drop a leftover estimator axis
        q = q[0]
    if q.ndim != 2 or q.shape[1] != N_QUANTILES:
        raise RuntimeError(
            f"Unexpected model output shape {q.shape}; "
            f"expected [n_rows, {N_QUANTILES}]."
        )

    price = q[:, _quantile_index(0.5)]
    low = q[:, _quantile_index(low_q)]
    high = q[:, _quantile_index(high_q)]

    # Undo the log transform if we used one.
    if context.log_target:
        price, low, high = np.expm1(price), np.expm1(low), np.expm1(high)

    # Guard the API contract low <= price <= high (numerical safety).
    low = np.minimum(low, price)
    high = np.maximum(high, price)

    return price, low, high

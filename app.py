"""CarQuantile — Phase 2 Streamlit UI.

A simple interface around the Phase 1 model wrapper (src/model.py):
choose a car, get a median price plus an 80% quantile range, and try
"Add a new sale" to adapt the in-context model without retraining.

Run me:
    streamlit run app.py

IMPORTANT (Windows, this machine): torch must be imported BEFORE
pandas/pyarrow, otherwise torch's c10.dll can fail to load (WinError 1114,
see NOTES.md §6). src/model.py already imports torch first, so we simply
import it before anything pandas-related happens.
"""

from __future__ import annotations

import time

# Import src.model FIRST: it imports torch before pandas (DLL load-order
# workaround for this machine, see NOTES.md §6).
import src.model as kumo

import matplotlib
matplotlib.use("Agg")                       # headless-safe backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker                    # noqa: F401  (used below)
import pandas as pd
import streamlit as st

from src import applog, data_prep, formatting

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_SIZE = kumo.DEFAULT_MODEL_SIZE   # benchmark-chosen; fits the 6 GB RTX 3050
DEFAULT_YEAR = 2018           # slider default, per the brief
DEFAULT_KM = 50_000           # slider default
KM_STEP = 1_000               # km slider step
LOW_Q, HIGH_Q = 0.1, 0.9      # 80% range = 10th..90th percentile

# Realistic dropdown defaults: the most common values in the dataset, so the
# form opens on a typical car instead of the alphabetically first one.
DEFAULT_BRAND = "maruti"
DEFAULT_FUEL = "Diesel"
DEFAULT_SELLER_TYPE = "Individual"
DEFAULT_TRANSMISSION = "Manual"
DEFAULT_OWNER = "First Owner"
DEFAULT_CITY = "Mumbai"                 # most common city in the dataset
DEFAULT_INSURANCE_STATUS = "Third Party"  # most common insurance status
DEFAULT_SERVICE_HISTORY = "Full"         # most common service history


# ---------------------------------------------------------------------------
# Cached startup: data, model, context  (run once per process)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading data...")
def _load_data() -> dict:
    """Load + clean + split the dataset, and collect UI dropdown/slider info."""
    train_df, _test_df = data_prep.load_and_prepare()
    return {"train_df": train_df, "options": data_prep.get_ui_options(train_df)}


@st.cache_resource(show_spinner="Loading model...")
def _load_everything() -> dict:
    """Load the Kumo Tabular model and build the context ONCE.

    Returns the model, the built Context tensors, and plain DataFrames:
      - ctx_df   : the *current* context table (what the model was built with)
      - orig_df  : a pristine copy, used by the "Reset context" button
    The copies are mirrored into st.session_state by the module-level code
    below (session state may not be touched inside cached functions).
    """
    data = _load_data()
    # Use the benchmark-chosen context size (None = every context row).
    orig_context_df = kumo.limit_context(data["train_df"])

    model = kumo.load_model(size=MODEL_SIZE)
    context = kumo.build_context(orig_context_df)

    # Encode the context ONCE here (fit = build the KV cache). Later
    # predictions then skip re-encoding, which is what makes repeat clicks
    # fast. If it fails we log it and fall back to encoding per prediction.
    try:
        context = kumo.fit_context(model, context)
    except Exception as exc:
        applog.log_exception("startup fit_context", exc)

    return {"model": model, "context": context,
            "ctx_df": orig_context_df.copy(),
            "orig_df": orig_context_df.copy()}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _default_index(options_list: list[str], preferred: str) -> int:
    """Index of the preferred dropdown default (fall back to the first)."""
    try:
        return options_list.index(preferred)
    except ValueError:
        return 0


def _label(row: dict) -> str:
    """Human-readable label for a car (dict of feature values)."""
    return (f"{row['brand']} {row['year']}, {int(row['km_driven']):,} km, "
            f"{row['fuel']}, {row['transmission']}, {row['owner']}, "
            f"{row['city']}")


def build_row(brand: str, year: int, km_driven: int, fuel: str,
              seller_type: str, transmission: str, owner: str,
              city: str, insurance_status: str, service_history: str,
              price: float | None = None) -> pd.DataFrame:
    """Build a one-row DataFrame with the context's exact schema.

    Columns, order and dtypes match the context table (verified in Phase 1):
    brand/fuel/seller_type/transmission/owner/city/insurance_status/
    service_history are strings, year/km_driven are ints, and selling_price
    (only when labeling a new sale) is float. predict() re-casts against the
    context schema, but matching dtypes here keeps the bug surface at zero.
    """
    row = {
        "brand": str(brand),
        "year": int(year),
        "km_driven": int(km_driven),
        "fuel": str(fuel),
        "seller_type": str(seller_type),
        "transmission": str(transmission),
        "owner": str(owner),
        "city": str(city),
        "insurance_status": str(insurance_status),
        "service_history": str(service_history),
    }
    if price is not None:                      # labeled row for "Add a new sale"
        row[data_prep.TARGET_COL] = float(price)
    cols = (data_prep.FEATURE_COLUMNS
            + ([data_prep.TARGET_COL] if price is not None else []))
    return pd.DataFrame([row])[cols]


def _validate_new_sale(df: pd.DataFrame) -> str | None:
    """Return a plain-language problem description, or None if the row is OK."""
    row = df.iloc[0]
    for col in data_prep.CATEGORICAL_COLUMNS:
        if not str(row[col]).strip():
            return f"Please choose a value for '{col}'."
    if row["year"] < data_prep.MIN_YEAR:
        return f"Year must be {data_prep.MIN_YEAR} or later."
    if row["km_driven"] < 0:
        return "Km driven cannot be negative."
    if row[data_prep.TARGET_COL] < data_prep.MIN_PRICE:
        return "Selling price must be a positive number."
    return None


def rebuild_context(new_df: pd.DataFrame):
    """Rebuild the Kumo context tensors from a (possibly extended) table."""
    context = kumo.build_context(new_df)
    # Re-encode it once, so predictions after an add/reset stay fast.
    try:
        context = kumo.fit_context(model, context)
    except Exception as exc:
        applog.log_exception("rebuild_context fit_context", exc)
    st.session_state["context"] = context
    st.session_state["ctx_df"] = new_df
    return context


def range_chart(price: float, low: float, high: float) -> "plt.Figure":
    """Small horizontal bar: the 80% range with the median marked on it."""
    fig, ax = plt.subplots(figsize=(9, 1.5))
    span = max(high - low, 1.0)
    pad = 0.08 * span                          # breathing room for the markers
    ax.barh([0], high - low, left=low, height=0.45, color="#90caf9",
            edgecolor="none", label="80% range (P10\u2013P90)")
    ax.plot([price, price], [-0.30, 0.30], color="#2e7d32", lw=3,
            label="Median estimate")
    ax.set_xlim(low - pad, high + pad)
    ax.set_yticks([])
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("Price (Rs)", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _p: formatting.format_inr(v)))
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=2, fontsize=8,
              frameon=False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CarQuantile",
    page_icon="🚗",
    layout="wide",
)

st.title("CarQuantile: Used Car Price Estimator")
st.caption("Median price and an 80% range, powered by NVIDIA Kumo Tabular. "
           "No model training.")

# Safety check: this app needs the NVIDIA GPU (Kumo Tabular is extremely
# slow on CPU). A CUDA device is always an NVIDIA card, so this guards the
# CPU-only-torch-wheel case.
if not kumo.cuda_available():
    st.error(
        "An NVIDIA GPU with CUDA is required to run this app.\n\n"
        "PyTorch cannot see a CUDA device right now. Most likely the CPU-only "
        "torch wheel is installed. Re-install it with:\n\n"
        "`pip install torch --index-url https://download.pytorch.org/whl/cu128`"
    )
    st.stop()

try:
    everything = _load_everything()
except Exception as exc:                  # e.g. CUDA OOM while tensorizing
    applog.log_exception("startup _load_everything", exc)
    st.error(applog.user_message(
        "Something went wrong while loading the model or building the "
        "context. Try closing other GPU apps and reloading the page.", exc))
    st.stop()
model = everything["model"]

# Mirror the startup objects into session state ONCE per browser session
# (not inside the cached function: Streamlit forbids session_state there).
if "context" not in st.session_state:
    st.session_state["context"] = everything["context"]
    st.session_state["ctx_df"] = everything["ctx_df"]
    st.session_state["orig_df"] = everything["orig_df"]

options = _load_data()["options"]

year_min, year_max = options["year"]
km_min, km_max = options["km_driven"]

# Snap the km slider's lower end onto the step grid (a dataset min of 1 km
# would otherwise give ugly stops like 1, 1,001, 2,001 ...).
km_min = ((km_min + KM_STEP - 1) // KM_STEP) * KM_STEP

# Sensible slider defaults even if the dataset changes.
default_year = min(max(DEFAULT_YEAR, year_min), year_max)
default_km = min(max(DEFAULT_KM, km_min), km_max)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

info = kumo.device_info()
ctx_now: kumo.Context = st.session_state["context"]

with st.sidebar:
    st.header("🚗 CarQuantile")
    st.subheader("Runtime")
    st.write(f"**GPU:** {info['name']}")          # CUDA device == the NVIDIA card
    st.write(f"**Total VRAM:** {info['vram_gb']} GB")
    st.write(f"**Model size:** {getattr(model, 'size', MODEL_SIZE).capitalize()}")
    st.write(f"**Context size:** {ctx_now.n_rows:,} labeled rows")

    with st.expander("Add a new sale (adapt the model without retraining)"):
        st.write("Teach the model a real sale you know about. It is appended "
                 "to the context, so later predictions use it — **no "
                 "retraining needed**.")
        st.caption("⚠️ Session-only: this changes the model's context for this "
                   "session and does **not** modify `data/cars.csv`.")

        # Same fields as the main form, but for the NEW sale being added
        # (separate widget keys with a sale_ prefix).
        st.selectbox("Brand", options["brand"],
                     index=_default_index(options["brand"], DEFAULT_BRAND),
                     key="sale_brand")
        st.slider("Year", year_min, year_max, value=default_year,
                  key="sale_year")
        st.slider("Km driven", km_min, km_max, value=default_km,
                  key="sale_km_driven", step=KM_STEP)
        st.selectbox("Fuel", options["fuel"],
                     index=_default_index(options["fuel"], DEFAULT_FUEL),
                     key="sale_fuel")
        st.selectbox("Seller type", options["seller_type"],
                     index=_default_index(options["seller_type"],
                                          DEFAULT_SELLER_TYPE),
                     key="sale_seller_type")
        st.selectbox("Transmission", options["transmission"],
                     index=_default_index(options["transmission"],
                                          DEFAULT_TRANSMISSION),
                     key="sale_transmission")
        st.selectbox("Owner", options["owner"],
                     index=_default_index(options["owner"], DEFAULT_OWNER),
                     key="sale_owner")
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox("City", options["city"],
                         index=_default_index(options["city"], DEFAULT_CITY),
                         key="sale_city")
        with c2:
            st.selectbox("Insurance status", options["insurance_status"],
                         index=_default_index(options["insurance_status"],
                                              DEFAULT_INSURANCE_STATUS),
                         key="sale_insurance_status")
        st.selectbox("Service history", options["service_history"],
                     index=_default_index(options["service_history"],
                                          DEFAULT_SERVICE_HISTORY),
                     key="sale_service_history")
        new_price = st.number_input(
            "Actual selling price (Rs)",
            min_value=float(data_prep.MIN_PRICE),
            value=500_000.0,
            step=10_000.0,
        )

        c1, c2 = st.columns(2)
        if c1.button("Add to context"):
            row = build_row(
                st.session_state.sale_brand, st.session_state.sale_year,
                st.session_state.sale_km_driven, st.session_state.sale_fuel,
                st.session_state.sale_seller_type,
                st.session_state.sale_transmission, st.session_state.sale_owner,
                st.session_state.sale_city,
                st.session_state.sale_insurance_status,
                st.session_state.sale_service_history,
                price=new_price,
            )
            problem = _validate_new_sale(row)
            if problem:
                st.error(problem)
            else:
                current: pd.DataFrame = st.session_state["ctx_df"]
                extended = kumo.add_sale(current, row)
                try:
                    rebuild_context(extended)
                except Exception as exc:      # friendly message, no traceback
                    applog.log_exception("Add to context", exc)
                    st.error(applog.user_message(
                        "Could not add that sale to the context. Please "
                        "check the values and try again.", exc))
                else:
                    st.success(
                        f"Added to context ({_label(row.iloc[0].to_dict())} at "
                        f"{formatting.format_rupees(new_price)}). Context is now "
                        f"{st.session_state['context'].n_rows:,} rows for this "
                        "session.")
        if c2.button("Reset context"):
            try:
                rebuild_context(st.session_state["orig_df"].copy())
            except Exception as exc:          # friendly message, no traceback
                applog.log_exception("Reset context", exc)
                st.error(applog.user_message(
                    "Could not reset the context. Please reload the page.", exc))
            else:
                st.success(f"Context restored to the original "
                           f"{st.session_state['context'].n_rows:,} rows.")

    st.caption("The context is the labeled table the model reads at inference "
               "time. Adding or resetting rows only affects this session.")

# ---------------------------------------------------------------------------
# Main input form (two columns)
# ---------------------------------------------------------------------------

st.subheader("Describe the car")
left, right = st.columns(2)

with left:
    st.selectbox("Brand", options["brand"],
                 index=_default_index(options["brand"], DEFAULT_BRAND),
                 key="brand")
    st.slider("Year", year_min, year_max, value=default_year, key="year")

with right:
    st.slider("Km driven", km_min, km_max, value=default_km,
              key="km_driven", step=KM_STEP)
    st.selectbox("Fuel", options["fuel"],
                 index=_default_index(options["fuel"], DEFAULT_FUEL),
                 key="fuel")

st.selectbox("Seller type", options["seller_type"],
             index=_default_index(options["seller_type"], DEFAULT_SELLER_TYPE),
             key="seller_type")
c1, c2 = st.columns(2)
with c1:
    st.selectbox("Transmission", options["transmission"],
                 index=_default_index(options["transmission"],
                                      DEFAULT_TRANSMISSION),
                 key="transmission")
with c2:
    st.selectbox("Owner", options["owner"],
                 index=_default_index(options["owner"], DEFAULT_OWNER),
                 key="owner")

c1, c2, c3 = st.columns(3)
with c1:
    st.selectbox("City", options["city"],
                 index=_default_index(options["city"], DEFAULT_CITY),
                 key="city")
with c2:
    st.selectbox("Insurance status", options["insurance_status"],
                 index=_default_index(options["insurance_status"],
                                      DEFAULT_INSURANCE_STATUS),
                 key="insurance_status")
with c3:
    st.selectbox("Service history", options["service_history"],
                 index=_default_index(options["service_history"],
                                      DEFAULT_SERVICE_HISTORY),
                 key="service_history")

estimate_clicked = st.button("Estimate price", type="primary")

if not estimate_clicked:
    # Shown whenever no result is on screen (the button is momentary, so
    # result and hint never appear at the same time).
    st.caption("Pick a car above and click **Estimate price**. The first "
               "prediction includes a one-off CUDA warm-up; later ones are "
               "much faster.")

# ---------------------------------------------------------------------------
# Estimate
# ---------------------------------------------------------------------------

if estimate_clicked:
    values = {  # plain dict of the form values, for labeling and building
        "brand": st.session_state.brand,
        "year": st.session_state.year,
        "km_driven": st.session_state.km_driven,
        "fuel": st.session_state.fuel,
        "seller_type": st.session_state.seller_type,
        "transmission": st.session_state.transmission,
        "owner": st.session_state.owner,
        "city": st.session_state.city,
        "insurance_status": st.session_state.insurance_status,
        "service_history": st.session_state.service_history,
    }
    query = build_row(**values)               # schema-matched one-row DataFrame
    warnings: list[str] = []                  # plain-language input warnings
    with st.spinner("Predicting..."):
        t0 = time.perf_counter()
        try:
            price, low, high = kumo.predict(
                model, st.session_state["context"], query,
                low_q=LOW_Q, high_q=HIGH_Q, warnings_out=warnings,
            )
        except Exception as exc:              # friendly message, no traceback
            applog.log_exception("Estimate price", exc)
            st.error(applog.user_message(
                "Prediction failed. Please try different inputs, and check "
                "that the NVIDIA GPU is available.", exc))
            st.stop()
        elapsed = time.perf_counter() - t0

    # Unseen brand, extreme year/km, OOM retries, clipped prices, ...
    # (dict.fromkeys keeps the order and removes duplicates.)
    for message in dict.fromkeys(warnings):
        st.warning(message)

    p, lo, hi = float(price[0]), float(low[0]), float(high[0])
    if not (lo <= p <= hi):                   # should never happen; be honest
        lo, p, hi = sorted([lo, p, hi])
        st.warning("The model returned an unordered range; values were "
                   "sorted for display.")

    st.subheader("Result")
    st.metric("Estimated price", "Rs " + formatting.format_inr(p))
    st.write(f"Likely between **Rs {formatting.format_inr(lo)}** and "
             f"**Rs {formatting.format_inr(hi)}** (80% range)")
    st.pyplot(range_chart(p, lo, hi))
    plt.close("all")                      # free the figure after rendering
    st.caption("Range = 10th to 90th percentile from the model's quantile "
               "output.")
    st.caption(f"Prediction took **{elapsed:.2f} s** · "
               f"Query: {_label(values)}")

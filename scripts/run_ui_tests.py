"""Manual UI test cases for the CarQuantile Streamlit app.

Writes results/ui_tests.md with every input tried, the value the app showed
and any warning/error text a user would see.

Sections:
  A. Widget cases - driven through Streamlit's AppTest, i.e. the exact script
     `streamlit run app.py` runs (dropdowns, sliders, "Estimate price").
  B. Cases the widgets cannot express (unseen brand, negative km, missing
     field) - run through app.build_row + model.predict, the same functions
     the Estimate button calls.
  C. Failure paths: no CUDA device, and CUDA out of memory (checked on a
     patched copy of app.py so the real app is untouched).
  D. Context caching: first prediction vs later predictions.
  E. "Add a new sale" / "Reset context": context size before and after.

Run from the project root:  python scripts/run_ui_tests.py
"""

from __future__ import annotations

import datetime as _dt
import gc
import io
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# The app and Streamlit print a lot of bare-mode warnings; keep the console
# readable and make sure unicode (rupee sign) does not crash on cp1252.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace")

from src import applog, data_prep, formatting, model as kumo  # noqa: E402

APP_PATH = ROOT / "app.py"
OUT_PATH = ROOT / "results" / "ui_tests.md"
LOG_PATH = ROOT / "logs" / "app.log"
TIMEOUT = 300

TEXT_KEYS = ("brand", "fuel", "seller_type", "transmission", "owner",
             "city", "insurance_status", "service_history")
NUM_KEYS = ("year", "km_driven")


# ---------------------------------------------------------------------------
# Widget cases (filled in from the dataset ranges)
# ---------------------------------------------------------------------------

def build_widget_cases(train_df) -> list[dict]:
    """The 12 cases the dropdowns/sliders can express."""
    opts = data_prep.get_ui_options(train_df)
    year_min, year_max = opts["year"]
    km_min, km_max = opts["km_driven"]

    counts = train_df["brand"].value_counts()
    rare_brand = str(counts.index[-1])           # least common brand
    luxury = "bmw" if "bmw" in opts["brand"] else opts["brand"][-1]

    def case(title, brand, year, km, fuel, seller, trans, owner, note,
             city=None, insurance=None, service=None):
        values = {"brand": brand, "year": year, "km_driven": km,
                  "fuel": fuel, "seller_type": seller,
                  "transmission": trans, "owner": owner}
        # The three newer dropdowns keep the form defaults unless a case
        # picks a specific value to exercise them.
        if city is not None:
            values["city"] = city
        if insurance is not None:
            values["insurance_status"] = insurance
        if service is not None:
            values["service_history"] = service
        return {"title": title, "note": note, "values": values}

    return [
        case("Typical car (form defaults)", "maruti", 2018, 50_000, "Diesel",
             "Individual", "Manual", "First Owner", "everyday case"),
        case("Very old car (oldest year)", "maruti", year_min, 80_000,
             "Petrol", "Individual", "Manual", "First Owner",
             "edge: oldest year in the data",
             city="Kolkata", insurance="Expired", service="Partial"),
        case("Very high km (max slider)", "hyundai", 2015, km_max, "Diesel",
             "Individual", "Manual", "Second Owner",
             "edge: maximum km_driven"),
        case("Minimum slider case", "maruti", year_min, km_min, "Petrol",
             "Individual", "Manual", "First Owner",
             "edge: year and km at the minimum"),
        case("Maximum slider case", "maruti", year_max, km_max, "Diesel",
             "Dealer", "Automatic", "First Owner",
             "edge: year and km at the maximum"),
        case(f"Rare brand ({rare_brand})", rare_brand, 2016, 60_000, "Diesel",
             "Dealer", "Manual", "First Owner",
             "edge: least common brand in the data"),
        case("Luxury automatic", luxury, 2019, 30_000, "Petrol", "Dealer",
             "Automatic", "First Owner", "high price bracket",
             city="Delhi", insurance="Comprehensive", service="Full"),
        case("Old + maximum km + third owner", "hyundai", year_min, km_max,
             "Diesel", "Individual", "Manual", "Third Owner",
             "edge: worst-case age and mileage",
             city="Jaipur", insurance="Expired", service="Not Recorded"),
        case("CNG + dealer", "maruti", 2020, 20_000, "CNG", "Dealer",
             "Manual", "First Owner", "unusual fuel/seller combination"),
        case("Nearly new automatic", "hyundai", year_max, 15_000, "Petrol",
             "Dealer", "Automatic", "First Owner", "newest year in the data"),
        case("High-mileage petrol", "honda", 2013, 180_000, "Petrol",
             "Individual", "Manual", "Second Owner", "high km, older car"),
        case("Diesel automatic sedan", "toyota", 2018, 70_000, "Diesel",
             "Dealer", "Automatic", "First Owner", "common premium setup"),
    ]


# ---------------------------------------------------------------------------
# Running one widget case
# ---------------------------------------------------------------------------

def _first(elements, predicate, default=None):
    for el in elements:
        try:
            if predicate(el.value if hasattr(el, "value") else ""):
                return el
        except Exception:
            continue
    return default


def run_widget_cases() -> tuple[list[dict], list[str]]:
    """Click through every case in the real app script."""
    from streamlit.testing.v1 import AppTest

    rows: list[dict] = []
    problems: list[str] = []

    train_df, _test = data_prep.load_and_prepare()
    cases = build_widget_cases(train_df)

    at = AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT)
    at.run()
    if at.exception:
        raise RuntimeError(f"app failed to start: {at.exception[0].value}")

    for i, case in enumerate(cases, 1):
        for key, value in case["values"].items():
            if key in TEXT_KEYS:
                at.selectbox(key=key).set_value(value)
            else:
                at.slider(key=key).set_value(value)
        btn = next(b for b in at.button if b.label == "Estimate price")
        btn.click()
        at.run()

        if at.exception:
            problems.append(f"case {i} ({case['title']}): "
                            f"traceback {at.exception[0].value}")
        errors = [e.value for e in at.error]
        warnings = [w.value for w in at.warning]
        metric = next((m.value for m in at.metric
                       if m.label == "Estimated price"), None)
        range_text = ""
        for md in at.markdown:
            if "Likely between" in str(md.value):
                range_text = re.sub(r"[*_]", "", str(md.value))
                break
        timing = ""
        for cap in at.caption:
            m = re.search(r"Prediction took \*\*([0-9.]+) s\*\*",
                          str(cap.value))
            if m:
                timing = m.group(1) + " s"
                break

        values = case["values"]
        rows.append({
            "n": i,
            "title": case["title"],
            "input": (f"{values['brand']}, {values['year']}, "
                      f"{values['km_driven']:,} km, {values['fuel']}, "
                      f"{values['seller_type']}, {values['transmission']}, "
                      f"{values['owner']}"
                      + (f", {values['city']}, {values['insurance_status']}, "
                         f"{values['service_history']}"
                         if "city" in values else "")),
            "estimate": metric or "(no result)",
            "range": range_text or "-",
            "messages": "; ".join(errors + warnings) or "none",
            "time": timing or "-",
            "note": case["note"],
        })
        if errors:
            problems.append(f"case {i} ({case['title']}): error shown "
                            f"'{errors[0][:120]}'")
    return rows, problems


# ---------------------------------------------------------------------------
# Cases the widgets cannot express
# ---------------------------------------------------------------------------

def run_model_layer_cases() -> list[dict]:
    """Unseen brand / negative km / missing field through the app's helpers."""
    rows: list[dict] = []

    # app.build_row builds the exact DataFrame the Estimate button builds;
    # importing app also loads the model and the context once.
    import app
    model = getattr(app, "model", None)
    context = None
    everything = getattr(app, "everything", None)
    if isinstance(everything, dict):
        context = everything.get("context")
    if model is None or context is None:            # fallback: build our own
        train_df, _ = data_prep.load_and_prepare()
        model = kumo.load_model(size=kumo.DEFAULT_MODEL_SIZE)
        context = kumo.build_context(
            kumo.limit_context(train_df))
        context = kumo.fit_context(model, context)

    base = dict(brand="maruti", year=2018, km_driven=50_000, fuel="Diesel",
                seller_type="Individual", transmission="Manual",
                owner="First Owner", city="Mumbai",
                insurance_status="Third Party", service_history="Full")

    # 1. unseen brand -> warning, still a result
    row = app.build_row(**{**base, "brand": "not_a_real_brand_xyz"})
    warnings: list[str] = []
    try:
        price, low, high = kumo.predict(model, context, row,
                                        warnings_out=warnings)
        result = (f"{formatting.format_rupees(float(price[0]))} "
                  f"(likely {formatting.format_rupees(float(low[0]))} - "
                  f"{formatting.format_rupees(float(high[0]))})")
        message = "; ".join(dict.fromkeys(warnings)) or "none"
    except Exception as exc:
        result, message = "no result", applog.user_message("failed", exc)
    rows.append({"case": "Unseen brand (not in any dropdown)",
                 "input": "brand='not_a_real_brand_xyz', rest as defaults",
                 "result": result, "message": message})

    # 2. negative km -> clear rejection
    row = app.build_row(**{**base, "km_driven": -500})
    try:
        kumo.predict(model, context, row)
        result, message = "unexpectedly predicted", "none"
    except ValueError as exc:
        result, message = "rejected", str(exc)
    except Exception as exc:
        result, message = "rejected", applog.user_message("failed", exc)
    rows.append({"case": "Negative km driven",
                 "input": "km_driven = -500",
                 "result": result, "message": message})

    # 3. missing field -> clear rejection
    row = app.build_row(**base)
    row["fuel"] = None
    try:
        kumo.predict(model, context, row)
        result, message = "unexpectedly predicted", "none"
    except ValueError as exc:
        result, message = "rejected", str(exc)
    except Exception as exc:
        result, message = "rejected", applog.user_message("failed", exc)
    rows.append({"case": "Missing field (fuel empty)",
                 "input": "fuel = None",
                 "result": result, "message": message})

    # 4. extreme year/km -> warning, still a result
    row = app.build_row(**{**base, "year": 1850, "km_driven": 900_000})
    warnings = []
    try:
        price, low, high = kumo.predict(model, context, row,
                                        warnings_out=warnings)
        result = (f"{formatting.format_rupees(float(price[0]))} "
                  f"(likely {formatting.format_rupees(float(low[0]))} - "
                  f"{formatting.format_rupees(float(high[0]))})")
        message = "; ".join(dict.fromkeys(warnings)) or "none"
    except Exception as exc:
        result, message = "no result", applog.user_message("failed", exc)
    rows.append({"case": "Extreme inputs (year 1850, 900,000 km)",
                 "input": "year=1850, km_driven=900000",
                 "result": result, "message": message})
    return rows


# ---------------------------------------------------------------------------
# Failure paths (patched copies of app.py - the real file is untouched)
# ---------------------------------------------------------------------------

_OOM_SNIPPET = """import src.model as kumo
if not hasattr(kumo, "_test_real_predict"):
    kumo._test_real_predict = kumo.predict     # keep the real one for later
def _predict_with_one_oom(*args, **kwargs):
    # Fail once per process. The flag lives on the shared src.model module,
    # so it survives Streamlit re-running the whole script on every click.
    if not getattr(kumo, "_test_oom_raised", False):
        kumo._test_oom_raised = True
        raise kumo.torch.cuda.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 2.00 GiB "
            "(GPU 0; 5.95 GiB total capacity)")
    return kumo._test_real_predict(*args, **kwargs)
kumo.predict = _predict_with_one_oom
"""


def _patched_app(mutate) -> "object":
    """Run a patched copy of app.py and return the AppTest handle."""
    from streamlit.testing.v1 import AppTest

    source = APP_PATH.read_text(encoding="utf-8")
    patched = mutate(source)
    at = AppTest.from_string(patched, default_timeout=TIMEOUT)
    at.run()
    return at


class _temporary_patch:
    """Run a patched app, then restore the real src.model functions.

    The patch is applied by executing modified source, which assigns
    `src.model.cuda_available` / `src.model.predict` on the *shared* module.
    Without the restore, the "no CUDA" patch would leak into every later
    check in this process.
    """

    def __init__(self, mutate):
        import src.model as _m
        self._module = _m
        self._mutate = mutate
        self._saved = {"cuda_available": _m.cuda_available,
                       "predict": _m.predict}

    def __enter__(self):
        return _patched_app(self._mutate)

    def __exit__(self, *exc):
        self._module.cuda_available = self._saved["cuda_available"]
        self._module.predict = self._saved["predict"]
        for attr in ("_test_real_predict", "_test_oom_raised"):
            if hasattr(self._module, attr):
                delattr(self._module, attr)
        return False


def check_no_cuda() -> dict:
    """CUDA missing -> friendly error, app stops, no traceback."""

    def mutate(src: str) -> str:
        return src.replace(
            "import src.model as kumo",
            "import src.model as kumo\nkumo.cuda_available = lambda: False", 1)

    with _temporary_patch(mutate) as at:
        errors = [e.value for e in at.error]
        traceback_shown = bool(at.exception) or any(
            "Traceback" in str(e) for e in errors)
        return {
            "shown": errors[0] if errors else "(no message)",
            "traceback": traceback_shown,
            "ok": bool(errors) and not traceback_shown
                  and "CUDA" in errors[0],
        }


def check_cuda_oom() -> dict:
    """First predict raises OOM -> friendly message, logged, retry works."""
    if LOG_PATH.exists():
        before = LOG_PATH.stat().st_size
    else:
        before = 0

    with _temporary_patch(lambda src: src.replace(
            "import src.model as kumo", _OOM_SNIPPET.rstrip(), 1)) as at:

        if not any(b.label == "Estimate price" for b in at.button):
            details = "; ".join(str(e.value)[:200] for e in at.exception) or \
                      "; ".join(e.value[:200] for e in at.error) or "unknown"
            return {"shown": f"(app did not start: {details})",
                    "traceback": bool(at.exception), "logged": False,
                    "recovered": False}

        btn = next(b for b in at.button if b.label == "Estimate price")
        btn.click()
        at.run()
        errors = [e.value for e in at.error]
        traceback_shown = bool(at.exception) or any(
            "Traceback" in str(e) for e in errors)

        time.sleep(0.5)
        new_log = ""
        if LOG_PATH.exists():
            # Slice BYTES: `before` comes from stat().st_size, while
            # read_text() normalises CRLF to LF and would shift every
            # character offset (that mismatch made this check report a
            # false "not logged").
            new_log = LOG_PATH.read_bytes()[before:].decode("utf-8",
                                                            errors="replace")

        # A second click must work again (the app recovered).
        btn = next(b for b in at.button if b.label == "Estimate price")
        btn.click()
        at.run()
        recovered = any(m.label == "Estimated price" for m in at.metric) \
            and not at.exception

        return {
            "shown": errors[0] if errors else "(no message)",
            "traceback": traceback_shown,
            "logged": ("OutOfMemoryError" in new_log
                       and "Estimate price" in new_log),
            "recovered": recovered,
        }


# ---------------------------------------------------------------------------
# Context caching + add/reset
# ---------------------------------------------------------------------------

def measure_context_cache() -> dict:
    """Time the first prediction against later ones (context encoded once)."""
    train_df, test_df = data_prep.load_and_prepare()
    model = kumo.load_model(size=kumo.DEFAULT_MODEL_SIZE)
    context = kumo.build_context(kumo.limit_context(train_df))

    t0 = time.perf_counter()
    context = kumo.fit_context(model, context)      # encode ONCE
    fit_seconds = time.perf_counter() - t0

    sample = test_df.head(1)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        kumo.predict(model, context, sample)
        times.append(time.perf_counter() - t0)

    # Reference: a context that is NOT fitted re-encodes on every call.
    raw = kumo.build_context(kumo.limit_context(train_df))
    t0 = time.perf_counter()
    kumo.predict(model, raw, sample)
    unfitted = time.perf_counter() - t0

    del model, context, raw
    gc.collect()
    if kumo.cuda_available():
        kumo.torch.cuda.empty_cache()
    return {"fit_seconds": fit_seconds, "fitted_times": times,
            "unfitted_seconds": unfitted}


def check_add_and_reset() -> dict:
    """Click "Add to context" and "Reset context" in the real app."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=TIMEOUT)
    at.run()
    before = _sidebar_context_size(at)

    at.selectbox(key="sale_brand").set_value("maruti")
    at.slider(key="sale_year").set_value(2018)
    at.slider(key="sale_km_driven").set_value(42_000)
    next(b for b in at.button if b.label == "Add to context").click()
    at.run()                                   # the button executes here
    add_msgs = [s.value for s in at.success]
    add_errors = [e.value for e in at.error] + \
                 [str(e.value) for e in at.exception]
    at.run()                                   # re-render without the click:
                                               # the sidebar shows the new size
    after_add = _sidebar_context_size(at)

    next(b for b in at.button if b.label == "Reset context").click()
    at.run()                                   # the button executes here
    reset_msgs = [s.value for s in at.success]
    reset_errors = [e.value for e in at.error] + \
                   [str(e.value) for e in at.exception]
    at.run()                                   # re-render: sidebar updated
    after_reset = _sidebar_context_size(at)

    return {"before": before, "after_add": after_add,
            "after_reset": after_reset,
            "add_message": add_msgs[0] if add_msgs else "-",
            "add_errors": "; ".join(add_errors) or "none",
            "reset_message": reset_msgs[0] if reset_msgs else "-",
            "reset_errors": "; ".join(reset_errors) or "none"}


def _sidebar_context_size(at) -> str:
    for md in at.sidebar.markdown:
        if "Context size" in str(md.value):
            return re.sub(r"[*_]", "", str(md.value)).split(":")[-1].strip()
    return "?"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(widget_rows, widget_problems, model_rows, no_cuda, oom,
                 cache, add_reset) -> Path:
    lines: list[str] = []
    a = lines.append

    a("# CarQuantile - manual UI test results")
    a("")
    a(f"Generated by `scripts/run_ui_tests.py` on "
      f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M')} "
      f"(Streamlit AppTest runs `app.py` itself).")
    a("")
    info = kumo.device_info()
    a(f"GPU: {info.get('name')} | CUDA available: {info.get('cuda')} | "
      f"model: {kumo.DEFAULT_MODEL_SIZE} | context limit: "
      f"{kumo.DEFAULT_CONTEXT_ROWS or 'all rows'} "
      f"({data_prep.load_and_prepare()[0].shape[0]:,} context rows in the "
      f"dataset).")
    a("")

    a("## A. Widget cases (dropdowns, sliders, 'Estimate price')")
    a("")
    a("| # | Case | Input | Estimated price | 80% range | Message shown | Time |")
    a("|---|------|-------|-----------------|-----------|---------------|------|")
    for r in widget_rows:
        a(f"| {r['n']} | {r['title']} | {r['input']} | {r['estimate']} | "
          f"{r['range']} | {r['messages']} | {r['time']} |")
    a("")
    if widget_problems:
        a("**Problems found:**")
        for p in widget_problems:
            a(f"- {p}")
        a("")
    else:
        a("No exceptions and no error banners in any of these cases; every "
          "estimate came back as `low <= price <= high`.")
        a("")

    a("## B. Cases the dropdowns cannot express (same code path as the button)")
    a("")
    a("| Case | Input | Result | Message shown to the user |")
    a("|------|-------|--------|---------------------------|")
    for r in model_rows:
        a(f"| {r['case']} | {r['input']} | {r['result']} | {r['message']} |")
    a("")

    a("## C. Failure paths")
    a("")
    a("### C1. No CUDA device")
    a("")
    a(f"- Message shown: **{no_cuda['shown']}**")
    a(f"- Raw traceback on screen: **{'YES - BUG' if no_cuda['traceback'] else 'no'}**")
    a(f"- Result: {'PASS' if no_cuda['ok'] else 'FAIL'}")
    a("")
    a("### C2. CUDA out of memory (simulated on the first predict)")
    a("")
    a(f"- Message shown: **{oom['shown']}**")
    a(f"- Raw traceback on screen: **{'YES - BUG' if oom['traceback'] else 'no'}**")
    a(f"- Details written to `logs/app.log`: **{'yes' if oom['logged'] else 'no'}**")
    a(f"- A second click returned a normal result: "
      f"**{'yes' if oom['recovered'] else 'no'}**")
    a("")

    a("## D. Context caching (encode once, reuse on every click)")
    a("")
    a(f"- Encoding the context once (`fit_context`): "
      f"**{cache['fit_seconds']:.2f} s** (one-off)")
    a("- Predictions with the cached context (seconds): "
      + ", ".join(f"{t:.3f}" for t in cache["fitted_times"]))
    a(f"- First of those (includes CUDA warm-up): "
      f"**{cache['fitted_times'][0]:.3f} s**")
    a(f"- Later ones: **{min(cache['fitted_times'][1:]):.3f} - "
      f"{max(cache['fitted_times'][1:]):.3f} s**")
    a(f"- Same prediction with a context that was NOT encoded: "
      f"**{cache['unfitted_seconds']:.3f} s** (re-encodes every time)")
    a("")

    a("## E. 'Add a new sale' and 'Reset context'")
    a("")
    a(f"- Context size in the sidebar before: **{add_reset['before']}**")
    a(f"- After **Add to context**: **{add_reset['after_add']}** "
      f"(+1 expected) - {add_reset['add_message']}")
    a(f"- Errors/tracebacks during add: {add_reset['add_errors']}")
    a(f"- After **Reset context**: **{add_reset['after_reset']}** "
      f"(back to the original) - {add_reset['reset_message']}")
    a(f"- Errors/tracebacks during reset: {add_reset['reset_errors']}")
    a("")

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    return OUT_PATH


def main() -> int:
    import os
    os.chdir(ROOT)          # data/cars.csv and logs/ are relative paths

    print("A. running widget cases ...")
    widget_rows, widget_problems = run_widget_cases()
    print(f"   {len(widget_rows)} cases done")

    print("B. running model-layer cases ...")
    model_rows = run_model_layer_cases()

    print("C1. checking the 'no CUDA' message ...")
    no_cuda = check_no_cuda()

    print("C2. checking the CUDA out-of-memory message ...")
    oom = check_cuda_oom()

    print("D. timing first vs later predictions ...")
    cache = measure_context_cache()

    print("E. clicking Add to context / Reset context ...")
    add_reset = check_add_and_reset()

    path = write_report(widget_rows, widget_problems, model_rows, no_cuda,
                        oom, cache, add_reset)
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

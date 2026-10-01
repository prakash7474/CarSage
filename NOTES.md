# NOTES — NVIDIA Kumo Tabular: verified API & install facts

Everything below was read directly from the official sources on **2026-10-01**.
Nothing here is guessed. Sources:

- Repo README: https://github.com/NVIDIA/structured-data-models
- Quickstart example: `examples/tabular/quickstart.py`
- Model source: `sdm/models/kumo/tabular/model.py`, `sdm/models/base.py`
- Blog / announcement: https://huggingface.co/blog/nvidia/kumo-tabular
- Weights: https://huggingface.co/nvidia/Kumo-Tabular (revision `v1.0.0`)

The task brief's demo snippet is **almost** the real API but not exact:
`drop_column("target")` is wrong — the real method is `drop_columns("target")`,
and the target is normally supplied as `y_context`, not left inside the context
table. The real calls are in "Verified API" below.

---

## 1. How to choose the model size (Small / Medium / Large)

`KumoTabular` takes a `size` argument. From `sdm/models/kumo/tabular/model.py`:

```python
model = sdm.models.KumoTabular(task=sdm.Task.regression, size="small", device="cuda")
```

- Allowed values: `"small"`, `"medium"`, `"large"`.
- **The library default is `"large"`** — so we must pass `size="small"` explicitly
  on a 6 GB GPU.
- Approx. parameter counts (from the blog): Small ~28M, Medium ~? , Large ~215M.
- Checkpoints downloaded on first use (`huggingface_hub`):
  `nvidia/Kumo-Tabular` @ revision `v1.0.0`
  - regression: `{size}/regressor.pt`
  - classification: `{size}/classifier.pt`
- Model kwargs per size (cell channels / ICL layers / heads):
  - small: cell_channels=128, num_icl_layers=12, num_icl_heads=8
  - medium: cell_channels=256, num_icl_layers=24, num_icl_heads=8
  - large: cell_channels=256, num_icl_layers=24, num_icl_heads=16, icl_channels=1024

**Pass `task=sdm.Task.regression`** to load *only* the regressor checkpoint.
If `task` is omitted, the model initialises both tasks and downloads the
classifier too.

## 2. How to get the quantile outputs (999 quantiles)

From the `KumoTabular` docstring:

> "For regression tasks, KumoTabular predicts 999 quantiles named `q001`
> through `q999`."

The forward pass returns a `sdm.TableTensor` whose numerical columns are exactly
`q001 ... q999` (the default recipe applies `SortQuantiles()` at the output, so
the columns are in ascending quantile order). The raw tensor is available as
`.numerical`:

```python
out = model(x_context=..., y_context=..., x_query=...)   # TableTensor
q = out.numerical                                        # torch.Tensor [R_query, 999]
q = q.detach().cpu().numpy()
```

Column `q{k}` is the quantile at level `k/1000`:

- point estimate  -> median -> `q500` -> index `499`
- 10th percentile -> `q100` -> index `99`
- 90th percentile -> `q900` -> index `899`

The library also applies `AverageEstimators(trim_fraction=0.2)` when several
estimators are used, so the quantiles already come from the model's own
distribution — no extra calibration is needed.

## 3. Verified API (regression example)

```python
import sdm

# 1. Tensorize a pandas DataFrame (lossless tensor view, kept on the GPU).
table = sdm.TableTensor.from_pandas(
    df=df,
    stypes=sdm.infer_stypes(df, overrides={"selling_price": "numerical"}),
    device="cuda",
)

# 2. Split into context (labeled) and query (unlabeled) parts.
x_context = table.drop_columns("selling_price")
y_context = table[:, "selling_price"]     # note: tuple indexing (rows, column)
x_query   = <TableTensor built the same way for the rows to predict>

# 3. Load the regressor and run one forward pass (no training).
model = sdm.models.KumoTabular(task=sdm.Task.regression, size="small", device="cuda")
pred  = model(x_context=x_context, y_context=y_context, x_query=x_query)
q     = pred.numerical                    # [R_query, 999]
```

Key facts confirmed in source:

- `TableTensor.from_pandas(df=..., stypes=..., device=...)`.
- Row slicing `table[:300]`, `table[300:]`; column select `table[:, "col"]`;
  drop with `.drop_columns("col")` (note the plural).
- `sdm.infer_stypes(df, overrides=...)` maps ints/floats -> `numerical`,
  strings/bools/categories -> `categorical`, datetimes -> `datetime`.
  Kumo Tabular supports **numerical and categorical feature columns only**
  (`supported_feature_stypes = {numerical}` for features; text/datetime/id are
  not consumed as features). So no timestamps/text in our feature set.
- `num_estimators=None` (the default) means **exactly 1 forward pass / 1 member**
  (verified in `sdm/processing/execution.py::_to_ensemble_table`). More
  estimators = more accuracy but more VRAM.
- KV caching: call `model.fit(x=..., y=...)` once, then `model.predict(x_query)`
  repeatedly to encode the context once and reuse it. (This is the Phase 3
  "encode context once" optimisation.) `model.clear()` resets it.
- Autocast: the official quickstart wraps calls in
  `torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda)`.

## 4. Install on Windows

Package facts from `pyproject.toml`:

- `requires-python = ">=3.11"` (classifiers list 3.11–3.14).
- Hard deps: `torch>=2.7`, `numpy`, `pyarrow`, `huggingface_hub`, `safetensors`,
  `typing_extensions`.
- `cudf` is *recommended but optional* (Linux only in the test group).
- **Triton is optional.** `sdm/_kernels/rmsnorm_cast.py` and
  `segment_multi_reduce.py` try to import a Triton kernel and silently fall back
  to a pure-PyTorch eager path on `ImportError`. So **no Triton / no
  triton-windows is required** — the library works on Windows with plain PyTorch.

### Recommended environment on this laptop

- This machine has Python **3.13** and **3.14** (no 3.11/3.12). Use **3.13**:
  the PyTorch CUDA wheel index has far better coverage for cp313 than cp314,
  and every dependency (streamlit, xgboost, scikit-learn) has 3.13 wheels.
- GPU: **RTX 3050 6 GB Laptop** (Ampere, sm_86), driver 592.82 / CUDA 13.1.
  Use the **cu128** PyTorch wheels (backward compatible with the driver).

### Exact commands (run from the project root)

```bash
# 1. Create/activate a Python 3.13 virtual environment
py -3.13 -m venv .venv
.venv/Scripts/activate            # Git Bash: source .venv/Scripts/activate
python -m pip install --upgrade pip

# 2. PyTorch with CUDA (Windows wheels come from PyTorch's own index)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 3. The NVIDIA structured-data-models library (provides `import sdm`)
pip install git+https://github.com/NVIDIA/structured-data-models.git

# 4. Everything else
pip install -r requirements.txt

# 5. Verify the GPU is visible to PyTorch
python -m src.gpu_check
```

> On Windows the PyPI `torch` wheel is CPU-only, which is why step 2 must use
> the PyTorch CUDA index. Weights download automatically to the Hugging Face
> cache on the first `KumoTabular(...)` call (needs internet).

### WSL2 (only if you ever hit a Linux-only dependency)

Not required here — Windows + plain PyTorch works because Triton is optional.
If you prefer WSL2: install the NVIDIA Windows driver (already present), run
`wsl --install -d Ubuntu`, then inside WSL follow the same steps with
`pip install torch --index-url https://download.pytorch.org/whl/cu128`. The
Windows driver exposes the GPU to WSL2 via `/dev/dxg`; do **not** install a
separate driver inside WSL.

## 5. Dataset

CarDekho used-car dataset (Kaggle), user `nehalbirla`:
https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

Use **`Car details v3.csv`** — its columns match the brief exactly:
`name, year, selling_price, km_driven, fuel, seller_type, transmission, owner,
mileage, engine, max_power, torque, seats`.

Download it (Kaggle login required) and save it as **`data/cars.csv`**.

## 6. Measured on this machine (Phase 1 smoke test, 2026-10-01)

Environment: Windows 11, Python 3.13.14, torch 2.11.0+cu128,
sdm 0.1.0rc2.dev186, RTX 3050 6 GB Laptop (5.0 GiB free).

- Weight download + model load (small): **~65 s once** (cached in
  `~/.cache/huggingface` afterwards).
- Context build, 200 rows: **0.1 s**.
- First prediction (CUDA warm-up), 20 rows: **7.8 s**.
- Warm predictions, 2,000-row context: **~0.25 s** for a batch of 1–200 rows
  (one forward pass). Fine for an interactive Streamlit app.
- VRAM used by Small + 200-row context + 20-row query: **~1.1 GiB**
  (4.9 GiB still free).
- Accuracy on the synthetic table: median MAE ≈ 13k on 200k–300k prices,
  10th–90th coverage 90% — the quantile output works as documented.

### Windows install pitfalls found while setting up (all solved)

This machine is managed by a Windows **Application Control policy**
(WDAC / Smart App Control in evaluation mode). It blocks native DLLs from
*brand-new* package wheels until they build reputation:

1. `pyarrow` 25.x wheel → `ImportError: DLL load failed ... Application
   Control policy has blocked this file`. **Fix: pin `pyarrow==24.0.0`**.
2. `scikit-learn` 1.9.x wheels (`_target_encoder_fast` DLL) → same block.
   **Fix: pin `scikit-learn==1.7.2`**.
3. Import order matters on this setup: if pandas 3.x (which loads pyarrow
   eagerly) is imported **before** torch, `torch/lib/c10.dll` fails with
   WinError 1114. `import torch` first and everything coexists — `src/model.py`
   already does this. With the pinned versions above the conflict is gone.

None of this requires WSL2; the library runs fine on native Windows because
its optional Triton kernels fall back to pure PyTorch.

## 7. Open questions / unknowns (to verify on the real dataset)

- Exact VRAM of Kumo Tabular-Small at ~5,000 context rows on this 6 GB GPU —
  measured in the Phase 3 benchmark (`python -m src.evaluate --context 2000 5000 all`).
- Whether modelling `log(selling_price)` improves MAE/RMSE — decided from
  `results/comparison.csv` in Phase 3 (both paths are implemented).

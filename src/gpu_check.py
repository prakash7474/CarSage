"""GPU check for CarQuantile / NVIDIA Kumo Tabular.

Run me first:   python -m src.gpu_check

I answer three questions:
  1. Can PyTorch see CUDA at all?
  2. Which GPU is it actually using (we want the NVIDIA card, NOT the AMD
     integrated one)?
  3. How much VRAM is free, so we know whether the Small model fits?

Kumo Tabular does not strictly *require* a GPU, but it is designed for one and
is very slow on CPU. On this laptop we target the RTX 3050 6 GB.
"""

from __future__ import annotations

import sys


def _print_header() -> None:
    print("=" * 62)
    print("CarQuantile GPU check")
    print("=" * 62)
    print(f"Python : {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")


def main() -> int:
    _print_header()

    # Import torch lazily so the error message is friendly if it is missing.
    try:
        import torch
    except ImportError:
        print("\n[FAIL] PyTorch is not installed.")
        print("Install it with:")
        print("  pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128")
        return 1

    print(f"PyTorch: {torch.__version__}")

    available = torch.cuda.is_available()
    print(f"\ntorch.cuda.is_available() : {available}")

    if not available:
        print("\n[FAIL] PyTorch cannot see a CUDA GPU.")
        print("Most likely you installed the CPU-only torch wheel from PyPI.")
        print("Re-install with the CUDA index:")
        print("  pip install torch --index-url "
              "https://download.pytorch.org/whl/cu128")
        return 1

    # CUDA build / driver versions.
    print(f"torch.version.cuda       : {torch.version.cuda}")
    print(f"GPU count                : {torch.cuda.device_count()}")

    device_index = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device_index)

    # Total and currently-free VRAM (bytes -> GiB).
    props = torch.cuda.get_device_properties(device_index)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    gib = 1024 ** 3

    print(f"Device index             : {device_index}")
    print(f"Device name              : {name}")
    print(f"Compute capability       : {props.major}.{props.minor}")
    print(f"Total VRAM               : {total_bytes / gib:.2f} GiB")
    print(f"Free VRAM                : {free_bytes / gib:.2f} GiB")

    # Guard against silently using the AMD integrated GPU: a CUDA device is
    # always NVIDIA, so if we got here we are on the NVIDIA card. Tell the
    # user anyway.
    if "NVIDIA" in name.upper():
        print("\n[OK] Using the NVIDIA GPU (good).")
    else:
        print(f"\n[WARN] CUDA device is {name!r} — expected an NVIDIA GPU.")

    if props.major < 7:
        print("[WARN] Compute capability < 7.0; fp16 may be slow/unsupported.")
    else:
        print("[OK] Compute capability supports fp16 autocast "
              "(used by the Kumo quickstart).")

    if free_bytes / gib < 2.0:
        print("[WARN] Less than ~2 GiB VRAM free — close other GPU apps. "
              "Use the Small model and a small context.")
    else:
        print("[OK] Enough free VRAM for the Small model and a few "
              "thousand context rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

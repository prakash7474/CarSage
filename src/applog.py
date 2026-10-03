"""File logging for the Streamlit app.

The UI must never show a raw traceback: it shows one plain sentence instead,
while the full traceback is appended to ``logs/app.log`` so the problem can be
investigated afterwards.

    from src import applog
    try:
        ...
    except Exception as exc:
        applog.log_exception("Estimate price", exc)
        st.error(applog.user_message("Prediction failed", exc))
"""

from __future__ import annotations

import logging
from pathlib import Path

#: Where the technical details go (created on first use).
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log"

_LOGGER_NAME = "carquantile"


def get_logger() -> logging.Logger:
    """Return the app logger, adding the file handler the first time."""
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_exception(where: str, exc: BaseException) -> None:
    """Append the full traceback to logs/app.log (never to the screen)."""
    get_logger().error("Error in %s: %s", where, exc, exc_info=True)


def user_message(short: str, exc: BaseException | None = None) -> str:
    """A short, human sentence plus a pointer to the log file.

    ``exc`` is never shown as a traceback - only its type name is used, and
    only to add the out-of-memory advice when that is what happened.
    """
    text = short
    if exc is not None:
        kind = type(exc).__name__
        detail = str(exc).lower()
        if isinstance(exc, ValueError):
            # Our own validation messages are written for humans - show them.
            text = str(exc)
        elif "memory" in kind.lower() or "memory" in detail:
            text = (
                f"{short}\n\nThe GPU ran out of memory: the cache was freed "
                "and a smaller setting was tried. Close other GPU apps "
                "(browser tabs, games) and reload the page if it repeats."
            )
        else:
            text = f"{short}\n\n(Problem: {kind}.)"
    return f"{text}\n\nTechnical details were saved to `{LOG_FILE}`."

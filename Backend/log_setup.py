"""Shared logging configuration for all MAYA backend modules."""
import logging
import os
import sys
from pathlib import Path

_LOG_FILE = Path(__file__).parent / "maya.log"
_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    level_name = os.environ.get("MAYA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)

    try:
        file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(fmt)
        handlers: list[logging.Handler] = [stream_handler, file_handler]
    except OSError:
        handlers = [stream_handler]

    logging.basicConfig(level=level, handlers=handlers, force=True)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)

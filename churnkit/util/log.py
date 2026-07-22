"""Console logging. Stage banners keep a long pipeline run readable."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


class _Formatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[90m",
        "INFO": "\033[0m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[1;31m",
    }
    RESET = "\033[0m"

    def __init__(self, color: bool):
        super().__init__("%(message)s")
        self.color = color

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        prefix = {"WARNING": "WARN  ", "ERROR": "ERROR ", "CRITICAL": "FATAL "}.get(
            record.levelname, "      "
        )
        line = f"{prefix}{msg}"
        if self.color:
            return f"{self.COLORS.get(record.levelname, '')}{line}{self.RESET}"
        return line


def setup(verbose: bool = False) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_Formatter(color=sys.stderr.isatty()))
    root = logging.getLogger("churnkit")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup()
    return logging.getLogger(f"churnkit.{name}")


def stage(title: str) -> None:
    log = get_logger("stage")
    log.info("")
    log.info("\033[1m%s\033[0m" if sys.stderr.isatty() else "%s", f"── {title} " + "─" * max(0, 60 - len(title)))

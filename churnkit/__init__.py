"""ChurnKit — a config-driven churn analysis pipeline.

Edit the YAML in config/, run `python run.py`, read output/report.html.

Modules are named for the order they run in, so the directory listing is the
pipeline:

    l01_config  l02_data   l03_panel   l04_features
    l05_leakage l06_splits l07_model   l08_drivers
    l09_causal  l10_experiment         l11_survival  l12_report

`pipeline.py` calls them in that order; `util/` holds the shared support code
(errors, logging, charts, messy-export parsers), none of which is a stage.
"""

from .pipeline import run
from .util.errors import (
    ChurnKitError,
    ConfigError,
    DataError,
    InsufficientDataError,
    LeakageError,
)

__all__ = [
    "run",
    "ChurnKitError",
    "ConfigError",
    "DataError",
    "InsufficientDataError",
    "LeakageError",
]
__version__ = "1.0.0"

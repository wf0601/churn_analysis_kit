"""Fixtures that build a minimal but real project on disk.

Tests go through the actual config loader and the actual pipeline stages rather than
constructing objects by hand — a leakage guard that only works when called directly
is not a guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


BASE_PIPELINE = {
    "run": {"output_dir": "output", "random_seed": 0},
    "panel": {
        "snapshot_mode": "rolling",
        "snapshot_frequency": "MS",
        "horizon_days": 30,
        "embargo_days": 7,
    },
    "split": {"strategy": "out_of_time", "test_fraction": 0.3, "cv_folds": 2},
    "leakage": {"on_block": "quarantine"},
    "survival": {"enabled": False},
    "causal": {"enabled": False},
}


def write_project(
    root: Path,
    customers: pd.DataFrame,
    events: dict[str, pd.DataFrame] | None = None,
    features: dict | None = None,
    pipeline: dict | None = None,
    target: dict | None = None,
    survivorship: dict | None = None,
) -> Path:
    """Materialise data + config under `root` and return the config directory."""
    data_dir = root / "data"
    config_dir = root / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    customers.to_csv(data_dir / "customers.csv", index=False)
    event_config = {}
    for name, frame in (events or {}).items():
        frame.to_csv(data_dir / f"{name}.csv", index=False)
        event_config[name] = {
            "path": f"data/{name}.csv",
            "id_column": "customer_id",
            "date_column": "event_date",
        }

    data_config = {
        "entity": {
            "path": "data/customers.csv",
            "id_column": "customer_id",
            "start_date_column": "signup_date",
        },
        "segments": [],
    }
    if event_config:
        data_config["events"] = event_config

    def dump(name: str, payload: dict) -> None:
        (config_dir / name).write_text(yaml.safe_dump(payload, sort_keys=False))

    dump("data.yaml", data_config)
    dump("target.yaml", {"target": target or {
        "mode": "event_date",
        "event_date_column": "contract_end_date",
        "observation_end_date": "2023-12-31",
    }})
    dump("survivorship.yaml", survivorship or {
        "data_export_date": "2023-12-31",
        "observation_starting_date": "2023-01-01",
        "left_truncation": "drop",
    })
    dump("feature.yaml", features or {
        "profile": {"temporal": "static",
                    "columns": [{"name": "region", "type": "categorical"}]},
    })
    merged = _deep_merge(BASE_PIPELINE, pipeline or {})
    dump("pipeline.yaml", merged)
    dump("causal.yaml", {"treatments": []})
    dump("experiment.yaml", {"enabled": False})
    return config_dir


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        out[key] = _deep_merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


@pytest.fixture
def customers() -> pd.DataFrame:
    """60 customers signing up in early 2023; half churn, half are censored.

    Churn offsets are spread widely on purpose so that an out-of-time split has
    events on both sides — a fixture where every churn lands in one quarter would
    make the split tests fail for reasons that have nothing to do with splitting.
    """
    rows = []
    for i in range(60):
        signup = pd.Timestamp("2023-01-03") + pd.Timedelta(days=2 * (i % 15))
        churn = (
            signup + pd.Timedelta(days=40 + 11 * (i % 26)) if i % 2 == 0 else pd.NaT
        )
        rows.append(
            {
                "customer_id": f"C{i:03d}",
                "signup_date": signup.date(),
                "contract_end_date": churn.date() if churn is not pd.NaT else None,
                "region": ["north", "south", "east"][i % 3],
                "plan_tier": ["basic", "premium"][i % 2],
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def project(tmp_path, customers):
    return lambda **kwargs: write_project(tmp_path, customers, **kwargs)

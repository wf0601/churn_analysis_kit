"""Config loading, defaulting and validation.

The user only edits YAML, so this module carries the burden of turning loose YAML
into a checked, fully-defaulted object — and of failing with a message that says
which file and which key to fix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .util.errors import ConfigError
from .util.log import get_logger

log = get_logger("config")

CONFIG_FILES = {
    "data": "data.yaml",
    "target": "target.yaml",
    "survivorship": "survivorship.yaml",
    "features": "feature.yaml",
    "experiment": "experiment.yaml",
    "causal": "causal.yaml",
    "pipeline": "pipeline.yaml",
}

DEFAULTS: dict[str, Any] = {
    "run": {"output_dir": "output", "random_seed": 42},
    "panel": {
        "snapshot_mode": "rolling",
        "snapshot_frequency": "MS",
        "snapshot_start": None,
        "snapshot_end": None,
        "snapshot_dates": [],
        "horizon_days": 90,
        "embargo_days": 7,
        "min_tenure_days": 0,
        "max_snapshots_per_entity": None,
    },
    "split": {
        "strategy": "out_of_time",
        "test_fraction": 0.25,
        "cv_folds": 4,
        "purge_days": 0,
    },
    "model": {
        "primary": "hist_gradient_boosting",
        "calibrate": True,
        "class_weight": "balanced",
        "max_iter": 300,
        "learning_rate": 0.06,
        "max_leaf_nodes": 31,
    },
    "drivers": {
        "top_k": 20,
        "method": "auto",
        "n_permutation_repeats": 10,
        "min_stability": 0.5,
    },
    "leakage": {
        "on_block": "quarantine",
        "denylist_patterns": [
            "churn", "cancel", "terminat", r"(^|_)end_date$", "expir",
            r"(^|_)exit", "reason", "refund", "final_", "closed",
            "win_?back", "retention_offer", "save_?desk", "survival",
            "is_active", "status",
        ],
        "allowlist_columns": [],
        "single_feature_auc_block": 0.90,
        "single_feature_auc_warn": 0.80,
        "missingness_auc_warn": 0.75,
        "model_auc_block": 0.995,
        "id_cardinality_ratio_warn": 0.9,
    },
    "survival": {"enabled": True, "cox": True, "by": None, "max_strata": 8},
    "causal": {"enabled": True},
}


# --------------------------------------------------------------------------- #
# structured views
# --------------------------------------------------------------------------- #
# Aggregations a feature group may ask for. Anything outside this set is rejected
# at config load, so a typo fails immediately instead of silently dropping a feature.
ALLOWED_AGGS = {
    "sum", "mean", "max", "min", "last", "first", "count", "median", "std", "nunique",
}
DEFAULT_AGGS = ["sum", "mean", "max", "last"]


@dataclass
class ColumnSpec:
    name: str
    type: str = "numeric"              # numeric | categorical | boolean | datetime
    aggs: list[str] | None = None      # overrides the group's aggs
    windows: list[int] | None = None   # overrides the group's windows


@dataclass
class FilterSpec:
    """A named subset of an event log, aggregated into its own feature family.

    This is how `failed_payments_12m` gets built from a raw billing log: filter to
    the failed rows, then count them over a window. The filter is a pandas query
    string evaluated against the event table.
    """
    name: str
    where: str
    description: str = ""


@dataclass
class DerivedSpec:
    """A feature computed from other features, after aggregation."""
    name: str
    expression: str
    description: str = ""


@dataclass
class FeatureGroup:
    name: str
    temporal: str                      # static | time_varying | derived
    source: str                        # "entity" or an events key
    columns: list[ColumnSpec] = field(default_factory=list)
    windows: list[int] = field(default_factory=list)
    aggs: list[str] = field(default_factory=lambda: list(DEFAULT_AGGS))
    filters: list[FilterSpec] = field(default_factory=list)
    derived: list[DerivedSpec] = field(default_factory=list)
    generate_trends: bool = False
    include_unfiltered: bool = True    # filters add families, they don't replace
    leakage_review: str = "normal"     # normal | strict

    @property
    def is_time_varying(self) -> bool:
        return self.temporal == "time_varying"

    @property
    def is_derived(self) -> bool:
        return self.temporal == "derived"

    @property
    def aggregation_window_days(self) -> int | None:
        """Longest window, for the lead-time arithmetic in panel construction."""
        return max(self.windows) if self.windows else None


@dataclass
class EventSource:
    name: str
    path: Path
    id_column: str
    date_column: str
    format: str = "auto"


@dataclass
class Config:
    root: Path
    raw: dict[str, Any]

    # data
    entity_path: Path = Path()
    entity_format: str = "auto"
    id_column: str = "customer_id"
    start_date_column: str = "signup_date"
    valid_from_column: str | None = None
    valid_to_column: str | None = None
    events: dict[str, EventSource] = field(default_factory=dict)
    segments: list[str] = field(default_factory=list)

    # target
    target_mode: str = "event_date"
    event_date_column: str | None = None
    label_column: str | None = None
    churn_value: Any = 1
    observation_end_date: pd.Timestamp | None = None
    decision_lead_days: int = 0

    # survivorship
    data_export_date: pd.Timestamp | None = None
    observation_starting_date: pd.Timestamp | None = None
    event_history_starts: pd.Timestamp | None = None
    left_truncation: str = "drop"

    # features
    groups: list[FeatureGroup] = field(default_factory=list)

    # blocks straight from pipeline.yaml
    run: dict = field(default_factory=dict)
    panel: dict = field(default_factory=dict)
    split: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    drivers: dict = field(default_factory=dict)
    leakage: dict = field(default_factory=dict)
    survival: dict = field(default_factory=dict)
    causal_run: dict = field(default_factory=dict)

    # causal / experiment
    causal: dict = field(default_factory=dict)
    experiment: dict = field(default_factory=dict)

    @property
    def output_dir(self) -> Path:
        p = self.root / self.run["output_dir"]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def seed(self) -> int:
        return int(self.run["random_seed"])

    @property
    def max_window_days(self) -> int:
        windows = [
            g.aggregation_window_days or 0 for g in self.groups if g.is_time_varying
        ]
        return max(windows) if windows else 0

    def group(self, name: str) -> FeatureGroup | None:
        return next((g for g in self.groups if g.name == name), None)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if v is None and k in out and isinstance(out[k], (dict, list)):
            continue  # blank YAML value must not wipe a structured default
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_date(value: Any, where: str) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        return pd.Timestamp(value).normalize()
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"{where}: cannot parse date {value!r} ({exc})") from exc


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{path.name}: expected a mapping at the top level")
    return loaded


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load(config_dir: str | Path = "config", root: str | Path | None = None) -> Config:
    config_dir = Path(config_dir)
    root_path = Path(root) if root else config_dir.parent
    raw = {key: _read_yaml(config_dir / fname) for key, fname in CONFIG_FILES.items()}

    pipeline = _deep_merge(DEFAULTS, raw["pipeline"])
    cfg = Config(root=root_path.resolve(), raw=raw)

    cfg.run = pipeline["run"]
    cfg.panel = pipeline["panel"]
    cfg.split = pipeline["split"]
    cfg.model = pipeline["model"]
    cfg.drivers = pipeline["drivers"]
    cfg.leakage = pipeline["leakage"]
    cfg.survival = pipeline["survival"]
    cfg.causal_run = pipeline["causal"]

    _load_data_block(cfg, raw["data"], root_path)
    _load_target_block(cfg, raw["target"])
    _load_survivorship_block(cfg, raw["survivorship"])
    _load_feature_block(cfg, raw["features"])

    cfg.causal = raw["causal"] or {}
    cfg.experiment = raw["experiment"] or {}

    _validate(cfg)
    return cfg


def _load_data_block(cfg: Config, data: dict, root: Path) -> None:
    if not data:
        raise ConfigError("data.yaml is empty — it must at least define `entity.path`")
    entity = data.get("entity") or {}
    if not entity.get("path"):
        raise ConfigError("data.yaml: `entity.path` is required")

    cfg.entity_path = (root / entity["path"]).resolve()
    cfg.entity_format = entity.get("format", "auto")
    cfg.id_column = entity.get("id_column", "customer_id")
    cfg.start_date_column = entity.get("start_date_column", "signup_date")
    cfg.valid_from_column = entity.get("valid_from_column")
    cfg.valid_to_column = entity.get("valid_to_column")

    for name, spec in (data.get("events") or {}).items():
        if not spec or not spec.get("path"):
            raise ConfigError(f"data.yaml: events.{name} needs a `path`")
        cfg.events[name] = EventSource(
            name=name,
            path=(root / spec["path"]).resolve(),
            id_column=spec.get("id_column", cfg.id_column),
            date_column=spec.get("date_column", "event_date"),
            format=spec.get("format", "auto"),
        )
    cfg.segments = list(data.get("segments") or [])


def _load_target_block(cfg: Config, target_raw: dict) -> None:
    target = (target_raw or {}).get("target") or {}
    cfg.target_mode = target.get("mode", "event_date")
    cfg.event_date_column = target.get("event_date_column")
    cfg.label_column = target.get("label_column")
    cfg.churn_value = target.get("churn_value", 1)
    cfg.observation_end_date = _parse_date(
        target.get("observation_end_date"), "target.yaml: observation_end_date"
    )
    cfg.decision_lead_days = int(target.get("decision_lead_days") or 0)


def _load_survivorship_block(cfg: Config, surv: dict) -> None:
    surv = surv or {}
    cfg.data_export_date = _parse_date(
        surv.get("data_export_date"), "survivorship.yaml: data_export_date"
    )
    cfg.observation_starting_date = _parse_date(
        surv.get("observation_starting_date"),
        "survivorship.yaml: observation_starting_date",
    )
    cfg.event_history_starts = _parse_date(
        surv.get("event_history_starts"),
        "survivorship.yaml: event_history_starts",
    )
    cfg.left_truncation = surv.get("left_truncation", "drop")


def _load_feature_block(cfg: Config, features: dict) -> None:
    if not features:
        raise ConfigError("feature.yaml is empty — define at least one feature group")

    for name, spec in features.items():
        if not isinstance(spec, dict):
            raise ConfigError(f"feature.yaml: group {name!r} must be a mapping")
        temporal = spec.get("temporal", "static")
        if temporal not in {"static", "time_varying", "derived"}:
            raise ConfigError(
                f"feature.yaml: {name}.temporal must be 'static', 'time_varying' or "
                f"'derived', got {temporal!r}"
            )

        if temporal == "derived":
            cfg.groups.append(_derived_group(name, spec))
            continue

        group_aggs = _validate_aggs(spec.get("aggs"), f"{name}.aggs") or list(DEFAULT_AGGS)
        group_windows = _validate_windows(
            spec.get("windows"), spec.get("aggregation_window_days"), f"{name}.windows"
        )

        columns = []
        for col in spec.get("columns") or []:
            if isinstance(col, str):
                columns.append(ColumnSpec(name=col))
            elif isinstance(col, dict) and "name" in col:
                columns.append(
                    ColumnSpec(
                        name=col["name"],
                        type=col.get("type", "numeric"),
                        aggs=_validate_aggs(col.get("aggs"), f"{name}.{col['name']}.aggs"),
                        windows=_validate_windows(
                            col.get("windows"), None, f"{name}.{col['name']}.windows"
                        ) or None,
                    )
                )
            else:
                raise ConfigError(
                    f"feature.yaml: {name}.columns entries must be a name or "
                    f"{{name, type, aggs, windows}}, got {col!r}"
                )

        filters = _parse_filters(name, spec.get("filters"))
        if not columns and not filters:
            log.warning(
                "feature.yaml: group %r has neither columns nor filters; skipping it", name
            )
            continue

        cfg.groups.append(
            FeatureGroup(
                name=name,
                temporal=temporal,
                source=spec.get("source", "entity" if temporal == "static" else name),
                columns=columns,
                windows=group_windows,
                aggs=group_aggs,
                filters=filters,
                generate_trends=bool(spec.get("generate_trends", False)),
                include_unfiltered=bool(spec.get("include_unfiltered", True)),
                leakage_review=spec.get("leakage_review", "normal"),
            )
        )


def _validate_aggs(value, where: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value:
        raise ConfigError(f"feature.yaml: {where} must be a non-empty list")
    unknown = [a for a in value if a not in ALLOWED_AGGS]
    if unknown:
        raise ConfigError(
            f"feature.yaml: {where} contains unsupported aggregation(s) {unknown}. "
            f"Available: {sorted(ALLOWED_AGGS)}"
        )
    return list(dict.fromkeys(value))


def _validate_windows(value, legacy, where: str) -> list[int]:
    if value is None and legacy is not None:
        value = [legacy]                 # `aggregation_window_days: 90` still works
    if value is None:
        return []
    if isinstance(value, (int, float)):
        value = [value]
    if not isinstance(value, list):
        raise ConfigError(f"feature.yaml: {where} must be a number or a list of numbers")
    windows = []
    for item in value:
        try:
            days = int(item)
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"feature.yaml: {where} entry {item!r} is not a number of days"
            ) from exc
        if days <= 0:
            raise ConfigError(f"feature.yaml: {where} entry {days} must be positive")
        windows.append(days)
    return sorted(dict.fromkeys(windows))


def _parse_filters(group: str, raw) -> list[FilterSpec]:
    if not raw:
        return []
    filters = []
    if isinstance(raw, dict):                       # {name: where} shorthand
        raw = [{"name": k, "where": v} for k, v in raw.items()]
    if not isinstance(raw, list):
        raise ConfigError(
            f"feature.yaml: {group}.filters must be a list of {{name, where}} entries"
        )
    for item in raw:
        if not isinstance(item, dict) or "name" not in item or "where" not in item:
            raise ConfigError(
                f"feature.yaml: {group}.filters entries need `name` and `where`, "
                f"got {item!r}"
            )
        name = str(item["name"])
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ConfigError(
                f"feature.yaml: {group}.filters name {name!r} must be alphanumeric or "
                f"underscores — it becomes part of the feature name"
            )
        filters.append(
            FilterSpec(name=name, where=str(item["where"]),
                       description=item.get("description", ""))
        )
    return filters


def _derived_group(name: str, spec: dict) -> FeatureGroup:
    entries = spec.get("features") or spec.get("derived") or []
    if not entries:
        raise ConfigError(
            f"feature.yaml: derived group {name!r} needs a `features` list of "
            f"{{name, expression}} entries"
        )
    derived = []
    for item in entries:
        if not isinstance(item, dict) or "name" not in item or "expression" not in item:
            raise ConfigError(
                f"feature.yaml: {name}.features entries need `name` and `expression`, "
                f"got {item!r}"
            )
        feature_name = str(item["name"])
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", feature_name):
            raise ConfigError(
                f"feature.yaml: derived feature name {feature_name!r} must be a valid "
                f"identifier"
            )
        derived.append(
            DerivedSpec(name=feature_name, expression=str(item["expression"]),
                        description=item.get("description", ""))
        )
    return FeatureGroup(
        name=name, temporal="derived", source="derived", derived=derived,
        leakage_review=spec.get("leakage_review", "normal"),
    )


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _validate(cfg: Config) -> None:
    if not cfg.entity_path.exists():
        raise ConfigError(
            f"data.yaml: entity file not found at {cfg.entity_path}. "
            "Run `python run.py demo` to generate a synthetic dataset to try the kit on."
        )

    if cfg.target_mode == "event_date":
        if not cfg.event_date_column:
            raise ConfigError(
                "target.yaml: mode is 'event_date' so `event_date_column` is required"
            )
    elif cfg.target_mode == "label":
        if not cfg.label_column:
            raise ConfigError(
                "target.yaml: mode is 'label' so `label_column` is required"
            )
    else:
        raise ConfigError(
            f"target.yaml: mode must be 'event_date' or 'label', got {cfg.target_mode!r}"
        )

    if cfg.left_truncation not in {"drop", "keep_flagged"}:
        raise ConfigError(
            "survivorship.yaml: left_truncation must be 'drop' or 'keep_flagged'"
        )

    p = cfg.panel
    if p["snapshot_mode"] not in {"rolling", "single"}:
        raise ConfigError("pipeline.yaml: panel.snapshot_mode must be 'rolling' or 'single'")
    if int(p["horizon_days"]) <= 0:
        raise ConfigError("pipeline.yaml: panel.horizon_days must be positive")
    if int(p["embargo_days"]) < 0:
        raise ConfigError("pipeline.yaml: panel.embargo_days cannot be negative")
    if int(p["embargo_days"]) == 0:
        log.warning(
            "panel.embargo_days is 0. Features may be measured right up to the "
            "prediction date, so death-spiral signal (failed payments, support "
            "escalations in the final days) will dominate the model and look like a "
            "cause. Consider 7-14 days."
        )

    if cfg.leakage["on_block"] not in {"quarantine", "fail"}:
        raise ConfigError("pipeline.yaml: leakage.on_block must be 'quarantine' or 'fail'")

    for pattern in cfg.leakage["denylist_patterns"]:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ConfigError(
                f"pipeline.yaml: leakage.denylist_patterns entry {pattern!r} is not a "
                f"valid regex ({exc})"
            ) from exc

    if not 0 < float(cfg.split["test_fraction"]) < 1:
        raise ConfigError("pipeline.yaml: split.test_fraction must be between 0 and 1")
    if int(cfg.split["cv_folds"]) < 2:
        raise ConfigError("pipeline.yaml: split.cv_folds must be at least 2")

    # cross-file coherence
    if (
        cfg.observation_starting_date is not None
        and cfg.observation_end_date is not None
        and cfg.observation_starting_date >= cfg.observation_end_date
    ):
        raise ConfigError(
            "observation_starting_date (survivorship.yaml) must be before "
            "observation_end_date (target.yaml)"
        )

    for group in cfg.groups:
        if group.is_derived:
            continue
        if group.source != "entity" and group.source not in cfg.events:
            log.warning(
                "feature.yaml: group %r wants source %r but data.yaml has no such event "
                "log. Falling back to the entity table — these features will be read "
                "as-of export, not as-of the prediction date.",
                group.name,
                group.source,
            )
            group.source = "entity"
            group.filters = []           # a filter cannot apply to a flat entity row
        if group.is_time_varying and group.source != "entity" and not group.windows:
            group.windows = [90]
            log.info("feature.yaml: group %r has no windows; using [90]", group.name)

    if not any(g.is_derived is False for g in cfg.groups):
        raise ConfigError(
            "feature.yaml defines only derived groups. Derived features are expressions "
            "over other features, so at least one static or time_varying group is needed."
        )

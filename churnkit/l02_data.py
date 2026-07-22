"""Data loading and the first round of sanity checks.

Anything that is a mismatch between config and disk should surface here with a
message naming the file and the column, rather than as a KeyError three stages later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .l01_config import Config
from .util.errors import DataError
from .util.log import get_logger

log = get_logger("data")


@dataclass
class Dataset:
    entity: pd.DataFrame
    events: dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _read(path: Path, fmt: str) -> pd.DataFrame:
    if not path.exists():
        raise DataError(f"file not found: {path}")
    if fmt == "auto":
        fmt = "parquet" if path.suffix.lower() in {".parquet", ".pq"} else "csv"
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _require_columns(df: pd.DataFrame, columns: list[str], where: str) -> None:
    missing = [c for c in columns if c and c not in df.columns]
    if missing:
        raise DataError(
            f"{where}: missing column(s) {missing}. Available: {sorted(df.columns)[:40]}"
        )


def _to_datetime(df: pd.DataFrame, column: str, where: str) -> pd.Series:
    parsed = pd.to_datetime(df[column], errors="coerce")
    bad = parsed.isna() & df[column].notna()
    if bad.any():
        log.warning(
            "%s: %d value(s) in %r could not be parsed as dates and became NaT",
            where, int(bad.sum()), column,
        )
    return parsed.dt.normalize()


def load(cfg: Config) -> Dataset:
    entity = _read(cfg.entity_path, cfg.entity_format)
    log.info("entity table: %s rows x %s cols from %s",
             f"{len(entity):,}", entity.shape[1], cfg.entity_path.name)

    required = [cfg.id_column, cfg.start_date_column]
    if cfg.target_mode == "event_date":
        required.append(cfg.event_date_column)
    else:
        required.append(cfg.label_column)
    _require_columns(entity, required, f"entity table ({cfg.entity_path.name})")

    entity[cfg.start_date_column] = _to_datetime(entity, cfg.start_date_column, "entity")
    if cfg.target_mode == "event_date":
        entity[cfg.event_date_column] = _to_datetime(
            entity, cfg.event_date_column, "entity"
        )
    for col in (cfg.valid_from_column, cfg.valid_to_column):
        if col:
            _require_columns(entity, [col], "entity table (SCD2 columns)")
            entity[col] = _to_datetime(entity, col, "entity")

    notes: list[str] = []

    dupes = int(entity[cfg.id_column].duplicated().sum())
    if dupes:
        if cfg.valid_from_column:
            notes.append(
                f"entity table is versioned (SCD2): {dupes:,} extra row(s) beyond one "
                f"per {cfg.id_column}; static features will be read as-of each snapshot"
            )
            log.info(notes[-1])
        else:
            raise DataError(
                f"entity table has {dupes:,} duplicate {cfg.id_column} value(s) but no "
                f"`valid_from_column`/`valid_to_column` in data.yaml. Either de-duplicate "
                f"to one row per customer, or declare the validity columns so the kit can "
                f"pick the row that was current at each prediction date."
            )

    missing_start = int(entity[cfg.start_date_column].isna().sum())
    if missing_start:
        notes.append(
            f"{missing_start:,} customer(s) have no {cfg.start_date_column}; dropped "
            f"(tenure and eligibility cannot be established without it)"
        )
        log.warning(notes[-1])
        entity = entity[entity[cfg.start_date_column].notna()].copy()

    if cfg.target_mode == "event_date":
        churn = entity[cfg.event_date_column]
        backwards = int((churn < entity[cfg.start_date_column]).sum())
        if backwards:
            notes.append(
                f"{backwards:,} customer(s) churn before they start; those churn dates "
                f"were set to null (treated as still active)"
            )
            log.warning(notes[-1])
            entity.loc[
                churn < entity[cfg.start_date_column], cfg.event_date_column
            ] = pd.NaT

    events: dict[str, pd.DataFrame] = {}
    for name, src in cfg.events.items():
        df = _read(src.path, src.format)
        _require_columns(df, [src.id_column, src.date_column], f"events.{name}")
        df[src.date_column] = _to_datetime(df, src.date_column, f"events.{name}")
        before = len(df)
        df = df[df[src.date_column].notna()]
        if len(df) < before:
            log.warning(
                "events.%s: dropped %s row(s) with an unparseable %s",
                name, f"{before - len(df):,}", src.date_column,
            )
        if src.id_column != cfg.id_column:
            df = df.rename(columns={src.id_column: cfg.id_column})
        df = df.sort_values([cfg.id_column, src.date_column])
        events[name] = df
        log.info("events.%s: %s rows from %s", name, f"{len(df):,}", src.path.name)

        orphans = ~df[cfg.id_column].isin(set(entity[cfg.id_column]))
        if orphans.any():
            share = orphans.mean()
            log.warning(
                "events.%s: %.1f%% of rows reference a %s that is not in the entity "
                "table; they are ignored", name, 100 * share, cfg.id_column,
            )

    return Dataset(entity=entity, events=events, notes=notes)


def resolve_timeline(cfg: Config, ds: Dataset) -> dict[str, pd.Timestamp]:
    """Pin down the observation window, inferring what the user left blank.

    Getting these two dates wrong is the single most expensive mistake in a churn
    study: too late an end date turns censored customers into fake survivors, too
    early a start date lets survivorship bias in. So infer, then say so out loud.
    """
    entity = ds.entity
    start_col, id_col = cfg.start_date_column, cfg.id_column

    latest = entity[start_col].max()
    if cfg.target_mode == "event_date":
        latest = max(latest, entity[cfg.event_date_column].max())
    for name, df in ds.events.items():
        latest = max(latest, df[cfg.events[name].date_column].max())

    export = cfg.data_export_date
    if export is None:
        export = latest
        log.warning(
            "survivorship.yaml: data_export_date is blank; inferred %s from the latest "
            "date in your data", export.date(),
        )

    obs_end = cfg.observation_end_date or export
    if cfg.observation_end_date is None:
        log.warning(
            "target.yaml: observation_end_date is blank; using %s. Every customer "
            "without a churn date is right-censored at that point, NOT retained.",
            obs_end.date(),
        )

    obs_start = cfg.observation_starting_date
    if obs_start is None:
        obs_start = entity[start_col].min()
        log.warning(
            "survivorship.yaml: observation_starting_date is blank; inferred %s. If "
            "customers who churned before that date were never exported, the kit cannot "
            "detect it and tenure will look protective.", obs_start.date(),
        )

    # The date from which event history is complete enough to fill a feature window.
    # It is a different question from when churn started being recorded: an export
    # can carry three years of events for a cohort whose churn is only observed from
    # last July. Conflating them either rejects every prediction date or silently
    # truncates windows.
    feature_start = cfg.event_history_starts or obs_start
    if cfg.event_history_starts is not None:
        earliest_event = min(
            (df[cfg.events[name].date_column].min() for name, df in ds.events.items()),
            default=None,
        )
        if earliest_event is not None and feature_start < earliest_event:
            log.warning(
                "survivorship.yaml: event_history_starts is %s but the earliest event "
                "in your logs is %s. Feature windows reaching before that are silently "
                "short, so early aggregations will look artificially low.",
                feature_start.date(), earliest_event.date(),
            )
        log.info(
            "event history assumed complete from %s (survivorship boundary is %s)",
            feature_start.date(), obs_start.date(),
        )

    if obs_end > export:
        log.warning(
            "observation_end_date (%s) is after data_export_date (%s). The tail of the "
            "window has no data to observe churn in, so late snapshots will "
            "under-report churn.", obs_end.date(), export.date(),
        )

    log.info(
        "timeline: observe %s -> %s (export %s), %s customers",
        obs_start.date(), obs_end.date(), export.date(), f"{entity[id_col].nunique():,}",
    )
    return {
        "observation_start": obs_start,
        "observation_end": obs_end,
        "data_export": export,
        "feature_start": feature_start,
    }

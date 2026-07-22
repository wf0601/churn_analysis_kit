"""Point-in-time feature construction.

Every feature is computed inside the window [T - w, T - embargo). The window end is
enforced with an assertion after each aggregation, so a future-dated row cannot slip
in through a mis-sorted or mis-parsed date column.

Raw datetimes are never emitted as features — they are converted to "days before T".
A raw timestamp lets a tree learn the calendar, which is how an out-of-time split
quietly turns into memorisation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .l01_config import Config, FeatureGroup
from .l02_data import Dataset
from .util.errors import ConfigError, DataError
from .util.log import get_logger
from .l03_panel import Panel

log = get_logger("features")

DERIVED_COLUMNS = {"tenure_days", "recency_days"}

# Functions a derived expression may call. Everything else — attribute access,
# imports, comprehensions, lambdas — is rejected before evaluation.
EXPRESSION_FUNCTIONS = {
    "log": np.log, "log1p": np.log1p, "exp": np.exp, "sqrt": np.sqrt,
    "abs": np.abs, "clip": np.clip, "where": np.where,
    "minimum": np.minimum, "maximum": np.maximum, "sign": np.sign,
    "isnull": pd.isna, "notnull": pd.notna, "fillna": lambda s, v: s.fillna(v),
}

MANY_FEATURES = 500


@dataclass
class FeatureMeta:
    name: str
    group: str
    source: str
    kind: str                     # numeric | categorical
    temporal: str
    window_days: int | None
    leakage_review: str
    derivation: str
    point_in_time: bool           # False = value could reflect post-T state
    depends_on: list[str] = field(default_factory=list)


@dataclass
class FeatureMatrix:
    X: pd.DataFrame
    meta: dict[str, FeatureMeta]
    skipped: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def numeric(self) -> list[str]:
        return [n for n, m in self.meta.items() if m.kind == "numeric" and n in self.X]

    @property
    def categorical(self) -> list[str]:
        return [n for n, m in self.meta.items() if m.kind == "categorical" and n in self.X]

    def drop(self, names: list[str]) -> None:
        keep = [n for n in names if n in self.X.columns]
        self.X = self.X.drop(columns=keep)

    def dictionary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "feature": m.name, "group": m.group, "source": m.source,
                    "kind": m.kind, "temporal": m.temporal,
                    "window_days": m.window_days, "leakage_review": m.leakage_review,
                    "derivation": m.derivation, "point_in_time": m.point_in_time,
                    "depends_on": ", ".join(m.depends_on),
                    "in_model": m.name in self.X.columns,
                }
                for m in self.meta.values()
            ]
        ).sort_values(["group", "feature"])

    def dependents_of(self, names: list[str]) -> list[str]:
        """Features that would be computed from any of `names`, transitively.

        Quarantining a leaking feature has to take its children with it: a ratio
        built on a leaking numerator carries exactly the same information.
        """
        doomed = set(names)
        while True:
            found = {
                name for name, meta in self.meta.items()
                if name not in doomed and set(meta.depends_on) & doomed
            }
            if not found:
                return sorted(doomed - set(names))
            doomed |= found


def build(cfg: Config, ds: Dataset, panel: Panel) -> FeatureMatrix:
    frame = panel.frame.reset_index(drop=True)
    X = pd.DataFrame(index=frame.index)
    meta: dict[str, FeatureMeta] = {}
    skipped: list[dict] = []
    notes: list[str] = []

    for group in cfg.groups:
        if group.is_derived:
            continue                      # derived groups run last, over everything else
        if group.source == "entity":
            _build_entity_group(cfg, ds, frame, group, X, meta, skipped, notes)
        else:
            _build_event_group(cfg, ds, frame, group, panel, X, meta, skipped)

    if X.empty or X.shape[1] == 0:
        raise DataError(
            "No features could be built. Check that the column names in feature.yaml "
            "match the columns in your data files."
        )
    _finalise_types(X, meta)
    X = X.copy()                      # de-fragment after many single-column inserts

    for group in cfg.groups:
        if group.is_derived:
            new = _build_derived_group(group, X, meta, skipped)
            if new:
                X = pd.concat([X, pd.DataFrame(new, index=X.index)], axis=1)
    _finalise_types(X, meta)

    log.info(
        "built %d features across %d groups (%d skipped)",
        X.shape[1], len(cfg.groups), len(skipped),
    )
    if X.shape[1] > MANY_FEATURES:
        log.warning(
            "%d features from %d rows. Every extra aggregation is another chance for "
            "one of them to correlate with the label by luck; prune the windows and "
            "aggs in feature.yaml rather than relying on the model to ignore them.",
            X.shape[1], len(frame),
        )
    if skipped:
        for item in skipped:
            log.warning("skipped %s.%s: %s", item["group"], item["column"], item["reason"])

    return FeatureMatrix(X=X, meta=meta, skipped=skipped, notes=notes)


# --------------------------------------------------------------------------- #
# entity-table features
# --------------------------------------------------------------------------- #
def _entity_asof(cfg: Config, ds: Dataset, frame: pd.DataFrame) -> pd.DataFrame:
    """Entity rows aligned to each panel row, honouring SCD2 validity if declared."""
    entity = ds.entity
    id_col = cfg.id_column

    if not cfg.valid_from_column:
        return (
            frame[["entity_id"]]
            .merge(entity, left_on="entity_id", right_on=id_col, how="left")
            .reset_index(drop=True)
        )

    vf, vt = cfg.valid_from_column, cfg.valid_to_column
    merged = frame[["entity_id", "feature_end"]].merge(
        entity, left_on="entity_id", right_on=id_col, how="left"
    )
    valid = (merged[vf] <= merged["feature_end"]) & (
        merged[vt].isna() | (merged[vt] > merged["feature_end"])
    )
    merged = merged[valid]
    # One version per panel row; keep the latest that was in force.
    merged = (
        merged.sort_values(vf)
        .groupby(["entity_id", "feature_end"], as_index=False)
        .last()
    )
    out = frame[["entity_id", "feature_end"]].merge(
        merged, on=["entity_id", "feature_end"], how="left"
    )
    return out.reset_index(drop=True)


def _build_entity_group(
    cfg: Config, ds: Dataset, frame: pd.DataFrame, group: FeatureGroup,
    X: pd.DataFrame, meta: dict, skipped: list, notes: list,
) -> None:
    aligned = _entity_asof(cfg, ds, frame)
    point_in_time = bool(cfg.valid_from_column)

    if group.is_time_varying and not point_in_time:
        msg = (
            f"group '{group.name}' is declared time_varying but is read from the entity "
            f"table, which has no version history. Its values are whatever was true at "
            f"export, not at the prediction date. If any of them are updated when a "
            f"customer churns, that is leakage the kit cannot detect. Move the group to "
            f"an event log, or add valid_from/valid_to columns in data.yaml."
        )
        log.warning(msg)
        notes.append(msg)

    for col in group.columns:
        name = f"{group.name}__{col.name}"

        if col.name in DERIVED_COLUMNS:
            series = _derived(col.name, frame, aligned, cfg)
            if series is None:
                skipped.append(
                    {"group": group.name, "column": col.name,
                     "reason": "not in data and not derivable"}
                )
                continue
            if col.name in aligned.columns:
                # A stored tenure/recency column is almost always computed at export
                # time, which means it encodes how long the customer ultimately
                # lasted — the answer, written into a feature. Recompute instead.
                msg = (
                    f"'{col.name}' exists as a column in {cfg.entity_path.name}, but the "
                    f"kit recomputed it as of each prediction date instead of reading "
                    f"it. A stored {col.name} is normally calculated when the export "
                    f"runs, so for churned customers it measures the full lifetime "
                    f"they turned out to have."
                )
                log.warning(msg)
                notes.append(msg)
            X[name] = series.values
            meta[name] = FeatureMeta(
                name, group.name, "derived", "numeric", group.temporal, None,
                group.leakage_review, f"recomputed as-of T ({col.name})", True,
            )
            continue

        if col.name not in aligned.columns:
            skipped.append(
                {"group": group.name, "column": col.name,
                 "reason": f"column not found in {cfg.entity_path.name}"}
            )
            continue

        raw = aligned[col.name]
        if col.type == "datetime":
            parsed = pd.to_datetime(raw, errors="coerce")
            name = f"{group.name}__{col.name}__days_before_T"
            X[name] = (frame["feature_end"].values - parsed.values) / np.timedelta64(1, "D")
            kind, derivation = "numeric", "days between the value and T - embargo"
        elif col.type == "boolean":
            X[name] = _to_bool(raw).values
            kind, derivation = "numeric", "as-of value (0/1)"
        elif col.type == "categorical":
            X[name] = raw.astype("object").where(raw.notna(), None).values
            kind, derivation = "categorical", "as-of value"
        else:
            X[name] = pd.to_numeric(raw, errors="coerce").values
            kind, derivation = "numeric", "as-of value"

        meta[name] = FeatureMeta(
            name, group.name, "entity", kind, group.temporal, None,
            group.leakage_review, derivation,
            point_in_time or group.temporal == "static",
        )


def _derived(column: str, frame: pd.DataFrame, aligned: pd.DataFrame, cfg: Config):
    if column == "tenure_days":
        return frame["tenure_days"]
    if column == "recency_days":
        for candidate in ("last_active_date", "last_seen_date", "last_login_date"):
            if candidate in aligned.columns:
                parsed = pd.to_datetime(aligned[candidate], errors="coerce")
                return pd.Series(
                    (frame["feature_end"].values - parsed.values) / np.timedelta64(1, "D")
                )
    return None


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    mapped = (
        series.astype("string").str.strip().str.lower()
        .map({"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0,
              "1": 1.0, "0": 0.0, "y": 1.0, "n": 0.0, "t": 1.0, "f": 0.0})
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric).astype(float)


# --------------------------------------------------------------------------- #
# event-log features
# --------------------------------------------------------------------------- #
def _feature_name(group: str, filter_name: str, column: str | None, suffix: str) -> str:
    parts = [group]
    if filter_name:
        parts.append(filter_name)
    if column:
        parts.append(column)
    parts.append(suffix)
    return "__".join(parts)


def _resolve_columns(
    group: FeatureGroup, events: pd.DataFrame, source_name: str, skipped: list
) -> tuple[list[tuple], list[tuple]]:
    """Bind each configured column to its aggregations and windows."""
    numeric, datetimes = [], []
    for col in group.columns:
        if col.name not in events.columns:
            if col.name not in DERIVED_COLUMNS:
                skipped.append(
                    {"group": group.name, "column": col.name,
                     "reason": f"column not found in {source_name}"}
                )
            continue
        windows = col.windows or group.windows
        if col.type == "datetime":
            datetimes.append((col.name, windows))
        elif col.type == "categorical":
            skipped.append(
                {"group": group.name, "column": col.name,
                 "reason": "categorical event columns are not aggregated directly — "
                           "use a `filters` entry to turn a category into its own "
                           "counted feature family"}
            )
        else:
            numeric.append((col.name, col.aggs or group.aggs, windows))
    return numeric, datetimes


def _apply_filter(events: pd.DataFrame, spec, group: FeatureGroup) -> pd.DataFrame:
    if not spec.where:
        return events
    try:
        return events.query(spec.where)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(
            f"feature.yaml: filter {group.name}.{spec.name} could not be applied. "
            f"`where: {spec.where}` failed with: {exc}. Available columns: "
            f"{sorted(events.columns)}"
        ) from exc


def _build_event_group(
    cfg: Config, ds: Dataset, frame: pd.DataFrame, group: FeatureGroup,
    panel: Panel, X: pd.DataFrame, meta: dict, skipped: list,
) -> None:
    events = ds.events[group.source]
    src = cfg.events[group.source]
    date_col = src.date_column
    windows = group.windows or [90]

    numeric_cols, datetime_cols = _resolve_columns(group, events, src.path.name, skipped)

    # Filters ADD feature families; they do not replace the base one. So a group
    # with a `failed` filter still produces the all-rows aggregation alongside it,
    # which is what makes ratios like failed/(failed+paid) expressible.
    # Set `include_unfiltered: false` when the base aggregate is meaningless.
    from .l01_config import FilterSpec  # noqa: PLC0415

    filters = list(group.filters)
    if group.include_unfiltered or not filters:
        filters = [FilterSpec(name="", where=""), *filters]
    subsets = {}
    for spec in filters:
        subset = _apply_filter(events, spec, group)
        subsets[spec.name] = subset
        if spec.where:
            log.info(
                "feature.yaml: %s.%s matched %s of %s %s events (%s)",
                group.name, spec.name, f"{len(subset):,}", f"{len(events):,}",
                group.source, spec.where,
            )
            if subset.empty:
                skipped.append(
                    {"group": group.name, "column": f"filter:{spec.name}",
                     "reason": f"`{spec.where}` matched no rows"}
                )

    blocks = []
    for _, rows in frame.groupby("snapshot_date", sort=True):
        feature_end = rows["feature_end"].iloc[0]
        pieces = [rows[["entity_id"]]]
        for spec in filters:
            subset = subsets[spec.name]
            if subset.empty:
                continue
            for window in windows:
                block = _aggregate_window(
                    subset, date_col, cfg.id_column, feature_end, window,
                    numeric_cols, datetime_cols, group, spec.name,
                )
                if group.generate_trends and numeric_cols:
                    block = block.merge(
                        _trend_block(subset, date_col, cfg.id_column, feature_end,
                                     window, numeric_cols, group, spec.name),
                        on="entity_id", how="outer",
                    )
                # merge() returns a fresh RangeIndex; restore the panel's row index
                # so the join below lines up with the right customer-date rows.
                piece = rows[["entity_id"]].merge(block, on="entity_id", how="left")
                piece.index = rows.index
                pieces.append(piece)

        merged = pieces[0]
        for piece in pieces[1:]:
            merged = merged.join(piece.drop(columns=["entity_id"]))
        merged.index = rows.index
        blocks.append(merged.drop(columns=["entity_id"]))

    if not blocks:
        return
    built = pd.concat(blocks).sort_index()
    for name in built.columns:
        X[name] = built[name]
        window = int(name.rsplit("_", 1)[-1].rstrip("d")) if name.rsplit("_", 1)[-1].endswith("d") else None
        meta[name] = FeatureMeta(
            name=name, group=group.name, source=group.source, kind="numeric",
            temporal=group.temporal, window_days=window,
            leakage_review=group.leakage_review,
            derivation=f"aggregated over [T-{window}d, T-{panel.embargo_days}d)",
            point_in_time=True,
        )


def _aggregate_window(
    events: pd.DataFrame, date_col: str, id_col: str, window_end: pd.Timestamp,
    window: int, numeric_cols: list[tuple], datetime_cols: list[tuple],
    group: FeatureGroup, filter_name: str,
) -> pd.DataFrame:
    suffix = f"{window}d"
    window_start = window_end - pd.Timedelta(days=window)
    in_window = events[
        (events[date_col] >= window_start) & (events[date_col] < window_end)
    ]
    # The guarantee this whole module exists to provide.
    if not in_window.empty:
        assert in_window[date_col].max() < window_end, (
            f"leakage guard tripped: group {group.name} pulled an event dated "
            f"{in_window[date_col].max()} into a window ending {window_end}"
        )

    names = [
        _feature_name(group.name, filter_name, None, f"event_count_{suffix}"),
        _feature_name(group.name, filter_name, None, f"recency_days_{suffix}"),
    ]
    for column, aggs, windows in numeric_cols:
        if window in windows:
            names += [
                _feature_name(group.name, filter_name, column, f"{agg}_{suffix}")
                for agg in aggs
            ]
    for column, windows in datetime_cols:
        if window in windows:
            names.append(
                _feature_name(group.name, filter_name, column, f"recency_days_{suffix}")
            )

    if in_window.empty:
        return pd.DataFrame(columns=["entity_id", *names])

    grouped = in_window.groupby(id_col)
    out = pd.DataFrame(index=grouped.size().index)
    out[names[0]] = grouped.size()
    out[names[1]] = (window_end - grouped[date_col].max()).dt.days

    for column, aggs, windows in numeric_cols:
        if window not in windows:
            continue
        values = grouped[column]
        for agg in aggs:
            out[_feature_name(group.name, filter_name, column, f"{agg}_{suffix}")] = (
                values.agg(agg)
            )

    for column, windows in datetime_cols:
        if window not in windows:
            continue
        latest = pd.to_datetime(grouped[column].max(), errors="coerce")
        out[_feature_name(group.name, filter_name, column, f"recency_days_{suffix}")] = (
            (window_end - latest).dt.days
        )

    return out.reset_index().rename(columns={id_col: "entity_id"})


def _trend_block(
    events: pd.DataFrame, date_col: str, id_col: str, window_end: pd.Timestamp,
    window: int, numeric_cols: list[tuple], group: FeatureGroup, filter_name: str,
) -> pd.DataFrame:
    """Recent half-window versus the half before it.

    Direction of travel is usually a stronger and more actionable signal than level,
    and it is far less confounded by customer size.
    """
    suffix = f"{window}d"
    columns = [c for c, _, windows in numeric_cols if window in windows]
    if not columns:
        return pd.DataFrame(columns=["entity_id"])

    mid = window_end - pd.Timedelta(days=window / 2)
    start = window_end - pd.Timedelta(days=window)
    recent = events[(events[date_col] >= mid) & (events[date_col] < window_end)]
    prior = events[(events[date_col] >= start) & (events[date_col] < mid)]

    r_sum = recent.groupby(id_col)[columns].sum() if not recent.empty else None
    p_sum = prior.groupby(id_col)[columns].sum() if not prior.empty else None

    ids = set()
    for part in (r_sum, p_sum):
        if part is not None:
            ids |= set(part.index)
    out = pd.DataFrame(index=pd.Index(sorted(ids), name=id_col))
    for column in columns:
        r = r_sum[column].reindex(out.index).fillna(0.0) if r_sum is not None else 0.0
        p = p_sum[column].reindex(out.index).fillna(0.0) if p_sum is not None else 0.0
        r = pd.Series(r, index=out.index, dtype=float)
        p = pd.Series(p, index=out.index, dtype=float)
        out[_feature_name(group.name, filter_name, column, f"trend_delta_{suffix}")] = r - p
        # Undefined rather than infinite when there is no baseline to compare against.
        out[_feature_name(group.name, filter_name, column, f"trend_ratio_{suffix}")] = (
            np.where(p > 0, r / p, np.nan)
        )
    return out.reset_index().rename(columns={id_col: "entity_id"})


# --------------------------------------------------------------------------- #
# derived features
# --------------------------------------------------------------------------- #
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Call, ast.Name, ast.Constant, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def _parse_expression(spec, available: set[str]) -> list[str]:
    """Check an expression before running it, and return the features it reads.

    Rejecting attribute access, imports and lambdas keeps a config file from
    becoming an arbitrary code-execution surface when it is shared between people.
    """
    try:
        tree = ast.parse(spec.expression, mode="eval")
    except SyntaxError as exc:
        raise ConfigError(
            f"feature.yaml: derived feature {spec.name!r} is not a valid expression "
            f"({exc.msg}): {spec.expression}"
        ) from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ConfigError(
                f"feature.yaml: derived feature {spec.name!r} uses {type(node).__name__}, "
                f"which is not allowed. Expressions may use feature names, numbers, "
                f"arithmetic, comparisons and these functions: "
                f"{sorted(EXPRESSION_FUNCTIONS)}."
            )
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in EXPRESSION_FUNCTIONS
        ):
            raise ConfigError(
                f"feature.yaml: derived feature {spec.name!r} calls something other than "
                f"an allowed function. Available: {sorted(EXPRESSION_FUNCTIONS)}."
            )

    referenced = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id not in EXPRESSION_FUNCTIONS
    }
    unknown = sorted(referenced - available)
    if unknown:
        raise ConfigError(
            f"feature.yaml: derived feature {spec.name!r} refers to unknown feature(s) "
            f"{unknown}. Run the pipeline once and read output/feature_dictionary.csv "
            f"for the exact names, or check the group it should come from."
        )
    return sorted(referenced)


def _build_derived_group(
    group: FeatureGroup, X: pd.DataFrame, meta: dict, skipped: list
) -> dict[str, pd.Series]:
    built: dict[str, pd.Series] = {}
    for spec in group.derived:
        depends_on = _parse_expression(spec, set(X.columns))
        namespace = {name: X[name] for name in depends_on}
        try:
            values = eval(  # noqa: S307 - AST-validated above, restricted namespace
                compile(ast.parse(spec.expression, mode="eval"), "<feature.yaml>", "eval"),
                {"__builtins__": {}},
                {**EXPRESSION_FUNCTIONS, **namespace},
            )
        except Exception as exc:  # noqa: BLE001
            skipped.append(
                {"group": group.name, "column": spec.name,
                 "reason": f"expression failed at runtime: {exc}"}
            )
            log.warning("derived feature %s failed: %s", spec.name, exc)
            continue

        name = f"{group.name}__{spec.name}"
        built[name] = pd.Series(values, index=X.index, dtype="float64")
        # A derived feature is only as point-in-time as its worst input.
        point_in_time = all(
            meta[dep].point_in_time for dep in depends_on if dep in meta
        )
        windows = [meta[dep].window_days for dep in depends_on if dep in meta]
        windows = [w for w in windows if w]
        meta[name] = FeatureMeta(
            name=name, group=group.name, source="derived", kind="numeric",
            temporal="derived", window_days=max(windows) if windows else None,
            leakage_review=group.leakage_review,
            derivation=spec.description or spec.expression,
            point_in_time=point_in_time, depends_on=depends_on,
        )
    return built


def _finalise_types(X: pd.DataFrame, meta: dict[str, FeatureMeta]) -> None:
    for name, m in meta.items():
        if name not in X.columns:
            continue
        if m.kind == "numeric":
            X[name] = pd.to_numeric(X[name], errors="coerce").astype(float)
            X[name] = X[name].replace([np.inf, -np.inf], np.nan)
        else:
            # np.nan rather than None: the encoders treat np.nan as missing, but a
            # None in an object column raises on some paths.
            X[name] = (
                X[name].apply(lambda v: np.nan if pd.isna(v) else str(v)).astype(object)
            )

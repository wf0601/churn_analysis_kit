"""Snapshot panel construction — the stage where leakage is prevented by design.

Every row is one (customer, prediction date T) pair with three separated regions:

    [ T - window , T - embargo )   features may look here, and nowhere else
    [ T - embargo , T ]            embargo: blanked out entirely
    ( T , T + horizon ]            label period

A customer only enters at T if they are genuinely at risk at T, and a row only
gets label 0 if the full horizon after T is actually observable in the data.
Censored time is never silently converted into retention.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .l01_config import Config
from .l02_data import Dataset
from .util.errors import InsufficientDataError
from .util.log import get_logger

log = get_logger("panel")


@dataclass
class Panel:
    frame: pd.DataFrame                       # one row per (entity, snapshot)
    survival: pd.DataFrame                    # one row per entity
    snapshot_dates: list[pd.Timestamp]
    timeline: dict[str, pd.Timestamp]
    horizon_days: int
    embargo_days: int
    diagnostics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def churn_rate(self) -> float:
        return float(self.frame["label"].mean())


def _effective_churn(cfg: Config, entity: pd.DataFrame) -> pd.Series:
    """Churn timestamps, shifted earlier by decision_lead_days where configured."""
    if cfg.target_mode == "label":
        return pd.Series(pd.NaT, index=entity.index, dtype="datetime64[ns]")
    churn = entity[cfg.event_date_column]
    if cfg.decision_lead_days:
        churn = churn - pd.Timedelta(days=cfg.decision_lead_days)
        log.info(
            "target.yaml: decision_lead_days=%d — churn treated as happening %d days "
            "before the recorded date", cfg.decision_lead_days, cfg.decision_lead_days,
        )
    return churn


def _snapshot_dates(cfg: Config, timeline: dict) -> list[pd.Timestamp]:
    horizon = int(cfg.panel["horizon_days"])
    obs_end = timeline["observation_end"]
    # Feature windows are bounded by how far the event history goes back, which is
    # not necessarily where churn observation starts.
    feature_start = timeline.get("feature_start", timeline["observation_start"])

    # Earliest T where a full feature window fits without reaching before the data.
    lead = cfg.max_window_days + int(cfg.panel["embargo_days"])
    default_start = feature_start + pd.Timedelta(days=lead)
    # Latest T where the full label horizon is still observable.
    default_end = obs_end - pd.Timedelta(days=horizon)

    if cfg.panel["snapshot_mode"] == "single":
        raw = cfg.panel["snapshot_dates"] or [default_end]
        dates = [pd.Timestamp(d).normalize() for d in raw]
    else:
        start = pd.Timestamp(cfg.panel["snapshot_start"] or default_start).normalize()
        end = pd.Timestamp(cfg.panel["snapshot_end"] or default_end).normalize()
        if start > end:
            raise InsufficientDataError(
                f"No usable prediction dates: the earliest date with a full "
                f"{cfg.max_window_days}-day feature window is {start.date()}, but the "
                f"latest date with a full {horizon}-day label horizon is {end.date()}. "
                f"Shorten panel.horizon_days or the aggregation windows in feature.yaml."
            )
        dates = list(pd.date_range(start, end, freq=cfg.panel["snapshot_frequency"]))
        if not dates:
            dates = [start]

    kept = []
    for date in sorted(set(dates)):
        if date + pd.Timedelta(days=horizon) > obs_end:
            log.warning(
                "snapshot %s dropped: its %d-day label window runs past "
                "observation_end_date (%s), so non-churners there are censored, not "
                "retained.", date.date(), horizon, obs_end.date(),
            )
            continue
        if date - pd.Timedelta(days=lead) < feature_start:
            log.warning(
                "snapshot %s dropped: its feature window would reach before %s, where "
                "event history begins, so features would be silently truncated and "
                "look artificially low. Set survivorship.yaml: event_history_starts if "
                "your logs go back further than churn observation does.",
                date.date(), feature_start.date(),
            )
            continue
        kept.append(date)

    if not kept:
        raise InsufficientDataError(
            "Every candidate prediction date was rejected. Event history from "
            f"{feature_start.date()} to {obs_end.date()} is too short for a "
            f"{cfg.max_window_days}-day feature window plus a {horizon}-day horizon. "
            f"Shorten the windows in feature.yaml, shorten panel.horizon_days, or set "
            f"survivorship.yaml: event_history_starts if your event logs reach further "
            f"back than observation_starting_date."
        )
    return kept


def build(cfg: Config, ds: Dataset, timeline: dict) -> Panel:
    entity = ds.entity
    id_col, start_col = cfg.id_column, cfg.start_date_column
    horizon = int(cfg.panel["horizon_days"])
    embargo = int(cfg.panel["embargo_days"])
    min_tenure = int(cfg.panel["min_tenure_days"])
    obs_start = timeline["observation_start"]

    notes: list[str] = []
    base = entity[[id_col, start_col]].copy()
    base["_churn"] = _effective_churn(cfg, entity).values
    base["_left_truncated"] = base[start_col] < obs_start

    n_truncated = int(base["_left_truncated"].sum())
    if n_truncated:
        share = n_truncated / len(base)
        msg = (
            f"{n_truncated:,} customers ({share:.1%}) started before "
            f"observation_starting_date ({obs_start.date()}). They are survivors of an "
            f"unobserved period — their churned peers are missing from the data."
        )
        if cfg.left_truncation == "drop":
            base = base[~base["_left_truncated"]].copy()
            notes.append(msg + " Dropped (survivorship.yaml: left_truncation=drop).")
        else:
            notes.append(
                msg + " Kept and flagged; survival uses a left-truncated fit, but "
                "classification metrics remain optimistic for long-tenured customers."
            )
        log.warning(notes[-1])
        if base.empty:
            raise InsufficientDataError(
                "Dropping left-truncated customers removed everyone. Set "
                "survivorship.yaml: observation_starting_date to the true start of "
                "your data, or use left_truncation: keep_flagged."
            )

    if cfg.target_mode == "label":
        return _build_label_mode(cfg, ds, base, timeline, notes)

    dates = _snapshot_dates(cfg, timeline)
    log.info(
        "%d prediction date(s) from %s to %s (horizon %dd, embargo %dd)",
        len(dates), dates[0].date(), dates[-1].date(), horizon, embargo,
    )

    rows = []
    per_snapshot = []
    for date in dates:
        at_risk = (
            (base[start_col] <= date - pd.Timedelta(days=min_tenure))
            & (base["_churn"].isna() | (base["_churn"] > date))
        )
        sub = base.loc[at_risk, [id_col, start_col, "_churn", "_left_truncated"]].copy()
        if sub.empty:
            log.warning("snapshot %s has no customers at risk; skipped", date.date())
            continue
        horizon_end = date + pd.Timedelta(days=horizon)
        sub["snapshot_date"] = date
        sub["label"] = (
            sub["_churn"].notna() & (sub["_churn"] <= horizon_end)
        ).astype(int)
        sub["tenure_days"] = (date - sub[start_col]).dt.days
        sub["feature_end"] = date - pd.Timedelta(days=embargo)
        rows.append(sub)
        per_snapshot.append(
            {
                "snapshot_date": date,
                "n_at_risk": len(sub),
                "n_churned": int(sub["label"].sum()),
                "churn_rate": float(sub["label"].mean()),
            }
        )

    if not rows:
        raise InsufficientDataError(
            "No customer was at risk at any prediction date. Check that "
            f"{start_col} and {cfg.event_date_column} overlap the observation window."
        )

    frame = pd.concat(rows, ignore_index=True)
    frame = frame.rename(columns={id_col: "entity_id", "_left_truncated": "left_truncated"})
    frame = frame.drop(columns=["_churn"])

    cap = cfg.panel["max_snapshots_per_entity"]
    if cap:
        rng = np.random.default_rng(cfg.seed)
        before = len(frame)
        frame = (
            frame.groupby("entity_id", group_keys=False)
            .apply(
                lambda g: g.sample(min(len(g), int(cap)), random_state=rng.integers(1e9)),
                include_groups=True,
            )
            .reset_index(drop=True)
        )
        log.info(
            "max_snapshots_per_entity=%s: kept %s of %s rows (reduces within-customer "
            "correlation between rows)", cap, f"{len(frame):,}", f"{before:,}",
        )

    _report_panel(frame, per_snapshot)
    survival = _survival_frame(cfg, base, timeline)

    return Panel(
        frame=frame,
        survival=survival,
        snapshot_dates=dates,
        timeline=timeline,
        horizon_days=horizon,
        embargo_days=embargo,
        diagnostics={"per_snapshot": pd.DataFrame(per_snapshot)},
        notes=notes,
    )


def _build_label_mode(
    cfg: Config, ds: Dataset, base: pd.DataFrame, timeline: dict, notes: list[str]
) -> Panel:
    """Single cross-section from a pre-computed label column.

    There is no date on the label, so the kit cannot verify that any feature was
    knowable before the customer churned. That warning belongs in the report, not
    just the console.
    """
    id_col, start_col = cfg.id_column, cfg.start_date_column
    obs_end = timeline["observation_end"]
    embargo = int(cfg.panel["embargo_days"])

    label_raw = ds.entity.set_index(id_col)[cfg.label_column]
    frame = base.rename(columns={id_col: "entity_id", "_left_truncated": "left_truncated"})
    frame = frame.drop(columns=["_churn"])
    frame["label"] = (
        frame["entity_id"].map(label_raw).astype("object") == cfg.churn_value
    ).astype(int)
    frame["snapshot_date"] = obs_end
    frame["tenure_days"] = (obs_end - frame[start_col]).dt.days
    frame["feature_end"] = obs_end - pd.Timedelta(days=embargo)

    warning = (
        "target.mode is 'label': churn has no timestamp, so features cannot be "
        "restricted to the period before each customer left. Any column that reacts to "
        "churn (final invoices, closed tickets, zeroed usage) will leak and the kit "
        "cannot prove otherwise. Switch to mode 'event_date' if you have a churn date."
    )
    log.warning(warning)
    notes.append(warning)

    survival = _survival_frame(cfg, base, timeline)
    return Panel(
        frame=frame,
        survival=survival,
        snapshot_dates=[obs_end],
        timeline=timeline,
        horizon_days=int(cfg.panel["horizon_days"]),
        embargo_days=embargo,
        diagnostics={
            "per_snapshot": pd.DataFrame(
                [{"snapshot_date": obs_end, "n_at_risk": len(frame),
                  "n_churned": int(frame["label"].sum()),
                  "churn_rate": float(frame["label"].mean())}]
            )
        },
        notes=notes,
    )


def _survival_frame(cfg: Config, base: pd.DataFrame, timeline: dict) -> pd.DataFrame:
    """Per-customer duration/event table with explicit right-censoring."""
    id_col, start_col = cfg.id_column, cfg.start_date_column
    obs_start, obs_end = timeline["observation_start"], timeline["observation_end"]

    out = pd.DataFrame({"entity_id": base[id_col].values})
    churn = base["_churn"]
    observed = churn.notna() & (churn <= obs_end)
    end_time = churn.where(observed, obs_end)
    out["duration_days"] = (end_time.values - base[start_col].values) / np.timedelta64(1, "D")
    out["event"] = observed.astype(int).values
    out["start_date"] = base[start_col].values
    out["left_truncated"] = base["_left_truncated"].values

    # Entry time for left-truncated fits: customers already alive at obs_start were
    # never at risk of being observed churning before it.
    entry = (obs_start - base[start_col]).dt.days.clip(lower=0)
    out["entry_days"] = entry.values if cfg.left_truncation == "keep_flagged" else 0.0

    out = out[out["duration_days"] > out["entry_days"]].copy()
    log.info(
        "survival table: %s customers, %s observed churn events (%.1f%% censored)",
        f"{len(out):,}", f"{int(out['event'].sum()):,}",
        100 * (1 - out["event"].mean()) if len(out) else 0.0,
    )
    return out


def _report_panel(frame: pd.DataFrame, per_snapshot: list[dict]) -> None:
    rate = frame["label"].mean()
    log.info(
        "panel: %s rows, %s customers, %s churn events (%.2f%% base rate)",
        f"{len(frame):,}", f"{frame['entity_id'].nunique():,}",
        f"{int(frame['label'].sum()):,}", 100 * rate,
    )
    if int(frame["label"].sum()) < 50:
        log.warning(
            "Only %d churn events in the panel. Driver rankings and causal estimates "
            "will be dominated by noise; widen the observation window or lengthen "
            "panel.horizon_days.", int(frame["label"].sum()),
        )
    rates = pd.DataFrame(per_snapshot)["churn_rate"]
    if len(rates) > 2 and rates.max() > 0 and rates.max() / max(rates.min(), 1e-9) > 3:
        log.warning(
            "Churn rate varies %.1fx across prediction dates (%.2f%% to %.2f%%). An "
            "out-of-time split will look unstable — this is a real regime change in "
            "your data, not a modelling artifact.",
            rates.max() / max(rates.min(), 1e-9), 100 * rates.min(), 100 * rates.max(),
        )

"""Survival analysis: when customers leave, not just whether.

Right-censoring is handled explicitly — a customer still active at the end of the
observation window contributes their time at risk and nothing more. Treating them as
"retained" is the classic way to under-report churn, and it is exactly what a plain
classification framing does.

Left truncation is honoured when configured: customers who were already alive when
the data starts enter the risk set at their current age, not at zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .l01_config import Config
from .l02_data import Dataset
from .util.log import get_logger
from .l03_panel import Panel

log = get_logger("survival")


@dataclass
class SurvivalReport:
    overall: pd.DataFrame
    strata: dict[str, pd.DataFrame] = field(default_factory=dict)
    logrank: pd.DataFrame = field(default_factory=pd.DataFrame)
    cox: pd.DataFrame = field(default_factory=pd.DataFrame)
    cox_diagnostics: dict = field(default_factory=dict)
    median_lifetime: float | None = None
    notes: list[str] = field(default_factory=list)


def _km(durations: np.ndarray, events: np.ndarray, entry: np.ndarray | None = None) -> pd.DataFrame:
    """Kaplan-Meier with optional left truncation, computed directly.

    Written out rather than delegated so the risk-set arithmetic is inspectable:
    at each event time, the denominator counts everyone under observation then —
    entered, not yet churned, not yet censored.
    """
    entry = np.zeros_like(durations, dtype=float) if entry is None else entry.astype(float)
    times = np.unique(durations[events == 1])
    rows, survival = [], 1.0
    for t in times:
        at_risk = int(np.sum((durations >= t) & (entry < t)))
        if at_risk == 0:
            continue
        n_events = int(np.sum((durations == t) & (events == 1)))
        survival *= 1 - n_events / at_risk
        # Greenwood's formula for the pointwise variance.
        rows.append({"time": float(t), "at_risk": at_risk, "events": n_events,
                     "survival": survival,
                     "_gw": n_events / (at_risk * max(at_risk - n_events, 1))})
    if not rows:
        return pd.DataFrame(columns=["time", "at_risk", "events", "survival",
                                     "ci_lower", "ci_upper"])
    out = pd.DataFrame(rows)
    cum = out["_gw"].cumsum()
    se = out["survival"] * np.sqrt(cum)
    out["ci_lower"] = (out["survival"] - 1.96 * se).clip(0, 1)
    out["ci_upper"] = (out["survival"] + 1.96 * se).clip(0, 1)
    return out.drop(columns=["_gw"])


def _median(curve: pd.DataFrame) -> float | None:
    below = curve[curve["survival"] <= 0.5]
    return float(below["time"].iloc[0]) if not below.empty else None


def run(cfg: Config, ds: Dataset, panel: Panel) -> SurvivalReport:
    surv = panel.survival
    if surv.empty:
        return SurvivalReport(overall=pd.DataFrame(), notes=["No survival data."])

    durations = surv["duration_days"].values
    events = surv["event"].values.astype(int)
    entry = surv["entry_days"].values

    overall = _km(durations, events, entry)
    median = _median(overall)
    notes: list[str] = []
    censored_share = 1 - float(events.mean())
    log.info(
        "Kaplan-Meier: %s customers, %.1f%% right-censored, median lifetime %s",
        f"{len(surv):,}", 100 * censored_share,
        f"{median:.0f} days" if median else "not reached within the window",
    )
    if median is None:
        notes.append(
            "Median lifetime is not reached inside the observation window — more than "
            "half of customers are still active. Any 'average lifetime' quoted from "
            "this data would be an extrapolation, not a measurement."
        )
    if censored_share > 0.9:
        notes.append(
            f"{censored_share:.0%} of customers are censored. Survival estimates in the "
            f"tail rest on very few events and the confidence band widens accordingly."
        )
        log.warning(notes[-1])

    strata, logrank_rows = {}, []
    by = cfg.survival["by"] or cfg.segments
    entity_index = ds.entity.set_index(cfg.id_column)
    for column in by or []:
        if column not in entity_index.columns:
            log.warning("survival stratum %r is not a column on the entity table", column)
            continue
        values = surv["entity_id"].map(entity_index[column])
        counts = values.value_counts()
        keep = counts[counts >= 30].head(int(cfg.survival["max_strata"])).index
        if len(keep) < 2:
            continue
        curves = {}
        for level in keep:
            mask = (values == level).values
            curve = _km(durations[mask], events[mask], entry[mask])
            if not curve.empty:
                curves[str(level)] = curve
        if len(curves) >= 2:
            strata[column] = pd.concat(
                [c.assign(stratum=k) for k, c in curves.items()], ignore_index=True
            )
            logrank_rows.append(_logrank(durations, events, entry, values.values, keep, column))

    cox, diagnostics = _cox(cfg, ds, surv, notes)

    return SurvivalReport(
        overall=overall, strata=strata,
        logrank=pd.DataFrame([r for r in logrank_rows if r]),
        cox=cox, cox_diagnostics=diagnostics, median_lifetime=median, notes=notes,
    )


def _logrank(durations, events, entry, values, levels, column) -> dict | None:
    try:
        from lifelines.statistics import multivariate_logrank_test  # noqa: PLC0415
    except ImportError:
        return None
    mask = pd.Series(values).isin(levels).values
    try:
        result = multivariate_logrank_test(
            durations[mask], pd.Series(values)[mask].astype(str), events[mask]
        )
        return {
            "column": column, "test_statistic": float(result.test_statistic),
            "p_value": float(result.p_value), "n_strata": len(levels),
            "significant": bool(result.p_value < 0.05),
        }
    except Exception as exc:  # noqa: BLE001
        log.debug("log-rank for %s failed: %s", column, exc)
        return None


def _cox(cfg: Config, ds: Dataset, surv: pd.DataFrame, notes: list[str]):
    """Cox proportional hazards on entity-level attributes.

    Only static attributes go in. Time-varying features would need a counting-process
    layout, and a *current* value of a time-varying covariate is measured after some
    of the survival time it is supposed to explain — a subtle, common leak.
    """
    if not cfg.survival.get("cox", True):
        return pd.DataFrame(), {}
    try:
        from lifelines import CoxPHFitter  # noqa: PLC0415
    except ImportError:
        notes.append("lifelines is not installed; the Cox model was skipped.")
        return pd.DataFrame(), {}

    static_cols: list[str] = []
    for group in cfg.groups:
        if group.temporal == "static" and group.source == "entity":
            static_cols += [c.name for c in group.columns]
    entity_index = ds.entity.set_index(cfg.id_column)
    static_cols = [c for c in dict.fromkeys(static_cols) if c in entity_index.columns]
    if not static_cols:
        return pd.DataFrame(), {}

    df = surv[["entity_id", "duration_days", "event", "entry_days"]].copy()
    attrs = entity_index.loc[
        entity_index.index.intersection(df["entity_id"]), static_cols
    ]
    df = df.merge(attrs, left_on="entity_id", right_index=True, how="inner")
    df = df.drop(columns=["entity_id"])

    for col in static_cols:
        if df[col].dtype == object or df[col].dtype.name == "category":
            top = df[col].value_counts().head(8).index
            df[col] = df[col].where(df[col].isin(top), "__other__")
    use_entry = cfg.left_truncation == "keep_flagged" and df["entry_days"].max() > 0
    if not use_entry:
        df = df.drop(columns=["entry_days"])

    df = pd.get_dummies(df, drop_first=True, dummy_na=False)
    protected = {"duration_days", "event", "entry_days"}
    varying = df.nunique(dropna=False) > 1
    df = df.loc[:, [c for c in df.columns if c in protected or varying[c]]]
    df = df.dropna()
    if df.empty or "event" not in df.columns:
        return pd.DataFrame(), {}

    try:
        fitter = CoxPHFitter(penalizer=0.1)
        fitter.fit(
            df, duration_col="duration_days", event_col="event",
            entry_col="entry_days" if use_entry else None,
        )
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Cox model did not converge ({exc}).")
        log.warning(notes[-1])
        return pd.DataFrame(), {}

    summary = fitter.summary.reset_index().rename(columns={"index": "covariate"})
    summary = summary[
        ["covariate", "coef", "exp(coef)", "se(coef)",
         "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
    ].rename(
        columns={
            "exp(coef)": "hazard_ratio", "exp(coef) lower 95%": "hr_lower",
            "exp(coef) upper 95%": "hr_upper", "p": "p_value",
        }
    ).sort_values("p_value")

    diagnostics = {"concordance": float(fitter.concordance_index_), "left_truncated": use_entry}
    log.info(
        "Cox model: concordance %.3f on %d covariates%s",
        diagnostics["concordance"], len(summary),
        " (left-truncated)" if use_entry else "",
    )

    ph_violations = _ph_check(fitter, df)
    if ph_violations:
        diagnostics["ph_violations"] = ph_violations
        notes.append(
            "Proportional-hazards assumption is violated for: "
            + ", ".join(ph_violations[:5])
            + ". Their hazard ratios are an average over a changing effect — the "
            "ranking is still informative, the point estimates are not."
        )
        log.warning(notes[-1])
    return summary, diagnostics


def _ph_check(fitter, df: pd.DataFrame) -> list[str]:
    try:
        import warnings  # noqa: PLC0415

        from lifelines.statistics import proportional_hazard_test  # noqa: PLC0415

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = proportional_hazard_test(fitter, df, time_transform="rank")
        table = result.summary
        return sorted({str(i[0]) for i in table[table["p"] < 0.01].index})
    except Exception:  # noqa: BLE001
        return []

"""Randomised experiment analysis.

When a real experiment exists it outranks everything observational in this kit, so
the report leads with it. But randomisation is a claim, not a guarantee: the sample
ratio and pre-period balance checks run first, and a failed check is reported above
the effect rather than beneath it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .l01_config import Config
from .l02_data import Dataset
from .util.log import get_logger
from .l03_panel import Panel

log = get_logger("experiment")


@dataclass
class VariantResult:
    variant: str
    n: int
    churned: int
    churn_rate: float
    lift_pp: float | None = None
    lift_ci: tuple[float, float] | None = None
    p_value: float | None = None
    adjusted_lift_pp: float | None = None


@dataclass
class ExperimentReport:
    enabled: bool
    results: list[VariantResult] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    control: str = ""
    horizon_days: int = 0
    notes: list[str] = field(default_factory=list)
    trustworthy: bool = True

    def table(self) -> pd.DataFrame:
        if not self.results:
            return pd.DataFrame()
        return pd.DataFrame(
            [
                {
                    "variant": r.variant, "n": r.n, "churned": r.churned,
                    "churn_rate_pct": round(100 * r.churn_rate, 2),
                    "lift_pp": None if r.lift_pp is None else round(r.lift_pp, 2),
                    "ci_lower_pp": None if r.lift_ci is None else round(r.lift_ci[0], 2),
                    "ci_upper_pp": None if r.lift_ci is None else round(r.lift_ci[1], 2),
                    "p_value": None if r.p_value is None else round(r.p_value, 4),
                    "adjusted_lift_pp": (
                        None if r.adjusted_lift_pp is None else round(r.adjusted_lift_pp, 2)
                    ),
                }
                for r in self.results
            ]
        )

    def checks_table(self) -> pd.DataFrame:
        return pd.DataFrame(self.checks) if self.checks else pd.DataFrame()


def run(cfg: Config, ds: Dataset, panel: Panel) -> ExperimentReport:
    spec = cfg.experiment or {}
    if not spec.get("enabled"):
        return ExperimentReport(enabled=False, notes=["No experiment configured."])

    variant_col = spec.get("variant_column")
    if not variant_col or variant_col not in ds.entity.columns:
        return ExperimentReport(
            enabled=False,
            notes=[
                f"experiment.yaml is enabled but variant_column {variant_col!r} is not "
                f"a column on the entity table; the stage was skipped."
            ],
        )

    horizon = int(spec.get("horizon_days") or cfg.panel["horizon_days"])
    assignment_date = (
        pd.Timestamp(spec["assignment_date"]).normalize()
        if spec.get("assignment_date") else None
    )
    start = pd.Timestamp(spec["start_date"]).normalize() if spec.get("start_date") else None
    end = pd.Timestamp(spec["end_date"]).normalize() if spec.get("end_date") else None
    alpha = float(spec.get("alpha", 0.05))

    entity = ds.entity.copy()
    eligible = entity[variant_col].notna()
    notes: list[str] = []

    if assignment_date is not None:
        # Snapshot cohort: every customer already active on one date was assigned.
        # Signup date is irrelevant to eligibility here — filtering on it, as a
        # rolling-enrolment experiment would, excludes the entire cohort.
        eligible &= entity[cfg.start_date_column] <= assignment_date
        if cfg.target_mode == "event_date":
            already_gone = entity[cfg.event_date_column] <= assignment_date
            dropped = int((eligible & already_gone.fillna(False)).sum())
            if dropped:
                notes.append(
                    f"{dropped:,} assigned customer(s) had already churned on or "
                    f"before {assignment_date.date()} and were excluded — they were "
                    f"never at risk during the experiment."
                )
                log.warning(notes[-1])
            eligible &= ~already_gone.fillna(False)
    else:
        # Rolling enrolment: customers join the experiment as they sign up.
        if start is not None:
            eligible &= entity[cfg.start_date_column] >= start
        if end is not None:
            eligible &= entity[cfg.start_date_column] <= end

    cohort = entity[eligible].copy()
    if cohort.empty:
        hint = (
            "Set experiment.assignment_date if everyone was assigned on one date "
            "(a snapshot cohort); start_date/end_date filter on SIGNUP date, which "
            "excludes customers who signed up before the experiment began."
        )
        return ExperimentReport(
            enabled=False,
            notes=[f"No customers fall inside the experiment window; stage skipped. {hint}"],
        )

    # Assignment time is the reference point, so the outcome window is the same
    # length for everyone regardless of when they entered.
    if assignment_date is not None:
        assign = pd.Series(assignment_date, index=cohort.index)
    elif start is None:
        assign = cohort[cfg.start_date_column]
    else:
        assign = cohort[cfg.start_date_column].clip(lower=start)
    obs_end = panel.timeline["observation_end"]
    window_end = assign + pd.Timedelta(days=horizon)

    incomplete = int((window_end > obs_end).sum())
    if incomplete:
        notes.append(
            f"{incomplete:,} of {len(cohort):,} assigned customers do not yet have a "
            f"full {horizon}-day window inside the data. They are excluded rather than "
            f"counted as retained — including them would bias the effect toward the "
            f"variant that was assigned earlier."
        )
        log.warning(notes[-1])
        keep = window_end <= obs_end
        cohort, assign, window_end = cohort[keep], assign[keep], window_end[keep]

    if cohort.empty:
        return ExperimentReport(
            enabled=False,
            notes=notes + [f"No customer has a complete {horizon}-day outcome window yet."],
        )

    if cfg.target_mode == "event_date":
        churn_date = cohort[cfg.event_date_column]
        churned = (churn_date.notna() & (churn_date > assign) & (churn_date <= window_end))
    else:
        churned = cohort[cfg.label_column] == cfg.churn_value
    cohort = cohort.assign(_churned=churned.astype(int))

    control = spec.get("control_value")
    variants = list(cohort[variant_col].astype(str).unique())
    control = str(control) if control is not None and str(control) in variants else variants[0]
    cohort[variant_col] = cohort[variant_col].astype(str)

    report = ExperimentReport(
        enabled=True, control=control, horizon_days=horizon, notes=notes
    )
    checks_cfg = spec.get("checks") or {}
    if checks_cfg.get("sample_ratio_mismatch", True):
        report.checks.append(_srm_check(cohort, variant_col))
    if checks_cfg.get("pre_period_balance", True):
        report.checks.extend(_balance_checks(cfg, cohort, variant_col, spec, control))
    report.trustworthy = all(c.get("passed", True) for c in report.checks)
    if not report.trustworthy:
        failed = [c["check"] for c in report.checks if not c.get("passed", True)]
        report.notes.append(
            f"Randomisation checks failed ({', '.join(failed)}). The comparison below is "
            f"no longer a clean experiment — read it as observational."
        )
        log.warning(report.notes[-1])

    base = cohort[cohort[variant_col] == control]
    p0, n0 = float(base["_churned"].mean()), len(base)
    report.results.append(
        VariantResult(variant=control, n=n0, churned=int(base["_churned"].sum()), churn_rate=p0)
    )

    covariates = [c for c in (spec.get("covariates") or []) if c in cohort.columns]
    for variant in sorted(v for v in cohort[variant_col].unique() if v != control):
        arm = cohort[cohort[variant_col] == variant]
        p1, n1 = float(arm["_churned"].mean()), len(arm)
        diff = p1 - p0
        se = np.sqrt(p1 * (1 - p1) / max(n1, 1) + p0 * (1 - p0) / max(n0, 1))
        z = diff / se if se > 0 else 0.0
        p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
        adjusted = _covariate_adjusted(cohort, variant_col, variant, control, covariates)
        report.results.append(
            VariantResult(
                variant=variant, n=n1, churned=int(arm["_churned"].sum()), churn_rate=p1,
                lift_pp=100 * diff,
                lift_ci=(100 * (diff - 1.96 * se), 100 * (diff + 1.96 * se)),
                p_value=p_value,
                adjusted_lift_pp=None if adjusted is None else 100 * adjusted,
            )
        )
        log.info(
            "%s vs %s: %+.2f pp churn (95%% CI %+.2f to %+.2f, p=%.4f, n=%s)",
            variant, control, 100 * diff, 100 * (diff - 1.96 * se),
            100 * (diff + 1.96 * se), p_value, f"{n1:,}",
        )
        if p_value >= alpha:
            mde = 100 * 2.8 * se
            report.notes.append(
                f"{variant}: no significant difference. With this sample the smallest "
                f"detectable effect is about {mde:.2f} pp — a real effect smaller than "
                f"that would not have shown up here."
            )
    return report


def _srm_check(cohort: pd.DataFrame, variant_col: str) -> dict:
    counts = cohort[variant_col].value_counts()
    expected = np.full(len(counts), counts.sum() / len(counts))
    chi2, p_value = stats.chisquare(counts.values, expected)
    passed = bool(p_value >= 0.001)
    return {
        "check": "sample_ratio_mismatch",
        "passed": passed,
        "p_value": round(float(p_value), 6),
        "detail": (
            f"Assignment counts: {counts.to_dict()}. "
            + ("Consistent with equal allocation." if passed else
               "Significantly uneven — assignment or logging is broken, and any effect "
               "measured here may be a selection artifact rather than the treatment.")
        ),
    }


def _balance_checks(
    cfg: Config, cohort: pd.DataFrame, variant_col: str, spec: dict, control: str
) -> list[dict]:
    candidates = [c for c in (spec.get("covariates") or []) if c in cohort.columns]
    if not candidates:
        candidates = [c for c in cfg.segments if c in cohort.columns]
    out = []
    for col in candidates[:10]:
        series = cohort[col]
        try:
            if pd.api.types.is_numeric_dtype(series):
                groups = [g[col].dropna().values for _, g in cohort.groupby(variant_col)]
                groups = [g for g in groups if len(g) > 5]
                if len(groups) < 2:
                    continue
                _, p_value = stats.f_oneway(*groups)
            else:
                table = pd.crosstab(cohort[variant_col], series.astype(str))
                if table.shape[1] < 2:
                    continue
                _, p_value, _, _ = stats.chi2_contingency(table)
        except Exception:  # noqa: BLE001
            continue
        passed = bool(p_value >= 0.01)
        out.append(
            {
                "check": f"pre_period_balance:{col}",
                "passed": passed,
                "p_value": round(float(p_value), 6),
                "detail": (
                    "Balanced across variants." if passed
                    else f"{col} differs across variants before treatment — the arms "
                         f"were not comparable to begin with."
                ),
            }
        )
    return out


def _covariate_adjusted(
    cohort: pd.DataFrame, variant_col: str, variant: str, control: str, covariates: list[str]
) -> float | None:
    """CUPED-style adjustment: same estimand, less variance, pre-treatment controls only."""
    if not covariates:
        return None
    subset = cohort[cohort[variant_col].isin([variant, control])].copy()
    design = pd.get_dummies(subset[covariates], drop_first=True, dummy_na=True)
    design = design.select_dtypes(include=[np.number]).astype(float)
    design = design.loc[:, design.std() > 0]
    if design.empty:
        return None
    treat = (subset[variant_col] == variant).astype(float).values
    y = subset["_churned"].values.astype(float)
    X = np.column_stack([np.ones(len(subset)), treat, design.fillna(design.mean()).values])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(beta[1])
    except np.linalg.LinAlgError:
        return None

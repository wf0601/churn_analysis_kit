"""Leakage detection.

Two complementary layers, because neither catches the other's failures:

  structural  — reasoning about time and definitions. Catches the outcome column
                being used as a feature, censored rows being scored as retained,
                survivorship truncation, feature windows that overrun the embargo.
                These are provable before any model is fit.

  statistical — reasoning about the numbers. Catches proxies that no denylist
                anticipates: a field only populated for churners, a column whose
                mere presence gives away the answer, an internal id that encodes
                signup batch. These are only visible once features meet labels.

Findings are levelled. BLOCK means the result is not trustworthy with that feature
present; under the default policy the feature is quarantined (dropped) and the run
continues, loudly. WARN means a human has to decide. INFO is bookkeeping.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .l01_config import Config
from .util.errors import LeakageError  # re-exported for pipeline use
from .l04_features import FeatureMatrix
from .util.log import get_logger
from .l03_panel import Panel

log = get_logger("leakage")

BLOCK, WARN, INFO = "BLOCK", "WARN", "INFO"
_ORDER = {BLOCK: 0, WARN: 1, INFO: 2}


@dataclass
class Finding:
    level: str
    code: str
    message: str
    columns: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    remedy: str = ""

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "columns": ", ".join(self.columns[:12]),
            "n_columns": len(self.columns),
            "remedy": self.remedy,
            **{f"evidence_{k}": v for k, v in self.evidence.items()},
        }


@dataclass
class LeakageReport:
    findings: list[Finding] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)
        level_log = {BLOCK: log.error, WARN: log.warning, INFO: log.info}[finding.level]
        level_log("[%s] %s", finding.code, finding.message)

    def by_level(self, level: str) -> list[Finding]:
        return [f for f in self.findings if f.level == level]

    @property
    def blocked(self) -> bool:
        return bool(self.by_level(BLOCK))

    def to_frame(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(columns=["level", "code", "message", "columns"])
        ordered = sorted(self.findings, key=lambda f: (_ORDER[f.level], f.code))
        return pd.DataFrame([f.as_dict() for f in ordered])

    def summary(self) -> str:
        n_b, n_w = len(self.by_level(BLOCK)), len(self.by_level(WARN))
        if not n_b and not n_w:
            return "No leakage findings."
        return f"{n_b} blocking finding(s), {n_w} warning(s), {len(self.quarantined)} feature(s) quarantined."


# --------------------------------------------------------------------------- #
# structural checks
# --------------------------------------------------------------------------- #
def audit_structure(
    cfg: Config, panel: Panel, fm: FeatureMatrix, report: LeakageReport
) -> None:
    _check_denylist(cfg, fm, report)
    _check_target_columns(cfg, fm, report)
    _check_embargo(cfg, panel, fm, report)
    _check_windows(cfg, panel, report)
    _check_censoring(cfg, panel, report)
    _check_truncation(cfg, panel, report)
    _check_point_in_time(fm, report)


def _check_denylist(cfg: Config, fm: FeatureMatrix, report: LeakageReport) -> None:
    patterns = [re.compile(p, re.IGNORECASE) for p in cfg.leakage["denylist_patterns"]]
    allow = {c.lower() for c in cfg.leakage["allowlist_columns"]}
    hits: dict[str, str] = {}

    for name in fm.meta:
        if name not in fm.X.columns:
            continue
        # A feature name is group__[filter__]column__agg_window. Check every segment
        # as well as the whole: an anchored pattern like `(^|_)end_date$` only matches
        # the bare column, and a filter name like `cancelled` only appears mid-name.
        segments = name.split("__")
        if {name.lower(), *(s.lower() for s in segments)} & allow:
            continue
        for pattern in patterns:
            if pattern.search(name) or any(pattern.search(s) for s in segments):
                hits[name] = pattern.pattern
                break

    if hits:
        report.add(
            Finding(
                level=BLOCK,
                code="OUTCOME_DERIVED_FEATURE",
                message=(
                    f"{len(hits)} feature(s) are named like the outcome or its clerical "
                    f"aftermath: {', '.join(sorted(hits)[:6])}"
                    f"{' …' if len(hits) > 6 else ''}. A column recording that a "
                    f"customer cancelled cannot be used to predict that they will."
                ),
                columns=sorted(hits),
                evidence={"matched_patterns": sorted(set(hits.values()))},
                remedy=(
                    "If one of these is genuinely knowable at the prediction date, add "
                    "its exact name to leakage.allowlist_columns in pipeline.yaml."
                ),
            )
        )


def _check_target_columns(cfg: Config, fm: FeatureMatrix, report: LeakageReport) -> None:
    target_cols = {c for c in (cfg.event_date_column, cfg.label_column) if c}
    hits = [
        name for name in fm.X.columns
        if "__" in name and set(name.split("__")) & target_cols
    ]
    if hits:
        report.add(
            Finding(
                level=BLOCK,
                code="TARGET_COLUMN_AS_FEATURE",
                message=(
                    f"The target column itself is being used as a feature: "
                    f"{', '.join(hits)}. This is the definition of the label."
                ),
                columns=hits,
                remedy="Remove the column from feature.yaml.",
            )
        )


def _check_embargo(
    cfg: Config, panel: Panel, fm: FeatureMatrix, report: LeakageReport
) -> None:
    if panel.embargo_days > 0:
        return
    strict = [
        n for n, m in fm.meta.items()
        if m.leakage_review == "strict" and n in fm.X.columns
    ]
    report.add(
        Finding(
            level=WARN if not strict else BLOCK,
            code="NO_EMBARGO",
            message=(
                "panel.embargo_days is 0, so features run right up to the prediction "
                "date. Signals from the final days before churn — a failed payment, an "
                "angry ticket, usage falling to zero — are symptoms of a decision "
                "already taken. They will top the driver ranking and they are not "
                "levers."
                + (f" {len(strict)} feature(s) are in groups marked leakage_review: "
                   f"strict, which makes this blocking." if strict else "")
            ),
            columns=strict,
            evidence={"embargo_days": 0},
            remedy="Set panel.embargo_days to 7-14 in pipeline.yaml and re-run.",
        )
    )


def _check_windows(cfg: Config, panel: Panel, report: LeakageReport) -> None:
    horizon = panel.horizon_days
    long_groups = [
        g.name for g in cfg.groups
        if g.is_time_varying and (g.aggregation_window_days or 0) > 4 * horizon
    ]
    if long_groups:
        report.add(
            Finding(
                level=INFO,
                code="LONG_FEATURE_WINDOW",
                message=(
                    f"Feature group(s) {', '.join(long_groups)} aggregate over a window "
                    f"more than 4x the {horizon}-day horizon. Not leakage, but rows for "
                    f"the same customer at nearby prediction dates will share most of "
                    f"their history, so effective sample size is well below row count."
                ),
                columns=long_groups,
            )
        )


def _check_censoring(cfg: Config, panel: Panel, report: LeakageReport) -> None:
    obs_end = panel.timeline["observation_end"]
    latest = max(panel.snapshot_dates)
    last_label_end = latest + pd.Timedelta(days=panel.horizon_days)
    if last_label_end > obs_end:
        report.add(
            Finding(
                level=BLOCK,
                code="CENSORED_AS_RETAINED",
                message=(
                    f"The last prediction date ({latest.date()}) has a label window "
                    f"ending {last_label_end.date()}, past observation_end_date "
                    f"({obs_end.date()}). Customers there are censored, not retained."
                ),
                evidence={"overrun_days": int((last_label_end - obs_end).days)},
                remedy="The kit normally trims these automatically; a manual "
                       "panel.snapshot_end in pipeline.yaml is overriding the trim.",
            )
        )
    else:
        report.add(
            Finding(
                level=INFO,
                code="CENSORING_HANDLED",
                message=(
                    f"Right-censoring handled: every label window closes on or before "
                    f"observation_end_date ({obs_end.date()}). Customers still active at "
                    f"the end of the data are censored in the survival model, not "
                    f"counted as retained."
                ),
            )
        )


def _check_truncation(cfg: Config, panel: Panel, report: LeakageReport) -> None:
    if "left_truncated" not in panel.frame.columns:
        return
    share = float(panel.frame["left_truncated"].mean())
    if share <= 0:
        return
    if cfg.left_truncation == "drop":
        report.add(
            Finding(
                level=INFO,
                code="SURVIVORSHIP_HANDLED",
                message=(
                    f"{share:.1%} of panel rows came from customers who predate the "
                    f"observation window; they were dropped per survivorship.yaml."
                ),
            )
        )
    else:
        report.add(
            Finding(
                level=WARN,
                code="SURVIVORSHIP_BIAS",
                message=(
                    f"{share:.1%} of panel rows belong to customers who started before "
                    f"observation_starting_date. Their early-churning peers were never "
                    f"recorded, so this group is pre-selected for loyalty. Long tenure "
                    f"will look protective when it is partly an artifact of who "
                    f"survived long enough to appear in the export."
                ),
                evidence={"share_of_rows": round(share, 4)},
                remedy="Set survivorship.yaml: left_truncation: drop to exclude them.",
            )
        )


def _check_point_in_time(fm: FeatureMatrix, report: LeakageReport) -> None:
    stale = [
        n for n, m in fm.meta.items()
        if n in fm.X.columns and not m.point_in_time
    ]
    if stale:
        report.add(
            Finding(
                level=WARN,
                code="NOT_POINT_IN_TIME",
                message=(
                    f"{len(stale)} feature(s) are read from the entity table as-of "
                    f"export rather than as-of the prediction date. If any of these "
                    f"fields is rewritten when a customer churns (plan downgraded to "
                    f"'none', mrr zeroed, payment method cleared), the model is reading "
                    f"the future and no statistical test here can prove it isn't."
                ),
                columns=sorted(stale),
                remedy=(
                    "Move these to an event log, or add valid_from_column/"
                    "valid_to_column to data.yaml so versions can be selected by date."
                ),
            )
        )


# --------------------------------------------------------------------------- #
# statistical checks
# --------------------------------------------------------------------------- #
def audit_statistics(
    cfg: Config, fm: FeatureMatrix, y: pd.Series, report: LeakageReport, seed: int = 42
) -> pd.DataFrame:
    """Screen every feature on its own against the label. Returns the scan table."""
    block_at = float(cfg.leakage["single_feature_auc_block"])
    warn_at = float(cfg.leakage["single_feature_auc_warn"])
    miss_at = float(cfg.leakage["missingness_auc_warn"])
    id_ratio = float(cfg.leakage["id_cardinality_ratio_warn"])

    if y.nunique() < 2:
        report.add(
            Finding(
                level=BLOCK, code="SINGLE_CLASS",
                message="The label has only one class; nothing can be learned or audited.",
            )
        )
        return pd.DataFrame()

    rows = []
    for name in fm.X.columns:
        meta = fm.meta[name]
        series = fm.X[name]
        auc = (
            _numeric_auc(series, y) if meta.kind == "numeric"
            else _categorical_auc(series, y, seed)
        )
        miss_auc = _missingness_auc(series, y)
        n_unique = int(series.nunique(dropna=True))
        rows.append(
            {
                "feature": name, "group": meta.group, "kind": meta.kind,
                "univariate_auc": auc, "missingness_auc": miss_auc,
                "n_unique": n_unique, "null_share": float(series.isna().mean()),
                # Continuous measurements are near-unique by nature — money and
                # durations are not identifiers, so they are exempt from the
                # cardinality check rather than warned about every run.
                "cardinality_ratio": (
                    0.0 if _is_continuous(series) else n_unique / max(len(series), 1)
                ),
                "leakage_review": meta.leakage_review,
            }
        )
    scan = pd.DataFrame(rows).sort_values("univariate_auc", ascending=False)

    _flag_univariate(scan, report, block_at, warn_at)
    _flag_missingness(scan, report, miss_at)
    _flag_degenerate(scan, report, id_ratio)
    return scan


def _is_continuous(series: pd.Series) -> bool:
    """True for real-valued measurements, false for counts, codes and categories."""
    if series.dtype == object:
        return False
    values = series.dropna()
    if values.empty:
        return False
    return bool((values != values.round()).mean() > 0.01)


def _numeric_auc(series: pd.Series, y: pd.Series) -> float:
    mask = series.notna()
    if mask.sum() < 20 or y[mask].nunique() < 2 or series[mask].nunique() < 2:
        return float("nan")
    auc = roc_auc_score(y[mask], series[mask])
    return float(max(auc, 1 - auc))


def _categorical_auc(series: pd.Series, y: pd.Series, seed: int) -> float:
    """Out-of-fold target encoding, so a high-cardinality column cannot fake signal."""
    mask = series.notna()
    if mask.sum() < 20 or y[mask].nunique() < 2:
        return float("nan")
    values = series[mask].astype(str).values
    target = y[mask].values
    encoded = np.full(len(target), float(target.mean()))
    splitter = StratifiedKFold(n_splits=min(5, int(min(np.bincount(target)))), shuffle=True, random_state=seed)
    try:
        for train_idx, test_idx in splitter.split(values.reshape(-1, 1), target):
            means = pd.Series(target[train_idx]).groupby(values[train_idx]).mean()
            encoded[test_idx] = (
                pd.Series(values[test_idx]).map(means).fillna(target[train_idx].mean()).values
            )
    except ValueError:
        return float("nan")
    if len(np.unique(encoded)) < 2:
        return float("nan")
    auc = roc_auc_score(target, encoded)
    return float(max(auc, 1 - auc))


def _missingness_auc(series: pd.Series, y: pd.Series) -> float:
    null_share = series.isna().mean()
    if null_share in (0.0,) or null_share > 0.999:
        return float("nan")
    auc = roc_auc_score(y, series.isna().astype(int))
    return float(max(auc, 1 - auc))


def _flag_univariate(
    scan: pd.DataFrame, report: LeakageReport, block_at: float, warn_at: float
) -> None:
    if scan.empty:
        return
    strict_warn = (scan["leakage_review"] == "strict") & (scan["univariate_auc"] >= warn_at)
    blocking = scan[(scan["univariate_auc"] >= block_at) | strict_warn]
    warning = scan[
        (scan["univariate_auc"] >= warn_at) & (scan["univariate_auc"] < block_at)
        & ~strict_warn
    ]

    if not blocking.empty:
        detail = ", ".join(
            f"{r.feature} (AUC {r.univariate_auc:.3f})"
            for r in blocking.head(6).itertuples()
        )
        report.add(
            Finding(
                level=BLOCK,
                code="SINGLE_FEATURE_SEPARATION",
                message=(
                    f"{len(blocking)} feature(s) separate churners on their own at or "
                    f"above the blocking threshold: {detail}. A single pre-churn "
                    f"behavioural signal essentially never does this; a field that is "
                    f"written when the customer leaves does."
                ),
                columns=blocking["feature"].tolist(),
                evidence={"threshold": block_at, "max_auc": round(float(blocking["univariate_auc"].max()), 4)},
                remedy=(
                    "Check when each field is written relative to cancellation. If it is "
                    "populated by the churn process, drop it from feature.yaml. If it is "
                    "legitimately available at the prediction date, allowlist it."
                ),
            )
        )
    if not warning.empty:
        detail = ", ".join(
            f"{r.feature} ({r.univariate_auc:.3f})" for r in warning.head(6).itertuples()
        )
        report.add(
            Finding(
                level=WARN,
                code="STRONG_SINGLE_FEATURE",
                message=(
                    f"{len(warning)} feature(s) are unusually predictive alone: {detail}. "
                    f"Could be a genuine dominant driver, could be a partial leak. Worth "
                    f"one minute of thought each about when the field is populated."
                ),
                columns=warning["feature"].tolist(),
                evidence={"threshold": warn_at},
            )
        )


def _flag_missingness(scan: pd.DataFrame, report: LeakageReport, miss_at: float) -> None:
    hits = scan[scan["missingness_auc"] >= miss_at]
    if hits.empty:
        return
    detail = ", ".join(
        f"{r.feature} ({r.missingness_auc:.3f})" for r in hits.head(6).itertuples()
    )
    report.add(
        Finding(
            level=WARN,
            code="MISSINGNESS_PREDICTS_TARGET",
            message=(
                f"For {len(hits)} feature(s), whether the value is present predicts "
                f"churn better than the value itself normally would: {detail}. That "
                f"usually means the record is created or deleted by a downstream "
                f"process that already knows the outcome."
            ),
            columns=hits["feature"].tolist(),
            evidence={"threshold": miss_at},
            remedy="Trace what writes the field. If a churn workflow touches it, drop it.",
        )
    )


def _flag_degenerate(scan: pd.DataFrame, report: LeakageReport, id_ratio: float) -> None:
    # A single distinct value alongside nulls is not constant — it is a presence
    # indicator ("had a cancellation enquiry in the window"), and the trees read the
    # null as its own branch. Only drop columns that genuinely never vary.
    constant = scan[
        (scan["n_unique"] == 0)
        | ((scan["n_unique"] <= 1) & (scan["null_share"] <= 0))
    ]["feature"].tolist()
    if constant:
        report.add(
            Finding(
                level=INFO, code="CONSTANT_FEATURE",
                message=f"{len(constant)} feature(s) never vary; dropped.",
                columns=constant,
            )
        )
    identifiers = scan[
        (scan["cardinality_ratio"] >= id_ratio) & (scan["n_unique"] > 50)
    ]["feature"].tolist()
    if identifiers:
        report.add(
            Finding(
                level=WARN, code="IDENTIFIER_LIKE_FEATURE",
                message=(
                    f"{len(identifiers)} feature(s) are near-unique per row: "
                    f"{', '.join(identifiers[:6])}. Identifiers often encode signup "
                    f"order or account batch, which correlates with cohort and therefore "
                    f"with churn — predictive, meaningless, and untransferable."
                ),
                columns=identifiers,
            )
        )


def audit_splits(
    train_ids: pd.Series, test_ids: pd.Series, report: LeakageReport
) -> None:
    overlap = set(train_ids) & set(test_ids)
    if not overlap:
        return
    share = len(overlap) / max(test_ids.nunique(), 1)
    report.add(
        Finding(
            level=INFO if share < 0.5 else WARN,
            code="ENTITY_SPANS_SPLIT",
            message=(
                f"{len(overlap):,} customers ({share:.0%} of the test set) appear in "
                f"both train and test at different prediction dates. That is expected "
                f"for a rolling panel and mirrors production, but it does make the "
                f"holdout score optimistic relative to a brand-new customer."
            ),
            evidence={"n_overlapping_entities": len(overlap)},
            remedy="Set panel.max_snapshots_per_entity: 1 for a stricter read.",
        )
    )


def audit_generalisation(
    cv_auc: float, holdout_auc: float, report: LeakageReport
) -> None:
    """A large CV-to-holdout drop is the classic signature of a time-shifted feature.

    Cross-validation resamples within the training period, so a feature whose meaning
    depends on when it was measured looks excellent there and falls apart on later
    dates. Distribution shift produces the same pattern, and the two are worth
    separating by hand — but either way the headline CV number is not the truth.
    """
    if not (np.isfinite(cv_auc) and np.isfinite(holdout_auc)):
        return
    gap = cv_auc - holdout_auc
    if gap < 0.10:
        return
    report.add(
        Finding(
            level=WARN if gap < 0.20 else BLOCK,
            code="CV_HOLDOUT_GAP",
            message=(
                f"Cross-validated AUC on the training period is {cv_auc:.3f} but "
                f"out-of-time AUC is {holdout_auc:.3f}, a drop of {gap:.3f}. Either a "
                f"feature means something different at different points in time (a "
                f"value computed at export date behaves exactly like this), or the "
                f"churn regime genuinely shifted between the two periods."
            ),
            evidence={"cv_auc": round(cv_auc, 4), "holdout_auc": round(holdout_auc, 4)},
            remedy=(
                "Compare the driver ranking against the univariate scan: a feature "
                "with a high standalone AUC that the model leans on hard is the first "
                "place to look."
            ),
        )
    )


def audit_model_performance(
    cfg: Config, auc: float, baseline_rate: float, report: LeakageReport
) -> None:
    threshold = float(cfg.leakage["model_auc_block"])
    if auc >= threshold:
        report.add(
            Finding(
                level=BLOCK,
                code="MODEL_TOO_GOOD",
                message=(
                    f"Out-of-time AUC is {auc:.4f}, at or above the {threshold} "
                    f"plausibility ceiling. Churn is a human decision made partly for "
                    f"reasons never recorded in a database; models this good are almost "
                    f"always reading the answer somewhere."
                ),
                evidence={"holdout_auc": round(auc, 4), "base_rate": round(baseline_rate, 4)},
                remedy=(
                    "Look at the top drivers in the report and ask, for each: was this "
                    "value knowable before the customer decided to leave?"
                ),
            )
        )
    elif auc >= 0.95:
        report.add(
            Finding(
                level=WARN, code="MODEL_SUSPICIOUSLY_GOOD",
                message=(
                    f"Out-of-time AUC is {auc:.4f}. Possible with strong behavioural "
                    f"data, but high enough to warrant reviewing the top drivers for "
                    f"anything that is a consequence of churn rather than a cause."
                ),
                evidence={"holdout_auc": round(auc, 4)},
            )
        )


# --------------------------------------------------------------------------- #
# enforcement
# --------------------------------------------------------------------------- #
def enforce(cfg: Config, fm: FeatureMatrix, report: LeakageReport) -> None:
    """Apply the configured policy to BLOCK findings that name specific columns."""
    doomed: list[str] = []
    for finding in report.by_level(BLOCK):
        doomed.extend(c for c in finding.columns if c in fm.X.columns)
    for finding in report.findings:
        if finding.code == "CONSTANT_FEATURE":
            doomed.extend(c for c in finding.columns if c in fm.X.columns)
    doomed = sorted(set(doomed))

    # A derived feature built on a leaking input carries the same information, so
    # dropping the input alone would leave the leak in place behind a ratio.
    dependents = [c for c in fm.dependents_of(doomed) if c in fm.X.columns]
    if dependents:
        report.add(
            Finding(
                level=BLOCK,
                code="DERIVED_FROM_BLOCKED_FEATURE",
                message=(
                    f"{len(dependents)} derived feature(s) are computed from a blocked "
                    f"input and are quarantined with it: {', '.join(dependents[:6])}"
                    f"{' …' if len(dependents) > 6 else ''}."
                ),
                columns=dependents,
                remedy="Rewrite the expression in feature.yaml without the blocked input.",
            )
        )
        doomed = sorted(set(doomed) | set(dependents))

    if not doomed:
        if report.blocked and cfg.leakage["on_block"] == "fail":
            raise LeakageError(
                "Blocking leakage findings that cannot be fixed by dropping a column:\n"
                + "\n".join(f"  [{f.code}] {f.message}" for f in report.by_level(BLOCK))
            )
        return

    if cfg.leakage["on_block"] == "fail":
        raise LeakageError(
            f"Leakage policy is 'fail' and {len(doomed)} feature(s) were blocked:\n"
            + "\n".join(f"  - {c}" for c in doomed)
            + "\n\nFix feature.yaml, or set leakage.on_block: quarantine to drop them "
              "automatically and continue."
        )

    fm.drop(doomed)
    report.quarantined = doomed
    log.error(
        "QUARANTINED %d feature(s) — they are excluded from every downstream stage: %s",
        len(doomed), ", ".join(doomed[:10]) + (" …" if len(doomed) > 10 else ""),
    )
    if fm.X.shape[1] == 0:
        raise LeakageError(
            "Quarantine removed every feature. The configured feature set is entirely "
            "outcome-derived — see the findings above."
        )

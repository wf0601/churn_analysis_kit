"""Causal effect estimation for the levers you actually control.

The model in drivers.py answers "what predicts churn". This module answers a
different question — "what would happen to churn if we changed this" — and the two
routinely disagree. Cancellation-page visits predict churn almost perfectly and
blocking the page prevents nothing.

Method: cross-fitted augmented inverse-probability weighting (AIPW). Doubly robust,
so the estimate survives misspecification of either the outcome model or the
propensity model. Cross-fitting keeps the nuisance models from overfitting into the
effect estimate.

What this cannot do: rule out an unmeasured confounder. Everything here is honest
about that, and the refutation tests are falsification checks, not proofs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from .l01_config import Config
from .l04_features import FeatureMatrix
from .util.log import get_logger
from .l07_model import _preprocessor

log = get_logger("causal")


@dataclass
class CausalEstimate:
    name: str
    treatment_feature: str
    definition: str
    n: int
    n_treated: int
    treated_share: float
    ate: float                       # risk difference, probability scale
    ate_ci: tuple[float, float]
    se: float
    p_value: float
    naive_difference: float          # unadjusted, for contrast
    confounders_used: int
    overlap_share: float
    max_smd_before: float
    max_smd_after: float
    refutations: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    valid: bool = True

    def as_row(self) -> dict:
        return {
            "treatment": self.name,
            "feature": self.treatment_feature,
            "definition": self.definition,
            "n": self.n,
            "treated_share": round(self.treated_share, 4),
            "naive_difference_pp": round(100 * self.naive_difference, 2),
            "causal_ate_pp": round(100 * self.ate, 2),
            "ci_lower_pp": round(100 * self.ate_ci[0], 2),
            "ci_upper_pp": round(100 * self.ate_ci[1], 2),
            "p_value": round(self.p_value, 4),
            "confounders": self.confounders_used,
            "overlap_share": round(self.overlap_share, 3),
            "max_smd_after": round(self.max_smd_after, 3),
            "placebo_ate_pp": (
                round(100 * self.refutations["placebo"]["ate"], 2)
                if "placebo" in self.refutations else None
            ),
            "subset_ate_pp": (
                round(100 * self.refutations["subset"]["ate"], 2)
                if "subset" in self.refutations else None
            ),
            "verdict": self.verdict,
        }

    @property
    def verdict(self) -> str:
        if not self.valid:
            return "not estimable"
        if self.p_value >= 0.05:
            return "no detectable effect"
        failed = [k for k, v in self.refutations.items() if not v.get("passed", True)]
        if failed:
            return f"effect found but failed {', '.join(failed)} check"
        return "effect supported"


@dataclass
class CausalReport:
    estimates: list[CausalEstimate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)

    def table(self) -> pd.DataFrame:
        if not self.estimates:
            return pd.DataFrame()
        return pd.DataFrame([e.as_row() for e in self.estimates])


# --------------------------------------------------------------------------- #
def run(cfg: Config, fm: FeatureMatrix, frame: pd.DataFrame) -> CausalReport:
    report = CausalReport()
    spec = cfg.causal or {}
    treatments = spec.get("treatments") or []
    if not treatments:
        report.notes.append("No treatments defined in causal.yaml; stage skipped.")
        log.info(report.notes[-1])
        return report

    estimator_cfg = spec.get("estimator") or {}
    refute_cfg = spec.get("refutations") or {}
    conf_cfg = spec.get("confounders") or {}
    y = frame["label"].values.astype(int)

    for entry in treatments:
        name = entry.get("name") or entry.get("feature")
        feature = entry.get("feature")
        if not feature or feature not in fm.X.columns:
            reason = (
                f"feature {feature!r} is not available"
                + (" (it was quarantined as leaky)" if feature else "")
                if feature else "no `feature` given"
            )
            report.skipped.append({"treatment": name, "reason": reason})
            log.warning("causal: skipping %s — %s", name, reason)
            continue

        treated, definition = _binarise(fm.X[feature], entry)
        if treated is None:
            report.skipped.append(
                {"treatment": name, "reason": f"could not binarise ({definition})"}
            )
            log.warning("causal: skipping %s — %s", name, definition)
            continue

        confounders = _select_confounders(cfg, fm, feature, conf_cfg)
        if not confounders:
            report.skipped.append({"treatment": name, "reason": "no confounders left after exclusions"})
            continue

        estimate = _estimate(
            cfg, fm, confounders, treated, y, name, feature, definition, estimator_cfg
        )
        if estimate.valid and refute_cfg:
            _refute(cfg, fm, confounders, treated, y, estimate, estimator_cfg, refute_cfg)
        report.estimates.append(estimate)
        if estimate.valid:
            log.info(
                "%s: ATE %+.2f pp (95%% CI %+.2f to %+.2f, p=%.3f) vs naive %+.2f pp — %s",
                name, 100 * estimate.ate, 100 * estimate.ate_ci[0],
                100 * estimate.ate_ci[1], estimate.p_value,
                100 * estimate.naive_difference, estimate.verdict,
            )
        else:
            log.warning(
                "%s: not estimable. %s", name,
                " ".join(estimate.warnings) or "insufficient overlap between arms.",
            )

    if conf_cfg.get("mode", "auto") == "auto":
        report.notes.append(
            "Confounders were selected automatically: every other pre-T feature. That "
            "set cannot distinguish a confounder from a mediator. Any variable that "
            "churn risk causes, rather than causes churn, will absorb part of the "
            "effect and bias the estimate toward zero. Review causal.yaml: "
            "confounders.exclude."
        )
    return report


# --------------------------------------------------------------------------- #
def _binarise(series: pd.Series, entry: dict) -> tuple[np.ndarray | None, str]:
    kind = entry.get("type", "threshold")
    values = pd.to_numeric(series, errors="coerce")
    if kind == "binary":
        treated = (values > 0).astype(int)
        definition = f"{series.name} > 0"
    elif kind == "continuous_median_split":
        median = values.median()
        treated = (values > median).astype(int)
        definition = f"{series.name} > median ({median:.3g})"
    else:
        threshold = float(entry.get("threshold", 0))
        direction = entry.get("direction", "greater")
        treated = (
            (values > threshold) if direction == "greater" else (values < threshold)
        ).astype(int)
        symbol = ">" if direction == "greater" else "<"
        definition = f"{series.name} {symbol} {threshold:g}"

    # A NaN feature value means "no events in the window", which is untreated —
    # but say so, because silently recoding missing to 0 is its own bias.
    treated = treated.where(values.notna(), 0).values.astype(int)
    share = treated.mean()
    if share < 0.01 or share > 0.99:
        return None, f"only {share:.2%} treated — no comparable group exists"
    return treated, definition


def _select_confounders(
    cfg: Config, fm: FeatureMatrix, treatment_feature: str, conf_cfg: dict
) -> list[str]:
    if conf_cfg.get("mode") == "explicit":
        return [c for c in (conf_cfg.get("explicit") or []) if c in fm.X.columns]

    # Exclusions accept globs, because one underlying column becomes several
    # features and excluding a mediator means excluding all of its aggregations.
    patterns = list(conf_cfg.get("exclude") or [])

    group, column = _treatment_scope(cfg, fm, treatment_feature)

    def redundant(name: str) -> bool:
        if (fm.meta[name].group if name in fm.meta else name.split("__")[0]) != group:
            return False
        if column is None:
            # The treatment is a per-subset count, and a base count is exactly the
            # sum of its filtered subsets. Anything else built from the same log
            # lets the propensity model reconstruct subset membership, which
            # manufactures determinism rather than adjustment.
            return True
        # Otherwise only the same underlying column is redundant — other columns
        # from the same source are usually genuine confounders and dropping them
        # would leave the estimate confounded.
        return column in name.split("__")

    dropped = {
        c for c in fm.X.columns
        if c == treatment_feature or any(fnmatch(c, p) for p in patterns) or redundant(c)
    }
    # A derived feature computed from an excluded input reintroduces it.
    dropped |= set(fm.dependents_of(sorted(dropped)))

    return [c for c in fm.X.columns if c not in dropped]


def _treatment_scope(cfg, fm: FeatureMatrix, name: str) -> tuple[str, str | None]:
    """Return (group, column) for the treatment; column None means 'the whole group'."""
    segments = name.split("__")
    group = fm.meta[name].group if name in fm.meta else segments[0]

    filter_names: set[str] = set()
    if cfg is not None:
        spec = cfg.group(group)
        if spec is not None:
            filter_names = {f.name for f in spec.filters}

    is_filtered = len(segments) > 1 and segments[1] in filter_names
    # event_count / recency_days describe a subset rather than a column, so they
    # carry the same partition problem whether or not a filter name is present.
    is_subset_measure = any(
        s.startswith(("event_count", "recency_days")) for s in segments[1:]
    )
    if is_filtered or is_subset_measure:
        return group, None

    column = segments[2] if is_filtered and len(segments) > 3 else (
        segments[1] if len(segments) > 2 else None
    )
    return group, column


def _nuisance_matrix(fm: FeatureMatrix, confounders: list[str]):
    numeric = [c for c in confounders if fm.meta[c].kind == "numeric"]
    categorical = [c for c in confounders if fm.meta[c].kind == "categorical"]
    return fm.X[numeric + categorical], numeric, categorical


def _fit_nuisances(
    cfg: Config, X: pd.DataFrame, numeric, categorical, treated, y, folds, clip: float
):
    """Cross-fitted propensity and outcome predictions."""
    n = len(y)
    e_hat = np.full(n, np.nan)
    mu0 = np.full(n, np.nan)
    mu1 = np.full(n, np.nan)

    def _make() -> Pipeline:
        return Pipeline(
            [
                ("pre", _preprocessor(numeric, categorical, for_linear=False)),
                ("est", HistGradientBoostingClassifier(
                    max_iter=200, learning_rate=0.08, random_state=cfg.seed)),
            ]
        )

    for train_idx, test_idx in folds:
        prop = _make()
        prop.fit(X.iloc[train_idx], treated[train_idx])
        e_hat[test_idx] = prop.predict_proba(X.iloc[test_idx])[:, 1]

        for arm, store in ((0, mu0), (1, mu1)):
            mask = treated[train_idx] == arm
            idx = train_idx[mask]
            if len(np.unique(y[idx])) < 2:
                store[test_idx] = float(y[idx].mean()) if len(idx) else float(y.mean())
                continue
            outcome = _make()
            outcome.fit(X.iloc[idx], y[idx])
            store[test_idx] = outcome.predict_proba(X.iloc[test_idx])[:, 1]

    return np.clip(e_hat, clip, 1 - clip), mu0, mu1


def _propensity_culprit(
    X: pd.DataFrame, treated: np.ndarray, e_hat: np.ndarray
) -> tuple[str, float] | None:
    """Name the single confounder that best explains treatment assignment."""
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    if len(np.unique(treated)) < 2:
        return None
    overall = roc_auc_score(treated, e_hat)
    if overall < 0.95:
        return None
    best, best_auc = None, 0.0
    for col in X.select_dtypes(include=[np.number]).columns:
        values = X[col].values.astype(float)
        mask = ~np.isnan(values)
        if mask.sum() < 50 or len(np.unique(treated[mask])) < 2:
            continue
        auc = roc_auc_score(treated[mask], values[mask])
        auc = max(auc, 1 - auc)
        if auc > best_auc:
            best, best_auc = col, auc
    return (f"{best} (alone: {best_auc:.3f})", float(overall)) if best else None


def _smd(X: pd.DataFrame, treated: np.ndarray, weights: np.ndarray | None) -> float:
    """Largest standardised mean difference across numeric confounders."""
    numeric = X.select_dtypes(include=[np.number])
    if numeric.empty:
        return float("nan")
    w = np.ones(len(X)) if weights is None else weights
    worst = 0.0
    for col in numeric.columns[:80]:
        values = numeric[col].values.astype(float)
        mask = ~np.isnan(values)
        if mask.sum() < 20:
            continue
        v, t, ww = values[mask], treated[mask], w[mask]
        for arm in (0, 1):
            if ww[t == arm].sum() <= 0:
                break
        else:
            m1 = np.average(v[t == 1], weights=ww[t == 1])
            m0 = np.average(v[t == 0], weights=ww[t == 0])
            pooled = np.sqrt((v[t == 1].var() + v[t == 0].var()) / 2)
            if pooled > 0:
                worst = max(worst, abs(m1 - m0) / pooled)
    return float(worst)


def _estimate(
    cfg: Config, fm: FeatureMatrix, confounders: list[str], treated: np.ndarray,
    y: np.ndarray, name: str, feature: str, definition: str, estimator_cfg: dict,
) -> CausalEstimate:
    clip = float(estimator_cfg.get("propensity_clip", 0.02))
    n_folds = int(estimator_cfg.get("cross_fit_folds", 5))
    min_overlap = float(estimator_cfg.get("min_overlap_share", 0.90))

    X, numeric, categorical = _nuisance_matrix(fm, confounders)
    n = len(y)
    warnings_: list[str] = []

    n_folds = min(n_folds, int(min(np.bincount(treated))), int(min(np.bincount(y))))
    n_folds = max(2, n_folds)
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.seed)
    strata = treated * 2 + y
    try:
        folds = [(tr, te) for tr, te in splitter.split(X, strata)]
    except ValueError:
        folds = [(tr, te) for tr, te in StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=cfg.seed).split(X, treated)]

    e_hat, mu0, mu1 = _fit_nuisances(cfg, X, numeric, categorical, treated, y, folds, clip)

    # Overlap / positivity: rows whose propensity sits at the clip boundary have no
    # counterfactual counterpart in the data. Extrapolating over them is where
    # observational estimates most often go quietly wrong.
    in_support = (e_hat > clip * 1.5) & (e_hat < 1 - clip * 1.5)
    overlap_share = float(in_support.mean())
    if overlap_share < min_overlap:
        warnings_.append(
            f"Only {overlap_share:.1%} of rows have a comparable counterpart in the "
            f"other arm (positivity). Customers at the extremes are being compared to "
            f"people who are not like them."
        )
        culprit = _propensity_culprit(X, treated, e_hat)
        if culprit:
            warnings_.append(
                f"Treatment is nearly determined by the adjustment set (propensity AUC "
                f"{culprit[1]:.3f}, driven by {culprit[0]}). When a confounder predicts "
                f"the treatment almost perfectly it is usually a restatement of it, and "
                f"there is no comparable untreated customer left to contrast against. "
                f"Add it to causal.yaml: confounders.exclude, or pick a treatment that "
                f"is not mechanically tied to it."
            )

    smd_before = _smd(X, treated, None)
    weights = np.where(treated == 1, 1 / e_hat, 1 / (1 - e_hat))
    smd_after = _smd(X, treated, weights)
    if smd_after > 0.1:
        warnings_.append(
            f"After weighting, the worst covariate imbalance is {smd_after:.2f} standard "
            f"deviations (want < 0.1). The two groups still differ systematically."
        )

    naive = float(y[treated == 1].mean() - y[treated == 0].mean())
    scores = (
        mu1 - mu0
        + treated * (y - mu1) / e_hat
        - (1 - treated) * (y - mu0) / (1 - e_hat)
    )
    valid_mask = in_support & np.isfinite(scores)
    if valid_mask.sum() < 50:
        return CausalEstimate(
            name=name, treatment_feature=feature, definition=definition, n=n,
            n_treated=int(treated.sum()), treated_share=float(treated.mean()),
            ate=float("nan"), ate_ci=(float("nan"), float("nan")), se=float("nan"),
            p_value=float("nan"), naive_difference=naive,
            confounders_used=len(confounders), overlap_share=overlap_share,
            max_smd_before=smd_before, max_smd_after=smd_after,
            warnings=warnings_ + ["Too few rows with overlap to estimate an effect."],
            valid=False,
        )

    ate = float(scores[valid_mask].mean())
    se = float(scores[valid_mask].std(ddof=1) / np.sqrt(valid_mask.sum()))

    n_boot = int(estimator_cfg.get("bootstrap", 0) or 0)
    if n_boot > 0:
        rng = np.random.default_rng(cfg.seed)
        pool = scores[valid_mask]
        draws = [
            float(rng.choice(pool, size=len(pool), replace=True).mean())
            for _ in range(n_boot)
        ]
        ci = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
        se = float(np.std(draws, ddof=1))
    else:
        ci = (ate - 1.96 * se, ate + 1.96 * se)

    from scipy import stats  # noqa: PLC0415

    p_value = float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else float("nan")
    naive = float(y[treated == 1].mean() - y[treated == 0].mean())

    if abs(naive) > 1e-9 and abs(ate) < abs(naive) * 0.4:
        warnings_.append(
            f"Adjustment shrank the raw gap from {100 * naive:+.1f} pp to "
            f"{100 * ate:+.1f} pp. Most of the apparent association was confounding — "
            f"which is exactly why the raw number should not be quoted."
        )

    return CausalEstimate(
        name=name, treatment_feature=feature, definition=definition, n=n,
        n_treated=int(treated.sum()), treated_share=float(treated.mean()),
        ate=ate, ate_ci=ci, se=se, p_value=p_value, naive_difference=naive,
        confounders_used=len(confounders), overlap_share=overlap_share,
        max_smd_before=smd_before, max_smd_after=smd_after, warnings=warnings_,
    )


def _refute(
    cfg: Config, fm: FeatureMatrix, confounders: list[str], treated: np.ndarray,
    y: np.ndarray, estimate: CausalEstimate, estimator_cfg: dict, refute_cfg: dict,
) -> None:
    rng = np.random.default_rng(cfg.seed)

    if refute_cfg.get("placebo_treatment", True):
        shuffled = rng.permutation(treated)
        placebo = _estimate(
            cfg, fm, confounders, shuffled, y, "placebo", estimate.treatment_feature,
            "shuffled", {**estimator_cfg, "bootstrap": 0},
        )
        # A placebo effect should be indistinguishable from zero. If it isn't, the
        # machinery is manufacturing effects and the real estimate means nothing.
        passed = bool(
            not np.isfinite(placebo.ate)
            or abs(placebo.ate) < max(abs(estimate.ate) * 0.25, 2 * placebo.se)
        )
        estimate.refutations["placebo"] = {
            "ate": placebo.ate, "p_value": placebo.p_value, "passed": passed,
            "explanation": (
                "Treatment was shuffled at random; a valid pipeline should find no "
                "effect. " + ("It didn't — good." if passed else
                              "It still found one, so the estimate is not trustworthy.")
            ),
        }
        if not passed:
            estimate.warnings.append(
                "Placebo test failed: a randomly assigned treatment produced an effect "
                f"of {100 * placebo.ate:+.2f} pp."
            )

    if refute_cfg.get("random_subset", True):
        fraction = float(refute_cfg.get("subset_fraction", 0.7))
        keep = rng.choice(len(y), size=int(len(y) * fraction), replace=False)
        sub_fm = FeatureMatrix(X=fm.X.iloc[keep].reset_index(drop=True), meta=fm.meta)
        subset = _estimate(
            cfg, sub_fm, confounders, treated[keep], y[keep], "subset",
            estimate.treatment_feature, "subset", {**estimator_cfg, "bootstrap": 0},
        )
        passed = bool(
            np.isfinite(subset.ate)
            and abs(subset.ate - estimate.ate) < max(2 * estimate.se, abs(estimate.ate) * 0.5)
        )
        estimate.refutations["subset"] = {
            "ate": subset.ate, "p_value": subset.p_value, "passed": passed,
            "explanation": (
                f"Re-estimated on a random {fraction:.0%} of rows. "
                + ("Stable." if passed else "The estimate moved more than its own "
                   "uncertainty allows, so it is driven by a subset of the data.")
            ),
        }
        if not passed:
            estimate.warnings.append(
                f"Subset test failed: effect moved from {100 * estimate.ate:+.2f} pp to "
                f"{100 * subset.ate:+.2f} pp on a random {fraction:.0%} sample."
            )

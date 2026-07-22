"""Risk driver detection.

Importance is measured on the held-out period, not on training data — a tree will
happily assign importance to whatever it overfit. Every driver is also checked for
stability across CV folds: a feature that tops the ranking in one fold and vanishes
in the next is noise, and shipping it as a "key driver" burns credibility.

The direction column answers "which way", the effect column answers "how much on the
churn rate". Neither answers "what happens if we change it" — that is causal.py, and
the report says so where a reader might forget.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .l01_config import Config
from .l04_features import FeatureMatrix
from .util.log import get_logger
from .l07_model import FittedModel
from .l06_splits import Split, cv_folds

log = get_logger("drivers")


@dataclass
class DriverReport:
    table: pd.DataFrame
    method: str
    stability: pd.DataFrame
    profiles: dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def top(self) -> pd.DataFrame:
        return self.table.head(20)


def detect(
    cfg: Config, fm: FeatureMatrix, frame: pd.DataFrame, split: Split, model: FittedModel
) -> DriverReport:
    features = model.numeric + model.categorical
    X = fm.X[features]
    y = frame["label"].values.astype(int)
    X_test, y_test = X.iloc[split.test_idx], y[split.test_idx]

    method = cfg.drivers["method"]
    importance, used = _importance(cfg, model, X_test, y_test, features, method)

    direction = _direction(X, y, features, split)
    effect = _effect_sizes(X, y, features, split, fm)
    stability = _stability(cfg, fm, frame, split, features)

    table = importance.merge(direction, on="feature", how="left")
    table = table.merge(effect, on="feature", how="left")
    table = table.merge(stability, on="feature", how="left")
    table["group"] = table["feature"].map(lambda f: fm.meta[f].group if f in fm.meta else "")
    table["window_days"] = table["feature"].map(
        lambda f: fm.meta[f].window_days if f in fm.meta else None
    )
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)

    min_stability = float(cfg.drivers["min_stability"])
    table["stable"] = table["stability"].fillna(0) >= min_stability

    notes = []
    top_k = int(cfg.drivers["top_k"])
    unstable = table.head(top_k)[~table.head(top_k)["stable"]]
    if len(unstable):
        notes.append(
            f"{len(unstable)} of the top {top_k} drivers did not hold their rank across "
            f"CV folds ({', '.join(unstable['feature'].head(5))}). Treat them as "
            f"hypotheses, not findings."
        )
        log.warning(notes[-1])

    profiles = _profiles(X, y, table.head(top_k)["feature"].tolist(), fm, split)

    log.info(
        "top drivers (%s): %s", used,
        ", ".join(table.head(5)["feature"].tolist()),
    )
    return DriverReport(table=table, method=used, stability=stability,
                        profiles=profiles, notes=notes)


# --------------------------------------------------------------------------- #
def _importance(
    cfg: Config, model: FittedModel, X_test: pd.DataFrame, y_test: np.ndarray,
    features: list[str], method: str,
) -> tuple[pd.DataFrame, str]:
    if method in ("auto", "shap"):
        shap_table = _shap_importance(model, X_test, features, cfg.seed)
        if shap_table is not None:
            return shap_table, "mean |SHAP| on the held-out period"
        if method == "shap":
            log.warning("SHAP unavailable; falling back to permutation importance")

    result = permutation_importance(
        model.pipeline, X_test, y_test,
        scoring="roc_auc",
        n_repeats=int(cfg.drivers["n_permutation_repeats"]),
        random_state=cfg.seed, n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": features,
            "importance": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    # Negative permutation importance means shuffling helped: pure noise.
    table["importance"] = table["importance"].clip(lower=0)
    return table, "permutation importance (drop in held-out AUC when shuffled)"


def _shap_importance(
    model: FittedModel, X_test: pd.DataFrame, features: list[str], seed: int
) -> pd.DataFrame | None:
    try:
        import shap  # noqa: PLC0415
    except ImportError:
        return None
    try:
        sample = X_test.sample(min(len(X_test), 4000), random_state=seed)
        transformed = model.pipeline.named_steps["pre"].transform(sample)
        explainer = shap.TreeExplainer(model.pipeline.named_steps["est"])
        values = explainer.shap_values(transformed, check_additivity=False)
        if isinstance(values, list):
            values = values[-1]
        values = np.asarray(values)
        if values.ndim == 3:
            values = values[:, :, -1]
        if values.shape[1] != len(features):
            return None
        return pd.DataFrame(
            {
                "feature": features,
                "importance": np.abs(values).mean(axis=0),
                "importance_std": np.abs(values).std(axis=0),
            }
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("SHAP failed (%s); using permutation importance", exc)
        return None


def _direction(
    X: pd.DataFrame, y: np.ndarray, features: list[str], split: Split
) -> pd.DataFrame:
    """Sign of the association, measured on the holdout only."""
    X_test, y_test = X.iloc[split.test_idx], y[split.test_idx]
    rows = []
    for name in features:
        series = X_test[name]
        if series.dtype == object:
            rows.append({"feature": name, "direction": "categorical", "auc_alone": np.nan})
            continue
        mask = series.notna()
        if mask.sum() < 20 or series[mask].nunique() < 2 or len(np.unique(y_test[mask])) < 2:
            rows.append({"feature": name, "direction": "unknown", "auc_alone": np.nan})
            continue
        auc = roc_auc_score(y_test[mask], series[mask])
        rows.append(
            {
                "feature": name,
                "direction": "higher -> more churn" if auc > 0.5 else "higher -> less churn",
                "auc_alone": float(max(auc, 1 - auc)),
            }
        )
    return pd.DataFrame(rows)


def _effect_sizes(
    X: pd.DataFrame, y: np.ndarray, features: list[str], split: Split, fm: FeatureMatrix
) -> pd.DataFrame:
    """Churn rate in the top vs bottom quintile of each feature, on the holdout.

    A plain, checkable number. It is an association, not an effect — the column name
    says 'spread' rather than 'effect' for that reason.
    """
    X_test, y_test = X.iloc[split.test_idx], y[split.test_idx]
    base = float(np.mean(y_test))
    rows = []
    for name in features:
        series = X_test[name]
        if series.dtype == object:
            grouped = pd.Series(y_test).groupby(series.astype(str).values).agg(["mean", "size"])
            grouped = grouped[grouped["size"] >= max(20, 0.01 * len(series))]
            if grouped.empty:
                rows.append({"feature": name, "churn_rate_low": np.nan,
                             "churn_rate_high": np.nan, "spread_pp": np.nan})
                continue
            rows.append(
                {
                    "feature": name,
                    "churn_rate_low": float(grouped["mean"].min()),
                    "churn_rate_high": float(grouped["mean"].max()),
                    "spread_pp": float(100 * (grouped["mean"].max() - grouped["mean"].min())),
                }
            )
            continue
        mask = series.notna()
        if mask.sum() < 40 or series[mask].nunique() < 5:
            rows.append({"feature": name, "churn_rate_low": np.nan,
                         "churn_rate_high": np.nan, "spread_pp": np.nan})
            continue
        values, target = series[mask].values, y_test[mask]
        lo, hi = np.quantile(values, 0.2), np.quantile(values, 0.8)
        low_rate = float(target[values <= lo].mean()) if (values <= lo).any() else np.nan
        high_rate = float(target[values >= hi].mean()) if (values >= hi).any() else np.nan
        rows.append(
            {
                "feature": name, "churn_rate_low": low_rate, "churn_rate_high": high_rate,
                "spread_pp": float(100 * (high_rate - low_rate)),
            }
        )
    out = pd.DataFrame(rows)
    out["base_rate"] = base
    return out


def _stability(
    cfg: Config, fm: FeatureMatrix, frame: pd.DataFrame, split: Split, features: list[str]
) -> pd.DataFrame:
    """Share of CV folds in which a feature lands in the top-k by permutation gain."""
    from .l07_model import build_pipeline  # local import avoids a cycle

    folds, _ = cv_folds(cfg, frame, split.train_idx)
    X_train = fm.X[features].iloc[split.train_idx]
    y_train = frame["label"].values.astype(int)[split.train_idx]
    top_k = int(cfg.drivers["top_k"])

    counts = pd.Series(0, index=features, dtype=float)
    usable = 0
    for tr, te in folds:
        if len(np.unique(y_train[te])) < 2:
            continue
        pipe = build_pipeline(cfg, fm.numeric, fm.categorical)
        pipe.fit(X_train.iloc[tr], y_train[tr])
        result = permutation_importance(
            pipe, X_train.iloc[te], y_train[te], scoring="roc_auc",
            n_repeats=3, random_state=cfg.seed, n_jobs=1,
        )
        ranked = pd.Series(result.importances_mean, index=features).nlargest(top_k)
        counts[ranked.index] += 1
        usable += 1

    if usable == 0:
        return pd.DataFrame({"feature": features, "stability": np.nan})
    return pd.DataFrame(
        {"feature": features, "stability": (counts / usable).reindex(features).values}
    )


def _profiles(
    X: pd.DataFrame, y: np.ndarray, features: list[str], fm: FeatureMatrix, split: Split
) -> dict[str, pd.DataFrame]:
    """Churn rate by decile (or category) for the headline drivers."""
    X_test, y_test = X.iloc[split.test_idx], y[split.test_idx]
    out: dict[str, pd.DataFrame] = {}
    for name in features[:10]:
        series = X_test[name]
        if series.dtype == object:
            grouped = (
                pd.DataFrame({"bucket": series.astype(str).values, "y": y_test})
                .groupby("bucket").agg(n=("y", "size"), churn_rate=("y", "mean"))
                .reset_index()
            )
            grouped = grouped[grouped["n"] >= 20].sort_values("churn_rate", ascending=False)
        else:
            mask = series.notna()
            if mask.sum() < 50 or series[mask].nunique() < 4:
                continue
            try:
                buckets = pd.qcut(series[mask], q=min(10, series[mask].nunique()),
                                  duplicates="drop")
            except ValueError:
                continue
            grouped = (
                pd.DataFrame({"bucket": buckets.astype(str), "y": y_test[mask.values]})
                .groupby("bucket", observed=True)
                .agg(n=("y", "size"), churn_rate=("y", "mean"))
                .reset_index()
            )
        if not grouped.empty:
            out[name] = grouped
    return out

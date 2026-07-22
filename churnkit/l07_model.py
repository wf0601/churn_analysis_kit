"""Model fitting and honest evaluation.

Preprocessing lives inside the estimator pipeline, never as a prior pass over the
whole frame. An imputer fitted on train+test has already seen the test distribution;
the resulting score is a slightly wrong number that nothing downstream can correct.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .l01_config import Config
from .l04_features import FeatureMatrix
from .util.log import get_logger
from .l06_splits import Split, cv_folds

log = get_logger("model")


@dataclass
class FittedModel:
    pipeline: Pipeline
    scorer: object                       # calibrated wrapper, or the pipeline itself
    numeric: list[str]
    categorical: list[str]
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    cv_metrics: pd.DataFrame
    test_predictions: pd.DataFrame
    calibration: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.scorer.predict_proba(X)[:, 1]


def _preprocessor(numeric: list[str], categorical: list[str], for_linear: bool):
    if for_linear:
        num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
        cat = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value="__missing__")),
                ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False)),
            ]
        )
    else:
        # Trees take NaN natively; imputing would erase a signal that is often real
        # (no event in the window is not the same as an average number of events).
        num = "passthrough"
        cat = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1,
            encoded_missing_value=-2,
        )
    return ColumnTransformer(
        [("num", num, numeric), ("cat", cat, categorical)],
        remainder="drop", verbose_feature_names_out=False,
    )


def build_pipeline(cfg: Config, numeric: list[str], categorical: list[str]) -> Pipeline:
    pre = _preprocessor(numeric, categorical, for_linear=False)
    n_num = len(numeric)
    categorical_mask = np.array([False] * n_num + [True] * len(categorical))
    estimator = HistGradientBoostingClassifier(
        max_iter=int(cfg.model["max_iter"]),
        learning_rate=float(cfg.model["learning_rate"]),
        max_leaf_nodes=int(cfg.model["max_leaf_nodes"]),
        categorical_features=categorical_mask if len(categorical_mask) else None,
        class_weight=cfg.model["class_weight"] or None,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=cfg.seed,
    )
    return Pipeline([("pre", pre), ("est", estimator)])


def _baseline_pipeline(cfg: Config, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("pre", _preprocessor(numeric, categorical, for_linear=True)),
            (
                "est",
                LogisticRegression(
                    max_iter=2000, class_weight=cfg.model["class_weight"] or None,
                    random_state=cfg.seed,
                ),
            ),
        ]
    )


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    base_rate = float(np.mean(y_true))
    order = np.argsort(-proba)
    metrics = {
        "n": int(len(y_true)),
        "base_rate": base_rate,
        "auc": float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "log_loss": float(log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6))),
    }
    for k in (5, 10, 20):
        top = max(1, int(len(y_true) * k / 100))
        captured = float(np.sum(y_true[order[:top]]))
        metrics[f"capture_at_{k}pct"] = captured / max(np.sum(y_true), 1)
        metrics[f"lift_at_{k}pct"] = (captured / top) / base_rate if base_rate > 0 else float("nan")
    return metrics


def _calibration_table(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.quantile(proba, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return pd.DataFrame(columns=["bin", "n", "mean_predicted", "observed"])
    idx = np.clip(np.digitize(proba, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b, "n": int(mask.sum()),
                "mean_predicted": float(proba[mask].mean()),
                "observed": float(y_true[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def fit(
    cfg: Config, fm: FeatureMatrix, frame: pd.DataFrame, split: Split
) -> FittedModel:
    numeric, categorical = fm.numeric, fm.categorical
    X = fm.X[numeric + categorical]
    y = frame["label"].values.astype(int)

    X_train, y_train = X.iloc[split.train_idx], y[split.train_idx]
    X_test, y_test = X.iloc[split.test_idx], y[split.test_idx]
    log.info(
        "fitting on %s rows (%s events) / testing on %s rows (%s events); "
        "%d numeric + %d categorical features",
        f"{len(X_train):,}", f"{y_train.sum():,}", f"{len(X_test):,}",
        f"{y_test.sum():,}", len(numeric), len(categorical),
    )

    pipeline = build_pipeline(cfg, numeric, categorical)
    pipeline.fit(X_train, y_train)

    scorer = pipeline
    notes: list[str] = []
    if cfg.model["calibrate"]:
        folds, n_splits = cv_folds(cfg, frame, split.train_idx)
        try:
            calibrated = CalibratedClassifierCV(
                clone(pipeline), method="isotonic", cv=folds,
            )
            calibrated.fit(X_train, y_train)
            scorer = calibrated
            log.info("probabilities calibrated with isotonic regression (%d folds)", n_splits)
        except Exception as exc:  # noqa: BLE001
            notes.append(f"Calibration failed ({exc}); using raw model scores.")
            log.warning(notes[-1])

    proba_test = scorer.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, proba_test)

    baseline = _baseline_pipeline(cfg, numeric, categorical)
    baseline.fit(X_train, y_train)
    baseline_metrics = evaluate(y_test, baseline.predict_proba(X_test)[:, 1])

    cv_metrics = _cross_validate(cfg, pipeline, X, y, frame, split)

    log.info(
        "holdout AUC %.4f (logistic baseline %.4f), PR-AUC %.4f, Brier %.4f, "
        "lift@10%% %.2fx",
        metrics["auc"], baseline_metrics["auc"], metrics["pr_auc"],
        metrics["brier"], metrics["lift_at_10pct"],
    )
    if metrics["auc"] - baseline_metrics["auc"] < 0.01:
        notes.append(
            "The gradient-boosted model barely beats a logistic regression. The signal "
            "in these features is essentially linear — prefer the simpler story when "
            "explaining drivers."
        )
        log.info(notes[-1])

    predictions = pd.DataFrame(
        {
            "entity_id": frame["entity_id"].values[split.test_idx],
            "snapshot_date": frame["snapshot_date"].values[split.test_idx],
            "label": y_test,
            "risk_score": proba_test,
        }
    ).sort_values("risk_score", ascending=False)

    return FittedModel(
        pipeline=pipeline, scorer=scorer, numeric=numeric, categorical=categorical,
        metrics=metrics, baseline_metrics=baseline_metrics, cv_metrics=cv_metrics,
        test_predictions=predictions,
        calibration=_calibration_table(y_test, proba_test), notes=notes,
    )


def _cross_validate(
    cfg: Config, pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray,
    frame: pd.DataFrame, split: Split,
) -> pd.DataFrame:
    folds, _ = cv_folds(cfg, frame, split.train_idx)
    X_train = X.iloc[split.train_idx]
    y_train = y[split.train_idx]
    rows = []
    for i, (tr, te) in enumerate(folds):
        if len(np.unique(y_train[te])) < 2:
            continue
        model = clone(pipeline)
        model.fit(X_train.iloc[tr], y_train[tr])
        scores = evaluate(y_train[te], model.predict_proba(X_train.iloc[te])[:, 1])
        rows.append({"fold": i, **scores})
    out = pd.DataFrame(rows)
    if not out.empty:
        log.info(
            "grouped CV on train: AUC %.4f ± %.4f across %d folds",
            out["auc"].mean(), out["auc"].std(ddof=0), len(out),
        )
        if len(out) > 1 and out["auc"].std(ddof=0) > 0.05:
            log.warning(
                "CV AUC swings by more than 0.05 between folds — the driver ranking "
                "below is not stable, treat it as a hypothesis list rather than a result."
            )
    return out

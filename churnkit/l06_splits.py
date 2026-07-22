"""Train/test splitting.

Out-of-time is the default and it is not a stylistic preference. A random split of a
churn panel scores the model on prediction dates it has already seen, in a market
whose churn regime it has already learned. That is the number that looks great in a
notebook and collapses in production.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .l01_config import Config
from .util.errors import InsufficientDataError
from .util.log import get_logger

log = get_logger("splits")


@dataclass
class Split:
    train_idx: np.ndarray
    test_idx: np.ndarray
    strategy: str
    cutoff: pd.Timestamp | None
    description: str


def make(cfg: Config, frame: pd.DataFrame) -> Split:
    strategy = cfg.split["strategy"]
    if strategy == "out_of_time" and frame["snapshot_date"].nunique() > 1:
        return _out_of_time(cfg, frame)
    if strategy == "out_of_time":
        log.warning(
            "split.strategy is out_of_time but there is only one prediction date; "
            "falling back to a customer-grouped random split. The holdout score will "
            "not tell you how the model survives a change in market conditions."
        )
    return _grouped_random(cfg, frame)


def _out_of_time(cfg: Config, frame: pd.DataFrame) -> Split:
    dates = np.sort(frame["snapshot_date"].unique())
    n_test = max(1, int(round(len(dates) * float(cfg.split["test_fraction"]))))
    n_test = min(n_test, len(dates) - 1)
    cutoff = pd.Timestamp(dates[len(dates) - n_test])

    purge = int(cfg.split["purge_days"])
    test_mask = frame["snapshot_date"] >= cutoff
    train_mask = frame["snapshot_date"] < cutoff - pd.Timedelta(days=purge)

    # A training row's label window extends horizon_days past its snapshot. Without
    # this purge the model would be trained on outcomes that happen after the test
    # period begins — future information, laundered through the label.
    horizon_purge = cutoff - pd.Timedelta(days=int(cfg.panel["horizon_days"]))
    overlap = int(((frame["snapshot_date"] < cutoff) & (frame["snapshot_date"] > horizon_purge)).sum())
    if overlap:
        train_mask &= frame["snapshot_date"] <= horizon_purge
        log.info(
            "purged %s training row(s) whose %d-day label window overlaps the test "
            "period", f"{overlap:,}", int(cfg.panel["horizon_days"]),
        )

    train_idx = np.flatnonzero(train_mask.values)
    test_idx = np.flatnonzero(test_mask.values)
    _validate(frame, train_idx, test_idx)

    description = (
        f"Out-of-time: trained on prediction dates up to "
        f"{frame.loc[train_idx, 'snapshot_date'].max().date()}, tested on "
        f"{cutoff.date()} onward ({n_test} of {len(dates)} dates held out)."
    )
    log.info(description)
    return Split(train_idx, test_idx, "out_of_time", cutoff, description)


def _grouped_random(cfg: Config, frame: pd.DataFrame) -> Split:
    groups = frame["entity_id"].values
    y = frame["label"].values
    n_splits = max(2, int(round(1 / float(cfg.split["test_fraction"]))))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
    train_idx, test_idx = next(splitter.split(frame, y, groups))
    _validate(frame, train_idx, test_idx)
    description = (
        f"Customer-grouped random split ({100 / n_splits:.0f}% held out). No customer "
        f"appears on both sides."
    )
    log.info(description)
    return Split(train_idx, test_idx, "grouped_random", None, description)


def _validate(frame: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    if len(train_idx) == 0 or len(test_idx) == 0:
        raise InsufficientDataError(
            "The split left one side empty. Lower split.test_fraction, or add more "
            "prediction dates by widening the observation window."
        )
    for side, idx in (("train", train_idx), ("test", test_idx)):
        events = int(frame["label"].values[idx].sum())
        if events == 0:
            raise InsufficientDataError(
                f"The {side} split contains no churn events. With a "
                f"{frame['label'].mean():.2%} base rate the panel is too small to split; "
                f"lengthen panel.horizon_days or widen the observation window."
            )
        if events < 25:
            log.warning(
                "%s split has only %d churn events — expect wide confidence intervals "
                "on every number in this report", side, events,
            )


def cv_folds(cfg: Config, frame: pd.DataFrame, idx: np.ndarray):
    """Inner CV folds, grouped by customer.

    Rows for the same customer at different prediction dates are near-duplicates. If
    they straddle a fold boundary, cross-validation measures memorisation.
    """
    y = frame["label"].values[idx]
    groups = frame["entity_id"].values[idx]
    n_splits = int(cfg.split["cv_folds"])
    n_groups = len(np.unique(groups))
    if n_groups < n_splits:
        n_splits = max(2, n_groups)
    minority = int(min(np.bincount(y.astype(int))))
    if minority < n_splits:
        n_splits = max(2, minority)
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.seed)
        return list(splitter.split(np.zeros(len(y)), y, groups)), n_splits
    except ValueError:
        splitter = GroupKFold(n_splits=n_splits)
        return list(splitter.split(np.zeros(len(y)), y, groups)), n_splits

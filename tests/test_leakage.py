"""Tests for the leakage guards.

Each test constructs data containing exactly one kind of leak and asserts the
matching guard fires. A guard that is never exercised against a real leak is
decoration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnkit import l01_config as config_mod
from churnkit import l02_data as data_mod
from churnkit import l03_panel as panel_mod
from churnkit import l04_features as features_mod
from churnkit import l05_leakage as leakage_mod
from churnkit.util.errors import LeakageError


def _stages(config_dir, run_audit: bool = True):
    cfg = config_mod.load(config_dir)
    ds = data_mod.load(cfg)
    timeline = data_mod.resolve_timeline(cfg, ds)
    pnl = panel_mod.build(cfg, ds, timeline)
    fm = features_mod.build(cfg, ds, pnl)
    report = leakage_mod.LeakageReport()
    if run_audit:
        leakage_mod.audit_structure(cfg, pnl, fm, report)
        leakage_mod.audit_statistics(cfg, fm, pnl.frame["label"], report, seed=0)
    return cfg, ds, pnl, fm, report


def codes(report) -> set[str]:
    return {f.code for f in report.findings}


# --------------------------------------------------------------------------- #
# time containment
# --------------------------------------------------------------------------- #
def test_features_stop_exactly_at_the_embargo_boundary(tmp_path, customers):
    """Each event carries its own date as its value, so the boundary is checkable.

    For a snapshot T with a 7-day embargo, the newest event any feature may include
    is the one dated T-8. One day later is a leak; one day earlier means the window
    is silently losing data.
    """
    from tests.conftest import write_project

    origin = pd.Timestamp("2023-01-01")
    days = pd.date_range(origin, "2023-12-31", freq="D")
    rows = [
        {"customer_id": cid, "event_date": day, "sessions": (day - origin).days}
        for cid in customers["customer_id"]
        for day in days
    ]
    config_dir = write_project(
        tmp_path, customers, events={"usage": pd.DataFrame(rows)},
        features={"usage": {"temporal": "time_varying", "source": "usage",
                            "aggregation_window_days": 30,
                            "columns": [{"name": "sessions", "type": "numeric"}]}},
    )
    _, _, pnl, fm, _ = _stages(config_dir, run_audit=False)

    frame = pnl.frame.reset_index(drop=True)
    observed = fm.X["usage__sessions__max_30d"]
    checked = 0
    for snapshot in pnl.snapshot_dates:
        mask = (frame["snapshot_date"] == snapshot).values
        values = observed[mask].dropna()
        if values.empty:
            continue
        newest_allowed = (snapshot - pd.Timedelta(days=8) - origin).days
        assert values.max() == newest_allowed, (
            f"snapshot {snapshot.date()}: newest event used encodes day "
            f"{values.max()}, expected {newest_allowed}"
        )
        checked += 1
    assert checked >= 5


def test_window_start_is_respected(tmp_path, customers):
    from tests.conftest import write_project

    rows = []
    for cid in customers["customer_id"]:
        # Far outside any 30-day window.
        rows.append({"customer_id": cid, "event_date": pd.Timestamp("2023-01-02"),
                     "sessions": 9_999})
    config_dir = write_project(
        tmp_path, customers, events={"usage": pd.DataFrame(rows)},
        features={"usage": {"temporal": "time_varying", "source": "usage",
                            "aggregation_window_days": 30,
                            "columns": [{"name": "sessions", "type": "numeric"}]}},
        pipeline={"panel": {"snapshot_start": "2023-06-01"}},
    )
    _, _, _, fm, _ = _stages(config_dir, run_audit=False)
    assert fm.X["usage__sessions__max_30d"].dropna().empty


# --------------------------------------------------------------------------- #
# labels, censoring, truncation
# --------------------------------------------------------------------------- #
def test_snapshots_past_the_observation_end_are_dropped(tmp_path, customers):
    from tests.conftest import write_project

    config_dir = write_project(tmp_path, customers)
    _, _, pnl, _, report = _stages(config_dir)
    obs_end = pnl.timeline["observation_end"]
    horizon = pd.Timedelta(days=pnl.horizon_days)
    assert all(date + horizon <= obs_end for date in pnl.snapshot_dates)
    assert "CENSORED_AS_RETAINED" not in codes(report)
    assert "CENSORING_HANDLED" in codes(report)


def test_still_active_customers_are_censored_not_retained(tmp_path, customers):
    from tests.conftest import write_project

    config_dir = write_project(tmp_path, customers)
    cfg, _, pnl, _, _ = _stages(config_dir, run_audit=False)
    never_churned = customers["contract_end_date"].isna().sum()
    censored = int((pnl.survival["event"] == 0).sum())
    assert censored >= never_churned
    assert pnl.survival["duration_days"].min() > 0


def test_left_truncated_customers_are_dropped(tmp_path, customers):
    from tests.conftest import write_project

    early = customers.copy()
    early.loc[:9, "signup_date"] = "2021-06-01"        # predates the window
    early.loc[:9, "contract_end_date"] = None          # still active, so only truncation removes them
    config_dir = write_project(tmp_path, early)
    _, _, pnl, _, _ = _stages(config_dir, run_audit=False)
    dropped = set(early.loc[:9, "customer_id"])
    assert not (set(pnl.frame["entity_id"]) & dropped)


def test_left_truncation_keep_flagged_warns(tmp_path, customers):
    from tests.conftest import write_project

    early = customers.copy()
    early.loc[:19, "signup_date"] = "2021-06-01"
    config_dir = write_project(
        tmp_path, early,
        survivorship={"data_export_date": "2023-12-31",
                      "observation_starting_date": "2023-01-01",
                      "left_truncation": "keep_flagged"},
    )
    _, _, _, _, report = _stages(config_dir)
    assert "SURVIVORSHIP_BIAS" in codes(report)


# --------------------------------------------------------------------------- #
# denylist / allowlist / target columns
# --------------------------------------------------------------------------- #
def test_outcome_named_column_is_blocked_and_quarantined(tmp_path, customers):
    from tests.conftest import write_project

    frame = customers.copy()
    frame["cancellation_reason"] = np.where(
        frame["contract_end_date"].notna(), "price", None
    )
    config_dir = write_project(
        tmp_path, frame,
        features={"profile": {"temporal": "static", "columns": [
            {"name": "region", "type": "categorical"},
            {"name": "cancellation_reason", "type": "categorical"},
        ]}},
    )
    cfg, _, _, fm, report = _stages(config_dir)
    assert "OUTCOME_DERIVED_FEATURE" in codes(report)
    leakage_mod.enforce(cfg, fm, report)
    assert "profile__cancellation_reason" not in fm.X.columns
    assert "profile__cancellation_reason" in report.quarantined


def test_allowlist_overrides_the_denylist(tmp_path, customers):
    from tests.conftest import write_project

    frame = customers.copy()
    frame["winback_offer_seen"] = (np.arange(len(frame)) % 3 == 0).astype(int)
    features = {"profile": {"temporal": "static", "columns": [
        {"name": "region", "type": "categorical"},
        {"name": "winback_offer_seen", "type": "numeric"},
    ]}}
    blocked_dir = write_project(tmp_path, frame, features=features)
    _, _, _, _, report = _stages(blocked_dir)
    assert "OUTCOME_DERIVED_FEATURE" in codes(report)

    allowed_dir = write_project(
        tmp_path, frame, features=features,
        pipeline={"leakage": {"allowlist_columns": ["profile__winback_offer_seen"]}},
    )
    _, _, _, fm, report = _stages(allowed_dir)
    assert "OUTCOME_DERIVED_FEATURE" not in codes(report)
    assert "profile__winback_offer_seen" in fm.X.columns


def test_target_column_used_as_feature_is_blocked(tmp_path, customers):
    from tests.conftest import write_project

    config_dir = write_project(
        tmp_path, customers,
        features={"profile": {"temporal": "static", "columns": [
            {"name": "region", "type": "categorical"},
            {"name": "contract_end_date", "type": "datetime"},
        ]}},
        pipeline={"leakage": {"denylist_patterns": ["nothing_matches_this"]}},
    )
    _, _, _, _, report = _stages(config_dir)
    assert "TARGET_COLUMN_AS_FEATURE" in codes(report)


# --------------------------------------------------------------------------- #
# statistical screens
# --------------------------------------------------------------------------- #
def test_row_level_proxy_is_blocked_and_quarantined(tmp_path, customers):
    """A column named innocently that happens to equal the label."""
    from tests.conftest import write_project

    config_dir = write_project(tmp_path, customers)
    cfg, _, pnl, fm, _ = _stages(config_dir, run_audit=False)

    labels = pnl.frame["label"].values
    fm.X["profile__engagement_index"] = labels * 0.98 + 0.01
    fm.meta["profile__engagement_index"] = features_mod.FeatureMeta(
        "profile__engagement_index", "profile", "entity", "numeric", "static",
        None, "normal", "test fixture", True,
    )

    report = leakage_mod.LeakageReport()
    scan = leakage_mod.audit_statistics(cfg, fm, pnl.frame["label"], report, seed=0)
    assert scan.loc[
        scan["feature"] == "profile__engagement_index", "univariate_auc"
    ].iloc[0] == pytest.approx(1.0)
    assert "SINGLE_FEATURE_SEPARATION" in codes(report)

    leakage_mod.enforce(cfg, fm, report)
    assert "profile__engagement_index" not in fm.X.columns


def test_eventual_churn_flag_surfaces_as_a_warning(tmp_path, customers):
    """A static "did they ever churn" column is diluted by the panel, not hidden.

    Spread across many prediction dates it lands under the blocking threshold, so it
    must still come out as a warning rather than passing silently.
    """
    from tests.conftest import write_project

    frame = customers.copy()
    frame["engagement_index"] = np.where(frame["contract_end_date"].notna(), 0.99, 0.01)
    config_dir = write_project(
        tmp_path, frame,
        features={"profile": {"temporal": "static", "columns": [
            {"name": "region", "type": "categorical"},
            {"name": "engagement_index", "type": "numeric"},
        ]}},
    )
    _, _, _, _, report = _stages(config_dir)
    assert "STRONG_SINGLE_FEATURE" in codes(report)


def test_missingness_that_predicts_the_target_is_flagged(tmp_path, customers):
    from tests.conftest import write_project

    frame = customers.copy()
    # The field is only populated for customers who never churned — a record written
    # by a downstream process that already knows the answer.
    frame["renewal_survey_score"] = np.where(
        frame["contract_end_date"].isna(), np.arange(len(frame)) % 5, np.nan
    )
    config_dir = write_project(
        tmp_path, frame,
        features={"profile": {"temporal": "static", "columns": [
            {"name": "region", "type": "categorical"},
            {"name": "renewal_survey_score", "type": "numeric"},
        ]}},
    )
    _, _, _, _, report = _stages(config_dir)
    assert "MISSINGNESS_PREDICTS_TARGET" in codes(report)


def test_stored_tenure_column_is_recomputed_not_read(tmp_path, customers):
    """A stored tenure encodes the lifetime the customer turned out to have."""
    from tests.conftest import write_project

    frame = customers.copy()
    end = pd.to_datetime(frame["contract_end_date"]).fillna(pd.Timestamp("2023-12-31"))
    frame["tenure_days"] = (end - pd.to_datetime(frame["signup_date"])).dt.days
    config_dir = write_project(
        tmp_path, frame,
        features={"lifecycle": {"temporal": "time_varying", "source": "entity",
                                "columns": [{"name": "tenure_days", "type": "numeric"}]}},
    )
    _, _, pnl, fm, _ = _stages(config_dir, run_audit=False)
    built = fm.X["lifecycle__tenure_days"].values
    expected = pnl.frame["tenure_days"].values
    assert np.allclose(built, expected), "the stored column was used instead of as-of T"
    assert fm.meta["lifecycle__tenure_days"].point_in_time


def test_constant_features_are_dropped(tmp_path, customers):
    from tests.conftest import write_project

    frame = customers.copy()
    frame["always_one"] = 1
    config_dir = write_project(
        tmp_path, frame,
        features={"profile": {"temporal": "static", "columns": [
            {"name": "region", "type": "categorical"},
            {"name": "always_one", "type": "numeric"},
        ]}},
    )
    cfg, _, _, fm, report = _stages(config_dir)
    leakage_mod.enforce(cfg, fm, report)
    assert "profile__always_one" not in fm.X.columns


def test_present_or_absent_indicators_are_kept(tmp_path, customers):
    """One distinct value plus nulls is a presence flag, not a constant.

    A filtered event count is 1-where-it-happened and null elsewhere; dropping it
    as "constant" would throw away the whole signal.
    """
    from tests.conftest import write_project

    frame = customers.copy()
    frame["one_or_missing"] = np.where(np.arange(len(frame)) % 3 == 0, 1.0, np.nan)
    config_dir = write_project(
        tmp_path, frame,
        features={"profile": {"temporal": "static", "columns": [
            {"name": "region", "type": "categorical"},
            {"name": "one_or_missing", "type": "numeric"},
        ]}},
    )
    cfg, _, _, fm, report = _stages(config_dir)
    leakage_mod.enforce(cfg, fm, report)
    assert "profile__one_or_missing" in fm.X.columns


# --------------------------------------------------------------------------- #
# policy
# --------------------------------------------------------------------------- #
def test_on_block_fail_aborts_the_run(tmp_path, customers):
    from tests.conftest import write_project

    frame = customers.copy()
    frame["churn_flag_internal"] = frame["contract_end_date"].notna().astype(int)
    config_dir = write_project(
        tmp_path, frame,
        features={"profile": {"temporal": "static", "columns": [
            {"name": "region", "type": "categorical"},
            {"name": "churn_flag_internal", "type": "numeric"},
        ]}},
        pipeline={"leakage": {"on_block": "fail"}},
    )
    cfg, _, _, fm, report = _stages(config_dir)
    with pytest.raises(LeakageError, match="churn_flag_internal"):
        leakage_mod.enforce(cfg, fm, report)


def test_no_embargo_warns(tmp_path, customers):
    from tests.conftest import write_project

    config_dir = write_project(tmp_path, customers, pipeline={"panel": {"embargo_days": 0}})
    _, _, _, _, report = _stages(config_dir)
    assert "NO_EMBARGO" in codes(report)


def test_no_embargo_blocks_when_a_group_is_strict(tmp_path, customers):
    from tests.conftest import write_project

    rows = [
        {"customer_id": cid, "event_date": month, "days_past_due": 3}
        for cid in customers["customer_id"]
        for month in pd.date_range("2023-01-01", "2023-12-01", freq="MS")
    ]
    config_dir = write_project(
        tmp_path, customers, events={"billing": pd.DataFrame(rows)},
        features={"billing": {"temporal": "time_varying", "source": "billing",
                              "aggregation_window_days": 90,
                              "leakage_review": "strict",
                              "columns": [{"name": "days_past_due", "type": "numeric"}]}},
        pipeline={"panel": {"embargo_days": 0}},
    )
    _, _, _, _, report = _stages(config_dir)
    finding = next(f for f in report.findings if f.code == "NO_EMBARGO")
    assert finding.level == leakage_mod.BLOCK


def test_generalisation_gap_is_flagged():
    report = leakage_mod.LeakageReport()
    leakage_mod.audit_generalisation(0.98, 0.74, report)
    assert "CV_HOLDOUT_GAP" in codes(report)

    clean = leakage_mod.LeakageReport()
    leakage_mod.audit_generalisation(0.72, 0.70, clean)
    assert "CV_HOLDOUT_GAP" not in codes(clean)


def test_implausible_model_performance_blocks(tmp_path, customers):
    from tests.conftest import write_project

    config_dir = write_project(tmp_path, customers)
    cfg = config_mod.load(config_dir)
    report = leakage_mod.LeakageReport()
    leakage_mod.audit_model_performance(cfg, 0.999, 0.05, report)
    assert "MODEL_TOO_GOOD" in codes(report)
    assert report.blocked

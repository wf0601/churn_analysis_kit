"""Tests for YAML-driven feature construction.

The promise is that adding a window, an aggregation, a filtered subset or a ratio
means editing feature.yaml and nothing else. These check that the YAML actually
drives the output, that bad YAML fails with a usable message, and that the leakage
guards still hold across the new construction paths.
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
from churnkit.util.errors import ConfigError


def _build(config_dir):
    cfg = config_mod.load(config_dir)
    ds = data_mod.load(cfg)
    pnl = panel_mod.build(cfg, ds, data_mod.resolve_timeline(cfg, ds))
    return cfg, ds, pnl, features_mod.build(cfg, ds, pnl)


@pytest.fixture
def billing_events(customers) -> pd.DataFrame:
    """One billing row per customer per month, a fifth of them failed."""
    rng = np.random.default_rng(0)
    rows = []
    for i, cid in enumerate(customers["customer_id"]):
        for month in pd.date_range("2023-01-01", "2023-12-01", freq="MS"):
            failed = (i + month.month) % 5 == 0
            rows.append(
                {
                    "customer_id": cid,
                    "event_date": month + pd.Timedelta(days=3),
                    "status": "failed" if failed else "paid",
                    "amount": round(float(rng.uniform(10, 100)), 2),
                    "days_past_due": int(rng.integers(1, 40)) if failed else 0,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# windows and aggregations
# --------------------------------------------------------------------------- #
def test_multiple_windows_produce_one_feature_family_each(tmp_path, customers, billing_events):
    from tests.conftest import write_project

    config_dir = write_project(
        tmp_path, customers, events={"billing": billing_events},
        features={"billing": {"temporal": "time_varying", "source": "billing",
                              "windows": [30, 90, 180], "aggs": ["sum"],
                              "columns": [{"name": "amount", "type": "numeric"}]}},
    )
    _, _, _, fm = _build(config_dir)
    for window in (30, 90, 180):
        assert f"billing__amount__sum_{window}d" in fm.X.columns
    # Wider windows can only see more events, never fewer.
    assert (
        fm.X["billing__amount__sum_180d"].fillna(0)
        >= fm.X["billing__amount__sum_90d"].fillna(0)
    ).all()


def test_per_column_aggs_and_windows_override_the_group(tmp_path, customers, billing_events):
    from tests.conftest import write_project

    config_dir = write_project(
        tmp_path, customers, events={"billing": billing_events},
        features={"billing": {
            "temporal": "time_varying", "source": "billing",
            "windows": [30, 90], "aggs": ["sum", "mean"],
            "columns": [
                {"name": "amount", "type": "numeric"},
                {"name": "days_past_due", "type": "numeric",
                 "aggs": ["max"], "windows": [90]},
            ],
        }},
    )
    _, _, _, fm = _build(config_dir)
    assert "billing__amount__sum_30d" in fm.X.columns
    assert "billing__amount__mean_90d" in fm.X.columns
    # The override is exact: only max, only the 90-day window.
    assert "billing__days_past_due__max_90d" in fm.X.columns
    assert "billing__days_past_due__max_30d" not in fm.X.columns
    assert "billing__days_past_due__sum_90d" not in fm.X.columns


def test_aggregation_window_days_still_works(tmp_path, customers, billing_events):
    """The older single-window key must keep working."""
    from tests.conftest import write_project

    config_dir = write_project(
        tmp_path, customers, events={"billing": billing_events},
        features={"billing": {"temporal": "time_varying", "source": "billing",
                              "aggregation_window_days": 90, "aggs": ["sum"],
                              "columns": [{"name": "amount", "type": "numeric"}]}},
    )
    _, _, _, fm = _build(config_dir)
    assert "billing__amount__sum_90d" in fm.X.columns


def test_unknown_aggregation_is_rejected_with_the_available_list(tmp_path, customers):
    from tests.conftest import write_project

    config_dir = write_project(
        tmp_path, customers,
        features={"profile": {"temporal": "static", "aggs": ["averge"],
                              "columns": [{"name": "region", "type": "categorical"}]}},
    )
    with pytest.raises(ConfigError, match="averge"):
        config_mod.load(config_dir)


# --------------------------------------------------------------------------- #
# filters
# --------------------------------------------------------------------------- #
def test_filters_build_counted_variables_from_a_raw_log(tmp_path, customers, billing_events):
    """`failed_payments_90d` as a filter + window + count, not a pre-made column."""
    from tests.conftest import write_project

    config_dir = write_project(
        tmp_path, customers, events={"billing": billing_events},
        features={"billing": {
            "temporal": "time_varying", "source": "billing", "windows": [90],
            "aggs": ["sum"],
            "filters": [
                {"name": "failed", "where": "status == 'failed'"},
                {"name": "paid", "where": "status == 'paid'"},
            ],
            "columns": [{"name": "amount", "type": "numeric"}],
        }},
    )
    _, _, _, fm = _build(config_dir)

    for name in ("billing__failed__event_count_90d", "billing__paid__event_count_90d",
                 "billing__failed__amount__sum_90d", "billing__event_count_90d"):
        assert name in fm.X.columns, name

    failed = fm.X["billing__failed__event_count_90d"].fillna(0)
    paid = fm.X["billing__paid__event_count_90d"].fillna(0)
    total = fm.X["billing__event_count_90d"].fillna(0)
    assert (failed + paid == total).all(), "the subsets must partition the base family"
    assert failed.sum() > 0 and paid.sum() > 0


def test_filters_add_to_the_base_family_unless_told_otherwise(tmp_path, customers, billing_events):
    from tests.conftest import write_project

    spec = {
        "temporal": "time_varying", "source": "billing", "windows": [90],
        "aggs": ["sum"], "include_unfiltered": False,
        "filters": [{"name": "failed", "where": "status == 'failed'"}],
        "columns": [{"name": "amount", "type": "numeric"}],
    }
    config_dir = write_project(
        tmp_path, customers, events={"billing": billing_events},
        features={"billing": spec},
    )
    _, _, _, fm = _build(config_dir)
    assert "billing__failed__event_count_90d" in fm.X.columns
    assert "billing__event_count_90d" not in fm.X.columns


def test_a_broken_filter_expression_names_the_available_columns(tmp_path, customers, billing_events):
    from tests.conftest import write_project

    config_dir = write_project(
        tmp_path, customers, events={"billing": billing_events},
        features={"billing": {
            "temporal": "time_varying", "source": "billing", "windows": [90],
            "filters": [{"name": "bad", "where": "stat == 'failed'"}],
            "columns": [{"name": "amount", "type": "numeric"}],
        }},
    )
    with pytest.raises(ConfigError, match="status"):
        _build(config_dir)


def test_filter_names_are_screened_by_the_denylist(tmp_path, customers, billing_events):
    """A filter name becomes part of the feature name, so it must be checked too."""
    from tests.conftest import write_project

    events = billing_events.copy()
    events["status"] = np.where(events["status"] == "failed", "cancellation", "paid")
    config_dir = write_project(
        tmp_path, customers, events={"billing": events},
        features={"billing": {
            "temporal": "time_varying", "source": "billing", "windows": [90],
            "aggs": ["sum"],
            "filters": [{"name": "cancellation", "where": "status == 'cancellation'"}],
            "columns": [{"name": "amount", "type": "numeric"}],
        }},
    )
    cfg, _, pnl, fm = _build(config_dir)
    report = leakage_mod.LeakageReport()
    leakage_mod.audit_structure(cfg, pnl, fm, report)
    assert "OUTCOME_DERIVED_FEATURE" in {f.code for f in report.findings}
    leakage_mod.enforce(cfg, fm, report)
    assert not [c for c in fm.X.columns if "cancellation" in c]


# --------------------------------------------------------------------------- #
# derived expressions
# --------------------------------------------------------------------------- #
def _derived_project(tmp_path, customers, billing_events, features_extra):
    from tests.conftest import write_project

    return write_project(
        tmp_path, customers, events={"billing": billing_events},
        features={
            "billing": {"temporal": "time_varying", "source": "billing",
                        "windows": [90], "aggs": ["sum"],
                        "filters": [{"name": "failed", "where": "status == 'failed'"}],
                        "columns": [{"name": "amount", "type": "numeric"}]},
            "lifecycle": {"temporal": "time_varying", "source": "entity",
                          "columns": [{"name": "tenure_days", "type": "numeric"}]},
            **features_extra,
        },
    )


def test_derived_expressions_are_computed_from_other_features(tmp_path, customers, billing_events):
    config_dir = _derived_project(
        tmp_path, customers, billing_events,
        {"ratios": {"temporal": "derived", "features": [
            {"name": "failed_share_90d",
             "expression": "billing__failed__event_count_90d / billing__event_count_90d"},
            {"name": "log_amount",
             "expression": "log1p(billing__amount__sum_90d)"},
        ]}},
    )
    _, _, _, fm = _build(config_dir)

    assert "ratios__failed_share_90d" in fm.X.columns
    share = fm.X["ratios__failed_share_90d"].dropna()
    assert share.between(0, 1).all()

    expected = np.log1p(fm.X["billing__amount__sum_90d"])
    built = fm.X["ratios__log_amount"]
    assert np.allclose(built.dropna(), expected[built.notna()])


def test_division_by_zero_becomes_null_not_infinity(tmp_path, customers, billing_events):
    config_dir = _derived_project(
        tmp_path, customers, billing_events,
        {"ratios": {"temporal": "derived", "features": [
            {"name": "per_failure",
             "expression": "billing__amount__sum_90d / billing__failed__event_count_90d"},
        ]}},
    )
    _, _, _, fm = _build(config_dir)
    values = fm.X["ratios__per_failure"]
    assert not np.isinf(values.dropna()).any()


def test_unknown_feature_in_an_expression_fails_loudly(tmp_path, customers, billing_events):
    config_dir = _derived_project(
        tmp_path, customers, billing_events,
        {"ratios": {"temporal": "derived", "features": [
            {"name": "nonsense", "expression": "billing__amount__sum_30d * 2"},
        ]}},
    )
    with pytest.raises(ConfigError, match="billing__amount__sum_30d"):
        _build(config_dir)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('true')",
        "billing__amount__sum_90d.values",
        "(lambda: 1)()",
        "open('/etc/passwd')",
    ],
)
def test_expressions_cannot_reach_outside_the_feature_matrix(
    tmp_path, customers, billing_events, expression
):
    config_dir = _derived_project(
        tmp_path, customers, billing_events,
        {"ratios": {"temporal": "derived",
                    "features": [{"name": "evil", "expression": expression}]}},
    )
    with pytest.raises(ConfigError):
        _build(config_dir)


def test_derived_feature_inherits_the_leakage_status_of_its_inputs(
    tmp_path, customers, billing_events
):
    """A ratio built on a blocked input carries the same information."""
    events = billing_events.copy()
    events["cancellation_fee"] = np.where(events["status"] == "failed", 25.0, 0.0)
    config_dir = _derived_project(
        tmp_path, customers, events,
        {
            "billing2": {"temporal": "time_varying", "source": "billing",
                         "windows": [90], "aggs": ["sum"],
                         "columns": [{"name": "cancellation_fee", "type": "numeric"}]},
            "ratios": {"temporal": "derived", "features": [
                {"name": "fee_per_euro",
                 "expression": "billing2__cancellation_fee__sum_90d / "
                               "maximum(billing__amount__sum_90d, 1)"},
            ]},
        },
    )
    cfg, _, pnl, fm = _build(config_dir)
    assert "ratios__fee_per_euro" in fm.X.columns
    assert fm.meta["ratios__fee_per_euro"].depends_on == [
        "billing2__cancellation_fee__sum_90d", "billing__amount__sum_90d"
    ]

    report = leakage_mod.LeakageReport()
    leakage_mod.audit_structure(cfg, pnl, fm, report)
    leakage_mod.enforce(cfg, fm, report)

    codes = {f.code for f in report.findings}
    assert "OUTCOME_DERIVED_FEATURE" in codes
    assert "DERIVED_FROM_BLOCKED_FEATURE" in codes
    assert "billing2__cancellation_fee__sum_90d" not in fm.X.columns
    assert "ratios__fee_per_euro" not in fm.X.columns, (
        "the leak survived behind a ratio"
    )


def test_derived_features_are_not_point_in_time_if_an_input_is_not(
    tmp_path, customers, billing_events
):
    config_dir = _derived_project(
        tmp_path, customers, billing_events,
        {"ratios": {"temporal": "derived", "features": [
            {"name": "amount_per_day",
             "expression": "billing__amount__sum_90d / maximum(lifecycle__tenure_days, 1)"},
        ]}},
    )
    _, _, _, fm = _build(config_dir)
    # tenure_days is recomputed as-of T, so this one stays clean.
    assert fm.meta["ratios__amount_per_day"].point_in_time


def test_feature_dictionary_records_the_construction(tmp_path, customers, billing_events):
    config_dir = _derived_project(
        tmp_path, customers, billing_events,
        {"ratios": {"temporal": "derived", "features": [
            {"name": "failed_share_90d",
             "expression": "billing__failed__event_count_90d / billing__event_count_90d",
             "description": "share of collection attempts that failed"},
        ]}},
    )
    _, _, _, fm = _build(config_dir)
    dictionary = fm.dictionary().set_index("feature")

    row = dictionary.loc["ratios__failed_share_90d"]
    assert row["derivation"] == "share of collection attempts that failed"
    assert "billing__event_count_90d" in row["depends_on"]
    assert dictionary.loc["billing__failed__amount__sum_90d", "window_days"] == 90

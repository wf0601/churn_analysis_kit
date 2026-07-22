"""Tests for the analysis stages and the end-to-end run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churnkit import l01_config as config_mod
from churnkit import l02_data as data_mod
from churnkit import l03_panel as panel_mod
from churnkit import l04_features as features_mod
from churnkit import l06_splits as splits_mod
from churnkit import l09_causal as causal_mod
from churnkit import l10_experiment as experiment_mod
from churnkit import l11_survival as survival_mod
from churnkit.util.errors import ConfigError, DataError, InsufficientDataError


# --------------------------------------------------------------------------- #
# config validation
# --------------------------------------------------------------------------- #
def test_missing_entity_file_names_the_file(tmp_path, customers):
    from tests.conftest import write_project

    config_dir = write_project(tmp_path, customers)
    (tmp_path / "data" / "customers.csv").unlink()
    with pytest.raises(ConfigError, match="entity file not found"):
        config_mod.load(config_dir)


def test_event_date_mode_requires_the_date_column(tmp_path, customers):
    from tests.conftest import write_project

    config_dir = write_project(
        tmp_path, customers, target={"mode": "event_date", "observation_end_date": "2023-12-31"}
    )
    with pytest.raises(ConfigError, match="event_date_column"):
        config_mod.load(config_dir)


def test_duplicate_ids_without_scd2_columns_are_rejected(tmp_path, customers):
    from tests.conftest import write_project

    doubled = pd.concat([customers, customers], ignore_index=True)
    config_dir = write_project(tmp_path, doubled)
    cfg = config_mod.load(config_dir)
    with pytest.raises(DataError, match="valid_from_column"):
        data_mod.load(cfg)


def test_horizon_longer_than_the_window_fails_clearly(tmp_path, customers):
    from tests.conftest import write_project

    config_dir = write_project(tmp_path, customers, pipeline={"panel": {"horizon_days": 900}})
    cfg = config_mod.load(config_dir)
    ds = data_mod.load(cfg)
    timeline = data_mod.resolve_timeline(cfg, ds)
    with pytest.raises(InsufficientDataError, match="horizon"):
        panel_mod.build(cfg, ds, timeline)


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #
def test_labels_match_a_hand_computed_answer(tmp_path):
    from tests.conftest import write_project

    frame = pd.DataFrame(
        [
            # churns 15 days after the 2023-06-01 snapshot -> label 1 there
            {"customer_id": "A", "signup_date": "2023-01-10",
             "contract_end_date": "2023-06-16", "region": "north"},
            # churns 60 days after -> outside a 30-day horizon -> label 0
            {"customer_id": "B", "signup_date": "2023-01-10",
             "contract_end_date": "2023-07-31", "region": "north"},
            # never churns -> label 0 everywhere
            {"customer_id": "C", "signup_date": "2023-01-10",
             "contract_end_date": None, "region": "south"},
        ]
    )
    config_dir = write_project(tmp_path, frame,
                               pipeline={"panel": {"snapshot_dates": ["2023-06-01"],
                                                   "snapshot_mode": "single"}})
    cfg = config_mod.load(config_dir)
    ds = data_mod.load(cfg)
    pnl = panel_mod.build(cfg, ds, data_mod.resolve_timeline(cfg, ds))

    at_snapshot = pnl.frame.set_index("entity_id")["label"].to_dict()
    assert at_snapshot == {"A": 1, "B": 0, "C": 0}


def test_churned_customers_leave_the_risk_set(tmp_path):
    from tests.conftest import write_project

    frame = pd.DataFrame(
        [{"customer_id": "A", "signup_date": "2023-01-10",
          "contract_end_date": "2023-03-01", "region": "north"}]
    )
    config_dir = write_project(tmp_path, frame)
    cfg = config_mod.load(config_dir)
    ds = data_mod.load(cfg)
    pnl = panel_mod.build(cfg, ds, data_mod.resolve_timeline(cfg, ds))
    assert pnl.frame["snapshot_date"].max() < pd.Timestamp("2023-03-01")


def test_decision_lead_days_shifts_the_label_earlier(tmp_path):
    from tests.conftest import write_project

    frame = pd.DataFrame(
        [{"customer_id": "A", "signup_date": "2023-01-10",
          "contract_end_date": "2023-07-10", "region": "north"}]
    )
    common = {"snapshot_mode": "single", "snapshot_dates": ["2023-06-05"]}
    plain = write_project(tmp_path, frame, pipeline={"panel": common})
    cfg = config_mod.load(plain)
    ds = data_mod.load(cfg)
    pnl = panel_mod.build(cfg, ds, data_mod.resolve_timeline(cfg, ds))
    assert pnl.frame["label"].iloc[0] == 0        # 35 days out, horizon is 30

    shifted = write_project(
        tmp_path, frame, pipeline={"panel": common},
        target={"mode": "event_date", "event_date_column": "contract_end_date",
                "observation_end_date": "2023-12-31", "decision_lead_days": 20},
    )
    cfg = config_mod.load(shifted)
    ds = data_mod.load(cfg)
    pnl = panel_mod.build(cfg, ds, data_mod.resolve_timeline(cfg, ds))
    assert pnl.frame["label"].iloc[0] == 1        # now 15 days out


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def test_out_of_time_split_purges_overlapping_label_windows(tmp_path, customers):
    from tests.conftest import write_project

    config_dir = write_project(tmp_path, customers)
    cfg = config_mod.load(config_dir)
    ds = data_mod.load(cfg)
    pnl = panel_mod.build(cfg, ds, data_mod.resolve_timeline(cfg, ds))
    split = splits_mod.make(cfg, pnl.frame)

    train_dates = pnl.frame["snapshot_date"].iloc[split.train_idx]
    test_dates = pnl.frame["snapshot_date"].iloc[split.test_idx]
    horizon = pd.Timedelta(days=cfg.panel["horizon_days"])
    assert train_dates.max() + horizon <= test_dates.min(), (
        "a training row's label window reaches into the test period"
    )


def test_grouped_random_split_keeps_customers_on_one_side(tmp_path, customers):
    from tests.conftest import write_project

    config_dir = write_project(tmp_path, customers,
                               pipeline={"split": {"strategy": "grouped_random"}})
    cfg = config_mod.load(config_dir)
    ds = data_mod.load(cfg)
    pnl = panel_mod.build(cfg, ds, data_mod.resolve_timeline(cfg, ds))
    split = splits_mod.make(cfg, pnl.frame)
    train_ids = set(pnl.frame["entity_id"].iloc[split.train_idx])
    test_ids = set(pnl.frame["entity_id"].iloc[split.test_idx])
    assert not (train_ids & test_ids)


def test_cv_folds_are_grouped_by_customer(tmp_path, customers):
    from tests.conftest import write_project

    config_dir = write_project(tmp_path, customers)
    cfg = config_mod.load(config_dir)
    ds = data_mod.load(cfg)
    pnl = panel_mod.build(cfg, ds, data_mod.resolve_timeline(cfg, ds))
    split = splits_mod.make(cfg, pnl.frame)
    folds, _ = splits_mod.cv_folds(cfg, pnl.frame, split.train_idx)
    ids = pnl.frame["entity_id"].values[split.train_idx]
    for train_idx, test_idx in folds:
        assert not (set(ids[train_idx]) & set(ids[test_idx]))


# --------------------------------------------------------------------------- #
# survival
# --------------------------------------------------------------------------- #
def test_kaplan_meier_matches_lifelines():
    lifelines = pytest.importorskip("lifelines")
    rng = np.random.default_rng(3)
    durations = rng.integers(1, 300, 400).astype(float)
    events = (rng.random(400) < 0.6).astype(int)

    ours = survival_mod._km(durations, events)
    theirs = lifelines.KaplanMeierFitter().fit(durations, events)
    expected = theirs.survival_function_at_times(ours["time"].values).values
    assert np.allclose(ours["survival"].values, expected, atol=1e-9)


def test_survival_curve_is_monotone_and_bounded():
    rng = np.random.default_rng(5)
    durations = rng.integers(1, 200, 300).astype(float)
    events = (rng.random(300) < 0.5).astype(int)
    curve = survival_mod._km(durations, events)
    assert curve["survival"].is_monotonic_decreasing
    assert curve["survival"].between(0, 1).all()
    assert (curve["ci_lower"] <= curve["survival"]).all()
    assert (curve["survival"] <= curve["ci_upper"]).all()


def test_left_truncation_raises_the_early_survival_estimate():
    """Ignoring delayed entry inflates the risk set and understates early survival."""
    durations = np.array([50.0, 60, 70, 200, 210, 220])
    events = np.array([1, 1, 1, 1, 1, 1])
    entry = np.array([0.0, 0, 0, 150, 150, 150])   # last three enter late

    naive = survival_mod._km(durations, events)
    truncated = survival_mod._km(durations, events, entry)
    at_100_naive = naive[naive["time"] <= 100]["survival"].iloc[-1]
    at_100_trunc = truncated[truncated["time"] <= 100]["survival"].iloc[-1]
    assert at_100_trunc < at_100_naive


# --------------------------------------------------------------------------- #
# causal
# --------------------------------------------------------------------------- #
def _confounded_data(n: int = 6000, true_effect: float = 0.06, seed: int = 11):
    rng = np.random.default_rng(seed)
    confounder = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.5, n)
    treated = (rng.random(n) < 1 / (1 + np.exp(-(0.9 * confounder)))).astype(int)
    base = np.clip(0.20 + 0.10 * confounder, 0.02, 0.9)
    y = (rng.random(n) < np.clip(base + true_effect * treated, 0.01, 0.99)).astype(int)

    X = pd.DataFrame({"g__confounder": confounder, "g__noise": noise,
                      "g__treatment": treated.astype(float)})
    meta = {
        name: features_mod.FeatureMeta(name, "g", "entity", "numeric", "static",
                                       None, "normal", "fixture", True)
        for name in X.columns
    }
    return features_mod.FeatureMatrix(X=X, meta=meta), treated, y


def test_aipw_recovers_a_known_effect_that_the_naive_gap_gets_wrong(tmp_path, customers):
    from tests.conftest import write_project

    cfg = config_mod.load(write_project(tmp_path, customers))
    fm, treated, y = _confounded_data(true_effect=0.06)

    estimate = causal_mod._estimate(
        cfg, fm, ["g__confounder", "g__noise"], treated, y,
        "t", "g__treatment", "fixture", {"cross_fit_folds": 4},
    )
    assert estimate.valid
    assert estimate.ate == pytest.approx(0.06, abs=0.025), estimate.ate
    assert abs(estimate.naive_difference - 0.06) > abs(estimate.ate - 0.06), (
        "the adjusted estimate should be closer to the truth than the raw gap"
    )
    assert estimate.ate_ci[0] < 0.06 < estimate.ate_ci[1]


def test_placebo_treatment_produces_no_effect(tmp_path, customers):
    from tests.conftest import write_project

    cfg = config_mod.load(write_project(tmp_path, customers))
    fm, treated, y = _confounded_data(true_effect=0.06)
    rng = np.random.default_rng(0)

    placebo = causal_mod._estimate(
        cfg, fm, ["g__confounder", "g__noise"], rng.permutation(treated), y,
        "placebo", "g__treatment", "fixture", {"cross_fit_folds": 4},
    )
    assert abs(placebo.ate) < 0.02, placebo.ate


def test_no_overlap_is_refused_rather_than_estimated(tmp_path, customers):
    """A confounder that determines treatment leaves nothing to compare against."""
    from tests.conftest import write_project

    cfg = config_mod.load(write_project(tmp_path, customers))
    fm, treated, y = _confounded_data()
    fm.X["g__mirror"] = treated.astype(float)     # a restatement of the treatment
    fm.meta["g__mirror"] = features_mod.FeatureMeta(
        "g__mirror", "g", "entity", "numeric", "static", None, "normal", "fixture", True
    )
    estimate = causal_mod._estimate(
        cfg, fm, ["g__confounder", "g__mirror"], treated, y,
        "t", "g__treatment", "fixture", {"cross_fit_folds": 4},
    )
    assert not estimate.valid
    assert any("propensity" in w.lower() or "positivity" in w.lower()
               for w in estimate.warnings)


def test_confounder_selection_drops_other_aggregations_of_the_treatment():
    fm_cols = ["support__complaints__sum_90d", "support__complaints__mean_90d",
               "usage__sessions__sum_30d", "contract__mrr"]
    fm = features_mod.FeatureMatrix(
        X=pd.DataFrame({c: [0.0] for c in fm_cols}),
        meta={c: features_mod.FeatureMeta(c, c.split("__")[0], "e", "numeric",
                                          "static", None, "normal", "", True)
              for c in fm_cols},
    )
    chosen = causal_mod._select_confounders(
        None, fm, "support__complaints__sum_90d",
        {"mode": "auto", "exclude": ["usage__sessions__sum_30d"]},
    )
    assert chosen == ["contract__mrr"]


def test_filtered_siblings_are_dropped_from_the_adjustment_set():
    """A base count equals the sum of its subsets, so siblings determine treatment."""
    fm_cols = ["billing__event_count_90d", "billing__failed__event_count_90d",
               "billing__paid__event_count_90d", "contract__mrr"]
    fm = features_mod.FeatureMatrix(
        X=pd.DataFrame({c: [0.0] for c in fm_cols}),
        meta={c: features_mod.FeatureMeta(c, c.split("__")[0], "e", "numeric",
                                          "static", None, "normal", "", True)
              for c in fm_cols},
    )
    chosen = causal_mod._select_confounders(
        None, fm, "billing__failed__event_count_90d", {"mode": "auto"}
    )
    assert chosen == ["contract__mrr"]


def test_derived_features_built_on_an_excluded_input_are_also_dropped():
    fm_cols = ["support__complaints__sum_90d", "usage__sessions__sum_30d",
               "contract__mrr"]
    meta = {
        c: features_mod.FeatureMeta(c, c.split("__")[0], "e", "numeric", "static",
                                    None, "normal", "", True)
        for c in fm_cols
    }
    meta["ratios__complaints_per_session"] = features_mod.FeatureMeta(
        "ratios__complaints_per_session", "ratios", "derived", "numeric", "derived",
        None, "normal", "", True,
        depends_on=["support__complaints__sum_90d", "usage__sessions__sum_30d"],
    )
    fm = features_mod.FeatureMatrix(
        X=pd.DataFrame({c: [0.0] for c in [*fm_cols, "ratios__complaints_per_session"]}),
        meta=meta,
    )
    chosen = causal_mod._select_confounders(
        None, fm, "support__complaints__sum_90d", {"mode": "auto"}
    )
    assert "ratios__complaints_per_session" not in chosen, (
        "the treatment came back in through a ratio"
    )
    assert set(chosen) == {"usage__sessions__sum_30d", "contract__mrr"}


def test_confounder_exclusion_accepts_globs():
    fm_cols = ["billing__days_past_due__sum_90d", "billing__days_past_due__mean_90d",
               "contract__mrr", "profile__region"]
    fm = features_mod.FeatureMatrix(
        X=pd.DataFrame({c: [0.0] for c in fm_cols}),
        meta={c: features_mod.FeatureMeta(c, c.split("__")[0], "e", "numeric",
                                          "static", None, "normal", "", True)
              for c in fm_cols},
    )
    chosen = causal_mod._select_confounders(
        None, fm, "billing__failed_payments__sum_90d",
        {"mode": "auto", "exclude": ["billing__days_past_due__*"]},
    )
    assert chosen == ["contract__mrr", "profile__region"]


# --------------------------------------------------------------------------- #
# experiment
# --------------------------------------------------------------------------- #
def test_sample_ratio_mismatch_is_detected():
    cohort = pd.DataFrame({"variant": ["a"] * 700 + ["b"] * 300})
    check = experiment_mod._srm_check(cohort, "variant")
    assert not check["passed"]

    balanced = pd.DataFrame({"variant": ["a"] * 505 + ["b"] * 495})
    assert experiment_mod._srm_check(balanced, "variant")["passed"]


def test_experiment_measures_a_real_treatment_effect(tmp_path):
    from tests.conftest import write_project

    rng = np.random.default_rng(2)
    n = 4000
    variant = rng.choice(["control", "treatment"], n)
    churn_p = np.where(variant == "treatment", 0.10, 0.20)
    churns = rng.random(n) < churn_p
    signup = pd.Timestamp("2023-02-01") + pd.to_timedelta(rng.integers(0, 30, n), "D")

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "signup_date": signup,
            "contract_end_date": np.where(
                churns, signup + pd.Timedelta(days=20), pd.NaT
            ),
            "region": rng.choice(["north", "south"], n),
            "variant": variant,
        }
    )
    config_dir = write_project(tmp_path, frame)
    (config_dir / "experiment.yaml").write_text(
        "enabled: true\nvariant_column: variant\ncontrol_value: control\n"
        "start_date: 2023-02-01\nend_date: 2023-03-05\nhorizon_days: 30\n"
    )
    cfg = config_mod.load(config_dir)
    ds = data_mod.load(cfg)
    pnl = panel_mod.build(cfg, ds, data_mod.resolve_timeline(cfg, ds))
    report = experiment_mod.run(cfg, ds, pnl)

    assert report.enabled and report.trustworthy
    treatment = next(r for r in report.results if r.variant == "treatment")
    assert treatment.lift_pp == pytest.approx(-10, abs=3)
    assert treatment.p_value < 0.001


def test_snapshot_cohort_experiment_needs_assignment_date(tmp_path):
    """Everyone assigned on one date, all signed up well before it.

    Filtering on signup date — the rolling-enrolment shape — throws the whole
    cohort away, so the two shapes must not be conflated.
    """
    from tests.conftest import write_project

    rng = np.random.default_rng(9)
    n = 3000
    variant = rng.choice(["control", "treatment"], n)
    signup = pd.Timestamp("2022-06-01") + pd.to_timedelta(rng.integers(0, 200, n), "D")
    assignment = pd.Timestamp("2023-03-01")
    churns = rng.random(n) < np.where(variant == "treatment", 0.08, 0.18)

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n)],
            "signup_date": signup,
            "contract_end_date": np.where(
                churns, assignment + pd.Timedelta(days=30), pd.NaT
            ),
            "region": rng.choice(["north", "south"], n),
            "variant": variant,
        }
    )
    config_dir = write_project(
        tmp_path, frame,
        target={"mode": "event_date", "event_date_column": "contract_end_date",
                "observation_end_date": "2023-12-31"},
        survivorship={"data_export_date": "2023-12-31",
                      "observation_starting_date": "2022-06-01",
                      "left_truncation": "keep_flagged"},
    )
    cfg = config_mod.load(config_dir)
    ds = data_mod.load(cfg)
    pnl = panel_mod.build(cfg, ds, data_mod.resolve_timeline(cfg, ds))

    # Wrong shape: signup-date filtering empties the cohort, and says why.
    (config_dir / "experiment.yaml").write_text(
        "enabled: true\nvariant_column: variant\ncontrol_value: control\n"
        "start_date: 2023-03-01\nend_date: 2023-03-01\nhorizon_days: 90\n"
    )
    wrong = experiment_mod.run(config_mod.load(config_dir), ds, pnl)
    assert not wrong.enabled
    assert "assignment_date" in wrong.notes[0]

    # Right shape.
    (config_dir / "experiment.yaml").write_text(
        "enabled: true\nvariant_column: variant\ncontrol_value: control\n"
        "assignment_date: 2023-03-01\nhorizon_days: 90\n"
    )
    report = experiment_mod.run(config_mod.load(config_dir), ds, pnl)
    assert report.enabled
    treatment = next(r for r in report.results if r.variant == "treatment")
    assert treatment.lift_pp == pytest.approx(-10, abs=3)
    assert treatment.p_value < 0.001


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_full_pipeline_produces_a_report(tmp_path, monkeypatch):
    from churnkit import pipeline as pipeline_mod
    from tests.conftest import write_project
    from tools.make_synthetic_data import generate

    generate(tmp_path / "data", n_customers=400, seed=1)
    customers = pd.read_csv(tmp_path / "data" / "customers.csv")
    events = {
        name: pd.read_csv(tmp_path / "data" / f"{name}_events.csv")
        for name in ("usage", "billing", "support")
    }
    config_dir = write_project(
        tmp_path, customers,
        events={f"{k}": v for k, v in events.items()},
        features={
            "contract": {"temporal": "static", "columns": [
                {"name": "plan_tier", "type": "categorical"},
                {"name": "mrr", "type": "numeric"},
                {"name": "auto_renew", "type": "boolean"},
            ]},
            "usage": {"temporal": "time_varying", "source": "usage",
                      "aggregation_window_days": 30, "generate_trends": True,
                      "columns": [{"name": "sessions", "type": "numeric"},
                                  {"name": "spend", "type": "numeric"}]},
            "support": {"temporal": "time_varying", "source": "support",
                        "aggregation_window_days": 90, "leakage_review": "strict",
                        "columns": [{"name": "complaints", "type": "numeric"}]},
        },
        pipeline={
            "panel": {"horizon_days": 90, "embargo_days": 7},
            "survival": {"enabled": True, "cox": False},
            "drivers": {"top_k": 8, "n_permutation_repeats": 2},
        },
        target={"mode": "event_date", "event_date_column": "contract_end_date",
                "observation_end_date": "2025-06-30"},
        survivorship={"data_export_date": "2025-06-30",
                      "observation_starting_date": "2022-01-01",
                      "left_truncation": "drop"},
    )
    monkeypatch.chdir(tmp_path)
    results = pipeline_mod.run(config_dir)

    report_path = results["report_path"]
    assert report_path.exists()
    html = report_path.read_text()
    assert "Data integrity and leakage" in html
    assert "Risk drivers" in html

    output = report_path.parent
    for name in ("feature_dictionary.csv", "risk_drivers.csv", "summary.json"):
        assert (output / name).exists(), name

    metrics = results["model"].metrics
    assert 0.4 < metrics["auc"] < 0.999
    assert results["features"].X.shape[1] > 5

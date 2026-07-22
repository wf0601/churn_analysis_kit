"""Tests for the messy-export parsers and the PLG ingest adapter.

The raw file is not in the repo, so these run against a synthetic export built to
contain every anomaly the case study description lists. That validates the parsers
against the documented contract; it does not prove they handle whatever else is
actually in the real file, which is what `--dry-run` and the coercion report are for.

The generic parsers live in churnkit/util/parsing.py and are tested there directly;
`ingest_plg` supplies only this export's column mapping and alias tables.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churnkit.util import parsing
from tools import ingest_plg as ingest

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# scalar parsers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("€15,50", 15.50),          # EU decimal comma with symbol
        ("EUR 15.50", 15.50),       # currency code, Anglo decimal
        ("$15.50", 15.50),
        ("15,50", 15.50),
        ("1.200,00", 1200.00),      # EU thousands + decimal
        ("1,200.00", 1200.00),      # Anglo thousands + decimal
        ("€180,00", 180.00),
        ("0,0", 0.0),
        ("0", 0.0),
        ("3", 3.0),
        ("-12,5", -12.5),
        ("1.234.567", 1234567.0),   # repeated separator can only be thousands
        ("", np.nan),
        ("Unknown", np.nan),
        ("N/A", np.nan),
    ],
)
def test_number_parsing_covers_every_documented_form(raw, expected):
    result = parsing.parse_number(raw)
    if isinstance(expected, float) and np.isnan(expected):
        assert np.isnan(result)
    else:
        assert result == pytest.approx(expected)


def test_ambiguous_separator_is_reported_not_hidden():
    """'1,250' is 1250 in one convention and 1.25 in the other."""
    flagged: list[str] = []
    value = parsing.parse_number("1,250", flagged)
    assert value == 1250.0          # documented choice: read as thousands
    assert flagged == ["1,250"]     # and surfaced, so a human can overrule it


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2022-03-04 00:00:00", "2022-03-04"),
        ("2022-03-04", "2022-03-04"),
        ("July 2018", "2018-07-01"),
        ("Jul 2018", "2018-07-01"),
        ("15 March 2022", "2022-03-15"),
        ("NA", None),
        ("", None),
    ],
)
def test_unambiguous_dates_parse(raw, expected):
    result = parsing.parse_one_date(raw, dayfirst=True)
    if expected is None:
        assert pd.isna(result)
    else:
        assert result == pd.Timestamp(expected)


def test_date_order_is_inferred_from_the_column():
    eu = pd.Series(["01/02/2022", "25/12/2022", "13/06/2022"])   # 25 and 13 => days
    us = pd.Series(["01/02/2022", "12/25/2022", "06/13/2022"])   # 25 and 13 => days, 2nd slot
    assert parsing.infer_dayfirst(eu)[0] is True
    assert parsing.infer_dayfirst(us)[0] is False

    mixed = pd.Series(["25/12/2022", "12/25/2022"])
    dayfirst, note = parsing.infer_dayfirst(mixed)
    assert note.startswith("MIXED"), "an unresolvable column must say so"


def test_boolean_encodings_normalise():
    assert parsing.parse_bool("Y") == 1.0
    assert parsing.parse_bool("no") == 0.0
    assert parsing.parse_bool("1") == 1.0
    assert parsing.parse_bool("FALSE") == 0.0
    assert np.isnan(parsing.parse_bool("NA"))


def test_engagement_history_handles_both_shapes():
    assert ingest.parse_engagement('{"email_opens": "5"}') == 5.0
    assert ingest.parse_engagement("opens=3") == 3.0
    assert ingest.parse_engagement("{'email_opens': 7}") == 7.0
    assert np.isnan(ingest.parse_engagement("missing"))


def test_capped_counts_keep_the_censoring_flag():
    assert parsing.parse_capped_count("10+") == (10.0, 1.0)
    assert parsing.parse_capped_count("5") == (5.0, 0.0)
    value, capped = parsing.parse_capped_count("missing")
    assert np.isnan(value) and np.isnan(capped)


def test_postcodes_are_coarsened_not_kept_raw():
    assert parsing.postcode_prefix("1012") == "10"
    assert parsing.postcode_prefix("SE-123 45") == "SE"
    assert parsing.postcode_prefix("EC1A 1BB") == "EC"


def test_win_dates_split_into_individual_events():
    assert len(parsing.split_dates("2022-03-04;2021-05-06")) == 2
    assert len(parsing.split_dates('["2022-03-04", "2021-05-06"]')) == 2
    assert parsing.split_dates("NA") == []


# --------------------------------------------------------------------------- #
# synthetic messy export
# --------------------------------------------------------------------------- #
PREDICTION_DATE = pd.Timestamp("2022-07-01")
OBSERVATION_END = pd.Timestamp("2023-03-31")


def make_messy_export(n: int = 1200, seed: int = 4) -> pd.DataFrame:
    """A raw export carrying every anomaly the case study description lists."""
    rng = np.random.default_rng(seed)

    tenure_years = rng.exponential(3.0, n).clip(0.1, 12)
    subscription_date = PREDICTION_DATE - pd.to_timedelta(tenure_years * 365, unit="D")
    cohort = rng.choice(["Control", "Variant_A", "Variant_B"], n, p=[0.34, 0.33, 0.33])

    failed = rng.poisson(0.35, n)
    complaints = rng.poisson(0.3, n)
    risk = rng.choice(["Low", "Medium", "High", "Very high"], n, p=[0.4, 0.3, 0.2, 0.1])
    risk_effect = pd.Series(risk).map(
        {"Low": -0.9, "Medium": -0.2, "High": 0.5, "Very high": 1.2}
    ).values
    # True effects: Variant B works, Variant A barely does.
    arm_effect = pd.Series(cohort).map(
        {"Control": 0.0, "Variant_A": -0.12, "Variant_B": -0.55}
    ).values

    logit = -1.1 + risk_effect + 0.45 * failed + 0.3 * complaints + arm_effect
    churned = rng.random(n) < 1 / (1 + np.exp(-logit))
    churn_offset = rng.integers(10, 273, n)
    churn_date = np.where(
        churned, PREDICTION_DATE + pd.to_timedelta(churn_offset, unit="D"), pd.NaT
    )

    def messy_money(values):
        out = []
        for i, v in enumerate(values):
            style = i % 4
            if style == 0:
                out.append(f"€{v:,.2f}".replace(",", "_").replace(".", ",").replace("_", "."))
            elif style == 1:
                out.append(f"EUR {v:.2f}")
            elif style == 2:
                out.append(f"${v:.2f}")
            else:
                out.append(f"{v:.2f}")
        return out

    def messy_date(values):
        out = []
        for i, v in enumerate(values):
            if pd.isna(v):
                out.append(rng.choice(["NA", "", "N/A"]))
            elif i % 3 == 0:
                out.append(pd.Timestamp(v).strftime("%Y-%m-%d %H:%M:%S"))
            elif i % 3 == 1:
                out.append(pd.Timestamp(v).strftime("%d/%m/%Y"))
            else:
                out.append(pd.Timestamp(v).strftime("%B %Y"))
        return out

    spend = rng.lognormal(2.6, 0.5, n)
    ages = rng.integers(19, 88, n).astype(object)
    ages[rng.random(n) < 0.05] = "Unknown"

    # Surveys and contacts straddle the prediction date on purpose — that is the
    # leakage trap the event-log split exists to defuse.
    survey_date = PREDICTION_DATE + pd.to_timedelta(rng.integers(-400, 250, n), unit="D")
    contact_date = PREDICTION_DATE + pd.to_timedelta(rng.integers(-400, 250, n), unit="D")

    return pd.DataFrame(
        {
            "customer_id": [f"cust{i % (n - 50):06x}" for i in range(n)],
            "subscription_id": [f"sub{i:06x}" for i in range(n)],
            "legacy_system_id": rng.choice(["SYS_OLD", "SYS_NEW"], n),
            "subscription_date": messy_date(subscription_date),
            "participant_age": ages,
            "marketing_channel": rng.choice(
                ["DM", "Direct mail", "Online paid", "referral", "TV"], n
            ),
            "country_code": rng.choice(["NL", "DE", "GB", "N/A", "nl"], n),
            "postcode_area": rng.choice(["1012", "3011", "SE-123 45", "EC1A 1BB"], n),
            "extra_draws_per_year": rng.choice(["0", "1", "2"], n),
            "Add-ons": rng.choice(["0", "1", "3", "0,0"], n),
            "payment_method": rng.choice(
                ["Direct debit", "Credit card", "iDEAL", "PayPal"], n
            ),
            "failed_payments_12m": [str(v) if rng.random() > 0.05 else "NaN" for v in failed],
            "monthly_spend_estimated": messy_money(spend),
            "donation_share_charity": [
                f"{v:.2f}" if i % 2 else f"{100 * v:.0f}"
                for i, v in enumerate(rng.uniform(0.2, 0.6, n))
            ],
            "engagement_history": [
                '{"email_opens": "%d"}' % v if i % 3 == 0
                else (f"opens={v}" if i % 3 == 1 else "missing")
                for i, v in enumerate(rng.poisson(4, n))
            ],
            "web_sessions_90d_raw": rng.choice(["0", "5", "10+", "missing", "3"], n),
            "c_service_contacts_12m": rng.choice(["0", "1", "2", "99"], n, p=[.5, .3, .18, .02]),
            "complaints_12m": complaints.astype(str),
            "lifetime_wins": rng.poisson(0.4, n).astype(str),
            "win_dates": [
                ";".join(
                    (PREDICTION_DATE - pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")
                    for d in rng.integers(30, 900, rng.integers(0, 3))
                ) or "NA"
                for _ in range(n)
            ],
            "campaign_cohort": [
                c if i % 5 else c.lower() for i, c in enumerate(cohort)
            ],
            "treatment_sent_flag": [
                ("0" if rng.random() < 0.15 else rng.choice(["1", "Y"]))
                if c != "Control" else rng.choice(["0", "N"])
                for c in cohort
            ],
            "offer_cost_eur": messy_money(np.where(cohort == "Control", 0.0, 20.0)),
            "baseline_churn_risk_band": risk,
            "historic_revenue_12m": messy_money(spend * 12),
            "churned": [rng.choice(["1", "Y"]) if c else rng.choice(["0", "N"]) for c in churned],
            "churn_date": messy_date(churn_date),
            "observation_end_date": OBSERVATION_END.strftime("%Y-%m-%d"),
            "revenue_next_12m_observed": messy_money(spend * 12 * (1 - churned)),
            "service_tier_upgrade_date": messy_date(
                np.where(rng.random(n) < 0.2,
                         PREDICTION_DATE + pd.Timedelta(days=120), pd.NaT)
            ),
            "satisfaction_score": rng.choice(["8", "3", "6", "NA", "10"], n),
            "survey_date": messy_date(survey_date),
            "last_contact_date": messy_date(contact_date),
            "last_contact_reason": rng.choice(
                ["billing", "cancellation_inquiry", "offer_question", "prize_claim",
                 "other", "NA"], n
            ),
        }
    )


@pytest.fixture
def ingested(tmp_path):
    raw_dir = tmp_path / "data" / "raw"
    raw_dir.mkdir(parents=True)
    raw_path = raw_dir / "advanced_case_study_data.csv"
    make_messy_export().to_csv(raw_path, index=False)
    out_dir = tmp_path / "data" / "plg"
    report = ingest.ingest(raw_path, out_dir, PREDICTION_DATE)
    return tmp_path, out_dir, report


def test_ingest_produces_every_expected_file(ingested):
    _, out_dir, _ = ingested
    for name in ("subscriptions.csv", "survey_events.csv", "contact_events.csv",
                 "win_events.csv", "upgrade_events.csv", "ingest_report.csv"):
        assert (out_dir / name).exists(), name


def test_money_columns_survive_the_mixed_formats(ingested):
    _, out_dir, _ = ingested
    subs = pd.read_csv(out_dir / "subscriptions.csv")
    spend = subs["monthly_spend_eur"]
    assert spend.notna().mean() > 0.98
    # lognormal(2.6, 0.5) sits around 13-14; a broken decimal separator would put
    # the mean out by a factor of 100.
    assert 5 < spend.median() < 40


def test_categories_are_collapsed_to_canonical_values(ingested):
    _, out_dir, _ = ingested
    subs = pd.read_csv(out_dir / "subscriptions.csv")
    assert set(subs["campaign_cohort"].dropna()) <= {"Control", "Variant_A", "Variant_B"}
    assert "direct_mail" in set(subs["marketing_channel"])
    assert set(subs["country_code"].dropna()) <= {"NL", "DE", "GB"}
    assert subs["postcode_prefix"].nunique() < 10


def test_donation_share_is_normalised_to_a_fraction(ingested):
    _, out_dir, _ = ingested
    subs = pd.read_csv(out_dir / "subscriptions.csv")
    share = subs["donation_share_charity"].dropna()
    assert share.between(0, 1).all(), "percentage-style values were not rescaled"


def test_capped_web_sessions_keep_their_flag(ingested):
    _, out_dir, _ = ingested
    subs = pd.read_csv(out_dir / "subscriptions.csv")
    capped = subs["web_sessions_capped"]
    assert capped.sum() > 0
    assert subs.loc[capped == 1, "web_sessions_90d"].eq(10).all()


def test_post_prediction_records_are_isolated_into_event_logs(ingested):
    """The whole reason the dated columns are split out."""
    _, out_dir, _ = ingested
    subs = pd.read_csv(out_dir / "subscriptions.csv")
    assert "satisfaction_score" not in subs.columns
    assert "last_contact_reason" not in subs.columns

    survey = pd.read_csv(out_dir / "survey_events.csv", parse_dates=["event_date"])
    after = (survey["event_date"] >= PREDICTION_DATE).sum()
    assert after > 0, "the fixture should contain post-prediction surveys"
    # They are still in the event log — the WINDOW is what excludes them, which the
    # pipeline test below verifies end to end.


def test_outcome_columns_are_not_carried_into_features(ingested):
    _, out_dir, _ = ingested
    subs = pd.read_csv(out_dir / "subscriptions.csv")
    for banned in ("revenue_next_12m_observed", "offer_cost_eur",
                   "service_tier_upgrade_date", "observation_end_date"):
        assert banned not in subs.columns, banned


def test_non_compliance_is_reported(ingested):
    _, _, report = ingested
    assert any("non-compliance" in w or "INTENTION-TO-TREAT" in w.upper()
               for w in report.warnings)


def test_repeat_customers_are_reported_as_a_clustering_risk(ingested):
    _, _, report = ingested
    assert any("more than one subscription" in w for w in report.warnings)


# --------------------------------------------------------------------------- #
# end to end through the real config
# --------------------------------------------------------------------------- #
def test_plg_config_runs_end_to_end(ingested, monkeypatch):
    from churnkit import pipeline as pipeline_mod

    tmp_path, out_dir, _ = ingested
    shutil.copytree(ROOT / "config_plg", tmp_path / "config_plg")
    monkeypatch.chdir(tmp_path)

    results = pipeline_mod.run(tmp_path / "config_plg")

    assert results["report_path"].exists()
    fm = results["features"]

    # The point-in-time guarantee, checked on the real config: no survey or contact
    # dated on or after T - embargo may influence any feature.
    survey = pd.read_csv(out_dir / "survey_events.csv", parse_dates=["event_date"])
    cutoff = PREDICTION_DATE - pd.Timedelta(days=7)
    assert (survey["event_date"] >= cutoff).any(), "fixture must contain trapped rows"
    in_window = survey[survey["event_date"] < cutoff]
    built = fm.X["survey__satisfaction_score__last_365d"].notna().sum()
    assert built <= len(in_window), "a post-cutoff survey reached a feature"

    # The experiment is the headline, and Variant B has a real effect by construction.
    experiment = results["experiment"]
    assert experiment.enabled
    variant_b = next(r for r in experiment.results if r.variant == "Variant_B")
    assert variant_b.lift_pp < 0
    assert variant_b.p_value < 0.05

    # The legacy risk band is allowlisted past the "churn" pattern, so it survives.
    assert "legacy_score__baseline_churn_risk_band" in fm.X.columns
    assert results["model"].metrics["auc"] > 0.55

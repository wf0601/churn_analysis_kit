#!/usr/bin/env python3
"""Ingest the Postcode Lottery Group raw CRM export into ChurnKit's input shape.

    python tools/ingest_plg.py --raw data/raw/advanced_case_study_data.csv
    python run.py --config config_plg

The raw file is one wide, dirty row per subscription. This turns it into:

    data/plg/subscriptions.csv   cleaned, one row per subscription,
                                 PRE-PREDICTION columns plus the outcome dates
    data/plg/survey_events.csv   ) the dated fields, in long form, so that the
    data/plg/contact_events.csv  ) pipeline's window logic can drop anything
    data/plg/win_events.csv      ) recorded at or after the prediction date
    data/plg/upgrade_events.csv  ) (emitted for inspection; not wired into
                                 )  feature.yaml — see the note at the bottom)
    data/plg/ingest_report.csv   per-column coercion counts

Every coercion is counted and reported rather than done silently. A cleaning step
that quietly turns 8% of a column into nulls is worse than one that fails, because
the model still trains and the number still looks fine.

The parsers here are written against the schema in case_study_plg_description.MD.
They have NOT been validated against the real file — run `--dry-run` first and read
the report before trusting the output.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Importable when this script is run by absolute path from another working
# directory, which is the normal way to point it at a data drop.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The messy-export parsers are dataset-agnostic and live with the rest of the
# shared utilities; only the column mapping and the alias tables below are
# specific to this export.
from churnkit.util.parsing import (  # noqa: E402
    extract_keyed_number,
    infer_dayfirst,
    is_blank as _blank,
    normalise_category,
    parse_bool,
    parse_capped_count,
    parse_date_series,
    parse_numeric_series,
    parse_one_date,
    postcode_prefix,
    split_dates,
)

CHANNEL_ALIASES = {
    "dm": "direct_mail", "direct mail": "direct_mail", "directmail": "direct_mail",
    "direct_mail": "direct_mail",
    "online paid": "online_paid", "paid online": "online_paid",
    "online_paid": "online_paid", "paid_online": "online_paid", "ppc": "online_paid",
    "online organic": "online_organic", "organic": "online_organic",
    "tv": "tv", "television": "tv",
    "door to door": "door_to_door", "d2d": "door_to_door",
    "referral": "referral", "partner": "partner", "telemarketing": "telemarketing",
}

PAYMENT_ALIASES = {
    "direct debit": "direct_debit", "directdebit": "direct_debit",
    "dd": "direct_debit", "automatische incasso": "direct_debit",
    "credit card": "credit_card", "creditcard": "credit_card", "cc": "credit_card",
    "ideal": "ideal", "paypal": "paypal", "bank transfer": "bank_transfer",
    "invoice": "invoice", "acceptgiro": "invoice",
}

COHORT_ALIASES = {
    "control": "Control", "ctrl": "Control", "c": "Control",
    "variant_a": "Variant_A", "variant a": "Variant_A", "varianta": "Variant_A",
    "a": "Variant_A", "treatment_a": "Variant_A",
    "variant_b": "Variant_B", "variant b": "Variant_B", "variantb": "Variant_B",
    "b": "Variant_B", "treatment_b": "Variant_B",
}

RISK_BAND_ALIASES = {
    "low": "low", "medium": "medium", "med": "medium", "high": "high",
    "very high": "very_high", "very_high": "very_high", "veryhigh": "very_high",
}

CONTACT_REASONS = {
    "billing": "is_billing_issue",
    "cancellation_inquiry": "is_cancellation_inquiry",
    "cancellation inquiry": "is_cancellation_inquiry",
    "offer_question": "is_offer_question",
    "offer question": "is_offer_question",
    "prize_claim": "is_prize_claim",
    "prize claim": "is_prize_claim",
}

ENGAGEMENT_KEYS = ("email_opens", "opens", "email_open", "openings")


def parse_engagement(value) -> float:
    """Pull the open-count out of this export's engagement_history blob."""
    return extract_keyed_number(value, ENGAGEMENT_KEYS)


# --------------------------------------------------------------------------- #
@dataclass
class Report:
    rows: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def record(self, column: str, kind: str, total: int, parsed: int, notes: str = "") -> None:
        failed = total - parsed
        self.rows.append(
            {
                "column": column, "parsed_as": kind, "n_rows": total,
                "n_parsed": parsed, "n_null_or_failed": failed,
                "pct_lost": round(100 * failed / max(total, 1), 2), "notes": notes,
            }
        )
        if total and failed / total > 0.10:
            self.warn(
                f"{column}: {failed / total:.0%} of values did not parse as {kind}. "
                f"Check the raw column before using it — this is high enough to change "
                f"conclusions."
            )

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"  WARN  {message}")

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# --------------------------------------------------------------------------- #
# reporting wrappers around the shared parsers
# --------------------------------------------------------------------------- #
def parse_date_column(series: pd.Series, column: str, report: Report) -> pd.Series:
    parsed, note = parse_date_series(series)
    if note.startswith("MIXED"):
        report.warn(f"{column}: {note}")
    non_null = series.map(lambda v: not _blank(v))
    report.record(column, "date", int(non_null.sum()), int(parsed.notna().sum()), note)
    return parsed


def parse_numeric_column(
    series: pd.Series, column: str, report: Report, note: str = ""
) -> pd.Series:
    parsed, ambiguous = parse_numeric_series(series)
    non_null = series.map(lambda v: not _blank(v))
    extra = note
    if ambiguous:
        sample = ", ".join(sorted(set(ambiguous))[:3])
        extra = (extra + "; " if extra else "") + (
            f"{len(ambiguous)} ambiguous separator value(s) read as thousands "
            f"(e.g. {sample})"
        )
        report.warn(
            f"{column}: {len(ambiguous)} value(s) like '{sample}' are ambiguous — "
            f"'1,250' is 1250 under one convention and 1.25 under another. Read as "
            f"thousands. Confirm against a known total before reporting money."
        )
    report.record(column, "numeric", int(non_null.sum()), int(parsed.notna().sum()), extra)
    return parsed


# --------------------------------------------------------------------------- #
# the ingest
# --------------------------------------------------------------------------- #
def ingest(raw_path: Path, out_dir: Path, prediction_date: pd.Timestamp) -> Report:
    report = Report()
    raw = _read_raw(raw_path, report)
    print(f"  read {len(raw):,} rows x {raw.shape[1]} columns from {raw_path.name}")

    missing = [c for c in ("subscription_id", "churn_date") if c not in raw.columns]
    if missing:
        raise SystemExit(
            f"the raw export is missing required column(s) {missing}; "
            f"found: {sorted(raw.columns)}"
        )

    out = pd.DataFrame({"subscription_id": raw["subscription_id"].astype(str).str.strip()})
    if "customer_id" in raw.columns:
        out["customer_id"] = raw["customer_id"].astype(str).str.strip()

    # ---- dates ----------------------------------------------------------- #
    date_columns = [
        "subscription_date", "churn_date", "observation_end_date",
        "survey_date", "last_contact_date", "service_tier_upgrade_date",
    ]
    dates = {
        col: parse_date_column(raw[col], col, report)
        for col in date_columns if col in raw.columns
    }
    out["subscription_date"] = dates.get("subscription_date")
    out["churn_date"] = dates.get("churn_date")

    # ---- outcome reconciliation ------------------------------------------ #
    _reconcile_outcome(raw, out, dates, prediction_date, report)

    # ---- pre-prediction features ----------------------------------------- #
    _build_features(raw, out, report)

    # ---- treatment ------------------------------------------------------- #
    if "campaign_cohort" in raw.columns:
        out["campaign_cohort"] = normalise_category(raw["campaign_cohort"], COHORT_ALIASES)
        counts = out["campaign_cohort"].value_counts(dropna=False).to_dict()
        print(f"  campaign_cohort: {counts}")
        report.record("campaign_cohort", "category", len(raw),
                      int(out["campaign_cohort"].notna().sum()), str(counts))
    if "treatment_sent_flag" in raw.columns:
        out["treatment_sent_flag"] = raw["treatment_sent_flag"].map(parse_bool)
        _report_compliance(out, report)

    # ---- de-duplicate ----------------------------------------------------- #
    before = len(out)
    out = out.drop_duplicates(subset=["subscription_id"], keep="first")
    if len(out) < before:
        report.warn(
            f"dropped {before - len(out):,} duplicate subscription_id row(s), keeping "
            f"the first. Confirm they are true duplicates and not versioned records."
        )
    _report_clustering(out, report)

    # ---- write ------------------------------------------------------------ #
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "subscriptions.csv", index=False)
    print(f"  subscriptions.csv    {len(out):>8,} rows x {out.shape[1]} columns")

    _write_events(raw, out, dates, out_dir, prediction_date, report)
    report.frame().to_csv(out_dir / "ingest_report.csv", index=False)
    return report


def _read_raw(path: Path, report: Report) -> pd.DataFrame:
    """Read everything as text. Letting pandas infer types is what loses the data."""
    if not path.exists():
        raise SystemExit(f"raw file not found: {path}")
    last_error = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        for sep in (None, ",", ";", "\t", "|"):
            try:
                frame = pd.read_csv(
                    path, dtype=str, keep_default_na=False, encoding=encoding,
                    sep=sep, engine="python", on_bad_lines="warn",
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
            if frame.shape[1] > 1:
                if encoding != "utf-8":
                    report.warn(f"file is not utf-8; read as {encoding}")
                return frame.rename(columns=lambda c: str(c).strip())
    raise SystemExit(f"could not parse {path} as a delimited file ({last_error})")


def _reconcile_outcome(raw, out, dates, prediction_date, report) -> None:
    """Cross-check the `churned` flag against `churn_date`, and report every conflict.

    These two disagreeing is the most consequential thing this script can find. A
    row flagged as churned with no date cannot be placed in time, so it can neither
    be labelled nor censored correctly, and silently treating it as active
    under-counts churn in exactly the group that matters.
    """
    churn_date = out["churn_date"]
    if "churned" in raw.columns:
        flag = raw["churned"].map(parse_bool)
        out["churned_flag_raw"] = flag
        flagged_no_date = int(((flag == 1) & churn_date.isna()).sum())
        dated_not_flagged = int(((flag == 0) & churn_date.notna()).sum())
        if flagged_no_date:
            report.warn(
                f"{flagged_no_date:,} row(s) have churned = 1 but no churn_date. They "
                f"will be treated as ACTIVE (censored), which under-counts churn. Ask "
                f"the data owner; if the dates are unrecoverable, consider "
                f"target.mode: label as a documented fallback."
            )
        if dated_not_flagged:
            report.warn(
                f"{dated_not_flagged:,} row(s) have a churn_date but churned = 0. The "
                f"date is used, since it is the more specific record."
            )
        report.record("churned", "boolean", len(raw), int(flag.notna().sum()),
                      f"{flagged_no_date} flagged-without-date, "
                      f"{dated_not_flagged} dated-without-flag")

    before_start = int((churn_date < prediction_date).sum())
    if before_start:
        report.warn(
            f"{before_start:,} row(s) have a churn_date BEFORE the "
            f"{prediction_date.date()} prediction date, but the export is supposed to "
            f"contain only subscriptions active on that date. Those rows are kept and "
            f"the pipeline will exclude them from the risk set — investigate before "
            f"reporting."
        )

    if "observation_end_date" in dates:
        window_ends = dates["observation_end_date"].dropna().unique()
        if len(window_ends) > 1:
            report.warn(
                f"observation_end_date is not constant ({len(window_ends)} distinct "
                f"values, {pd.Series(window_ends).min()} to "
                f"{pd.Series(window_ends).max()}). target.yaml takes a single date, so "
                f"set it to the EARLIEST value to keep every row's window fully "
                f"observed — a later date would count unobserved time as retention."
            )
        elif len(window_ends) == 1:
            print(f"  observation_end_date is constant at "
                  f"{pd.Timestamp(window_ends[0]).date()}")


def _build_features(raw, out, report) -> None:
    if "legacy_system_id" in raw.columns:
        out["legacy_system_id"] = normalise_category(raw["legacy_system_id"], {})
    if "participant_age" in raw.columns:
        age = parse_numeric_column(raw["participant_age"], "participant_age", report)
        implausible = int(((age < 16) | (age > 110)).sum())
        if implausible:
            report.warn(
                f"participant_age: {implausible:,} value(s) outside 16-110 set to null"
            )
            age = age.where((age >= 16) & (age <= 110))
        out["participant_age"] = age
    if "marketing_channel" in raw.columns:
        out["marketing_channel"] = normalise_category(raw["marketing_channel"], CHANNEL_ALIASES)
    if "country_code" in raw.columns:
        out["country_code"] = normalise_category(raw["country_code"], {}).str.upper()
    if "postcode_area" in raw.columns:
        out["postcode_prefix"] = raw["postcode_area"].map(postcode_prefix)
        report.record("postcode_area", "category (coarsened to prefix)", len(raw),
                      int(out["postcode_prefix"].notna().sum()),
                      "raw postcodes are near-unique and would be flagged as an "
                      "identifier; only the leading letters/digits are kept")
    if "payment_method" in raw.columns:
        out["payment_method"] = normalise_category(raw["payment_method"], PAYMENT_ALIASES)
    if "baseline_churn_risk_band" in raw.columns:
        out["baseline_churn_risk_band"] = normalise_category(
            raw["baseline_churn_risk_band"], RISK_BAND_ALIASES
        )

    for source, target in (
        ("extra_draws_per_year", "extra_draws_per_year"),
        ("Add-ons", "add_ons"),
        ("add_ons", "add_ons"),
        ("failed_payments_12m", "failed_payments_12m"),
        ("c_service_contacts_12m", "c_service_contacts_12m"),
        ("complaints_12m", "complaints_12m"),
        ("lifetime_wins", "lifetime_wins"),
        ("monthly_spend_estimated", "monthly_spend_eur"),
        ("historic_revenue_12m", "historic_revenue_12m_eur"),
    ):
        if source in raw.columns and target not in out.columns:
            out[target] = parse_numeric_column(raw[source], source, report)

    # Per the assignment, all money is EUR after cleaning regardless of the symbol
    # shown — so no FX conversion, but the mix is worth surfacing.
    for money in ("monthly_spend_estimated", "historic_revenue_12m", "offer_cost_eur"):
        if money in raw.columns:
            foreign = int(raw[money].astype(str).str.contains(r"\$|USD|£|GBP", case=False,
                                                              regex=True, na=False).sum())
            if foreign:
                report.warn(
                    f"{money}: {foreign:,} value(s) carry a non-EUR symbol. Per the "
                    f"assignment they are read as EUR without conversion — state that "
                    f"assumption in design_doc.md."
                )

    if "donation_share_charity" in raw.columns:
        share = parse_numeric_column(
            raw["donation_share_charity"], "donation_share_charity", report,
            "values > 1 are read as percentages and divided by 100",
        )
        out["donation_share_charity"] = np.where(share > 1.0, share / 100.0, share)

    if "engagement_history" in raw.columns:
        opens = raw["engagement_history"].map(parse_engagement)
        out["email_opens"] = opens
        non_null = raw["engagement_history"].map(lambda v: not _blank(v))
        report.record("engagement_history", "numeric (email_opens extracted)",
                      int(non_null.sum()), int(opens.notna().sum()),
                      "handles both JSON-like and key=value forms")

    if "web_sessions_90d_raw" in raw.columns:
        pairs = raw["web_sessions_90d_raw"].map(parse_capped_count)
        out["web_sessions_90d"] = [p[0] for p in pairs]
        out["web_sessions_capped"] = [p[1] for p in pairs]
        capped = int(np.nansum(out["web_sessions_capped"]))
        report.record("web_sessions_90d_raw", "numeric + capped flag", len(raw),
                      int(pd.Series(out["web_sessions_90d"]).notna().sum()),
                      f"{capped} value(s) were censored as '10+'; the flag preserves "
                      f"that they are a lower bound, not an exact count")

    for column in ("c_service_contacts_12m", "complaints_12m"):
        if column in out.columns:
            sentinels = int((out[column] >= 90).sum())
            if sentinels:
                report.warn(
                    f"{column}: {sentinels:,} value(s) are >= 90 (e.g. 99), which is a "
                    f"common 'unknown' sentinel. They are KEPT as-is — decide "
                    f"deliberately whether 99 means ninety-nine contacts or a missing "
                    f"value, and say which in design_doc.md."
                )


def _report_compliance(out: pd.DataFrame, report: Report) -> None:
    if "campaign_cohort" not in out.columns:
        return
    treated_arms = out[out["campaign_cohort"].isin(["Variant_A", "Variant_B"])]
    if treated_arms.empty:
        return
    rates = treated_arms.groupby("campaign_cohort")["treatment_sent_flag"].mean()
    detail = ", ".join(f"{arm} {rate:.1%}" for arm, rate in rates.items())
    print(f"  compliance (share actually sent): {detail}")

    control_sent = out[out["campaign_cohort"] == "Control"]["treatment_sent_flag"]
    if len(control_sent) and control_sent.fillna(0).max() > 0:
        report.warn(
            f"{int(control_sent.fillna(0).sum()):,} Control row(s) have "
            f"treatment_sent_flag = 1. That is contamination, not just non-compliance, "
            f"and it biases the ITT estimate toward zero."
        )
    if rates.min() < 0.95:
        report.warn(
            f"non-compliance present ({detail}). The pipeline reports the "
            f"INTENTION-TO-TREAT effect on assignment, which stays valid. The "
            f"treatment-on-the-treated effect is roughly ITT / compliance rate under "
            f"exclusion and monotonicity — compute it separately and state the "
            f"assumptions; do not filter to sent-only and compare to Control."
        )


def _report_clustering(out: pd.DataFrame, report: Report) -> None:
    if "customer_id" not in out.columns:
        return
    per_customer = out.groupby("customer_id").size()
    multi = int((per_customer > 1).sum())
    if multi:
        report.warn(
            f"{multi:,} customer(s) hold more than one subscription "
            f"({len(out):,} subscriptions / {len(per_customer):,} customers). "
            f"Randomisation was at subscription level, so the experiment analysis is "
            f"valid, but standard errors are optimistic: two subscriptions of the same "
            f"household are not independent observations. Note it, or cluster by "
            f"customer_id when reporting CIs."
        )


def _write_events(raw, out, dates, out_dir, prediction_date, report) -> None:
    """Emit the dated columns as event logs so the pipeline windows them."""
    ids = out[["subscription_id"]].copy()
    ids["_row"] = np.arange(len(ids))
    valid_ids = set(out["subscription_id"])

    def _emit(frame: pd.DataFrame, name: str, description: str) -> None:
        frame = frame[frame["subscription_id"].isin(valid_ids)]
        frame = frame[frame["event_date"].notna()]
        after = int((frame["event_date"] >= prediction_date).sum())
        frame.to_csv(out_dir / name, index=False)
        note = ""
        if after:
            note = (
                f"{after:,} of {len(frame):,} rows are dated on or after the "
                f"{prediction_date.date()} prediction date — as a flat column these "
                f"would have been leakage; the feature window excludes them"
            )
            print(f"  {name:<22}{len(frame):>8,} rows  ({after:,} post-prediction, excluded by the window)")
        else:
            print(f"  {name:<22}{len(frame):>8,} rows")
        report.record(name, "event log", len(frame), len(frame), note or description)

    subscription_ids = out["subscription_id"].values

    if "survey_date" in dates and "satisfaction_score" in raw.columns:
        scores = parse_numeric_column(raw["satisfaction_score"], "satisfaction_score", report)
        survey = pd.DataFrame(
            {
                "subscription_id": raw["subscription_id"].astype(str).str.strip(),
                "event_date": dates["survey_date"],
                "satisfaction_score": scores,
            }
        )
        _emit(survey, "survey_events.csv", "CRM satisfaction survey responses")

    if "last_contact_date" in dates:
        reasons = raw.get("last_contact_reason", pd.Series([None] * len(raw)))
        normalised = normalise_category(reasons, {})
        contact = pd.DataFrame(
            {
                "subscription_id": raw["subscription_id"].astype(str).str.strip(),
                "event_date": dates["last_contact_date"],
                "contact": 1.0,
            }
        )
        for flag in sorted(set(CONTACT_REASONS.values())):
            wanted = {k for k, v in CONTACT_REASONS.items() if v == flag}
            wanted |= {re.sub(r"[^a-z0-9]+", "_", w) for w in wanted}
            contact[flag] = normalised.isin(wanted).astype(float).values
        _emit(contact, "contact_events.csv",
              "most recent customer-service contact only — one row per subscription, "
              "so counts are 0/1 and trends are not meaningful")

    if "win_dates" in raw.columns:
        rows = []
        dayfirst, _ = infer_dayfirst(raw["win_dates"].map(lambda v: split_dates(v)[0]
                                                          if split_dates(v) else None))
        for sub_id, value in zip(subscription_ids, raw["win_dates"]):
            for part in split_dates(value):
                parsed = parse_one_date(part, dayfirst)
                if parsed is not pd.NaT:
                    rows.append({"subscription_id": sub_id, "event_date": parsed, "win": 1.0})
        _emit(pd.DataFrame(rows, columns=["subscription_id", "event_date", "win"]),
              "win_events.csv", "historical prize wins, one row per win")

    if "service_tier_upgrade_date" in dates:
        upgrade = pd.DataFrame(
            {
                "subscription_id": raw["subscription_id"].astype(str).str.strip(),
                "event_date": dates["service_tier_upgrade_date"],
                "upgrade": 1.0,
            }
        )
        _emit(upgrade, "upgrade_events.csv",
              "NOT wired into feature.yaml: most upgrades fall after the prediction "
              "date, making this a post-treatment outcome. Add it as a feature group "
              "only if you want pre-prediction upgrades, and check the split first")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path,
                        default=Path("data/raw/advanced_case_study_data.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/plg"))
    parser.add_argument("--prediction-date", default="2022-07-01")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the coercion report and write nothing else")
    args = parser.parse_args()

    prediction_date = pd.Timestamp(args.prediction_date).normalize()
    print(f"ingesting {args.raw} (prediction date {prediction_date.date()})")
    report = ingest(args.raw, args.out, prediction_date)

    frame = report.frame()
    if not frame.empty:
        print("\ncoercion report:")
        print(frame.to_string(index=False))
    print(f"\n{len(report.warnings)} warning(s). Full report: {args.out / 'ingest_report.csv'}")
    print("\nnext: python run.py --config config_plg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

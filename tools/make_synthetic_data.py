#!/usr/bin/env python3
"""Generate a synthetic subscription dataset shaped like the default config.

The point is to exercise the whole pipeline on data whose true structure is known,
including the failure modes the kit is built to catch:

  * churn driven by latent account health, which is never written to any table —
    so no model can be perfect, and an AUC near 1 would mean something is wrong;
  * a death-spiral pattern (collections activity spikes in the days before
    cancellation) that leaks badly at embargo 0 and is defused at embargo 7-14;
  * right-censoring: most customers are still active when the export is taken;
  * a randomised retention experiment on the 2024 signup cohorts.

Run with `python run.py demo`, or directly for a different size/seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OBSERVATION_START = pd.Timestamp("2022-01-01")
DATA_EXPORT = pd.Timestamp("2025-06-30")
EXPERIMENT_START = pd.Timestamp("2024-01-01")
EXPERIMENT_END = pd.Timestamp("2024-09-30")

PLAN_TIERS = ["basic", "standard", "premium"]
PLAN_WEIGHTS = [0.45, 0.38, 0.17]
REGIONS = ["north", "south", "east", "west"]
CHANNELS = ["organic", "paid_search", "referral", "partner", "outbound"]
PAYMENT_METHODS = ["card", "direct_debit", "invoice", "paypal"]


def generate(out_dir: Path, n_customers: int = 3000, seed: int = 7) -> dict[str, Path]:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    customers = _make_customers(rng, n_customers)
    usage, billing, support, churn_dates = _simulate(rng, customers)

    customers["contract_end_date"] = churn_dates
    customers["plan_change_count"] = rng.poisson(0.4, len(customers))
    customers["prior_winback"] = rng.random(len(customers)) < 0.06

    entity_columns = [
        "customer_id", "signup_date", "contract_end_date", "birthyear", "gender",
        "region", "acquisition_channel", "plan_tier", "mrr", "contract_type",
        "auto_renew", "payment_method", "plan_change_count", "prior_winback",
        "tenure_days", "variant",
    ]
    customers["tenure_days"] = (
        customers["contract_end_date"].fillna(DATA_EXPORT) - customers["signup_date"]
    ).dt.days

    paths = {}
    for name, frame, columns in (
        ("customers.csv", customers, entity_columns),
        ("usage_events.csv", usage, None),
        ("billing_events.csv", billing, None),
        ("support_events.csv", support, None),
    ):
        path = out_dir / name
        (frame[columns] if columns else frame).to_csv(path, index=False)
        paths[name] = path

    churned = int(customers["contract_end_date"].notna().sum())
    print(
        f"  customers.csv       {len(customers):>8,} rows  "
        f"({churned:,} churned, {len(customers) - churned:,} censored at "
        f"{DATA_EXPORT.date()})"
    )
    for name in ("usage_events.csv", "billing_events.csv", "support_events.csv"):
        frame = {"usage_events.csv": usage, "billing_events.csv": billing,
                 "support_events.csv": support}[name]
        print(f"  {name:<20}{len(frame):>8,} rows")
    print(f"  written to {out_dir.resolve()}")
    return paths


def _make_customers(rng: np.random.Generator, n: int) -> pd.DataFrame:
    span_days = (pd.Timestamp("2024-09-01") - OBSERVATION_START).days
    signup = OBSERVATION_START + pd.to_timedelta(rng.integers(0, span_days, n), unit="D")

    plan = rng.choice(PLAN_TIERS, n, p=PLAN_WEIGHTS)
    contract_type = np.where(rng.random(n) < 0.62, "monthly", "annual")
    base_mrr = {"basic": 25, "standard": 65, "premium": 180}
    mrr = np.array([base_mrr[p] for p in plan]) * rng.lognormal(0, 0.22, n)

    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "signup_date": signup,
            "birthyear": rng.integers(1955, 2004, n),
            "gender": rng.choice(["f", "m", "x"], n, p=[0.47, 0.47, 0.06]),
            "region": rng.choice(REGIONS, n, p=[0.3, 0.25, 0.25, 0.2]),
            "acquisition_channel": rng.choice(CHANNELS, n, p=[0.3, 0.28, 0.16, 0.14, 0.12]),
            "plan_tier": plan,
            "mrr": np.round(mrr, 2),
            "contract_type": contract_type,
            "auto_renew": rng.random(n) < 0.72,
            "payment_method": rng.choice(PAYMENT_METHODS, n, p=[0.55, 0.22, 0.13, 0.10]),
        }
    )

    # Randomised retention experiment on the 2024 signup cohorts only.
    in_experiment = (frame["signup_date"] >= EXPERIMENT_START) & (
        frame["signup_date"] <= EXPERIMENT_END
    )
    frame["variant"] = np.where(
        in_experiment,
        np.where(rng.random(len(frame)) < 0.5, "treatment", "control"),
        None,
    )
    return frame


def _simulate(rng: np.random.Generator, customers: pd.DataFrame):
    """Month-by-month simulation over the alive cohort.

    Churn is generated from a latent health state that is never exported. Every
    observable — sessions, spend, tickets, failed payments — is a noisy read on that
    state, which is what makes this a realistic ceiling rather than a solvable puzzle.
    """
    n = len(customers)
    signup = customers["signup_date"].values.astype("datetime64[D]")
    plan_effect = customers["plan_tier"].map(
        {"basic": 0.42, "standard": 0.0, "premium": -0.38}
    ).values
    monthly = (customers["contract_type"] == "monthly").values
    auto_renew = customers["auto_renew"].values
    invoice = (customers["payment_method"] == "invoice").values
    region_effect = customers["region"].map(
        {"north": -0.06, "south": 0.05, "east": 0.12, "west": -0.04}
    ).values
    treated = (customers["variant"] == "treatment").values

    intensity = rng.lognormal(0.0, 0.55, n)            # how heavily they use the product
    health = rng.normal(0.0, 1.0, n)                   # latent, never exported
    alive = np.ones(n, dtype=bool)
    churn_month = np.full(n, np.datetime64("NaT"), dtype="datetime64[D]")

    usage_rows, billing_rows, support_rows = [], [], []
    months = pd.date_range(OBSERVATION_START, DATA_EXPORT, freq="MS")

    for month in months:
        started = signup <= np.datetime64(month.date())
        active = alive & started
        idx = np.flatnonzero(active)
        if idx.size == 0:
            continue

        # Health drifts; a slow decline is what actually precedes churn.
        health[idx] = 0.93 * health[idx] + rng.normal(0, 0.42, idx.size)
        tenure_months = (
            (np.datetime64(month.date()) - signup[idx]).astype(int) / 30.4
        ).clip(0, None)

        # ---- weekly usage events -------------------------------------------
        engagement = np.exp(0.55 * health[idx]) * intensity[idx]
        for week in range(4):
            date = month + pd.Timedelta(days=7 * week + int(rng.integers(0, 5)))
            if date > DATA_EXPORT:
                continue
            sessions = rng.poisson(np.clip(2.2 * engagement, 0.05, 60))
            active_week = sessions > 0
            if not active_week.any():
                continue
            sub = idx[active_week]
            usage_rows.append(
                pd.DataFrame(
                    {
                        "customer_id": customers["customer_id"].values[sub],
                        "event_date": date,
                        "last_active_date": date,
                        "sessions": sessions[active_week],
                        "usage_volume": np.round(
                            sessions[active_week] * rng.lognormal(1.1, 0.5, sub.size), 2
                        ),
                        "spend": np.round(
                            customers["mrr"].values[sub] / 4
                            * rng.lognormal(0, 0.15, sub.size), 2
                        ),
                    }
                )
            )

        # ---- monthly billing ------------------------------------------------
        fail_p = np.clip(0.025 + 0.055 * np.maximum(-health[idx], 0) + 0.05 * invoice[idx], 0, 0.6)
        failed = rng.random(idx.size) < fail_p
        past_due = np.where(failed, rng.integers(1, 45, idx.size), 0)
        billing_rows.append(
            pd.DataFrame(
                {
                    "customer_id": customers["customer_id"].values[idx],
                    "event_date": month + pd.Timedelta(days=int(rng.integers(1, 6))),
                    # Raw status, so feature.yaml can build "failed payments in the
                    # last 90 days" with a filter instead of it arriving pre-counted.
                    "status": np.where(failed, "failed", "paid"),
                    "amount": np.round(customers["mrr"].values[idx], 2),
                    "failed_payments": failed.astype(int),
                    "days_past_due": past_due,
                }
            )
        )

        # ---- monthly support -------------------------------------------------
        ticket_lam = np.clip(0.35 + 0.45 * np.maximum(-health[idx], 0), 0.02, 6)
        tickets = rng.poisson(ticket_lam)
        has_ticket = tickets > 0
        if has_ticket.any():
            sub = idx[has_ticket]
            complaints = rng.binomial(
                tickets[has_ticket],
                np.clip(0.16 + 0.2 * np.maximum(-health[sub], 0), 0, 0.85),
            )
            support_rows.append(
                pd.DataFrame(
                    {
                        "customer_id": customers["customer_id"].values[sub],
                        "event_date": month + pd.Timedelta(days=int(rng.integers(3, 26))),
                        "channel": rng.choice(["phone", "email", "chat"], sub.size),
                        "ticket_count": tickets[has_ticket],
                        "complaints": complaints,
                        "nps": np.clip(
                            np.round(7 + 1.6 * health[sub] - 1.4 * complaints
                                     + rng.normal(0, 1.5, sub.size)), 0, 10,
                        ).astype(int),
                    }
                )
            )

        # ---- churn hazard ----------------------------------------------------
        logit = (
            -3.55
            + plan_effect[idx]
            + 0.40 * monthly[idx]
            - 0.55 * auto_renew[idx]
            + region_effect[idx]
            - 0.78 * health[idx]
            + 0.55 * failed
            - 0.42 * np.log1p(tenure_months)          # early months are the risky ones
            - 0.35 * treated[idx]                      # the experiment's true effect
        )
        churning = rng.random(idx.size) < 1 / (1 + np.exp(-logit))
        if churning.any():
            leaving = idx[churning]
            day_offset = rng.integers(0, 28, leaving.size)
            dates = np.datetime64(month.date()) + day_offset.astype("timedelta64[D]")
            in_window = dates <= np.datetime64(DATA_EXPORT.date())
            leaving, dates = leaving[in_window], dates[in_window]
            churn_month[leaving] = dates
            alive[leaving] = False

            # Collections activity in the final days. This is the death-spiral
            # pattern: a consequence of a decision already taken, not a cause. It
            # leaks hard at embargo 0 and is largely defused at embargo 7+.
            if leaving.size:
                final_offset = rng.integers(1, 6, leaving.size)
                billing_rows.append(
                    pd.DataFrame(
                        {
                            "customer_id": customers["customer_id"].values[leaving],
                            "event_date": pd.to_datetime(dates)
                            - pd.to_timedelta(final_offset, unit="D"),
                            "failed_payments": rng.integers(1, 4, leaving.size),
                            "days_past_due": rng.integers(30, 95, leaving.size),
                        }
                    )
                )

    usage = pd.concat(usage_rows, ignore_index=True) if usage_rows else pd.DataFrame()
    billing = pd.concat(billing_rows, ignore_index=True) if billing_rows else pd.DataFrame()
    support = pd.concat(support_rows, ignore_index=True) if support_rows else pd.DataFrame()

    churn_dates = pd.to_datetime(pd.Series(churn_month))
    # Events dated after a customer left would be impossible; trim them.
    ended = pd.Series(churn_dates.values, index=customers["customer_id"].values)
    for frame in (usage, billing, support):
        if frame.empty:
            continue
        limit = frame["customer_id"].map(ended)
        frame.drop(frame.index[limit.notna() & (frame["event_date"] > limit)], inplace=True)

    return (
        usage.sort_values(["customer_id", "event_date"]).reset_index(drop=True),
        billing.sort_values(["customer_id", "event_date"]).reset_index(drop=True),
        support.sort_values(["customer_id", "event_date"]).reset_index(drop=True),
        churn_dates,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data", type=Path)
    parser.add_argument("--customers", default=3000, type=int)
    parser.add_argument("--seed", default=7, type=int)
    args = parser.parse_args()
    generate(args.out, args.customers, args.seed)

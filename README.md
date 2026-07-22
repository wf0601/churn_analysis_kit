# ChurnKit

A churn analysis pipeline you drive from YAML. Point six config files at your data,
run one command, and get a report that covers risk drivers, causal effects, survival,
and — first, before any of the results — what the pipeline found wrong with your data.

```bash
pip install -r requirements.txt
python run.py demo      # generate synthetic data and analyse it end to end
python run.py check     # validate config + data without building anything
python run.py           # analyse your own data
open output/report.html
```

## What you edit

Six files in `config/`. Nothing else.

| File | What it decides |
|---|---|
| `data.yaml` | where your files are, and the id / date columns |
| `target.yaml` | what churn *is* — a churn date, or a pre-computed label |
| `survivorship.yaml` | when churn observation starts, when event history starts, and what to do about customers who predate either |
| `feature.yaml` | how features are constructed: windows, aggregations, filtered subsets, derived expressions |
| `pipeline.yaml` | horizon, embargo, split strategy, leakage policy |
| `causal.yaml` | the handful of drivers you want a causal answer for |
| `experiment.yaml` | optional: an A/B test to analyse instead of guessing |

## Feature construction is config, not code

Aggregations live in `feature.yaml`. Adding a window, an aggregation, a filtered
subset or a ratio means editing YAML — `features.py` does not change.

```yaml
billing:
  temporal: time_varying
  source: billing
  windows: [90, 180]          # one family per window
  aggs: [sum, max]            # sum, mean, max, min, last, first, count, median, std, nunique
  filters:                    # named row subsets, each its own feature family
    - name: failed
      where: "status == 'failed'"
    - name: paid
      where: "status == 'paid'"
  columns:
    - {name: amount,        type: numeric, aggs: [sum]}          # per-column override
    - {name: days_past_due, type: numeric, windows: [180]}       # ditto
```

`filters` is how a raw event log becomes the counted variables people expect.
"Failed payments in the last 90 days" is a filter plus a window plus a count —
`billing__failed__event_count_90d` — and it belongs in the config rather than in a
SQL view nobody can find. Filters **add** families alongside the unfiltered one, so
shares and ratios are expressible; set `include_unfiltered: false` when the base
aggregate is meaningless.

Post-aggregation features go in a `derived` group:

```yaml
ratios:
  temporal: derived
  features:
    - name: failed_payment_share_90d
      expression: >
        billing__failed__event_count_90d /
        (billing__failed__event_count_90d + billing__paid__event_count_90d)
    - name: tickets_per_year_of_tenure
      expression: support__ticket_count__sum_90d * 4 / maximum(lifecycle__tenure_days / 365, 0.25)
```

Expressions may use feature names, numbers, arithmetic, comparisons and a fixed
function list (`log`, `log1p`, `exp`, `sqrt`, `abs`, `clip`, `where`, `minimum`,
`maximum`, `sign`, `isnull`, `notnull`, `fillna`). They are parsed and checked
before evaluation: attribute access, calls to anything else, imports and lambdas
are rejected, so a shared config file is not an execution surface. Division by zero
becomes null rather than infinity.

Two properties that matter more than the convenience:

- **A derived feature inherits its inputs' leakage status.** If an input is
  quarantined, everything computed from it is quarantined too and reported as
  `DERIVED_FROM_BLOCKED_FEATURE`. Otherwise a leak survives behind a ratio.
- **Filter names are screened by the denylist**, because they become part of the
  feature name — a `cancellation` filter is caught the same way a `cancellation`
  column would be.

Feature names: `<group>__<column>__<agg>_<window>d`, with `__<filter>__` inserted
after the group when a filter applies, plus `event_count` and `recency_days` per
subset per window. `output/feature_dictionary.csv` lists every one with how it was
built and what it depends on.

## What runs

```
config → data → panel → features → LEAKAGE AUDIT → model → drivers
                                                        ↘ causal / experiment
                                                        ↘ survival → report
```

The leakage audit sits between feature construction and everything else, so nothing
downstream can see a feature that failed it.

## The leakage model

Churn leakage is not one bug, it is a family. Each member gets its own guard.

**Time is enforced by construction, not by convention.** Every row is a customer at a
prediction date `T`, and the timeline is split into three regions that never touch:

```
[ T-window ............ T-embargo )   [ T-embargo .. T ]   ( T ....... T+horizon ]
  features may look here                  embargo           label measured here
```

Aggregations are recomputed inside each window and asserted against the window end,
so a mis-sorted or mis-parsed date cannot slip a future row into a feature.

**The embargo is the setting people skip and shouldn't.** Churn is usually decided
before it is recorded. Failed payments, angry tickets and usage collapse in the final
days are consequences of a decision already taken — they will top your driver ranking
and they are not levers. `panel.embargo_days: 7` blanks that window out. Groups marked
`leakage_review: strict` in `feature.yaml` refuse to run without one.

**Censoring is never turned into retention.** A row gets label 0 only if the full
horizon after `T` sits inside the observed data. Prediction dates whose label window
runs past `observation_end_date` are dropped, loudly, rather than counted as
customers who stayed.

**Survivorship is handled explicitly.** Customers who started before
`observation_starting_date` are survivors of a period you cannot see — their
early-churning peers were never exported. `left_truncation: drop` excludes them;
`keep_flagged` keeps them and uses a left-truncated survival fit, with the caveat
printed in the report. Note that "when churn started being recorded" and "how far
back the event logs reach" are different dates — set `event_history_starts` when
they differ, or a long feature window will be measured against the wrong boundary.

**Point-in-time correctness is checked, not assumed.** A `time_varying` group read
from a flat entity table gets a `NOT_POINT_IN_TIME` warning, because its values are
whatever was true at export. Declare `valid_from_column`/`valid_to_column` in
`data.yaml` and the kit selects the version in force at each `T` instead. Stored
`tenure_days` columns are always recomputed — a stored tenure is calculated at export
time, so for churned customers it encodes the lifetime they turned out to have.

**Statistical screens catch what no denylist anticipates.** Every feature is scored
alone against the label; near-perfect separation is a leak, not a discovery. Whether a
field is *missing* is scored too, because a record created by a churn workflow gives
itself away that way. Identifier-like and constant columns are flagged. A holdout AUC
above `leakage.model_auc_block` fails the run outright, and a large gap between
cross-validated and out-of-time AUC is flagged as the signature of a feature whose
meaning shifts with the calendar.

**Splits are out-of-time by default.** A random split scores the model on dates it has
already seen. Training rows whose label window overlaps the test period are purged.
Inner CV folds are grouped by customer, so near-duplicate rows can't straddle a fold.
Preprocessing lives inside the estimator pipeline and is refitted per fold.

Findings are levelled `BLOCK` / `WARN` / `INFO`. The default policy is
`leakage.on_block: quarantine` — the offending feature is dropped, the run continues,
and the report says so at the top. Set `fail` to abort instead.

### When the guard is wrong

It will sometimes be. The demo blocks `lifecycle__prior_winback` because the name
matches `win_?back`, even though "this customer was previously won back" is genuinely
knowable at `T`. That is the intended shape of the trade: the denylist is deliberately
eager, and `leakage.allowlist_columns` is the escape hatch. Use it when you can say
when the field is written — not because the AUC drops without it.

## Reading the output

`output/report.html` is self-contained. Alongside it:

| File | Contents |
|---|---|
| `leakage_findings.csv` | every finding, with level and remedy |
| `leakage_scan.csv` | per-feature univariate and missingness AUC |
| `feature_dictionary.csv` | every feature, how it was derived, whether it survived |
| `risk_drivers.csv` | ranked drivers with direction, spread and fold stability |
| `risk_scores_holdout.csv` | scored customers from the held-out period |
| `causal_effects.csv` | adjusted effects, CIs, refutation results |
| `survival_curve.csv`, `survival_cox.csv` | Kaplan-Meier and hazard ratios |
| `summary.json` | machine-readable summary for downstream jobs |

Drivers are **associations**. Causal estimates are the ones you can act on, and even
those cannot rule out an unmeasured confounder — the report says so where it matters.

## Causal analysis

`causal.yaml` names the levers worth a real answer. Each gets a cross-fitted
doubly-robust (AIPW) estimate with:

- overlap / positivity diagnostics — if treatment is nearly determined by the
  adjustment set, the kit refuses to produce a number and names the culprit column;
- covariate balance before and after weighting;
- a **placebo test** (shuffle the treatment; a valid pipeline must find nothing) and a
  **subset test** (re-estimate on 70% of rows; the effect should hold).

Two classes of redundancy are excluded automatically, because they are mechanical
rather than a judgement call: other aggregations of the treatment's own column, and
— when the treatment is a filtered subset count — everything else built from the
same event log, since a base count equals the sum of its subsets and would make the
treatment perfectly predictable. Derived features computed from anything excluded go
too.

The one thing automation cannot do is tell a confounder from a mediator. Auto-selected
confounders include everything else measured before `T`, and anything on the causal
path will absorb part of the effect you are measuring. `confounders.exclude` is where
your domain knowledge goes, and it is the field most worth your time.

In the demo, support complaints show a `+2.20 pp` raw association with churn and no
detectable effect after adjustment — which is correct, because in the simulator
complaints and churn are both symptoms of a latent health state that is never
exported. That gap between the two numbers is the whole reason this stage exists.

## Data shapes

**Entity table + event logs** (recommended). One row per customer, plus long-format
event logs keyed by customer and date. Time-varying features are recomputed inside
each window, so point-in-time correctness is provable.

**Entity table only.** Everything is read as-of export. It works, and the kit warns
on every group where that assumption is doing load-bearing work.

For `target.mode: label` (a churn flag with no date), the kit runs a single
cross-section and reports that it cannot verify any feature was knowable before the
customer left. Prefer `event_date` whenever you have a date.

## Worked example: case study

`config_plg/` is a complete, runnable config for a specific churn analysis problem.

```bash
python tools/ingest_plg.py --raw data/raw/advanced_case_study_data.csv
python run.py --config config_plg
open output_plg/report.html
```

The ingest step does two jobs. It cleans (mixed decimal separators, currency
symbols, US/EU/text dates inferred per column, `Y/N` booleans, `10+` capped counts,
JSON-ish engagement blobs, near-unique postcodes coarsened) and reports every
coercion to `data/plg/ingest_report.csv` rather than doing it silently. Then it
splits the wide export into an entity table plus **event logs** for the dated
fields — `survey_date`, `last_contact_date`, `win_dates`,
`service_tier_upgrade_date`. That second job is the leakage-critical one: the
schema's own example has a survey dated 2023-01-14, six months after the
2022-07-01 prediction date. Flat columns, that is invisible leakage; as dated
events, the feature window drops them and says how many.

Key config decisions, all in one place and all reversible:

| Setting | Value | Why |
|---|---|---|
| `snapshot_mode` | `single`, 2022-07-01 | the assignment's own framing — one index date |
| `horizon_days` | 273 | the full observed window, so the model target and the experiment outcome are the same quantity |
| `left_truncation` | `keep_flagged` | every row is an active subscription on 2022-07-01, so the cohort is survivorship-selected by construction; `drop` would delete all of it |
| `split.strategy` | `grouped_random` | one index date means no later period to hold out — worth stating rather than falling back silently |
| `assignment_date` | 2022-07-01 | snapshot cohort, not rolling enrolment |
| `allowlist_columns` | `baseline_churn_risk_band`, `is_cancellation_inquiry` | both pre-prediction, both trip a denylist pattern |

**What it does not do.** Three parts of that assignment are outside this kit and
need separate work: CATE/uplift modelling for the optional heterogeneity task; the
complier-average (treatment-on-the-treated) effect implied by
`treatment_sent_flag`, where only intention-to-treat is reported; and the economic
layer that turns uplift and `offer_cost_eur` into an expected-value targeting rule.

## Layout

Stage modules are numbered in the order they run, so the directory listing is the
pipeline and the console banners match the filenames.

```
churnkit/
  pipeline.py           calls l01 → l12 in order
  l01_config.py         load, default and validate the YAML
  l02_data.py           read the files, pin down the observation window
  l03_panel.py          snapshot construction, labels, censoring, truncation
  l04_features.py       windows, aggregations, filters, derived expressions
  l05_leakage.py        structural + statistical guards, quarantine policy
  l06_splits.py         out-of-time and grouped splits with purging
  l07_model.py          fit, calibrate, honest metrics
  l08_drivers.py        importance, direction, fold stability
  l09_causal.py         cross-fitted AIPW with refutations
  l10_experiment.py     SRM, balance, ITT, CUPED adjustment
  l11_survival.py       Kaplan-Meier, log-rank, Cox
  l12_report.py         HTML + CSV + JSON
  util/                 shared support — none of it is a stage
    errors.py           exception types
    log.py              console logging and stage banners
    charts.py           figures
    parsing.py          messy-export parsers (separators, dates, booleans, …)
tools/
  make_synthetic_data.py   demo dataset generator
  ingest_plg.py            case-study adapter: column mapping + alias tables
```

`util/parsing.py` is dataset-agnostic on purpose. Mixed decimal separators, currency
symbols, US/EU date order and censored counts turn up in every legacy extract, so a
new ingest adapter supplies only its column mapping and reuses the parsers.

Requires Python 3.10+. `lifelines` and `shap` are optional — the kit degrades to
built-in Kaplan-Meier and permutation importance without them.

# Data Scientist Take-Home Assignment
## Retention & Uplift Optimiser

## 1 — Business Scenario

The Postcode Lottery Group raises funds for charities through lottery participation. 
Every 
retained participant directly contributes to more funds for environment, health, and 
community causes.

We have a subscription based lottery. Customers subscribe using their postcodes as 
their lottery numbers, and each month there is a draw with various prizes. Customers 
participate in these draws as long as they have an active subscription. Each year 
there are 2 extra draws on top of regular 12 draws.

Historically, our retention strategy was reactive: we predicted who was likely to 
churn and then manually decided what to do. We now want to move to a **proactive, 
test-and-learn approach**:

1.  We want to **predict churn**.
2.  We want to **understand which interventions actually work** (and for whom).
3.  We want to roll this out in a way that is **robust, reproducible, and ready for 
    production**.

### The Experiment
Recently, we ran a randomized controlled experiment with three groups:
* **Control:** No special treatment.
* **Variant A:** A standard discount offer.
* **Variant B:** A value-add offer (e.g., extra draws, bonus tickets, or 
  charity-related perks).

*Note: Each active subscription at the start of the experiment was randomly assigned 
to a cohort, but due to operational constraints, not everyone in a treatment cohort 
actually received the communication.*

### Your Goal
Build a production-grade data science pipeline that:
1.  **Cleans and standardizes** a messy CRM export.
2.  **Builds robust models** to predict churn probability.
3.  **Quantifies incremental impact** of Variant A and Variant B vs Control.

---

## 2 — The Data

**File:** `data/raw/advanced_case_study_data.csv`

Data come from 2 different main sources, given below as legacy_system_id. Each row 
represents one lottery subscription at the start of the experiment. Each row is a 
snapshot of one active subscription as of the experiment start date (1 July 2022). 
The observation window runs from that date to the `observation_end_date`.
This file is a raw legacy export and is **not in an ingest-ready format**. You 
should assume:
* Mixed decimal separators (`.` and `,`).
* Mixed thousand separators.
* Mixed date formats (US vs EU, text months).
* Mixed encodings for booleans and categories.
* Embedded JSON-like strings in some columns.
* And other anomalies that can appear in a data export.

### Data Schema

|    | Column Name                 | Description                                           | Example Values                      |
|----|:----------------------------|:------------------------------------------------------|:------------------------------------|
| 1  | `customer_id`               | Unique customer identifier                            | `b7a782741f667201b5`                |
| 2  | `subscription_id`           | Unique subscription identifier                        | `7db88cdd3c295d2276`                |
| 3  | `legacy_system_id`          | Data Source system identifier                         | `SYS_OLD`, `SYS_NEW`                |
| 4  | `subscription_date`         | Original subscription datetime                        | `2022-03-04 00:00:00`, `July 2018`  |
| 5  | `participant_age`           | Age at experiment start                               | `69`, `32`, `Unknown`               |
| 6  | `marketing_channel`         | Acquisition channel                                   | `DM`, `Direct mail`, `Online paid`  |
| 7  | `country_code`              | Country of lottery                                    | `NL`, `DE`, `GB`, `N/A`             |
| 8  | `postcode_area`             | Coarse location of subscription owner                 | `1012`, `SE-123 45`                 |
| 9  | `extra_draws_per_year`      | Opt-in for extra "13th/14th" month draws              | `0` (None), `1`, `2` (Both)         |
| 10 | `Add-ons`                   | Number of add-ons                                     | `0`, `1`, `3`, `0,0`                |
| 11 | `payment_method`            | Primary payment method                                | `Direct debit`, `Credit card`       |
| 12 | `failed_payments_12m`       | Count of failed payments in last 12 months            | `0`, `1`, `NaN`                     |
| 13 | `monthly_spend_estimated`   | Estimated monthly net spend (pre-tax)                 | `€15,50`, `EUR 15.50`, `$15.50`, `1,250` |
| 14 | `donation_share_charity`    | Share of monthly spend that goes to charity           | `0.40`, `40`                        |
| 15 | `engagement_history`        | Recent engagement in semi-structured format           | `{"email_opens": "5"}`, `opens=3`   |
| 16 | `web_sessions_90d_raw`      | Web sessions in last 90 days                          | `0`, `5`, `10+`, `missing`          |
| 17 | `c_service_contacts_12m`    | Number of customer service contacts in last 12 months | `0`, `1`, `99`                      |
| 18 | `complaints_12m`            | Number of complaints in last 12 months                | `0`, `2`, `5`                       |
| 19 | `lifetime_wins`             | Total number of historical prizes won                 | `0`, `1`, `15`                      |
| 20 | `win_dates`                 | Dates of prizes won                                   | `2022-03-04 00:00:00`, `July 2018`  |
| 21 | `campaign_cohort`           | Randomized experiment group                           | `Control`, `Variant_A`, `Variant_B` |
| 22 | `treatment_sent_flag`       | Whether the campaign communication was actually sent  | `1`, `0`, `Y`, `N`                  |
| 23 | `offer_cost_eur`            | Marketing + incentive cost                            | `0`, `€20,00`, `20.0`               |
| 24 | `baseline_churn_risk_band`  | Legacy rule-based churn band at experiment start      | `Low`, `Medium`, `High`, `Very high` |
| 25 | `historic_revenue_12m`      | Observed revenue last 12 months (before experiment)   | `€180,00`, `1.200,00`               |
| 26 | `churned`                   | Whether subscription churned in follow-up window      | `1`, `0`, `Y`, `N`                  |
| 27 | `churn_date`                | Date of churn (if any)                                | `2023-01-15`, `NA`                  |
| 28 | `observation_end_date`      | End date of follow-up observation                     | `2023-03-31`                        |
| 29 | `revenue_next_12m_observed` | Observed revenue (after exp) over 12-month follow-up  | `€600,00`, `0`, `NA`                |
| 30 | `service_tier_upgrade_date` | Date of service tier upgrade, if any occurred         | `2022-11-03`, `NA`                  |
| 31 | `satisfaction_score`        | Customer satisfaction score from CRM survey (1–10)    | `8`, `3`, `NA`                      |
| 32 | `survey_date`               | Date the satisfaction survey was completed            | `2022-05-10`, `2023-01-14`, `NA`    |
| 33 | `last_contact_date`         | Date of the most recent customer service contact      | `2022-06-01`, `2022-09-18`, `NA`    |
| 34 | `last_contact_reason`       | Reason for the most recent customer service contact   | `billing`, `cancellation_inquiry`, `offer_question`, `prize_claim`, `other`, `NA` |
### Assumptions
* The experiment assignment (`campaign_cohort`) is randomized at the subscription level.
* Some participants in Variant A/B did not actually receive the treatment due to 
  operational issues (`treatment_sent_flag`), which you should consider when 
  defining causal estimands.
* All monetary values are intended to be interpreted in **EUR** after cleaning and 
  conversion, even if the raw export shows $ or USD.



# 3 — Your Assignment: What You Must Build

**Timeline:** 1 Week.

**Deliverables:** 
* A Git repository (GitHub/GitLab) with reproducible code and a `design_doc.md`.
* A short presentation consisting of two parts:
  1. A technical section for team members explaining the solution you made. You should introduce your solution, discuss pros and cons compared to alternatives and describe next steps given more time or data.
  2. A non-technical section for stakeholders showcasing your solution. Imagine the stakeholder is a retention marketeer aiming to reduce churn. How can your model help them? And what is a specific piece of advice you can give them based on your work?

You have **one week** to work on this assignment. We expect a professional, 
reproducible repository (GitHub/GitLab-style) with code, analysis, and a short 
description. You will be invited to present your work if your solution meets our evaluation criteria.

**Tech Stack:** You are free to choose your stack (e.g., Python with Pandas / 
scikit-learn / statsmodels, etc.), but you should:
* Follow good MLOps practices.
* Make your work easy to rerun from scratch.

It is perfectly fine to use AI assistance in the completion of the case study. 
We do still expect you to understand the code that was created and be able to explain any choices made, by you or the model. 
We also ask you to include a short explanation of your process. How did you prompt? What did your interation process look like? How did you validate the outcomes?

## Key Framing Requirement — Churn Target and Timing

This assignment intentionally does **not** prescribe one single churn definition.
You may define churn in the way you believe is most appropriate for the business
problem. Your definition must be explicit, temporally valid, and consistently
implemented. You must clearly define the prediction date, eligible population,
churn horizon, censoring rules, excluded post-treatment variables, and how your
churn definition relates to the experiment outcome.

## Part A — Data Engineering & Pipeline 

### 1. Robust Ingestion
Build a single entry point (e.g., CLI or main function) that:
* Reads the raw data.
* **Handles robustness:** Manage mixed encodings, decimal separators, and date formats.
* **Normalizes:** Standardize booleans and categorical encodings.
* *Note:* A simple `pd.read_csv()` with default options is expected to produce 
  incorrect types; your code should make this robust and reproducible.

### 2. Cleaning & Standardization
* Standardise all columns properly

### 3. Feature Engineering
* **Output:** Create a clean training dataset and write it to a file including:
    * All features you decide to use.
    * Treatment assignment variables.
    * Target variables.
    * Any metadata needed to reproduce your prediction date, eligibility rules,
      churn horizon, censoring decisions, and train/test split.
* Clearly separate pre-treatment/pre-prediction features from treatment variables
  and post-treatment outcomes.

---

## Part B — Predictive Modeling 

### 1. Churn Propensity Model
* **Baseline:** Build at least one baseline model (e.g., regularized logistic 
  regression, GLM) to predict your defined churn target.
* **Advanced:** Build at least one more advanced model (e.g., Random Forest, 
  Gradient Boosting, XGBoost, LightGBM).
* **Documentation:** Clearly describe:
    * Your churn definition and why it is appropriate for this business problem.
    * The prediction date or index date used for model training and scoring.
    * How you approach training.
    * Which features you include or exclude (and why).
    * How you prevent data leakage, especially from post-treatment variables,
      future outcomes, dates that reveal the label, or revenue observed after the
      experiment started.
* **Evaluation:** Compare models using appropriate metrics (e.g., ROC AUC, PR AUC, 
  log loss, calibration plots, lift curves, etc.) and motivate your choice/conclusions.

---

## Part C — Causal & Uplift Analysis 

### 1. A/B Test Analysis
* Treat `campaign_cohort` as the treatment assignment.
* Estimate the **average treatment effect** on the churn outcome you choose for
  the experiment analysis for:
    * Variant A vs Control.
    * Variant B vs Control.
* State whether this is the same churn outcome used by the predictive model. If
  it is not the same, explain why the difference is justified.
* Because not everyone assigned to Variant A/B actually received the
  communication, define the causal estimand carefully. At minimum, report the
  intention-to-treat effect based on randomized assignment. You may also discuss
  treatment-on-the-treated or per-protocol effects if you clearly state the
  additional assumptions.
* **Checks:** Verify the results, should we be concerned about bias.
* **Output:** Provide point estimates and confidence intervals, and interpret them.

### 2. Treatment Effect Heterogeneity (Segments) *(Optional)*
This sub-task is optional. A full implementation is not required, but candidates who engage with it -- either by running a heterogeneity analysis or by discussing how they would approach it and why -- will receive credit for that engagement.
* Identify whether certain segments react differently to Variant A or B.
* **Method:** Choose the appropriate method to conduct this analysis and motivate 
  your choice. If you discuss this conceptually rather than implement it, explain which method you would use, why, and what risks (e.g. spurious subgroup findings, multiple testing) you would control for.
* **Summary:** Define which segments are particularly well suited for Variant A vs 
  Variant B.

---

## Part D — Documentation

### 1. Design Document (`design_doc.md`)
Briefly describe:
* **Pipeline Architecture:** Ingestion, cleaning, feature engineering.
* **Modeling Choices:** Rationale for algorithms, key features, regularization, and 
  leakage prevention.
* **Problem Framing:** Your churn definition, prediction date, eligible
  population, churn horizon, censoring rules, and leakage controls.
* **Causal Approach:** How you interpret your results, including whether the
  causal outcome matches the predictive target.
* **Operational Recommendation:** How you would combine churn propensity, uplift
  estimates, intervention costs, and expected value into a concrete retention
  recommendation.

### What to expect at the review session:
We will start with your presentation, with about 10 minutes per section. We will then ask some detailed questions about various parts of your solution like: 
* Questions about overall architecture and pipeline.
* Clarifying questions on **key technical decisions** (e.g., choice of uplift model, 
  handling dirty data).
* Questions on model and analysis quality, risks and performance.

"""Report assembly: one self-contained HTML file, plus CSVs and a JSON summary.

Ordering is deliberate. Data integrity comes before results, because a reader who
sees an AUC first will remember the AUC. If the run quarantined a leaking feature or
could not verify point-in-time correctness, that is the first thing on the page.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .util import charts
from .util.log import get_logger

log = get_logger("report")

CSS = """
:root{--surface:#ffffff;--surface-2:#f7f7f5;--ink:#0b0b0b;--ink-2:#52514e;
--ink-3:#84837d;--line:#e8e7e3;--blue:#2a78d6;--red:#e34948;--amber:#eda100;
--green:#1baf7a;}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-2);color:var(--ink);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;}
.wrap{max-width:940px;margin:0 auto;padding:40px 24px 80px}
header{margin-bottom:32px}
h1{font-size:28px;line-height:1.25;margin:0 0 6px;letter-spacing:-.02em}
h2{font-size:19px;margin:44px 0 4px;letter-spacing:-.01em;
padding-top:20px;border-top:1px solid var(--line)}
h3{font-size:14px;margin:26px 0 8px;color:var(--ink-2);text-transform:uppercase;
letter-spacing:.06em;font-weight:600}
p{margin:10px 0;color:var(--ink-2)}
.lede{color:var(--ink-3);font-size:14px;margin:0}
section{background:var(--surface);border:1px solid var(--line);border-radius:12px;
padding:4px 26px 26px;margin-bottom:18px}
section>h2:first-child{border-top:none;margin-top:22px;padding-top:0}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;
margin:18px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;
padding:14px 16px}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.06em;
color:var(--ink-3);font-weight:600}
.tile .v{font-size:26px;font-weight:600;letter-spacing:-.02em;margin-top:4px;
overflow-wrap:anywhere}
.tile .v.long{font-size:15px;line-height:1.35;margin-top:7px}
.tile .s{font-size:12px;color:var(--ink-3);margin-top:2px}
.finding{border-left:3px solid var(--line);padding:12px 16px;margin:10px 0;
background:var(--surface-2);border-radius:0 8px 8px 0}
.finding.BLOCK{border-left-color:var(--red)}
.finding.WARN{border-left-color:var(--amber)}
.finding.INFO{border-left-color:var(--blue)}
.finding .code{font:600 11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;
letter-spacing:.04em;color:var(--ink-3)}
.finding .msg{margin:4px 0 0;color:var(--ink)}
.finding .fix{margin:8px 0 0;font-size:13px;color:var(--ink-2)}
.finding .cols{margin:8px 0 0;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;
color:var(--ink-3);word-break:break-all}
.tablewrap{overflow-x:auto;margin:14px 0;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:480px}
th{text-align:left;font-weight:600;color:var(--ink-2);border-bottom:1px solid var(--line);
padding:8px 10px;white-space:nowrap;font-size:12px;text-transform:uppercase;
letter-spacing:.04em}
td{padding:7px 10px;border-bottom:1px solid var(--line);color:var(--ink);
white-space:nowrap}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tbody tr:last-child td{border-bottom:none}
figure{margin:18px 0}
figure img{width:100%;height:auto;display:block}
figcaption{font-size:12px;color:var(--ink-3);margin-top:6px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}
.badge{display:inline-block;font-size:11px;font-weight:600;padding:2px 8px;
border-radius:999px;letter-spacing:.03em}
.badge.ok{background:#e6f6ef;color:#0d6b4a}
.badge.warn{background:#fdf3dc;color:#7a5300}
.badge.bad{background:#fdeaea;color:#a12220}
code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--surface-2);
padding:1px 5px;border-radius:4px}
.note{font-size:13px;color:var(--ink-2);background:var(--surface-2);
border-radius:8px;padding:12px 14px;margin:10px 0}
footer{margin-top:40px;font-size:12px;color:var(--ink-3);text-align:center}
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4g}" if abs(value) < 1000 else f"{value:,.0f}"
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value)


def _table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame is None or frame.empty:
        return '<p class="lede">No rows.</p>'
    view = frame.head(max_rows)
    head = "".join(f"<th>{_esc(c)}</th>" for c in view.columns)
    body = []
    for _, row in view.iterrows():
        cells = []
        for col in view.columns:
            value = row[col]
            numeric = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
            cells.append(f'<td class="{"num" if numeric else ""}">{_esc(_fmt(value))}</td>')
        body.append(f"<tr>{''.join(cells)}</tr>")
    more = (
        f'<figcaption>Showing {max_rows} of {len(frame):,} rows — the full table is in '
        f"the CSVs beside this file.</figcaption>" if len(frame) > max_rows else ""
    )
    return (
        f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>{more}"
    )


def _figure(png: str | None, caption: str) -> str:
    if not png:
        return ""
    return (
        f'<figure><img src="data:image/png;base64,{png}" alt="{_esc(caption)}">'
        f"<figcaption>{_esc(caption)}</figcaption></figure>"
    )


def _tile(label: str, value: str, sub: str = "") -> str:
    # Long values (feature names) get a smaller step so they wrap inside the card
    # instead of running past its edge.
    size = " long" if len(value) > 14 else ""
    return (
        f'<div class="tile"><div class="k">{_esc(label)}</div>'
        f'<div class="v{size}">{_esc(value)}</div>'
        f'<div class="s">{_esc(sub)}</div></div>'
    )


def _note(text: str) -> str:
    return f'<div class="note">{_esc(text)}</div>'


# --------------------------------------------------------------------------- #
def build(results: dict, cfg) -> Path:
    out = cfg.output_dir
    sections = [
        _header(results, cfg),
        _integrity(results, cfg),
        _panel_section(results, cfg),
        _model_section(results),
        _drivers_section(results),
        _causal_section(results),
        _experiment_section(results),
        _survival_section(results),
        _appendix(results, cfg),
    ]
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Churn analysis — {_esc(cfg.entity_path.stem)}</title>
<style>{CSS}</style></head><body><div class="wrap">
{''.join(s for s in sections if s)}
<footer>Generated by ChurnKit on {generated} · every table on this page is also a CSV
in this folder · re-run with <code>python run.py</code></footer>
</div></body></html>"""

    path = out / "report.html"
    path.write_text(document, encoding="utf-8")
    _write_artifacts(results, out)
    log.info("report written to %s", path)
    return path


def _header(results: dict, cfg) -> str:
    panel = results["panel"]
    model = results.get("model")
    leak = results["leakage"]
    survival = results.get("survival")
    drivers = results.get("drivers")

    n_block = len(leak.by_level("BLOCK"))
    n_warn = len(leak.by_level("WARN"))
    integrity = (
        ("bad", f"{n_block} blocking") if n_block
        else ("warn", f"{n_warn} warnings") if n_warn
        else ("ok", "clean")
    )
    tiles = [
        _tile("Churn base rate", f"{100 * panel.churn_rate:.2f}%",
              f"within {panel.horizon_days} days of the prediction date"),
        _tile("Customers", f"{panel.frame['entity_id'].nunique():,}",
              f"{len(panel.frame):,} customer-date rows"),
    ]
    if model:
        tiles.append(
            _tile("Holdout AUC", f"{model.metrics['auc']:.3f}",
                  f"logistic baseline {model.baseline_metrics['auc']:.3f}")
        )
        tiles.append(
            _tile("Lift @ top 10%", f"{model.metrics['lift_at_10pct']:.1f}x",
                  f"captures {model.metrics['capture_at_10pct']:.0%} of churn")
        )
    if survival and survival.median_lifetime:
        tiles.append(_tile("Median lifetime", f"{survival.median_lifetime:,.0f} d",
                           "Kaplan-Meier, censoring respected"))
    if drivers is not None and not drivers.table.empty:
        top = drivers.table.iloc[0]
        tiles.append(_tile("Top driver", str(top["feature"]),
                           str(top.get("direction", ""))))
    tiles.append(
        _tile("Data integrity", integrity[1],
              f"{len(leak.quarantined)} feature(s) quarantined")
    )

    return f"""<header>
<h1>Churn analysis</h1>
<p class="lede">{_esc(cfg.entity_path.name)} · observation window
{_fmt(panel.timeline['observation_start'])} to {_fmt(panel.timeline['observation_end'])}
· {panel.horizon_days}-day horizon · {panel.embargo_days}-day embargo</p>
<div class="tiles">{''.join(tiles)}</div>
</header>"""


def _integrity(results: dict, cfg) -> str:
    leak = results["leakage"]
    fm = results["features"]
    blocks = leak.by_level("BLOCK")
    warns = leak.by_level("WARN")
    infos = leak.by_level("INFO")

    parts = ['<section><h2>Data integrity and leakage</h2>']
    parts.append(
        "<p>Leakage is the failure mode that makes a churn model look excellent and "
        "be useless: a feature that is only knowable because the customer already "
        "left. These checks run before any result below, and anything blocking was "
        "removed from every downstream stage.</p>"
    )

    if not blocks and not warns:
        parts.append(
            '<p><span class="badge ok">clean</span> No structural or statistical '
            "leakage signals were found. That is not proof of absence — it means the "
            "automated checks had nothing to say.</p>"
        )

    for finding in blocks + warns + infos:
        cols = ""
        if finding.columns:
            shown = ", ".join(finding.columns[:12])
            extra = f" (+{len(finding.columns) - 12} more)" if len(finding.columns) > 12 else ""
            cols = f'<p class="cols">{_esc(shown)}{_esc(extra)}</p>'
        fix = f'<p class="fix"><strong>Fix:</strong> {_esc(finding.remedy)}</p>' if finding.remedy else ""
        parts.append(
            f'<div class="finding {finding.level}"><span class="code">'
            f"{finding.level} · {_esc(finding.code)}</span>"
            f'<p class="msg">{_esc(finding.message)}</p>{fix}{cols}</div>'
        )

    if leak.quarantined:
        parts.append(
            f"<h3>Quarantined features ({len(leak.quarantined)})</h3>"
            + _note(
                "These were dropped before modelling. If you believe one is legitimate, "
                "add its exact name to leakage.allowlist_columns in pipeline.yaml and "
                "re-run — but be able to say when the field is written first."
            )
            + f'<p class="cols">{_esc(", ".join(leak.quarantined))}</p>'
        )

    scan = results.get("leakage_scan")
    if scan is not None and not scan.empty:
        parts.append("<h3>Univariate leakage scan</h3>")
        parts.append(
            "<p>Every feature scored on its own against the label. A single feature "
            "with an AUC near 1 is almost never a real behavioural signal.</p>"
        )
        view = scan.head(15)[
            ["feature", "univariate_auc", "missingness_auc", "null_share", "n_unique"]
        ]
        parts.append(_table(view, max_rows=15))

    if fm.skipped:
        parts.append("<h3>Columns in feature.yaml that were not found</h3>")
        parts.append(_table(pd.DataFrame(fm.skipped), max_rows=25))

    for note in results["panel"].notes + fm.notes + results.get("data_notes", []):
        parts.append(_note(note))

    parts.append("</section>")
    return "".join(parts)


def _panel_section(results: dict, cfg) -> str:
    panel = results["panel"]
    split = results.get("split")
    per_snapshot = panel.diagnostics.get("per_snapshot")

    window = cfg.max_window_days
    diagram = (
        f"<pre style='font:12px ui-monospace,Menlo,monospace;color:#52514e;"
        f"background:#f7f7f5;padding:14px;border-radius:8px;overflow-x:auto'>"
        f"  [ T-{window}d ............ T-{panel.embargo_days}d )   "
        f"[ T-{panel.embargo_days}d .. T ]   ( T ....... T+{panel.horizon_days}d ]\n"
        f"   features may look here        embargo         label measured here"
        f"</pre>"
    )

    parts = [
        "<section><h2>How the panel was built</h2>",
        "<p>Each row is one customer at one prediction date <code>T</code>. A customer "
        "enters only if they were genuinely still active at <code>T</code>, and gets a "
        "label of 0 only if the full horizon after <code>T</code> is inside the "
        "observed data — never because the data simply ran out.</p>",
        diagram,
        _figure(charts.churn_over_time(per_snapshot), "Churn rate by prediction date"),
    ]
    if per_snapshot is not None and not per_snapshot.empty:
        parts.append(_table(per_snapshot, max_rows=30))
    if split is not None:
        parts.append(_note(split.description))
    parts.append("</section>")
    return "".join(parts)


def _model_section(results: dict) -> str:
    model = results.get("model")
    if model is None:
        return ""
    metrics = pd.DataFrame(
        [
            {"metric": k, "model": v, "logistic baseline": model.baseline_metrics.get(k)}
            for k, v in model.metrics.items()
        ]
    )
    parts = [
        "<section><h2>Model performance</h2>",
        "<p>Measured on held-out prediction dates the model never saw. AUC answers "
        "'can it rank', calibration answers 'are the probabilities real', and lift "
        "answers the only question an operator asks: if we work the top decile, how "
        "much churn do we actually reach.</p>",
        '<div class="grid2">',
        _figure(charts.calibration(model.calibration), "Predicted vs observed risk"),
        _figure(charts.lift_curve(model.test_predictions), "Churn rate by risk decile"),
        "</div>",
        _table(metrics, max_rows=20),
    ]
    if not model.cv_metrics.empty:
        parts.append("<h3>Cross-validation on the training period</h3>")
        parts.append(
            "<p>Folds are grouped by customer, so no customer appears on both sides "
            "of a fold. Spread across folds is the honest measure of how much of the "
            "headline number is luck.</p>"
        )
        parts.append(_table(model.cv_metrics[["fold", "n", "auc", "pr_auc", "brier"]]))
    for note in model.notes:
        parts.append(_note(note))
    parts.append("</section>")
    return "".join(parts)


def _drivers_section(results: dict) -> str:
    drivers = results.get("drivers")
    if drivers is None or drivers.table.empty:
        return ""
    columns = [
        c for c in
        ["rank", "feature", "group", "importance", "direction", "churn_rate_low",
         "churn_rate_high", "spread_pp", "stability", "stable"]
        if c in drivers.table.columns
    ]
    parts = [
        "<section><h2>Risk drivers</h2>",
        f"<p>Ranked by {_esc(drivers.method)}. These are the features the model leans "
        "on — associations, not levers. A driver at the top of this list may be a "
        "symptom of a decision already made, or a marker for something else entirely. "
        "The causal section below is where a subset of them gets tested properly.</p>",
        _figure(charts.driver_importance(drivers.table), "Driver importance"),
        _table(drivers.table[columns], max_rows=25),
    ]
    for note in drivers.notes:
        parts.append(_note(note))
    if drivers.profiles:
        parts.append("<h3>How churn varies across the leading drivers</h3>")
        for name, frame in list(drivers.profiles.items())[:4]:
            parts.append(_figure(charts.driver_profile(name, frame), f"Churn rate across {name}"))
    parts.append("</section>")
    return "".join(parts)


def _causal_section(results: dict) -> str:
    causal = results.get("causal")
    if causal is None:
        return ""
    if not causal.estimates and not causal.skipped:
        return ""
    parts = [
        "<section><h2>Causal analysis</h2>",
        "<p>Doubly-robust (AIPW) estimates with cross-fitting. The diamond is the raw "
        "difference between customers who had the condition and those who did not; the "
        "dot is what remains after adjusting for everything else measured before "
        "<code>T</code>. When they diverge, the raw number was mostly confounding.</p>",
        _figure(charts.causal_forest(causal.estimates), "Adjusted effect vs raw difference"),
    ]
    table = causal.table()
    if not table.empty:
        lead = [
            "treatment", "verdict", "causal_ate_pp", "ci_lower_pp", "ci_upper_pp",
            "p_value", "naive_difference_pp",
        ]
        ordered = [c for c in lead if c in table.columns]
        ordered += [c for c in table.columns if c not in ordered]
        parts.append(_table(table[ordered], max_rows=25))

    for est in causal.estimates:
        if not est.refutations and not est.warnings:
            continue
        parts.append(f"<h3>{_esc(est.name)} — <code>{_esc(est.definition)}</code></h3>")
        if est.refutations:
            rows = pd.DataFrame(
                [
                    {
                        "test": k,
                        "estimate_pp": (
                            round(100 * v["ate"], 2) if v["ate"] == v["ate"] else None
                        ),
                        "passed": v["passed"],
                        "what it means": v["explanation"],
                    }
                    for k, v in est.refutations.items()
                ]
            )
            parts.append(_table(rows))
        for warning in est.warnings:
            parts.append(_note(warning))
    if causal.skipped:
        parts.append("<h3>Skipped treatments</h3>")
        parts.append(_table(pd.DataFrame(causal.skipped)))
    for note in causal.notes:
        parts.append(_note(note))
    parts.append(
        _note(
            "None of this rules out an unmeasured confounder. A doubly-robust estimate "
            "corrects for what is in the data; if the real reason customers complain "
            "and leave is something you never recorded, no estimator here can see it. "
            "Treat a surviving effect as the best available evidence for a test, not "
            "as a settled fact."
        )
    )
    parts.append("</section>")
    return "".join(parts)


def _experiment_section(results: dict) -> str:
    exp = results.get("experiment")
    if exp is None or not exp.enabled:
        return ""
    parts = [
        "<section><h2>Experiment</h2>",
        f"<p>Randomised comparison against <code>{_esc(exp.control)}</code> over a "
        f"{exp.horizon_days}-day window from assignment. This outranks every "
        f"observational estimate above.</p>",
    ]
    checks = exp.checks_table()
    if not checks.empty:
        badge = "ok" if exp.trustworthy else "bad"
        label = "randomisation looks sound" if exp.trustworthy else "randomisation is broken"
        parts.append(f'<h3>Validity checks <span class="badge {badge}">{label}</span></h3>')
        parts.append(_table(checks))
    parts.append(_table(exp.table()))
    for note in exp.notes:
        parts.append(_note(note))
    parts.append("</section>")
    return "".join(parts)


def _survival_section(results: dict) -> str:
    survival = results.get("survival")
    if survival is None or survival.overall.empty:
        return ""
    parts = [
        "<section><h2>Survival and lifetime</h2>",
        "<p>Customers still active at the end of the observation window are censored — "
        "they contribute the time they were observed and nothing more. Counting them as "
        "retained is what makes naive lifetime estimates too optimistic.</p>",
        _figure(charts.survival_curve(survival.overall), "Kaplan-Meier survival curve"),
    ]
    for column in list(survival.strata)[:3]:
        parts.append(
            _figure(
                charts.survival_curve(survival.overall, survival.strata, column),
                f"Survival by {column}",
            )
        )
    if not survival.logrank.empty:
        parts.append("<h3>Log-rank tests</h3>")
        parts.append(_table(survival.logrank))
    if not survival.cox.empty:
        concordance = survival.cox_diagnostics.get("concordance")
        parts.append("<h3>Cox proportional hazards</h3>")
        parts.append(
            f"<p>Concordance {concordance:.3f}. A hazard ratio above 1 means faster "
            f"churn. Static attributes only — a current value of a time-varying "
            f"covariate is measured after part of the survival time it would be "
            f"explaining.</p>" if concordance else ""
        )
        parts.append(_table(survival.cox, max_rows=25))
    for note in survival.notes:
        parts.append(_note(note))
    parts.append("</section>")
    return "".join(parts)


def _appendix(results: dict, cfg) -> str:
    fm = results["features"]
    return (
        "<section><h2>Appendix</h2>"
        "<h3>Feature dictionary</h3>"
        "<p>Every feature the pipeline built, how it was derived, and whether it "
        "survived the leakage screen. Copy exact names from here into causal.yaml.</p>"
        + _table(fm.dictionary(), max_rows=60)
        + "<h3>Run configuration</h3>"
        + _table(
            pd.DataFrame(
                [
                    {"setting": "horizon_days", "value": cfg.panel["horizon_days"]},
                    {"setting": "embargo_days", "value": cfg.panel["embargo_days"]},
                    {"setting": "snapshot_mode", "value": cfg.panel["snapshot_mode"]},
                    {"setting": "split.strategy", "value": cfg.split["strategy"]},
                    {"setting": "left_truncation", "value": cfg.left_truncation},
                    {"setting": "leakage.on_block", "value": cfg.leakage["on_block"]},
                    {"setting": "decision_lead_days", "value": cfg.decision_lead_days},
                    {"setting": "random_seed", "value": cfg.seed},
                ]
            ),
            max_rows=20,
        )
        + "</section>"
    )


# --------------------------------------------------------------------------- #
def _write_artifacts(results: dict, out: Path) -> None:
    def dump(frame, name: str) -> None:
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.to_csv(out / name, index=False)

    leak = results["leakage"]
    panel = results["panel"]
    dump(leak.to_frame(), "leakage_findings.csv")
    dump(results.get("leakage_scan"), "leakage_scan.csv")
    dump(results["features"].dictionary(), "feature_dictionary.csv")
    dump(panel.diagnostics.get("per_snapshot"), "panel_by_snapshot.csv")

    if results.get("drivers") is not None:
        dump(results["drivers"].table, "risk_drivers.csv")
    if results.get("model") is not None:
        dump(results["model"].test_predictions, "risk_scores_holdout.csv")
        dump(results["model"].cv_metrics, "cv_metrics.csv")
    if results.get("survival") is not None:
        dump(results["survival"].overall, "survival_curve.csv")
        dump(results["survival"].cox, "survival_cox.csv")
    if results.get("causal") is not None:
        dump(results["causal"].table(), "causal_effects.csv")
    if results.get("experiment") is not None and results["experiment"].enabled:
        dump(results["experiment"].table(), "experiment_results.csv")

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "observation_window": {
            k: str(v.date()) for k, v in panel.timeline.items()
        },
        "panel": {
            "rows": int(len(panel.frame)),
            "customers": int(panel.frame["entity_id"].nunique()),
            "churn_base_rate": round(panel.churn_rate, 6),
            "horizon_days": panel.horizon_days,
            "embargo_days": panel.embargo_days,
            "snapshots": [str(d.date()) for d in panel.snapshot_dates],
        },
        "leakage": {
            "blocking": len(leak.by_level("BLOCK")),
            "warnings": len(leak.by_level("WARN")),
            "quarantined": leak.quarantined,
        },
    }
    if results.get("model") is not None:
        summary["model"] = {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in results["model"].metrics.items()
        }
    if results.get("drivers") is not None and not results["drivers"].table.empty:
        summary["top_drivers"] = (
            results["drivers"].table.head(10)[["feature", "importance", "direction"]]
            .to_dict("records")
        )
    if results.get("causal") is not None:
        summary["causal"] = [
            {k: (v if not is_dataclass(v) else asdict(v)) for k, v in e.as_row().items()}
            for e in results["causal"].estimates
        ]
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

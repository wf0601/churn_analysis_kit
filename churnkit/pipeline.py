"""Pipeline orchestration.

The stage modules are numbered in the order this function calls them, so the file
listing, the imports below and the console banners all read the same way:

    l01_config      load and validate the YAML
    l02_data        read the files, pin down the observation window
    l03_panel       build (customer, prediction date) rows with labels
    l04_features    aggregate inside each window; build derived features
    l05_leakage     audit and quarantine  <- the gate everything else sits behind
    l06_splits      out-of-time or grouped split, with purging
    l07_model       fit, calibrate, score
    l08_drivers     importance, direction, fold stability
    l09_causal      cross-fitted AIPW with refutations
    l10_experiment  ITT analysis of a randomised test
    l11_survival    Kaplan-Meier, log-rank, Cox
    l12_report      HTML + CSV + JSON

Supporting code lives in util/ and is not a stage.

Stage order is a dependency order, not a preference. Leakage enforcement sits
between feature construction and modelling so that nothing downstream — not the
model, not the drivers, not the causal estimates — can ever see a quarantined
column.
"""

from __future__ import annotations

from pathlib import Path

# Imported in run order, which is also the l-number order of the modules.
from . import l01_config as config_mod
from . import l02_data as data_mod
from . import l03_panel as panel_mod
from . import l04_features as features_mod
from . import l05_leakage as leakage_mod
from . import l06_splits as splits_mod
from . import l07_model as model_mod
from . import l08_drivers as drivers_mod
from . import l09_causal as causal_mod
from . import l10_experiment as experiment_mod
from . import l11_survival as survival_mod
from . import l12_report as report_mod
from .util.log import get_logger, stage

log = get_logger("pipeline")


def run(config_dir: str | Path = "config") -> dict:
    stage("l01  Configuration")
    cfg = config_mod.load(config_dir)

    stage("l02  Data")
    dataset = data_mod.load(cfg)
    timeline = data_mod.resolve_timeline(cfg, dataset)

    stage("l03  Panel construction")
    panel = panel_mod.build(cfg, dataset, timeline)

    stage("l04  Feature engineering")
    fm = features_mod.build(cfg, dataset, panel)

    stage("l05  Leakage audit")
    leak = leakage_mod.LeakageReport()
    leakage_mod.audit_structure(cfg, panel, fm, leak)
    y = panel.frame["label"]
    scan = leakage_mod.audit_statistics(cfg, fm, y, leak, seed=cfg.seed)
    leakage_mod.enforce(cfg, fm, leak)
    if leak.quarantined and not scan.empty:
        scan = scan[~scan["feature"].isin(leak.quarantined)]
    log.info(leak.summary())

    results: dict = {
        "config": cfg,
        "dataset": dataset,
        "panel": panel,
        "features": fm,
        "leakage": leak,
        "leakage_scan": scan,
        "data_notes": dataset.notes,
    }

    stage("l06  Train/test split")
    split = splits_mod.make(cfg, panel.frame)
    leakage_mod.audit_splits(
        panel.frame["entity_id"].iloc[split.train_idx],
        panel.frame["entity_id"].iloc[split.test_idx],
        leak,
    )
    stage("l07  Model")
    fitted = model_mod.fit(cfg, fm, panel.frame, split)
    leakage_mod.audit_model_performance(
        cfg, fitted.metrics["auc"], fitted.metrics["base_rate"], leak
    )
    if not fitted.cv_metrics.empty:
        leakage_mod.audit_generalisation(
            float(fitted.cv_metrics["auc"].mean()), fitted.metrics["auc"], leak
        )
    if cfg.leakage["on_block"] == "fail" and leak.by_level("BLOCK"):
        raise leakage_mod.LeakageError(
            "Blocking finding(s) raised by the model audit:\n"
            + "\n".join(f"  [{f.code}] {f.message}" for f in leak.by_level("BLOCK"))
        )
    results["split"] = split
    results["model"] = fitted

    stage("l08  Risk drivers")
    results["drivers"] = drivers_mod.detect(cfg, fm, panel.frame, split, fitted)

    stage("l09  Causal analysis")
    if cfg.causal_run.get("enabled", True):
        results["causal"] = causal_mod.run(cfg, fm, panel.frame)
    else:
        log.info("causal analysis disabled in pipeline.yaml")
    stage("l10  Experiment")
    results["experiment"] = experiment_mod.run(cfg, dataset, panel)

    stage("l11  Survival")
    if cfg.survival.get("enabled", True):
        results["survival"] = survival_mod.run(cfg, dataset, panel)
    else:
        log.info("survival analysis disabled in pipeline.yaml")

    stage("l12  Report")
    path = report_mod.build(results, cfg)
    results["report_path"] = path

    _final_word(results, leak)
    return results


def _final_word(results: dict, leak) -> None:
    log.info("")
    if leak.by_level("BLOCK"):
        log.error(
            "This run had %d blocking leakage finding(s). Read the 'Data integrity' "
            "section of the report before quoting any number from it.",
            len(leak.by_level("BLOCK")),
        )
    model = results.get("model")
    if model is not None:
        log.info(
            "Holdout AUC %.3f · lift@10%% %.1fx · %s",
            model.metrics["auc"], model.metrics["lift_at_10pct"], leak.summary(),
        )
    log.info("Report: %s", results["report_path"])

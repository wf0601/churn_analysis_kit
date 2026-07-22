#!/usr/bin/env python3
"""ChurnKit entry point.

    python run.py                 run the pipeline against config/
    python run.py demo            generate synthetic data, then run
    python run.py check           validate config and data, build nothing
    python run.py --config other/ use a different config directory
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from churnkit.util.errors import ChurnKitError
from churnkit.util.log import get_logger, setup, stage

log = get_logger("cli")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run.py", description="Config-driven churn analysis with leakage guards."
    )
    parser.add_argument(
        "command", nargs="?", default="run", choices=["run", "demo", "check"],
        help="run the pipeline (default), generate demo data first, or validate only",
    )
    parser.add_argument("--config", default="config", help="config directory")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    setup(verbose=args.verbose)

    try:
        if args.command == "demo":
            from tools.make_synthetic_data import generate  # noqa: PLC0415

            stage("l00  Synthetic data")
            generate(Path("data"))

        if args.command == "check":
            return _check(args.config)

        from churnkit.pipeline import run  # noqa: PLC0415

        results = run(args.config)
        print(f"\nReport: {results['report_path']}")
        return 0

    except ChurnKitError as exc:
        log.error("")
        log.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


def _check(config_dir: str) -> int:
    from churnkit import l01_config as config_mod  # noqa: PLC0415
    from churnkit import l02_data as data_mod  # noqa: PLC0415

    stage("Configuration")
    cfg = config_mod.load(config_dir)
    log.info("config loaded: %d feature group(s), %d event source(s)",
             len(cfg.groups), len(cfg.events))

    stage("Data")
    dataset = data_mod.load(cfg)
    data_mod.resolve_timeline(cfg, dataset)

    missing = []
    entity_cols = set(dataset.entity.columns)
    for group in cfg.groups:
        available = (
            entity_cols if group.source == "entity"
            else set(dataset.events[group.source].columns)
        )
        for col in group.columns:
            if col.name not in available and col.name not in {"tenure_days", "recency_days"}:
                missing.append(f"{group.name}.{col.name} (source: {group.source})")

    if missing:
        log.warning("columns in feature.yaml with no matching data column:")
        for item in missing:
            log.warning("  - %s", item)
    else:
        log.info("every column in feature.yaml resolves to a real data column")

    log.info("")
    log.info("config and data check passed — run `python run.py` to produce the report")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Chart rendering.

One committed light look, thin marks, recessive axes, and a table view for every
figure (the report writes the same data to CSV). Categorical hues are assigned in a
fixed order and never cycled — a series keeps its colour when another is filtered out.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, PercentFormatter  # noqa: E402

# Validated categorical order (light surface). Slots are used in order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#84837d"
GRID = "#e8e7e3"
SURFACE = "#ffffff"
CRITICAL = "#e34948"
GOOD = "#1baf7a"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "600",
        "axes.labelsize": 9,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
    }
)


def _clean(ax, *, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _encode(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def churn_over_time(per_snapshot) -> str | None:
    if per_snapshot is None or len(per_snapshot) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    dates = per_snapshot["snapshot_date"]
    ax.plot(dates, per_snapshot["churn_rate"], color=SERIES[0], linewidth=2,
            marker="o", markersize=4, zorder=3)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_ylim(bottom=0)
    ax.set_title("Churn rate by prediction date")
    ax.set_ylabel("share churning within the horizon")
    _clean(ax)
    fig.autofmt_xdate(rotation=0, ha="center")
    return _encode(fig)


def driver_importance(table, top_k: int = 15) -> str | None:
    if table is None or table.empty:
        return None
    top = table.head(top_k).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, max(2.6, 0.32 * len(top) + 0.9)))
    colors = [
        SERIES[0] if stable else INK_MUTED
        for stable in top.get("stable", [True] * len(top))
    ]
    ax.barh(top["feature"], top["importance"], color=colors, height=0.62, zorder=3)
    ax.set_title("Risk drivers, ranked on the held-out period")
    ax.set_xlabel("importance")
    ax.tick_params(axis="y", length=0, labelsize=8)
    _clean(ax, grid_axis="x")
    if (~top.get("stable", True)).any():
        ax.plot([], [], color=SERIES[0], linewidth=6, label="stable across folds")
        ax.plot([], [], color=INK_MUTED, linewidth=6, label="unstable — treat as hypothesis")
        ax.legend(loc="lower right")
    return _encode(fig)


def calibration(table) -> str | None:
    if table is None or table.empty:
        return None
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    limit = max(table["mean_predicted"].max(), table["observed"].max()) * 1.1
    ax.plot([0, limit], [0, limit], color=INK_MUTED, linewidth=1,
            linestyle=(0, (4, 3)), zorder=2)
    ax.plot(table["mean_predicted"], table["observed"], color=SERIES[0],
            linewidth=2, marker="o", markersize=5, zorder=3)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_title("Calibration")
    ax.set_xlabel("predicted risk")
    ax.set_ylabel("observed churn rate")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _clean(ax)
    return _encode(fig)


def lift_curve(predictions) -> str | None:
    if predictions is None or predictions.empty:
        return None
    ordered = predictions.sort_values("risk_score", ascending=False)
    n = len(ordered)
    deciles = []
    for d in range(10):
        lo, hi = int(n * d / 10), int(n * (d + 1) / 10)
        chunk = ordered.iloc[lo:hi]
        if len(chunk):
            deciles.append({"decile": d + 1, "churn_rate": chunk["label"].mean()})
    if len(deciles) < 2:
        return None
    import pandas as pd  # noqa: PLC0415

    frame = pd.DataFrame(deciles)
    base = ordered["label"].mean()
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.bar(frame["decile"], frame["churn_rate"], color=SERIES[0], width=0.62, zorder=3)
    ax.axhline(base, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=4)
    ax.annotate(
        f"base rate {base:.1%}", xy=(10.4, base), xytext=(0, 3),
        textcoords="offset points", ha="right", va="bottom",
        fontsize=8, color=INK_SECONDARY,
    )
    ax.set_title("Churn rate by risk decile")
    ax.set_xlabel("risk decile (1 = highest scored)")
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _clean(ax)
    return _encode(fig)


def survival_curve(overall, strata: dict | None = None, column: str | None = None) -> str | None:
    if overall is None or overall.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    if strata and column and column in strata:
        frame = strata[column]
        levels = list(dict.fromkeys(frame["stratum"]))[: len(SERIES)]
        for i, level in enumerate(levels):
            part = frame[frame["stratum"] == level]
            ax.step(part["time"], part["survival"], where="post",
                    color=SERIES[i], linewidth=2, zorder=3, label=str(level))
            # Direct label at the line end: identity never rests on colour alone.
            ax.annotate(
                str(level),
                xy=(part["time"].iloc[-1], part["survival"].iloc[-1]),
                xytext=(4, 0), textcoords="offset points", va="center",
                fontsize=8, color=INK_SECONDARY,
            )
        ax.legend(loc="lower left", ncols=min(3, len(levels)))
        title = f"Survival by {column}"
    else:
        ax.fill_between(overall["time"], overall["ci_lower"], overall["ci_upper"],
                        color=SERIES[0], alpha=0.15, linewidth=0, zorder=2)
        ax.step(overall["time"], overall["survival"], where="post",
                color=SERIES[0], linewidth=2, zorder=3)
        title = "Survival curve (Kaplan-Meier, 95% band)"

    ax.set_title(title)
    ax.set_xlabel("days since signup")
    ax.set_ylabel("share still active")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    _clean(ax)
    return _encode(fig)


def causal_forest(estimates) -> str | None:
    usable = [e for e in (estimates or []) if e.valid and e.ate == e.ate]
    if not usable:
        return None
    fig, ax = plt.subplots(figsize=(7.2, max(2.4, 0.75 * len(usable) + 1.2)))
    ys = range(len(usable))
    for y, est in zip(ys, usable):
        lo, hi = 100 * est.ate_ci[0], 100 * est.ate_ci[1]
        significant = est.p_value < 0.05
        color = SERIES[0] if significant else INK_MUTED
        ax.plot([lo, hi], [y, y], color=color, linewidth=2, solid_capstyle="round", zorder=3)
        ax.plot([100 * est.ate], [y], marker="o", markersize=8, color=color,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.plot([100 * est.naive_difference], [y], marker="D", markersize=6,
                color=SERIES[1], markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    ax.axvline(0, color=INK_MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
    ax.set_ylim(-0.6, len(usable) - 0.4)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([e.name for e in usable], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("effect on churn rate (percentage points)")
    ax.set_title("Causal effect vs raw association", pad=26)  # room for the legend row
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else f"{v:+.1f}")
    )
    ax.plot([], [], marker="o", linestyle="none", color=SERIES[0],
            label="adjusted effect (95% CI)")
    ax.plot([], [], marker="D", linestyle="none", color=SERIES[1],
            label="raw difference")
    # Above the plot area: inside it, the legend lands on top of a confidence bar.
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncols=2, borderaxespad=0)
    ax.invert_yaxis()
    _clean(ax, grid_axis="x")
    return _encode(fig)


def driver_profile(name: str, frame) -> str | None:
    if frame is None or frame.empty or len(frame) < 2:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    labels = [str(b) for b in frame["bucket"]]
    ax.bar(range(len(frame)), frame["churn_rate"], color=SERIES[0], width=0.62, zorder=3)
    ax.set_xticks(range(len(frame)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_title(f"Churn rate across {name}")
    _clean(ax)
    return _encode(fig)

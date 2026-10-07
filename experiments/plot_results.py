"""Draws the report's data figures from results/summary/ and the stored traces.

    python experiments/plot_results.py

Called by analyse_results.py.  Colours: one fixed colour per method (KERB blue,
FF-Repair orange, Tabu-WarmStart aqua, DSATUR-Recompute violet; the set was
checked for colour-vision deficiency separation), an ordinal blue ramp for
KERB's repair steps, thin marks, a legend whenever there are two or more
series, and text in neutral ink.  Every figure has a table with the same
numbers in the report or in results/summary/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import config as C  # noqa: E402

SUMMARY = C.RESULTS / "summary"
RAW = C.RESULTS / "raw"
FIG = ROOT / "figures"
MAIN = f"KERB-R{C.MAIN_BUDGET}"

SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
METHOD_COLOUR = {MAIN: "#2a78d6", "FF-Repair": "#eb6834", "Tabu-WarmStart": "#1baf7a",
                 "DSATUR-Recompute": "#4a3aa7"}
METHOD_LABEL = {MAIN: f"KERB (R = {C.MAIN_BUDGET})", "FF-Repair": "FF-Repair",
                "Tabu-WarmStart": "Tabu-WarmStart", "DSATUR-Recompute": "DSATUR-Recompute"}
METHODS = list(METHOD_COLOUR)
REPAIR_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]   # ordinal, validated light->dark
WIDTH = 5.7                                                   # inches, fits the A4 text block
LW, MS, RING = 1.5, 6.0, 1.5                                  # 2 px lines, 8 px markers, 2 px ring


def style() -> None:
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.size": 8.5, "axes.labelsize": 8.5,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.minor.size": 0, "ytick.minor.size": 0,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True, "axes.axisbelow": True,
        "legend.frameon": False, "legend.labelcolor": INK2, "text.color": INK,
        "savefig.dpi": 200, "figure.dpi": 100,
    })


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


def _read(name: str) -> pd.DataFrame:
    path = SUMMARY / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _dot(ax, x, y, colour, label=None, zorder=3, size=MS):
    ax.plot(x, y, "o", color=colour, ms=size, mec=SURFACE, mew=RING, label=label, zorder=zorder, ls="none")


DODGE = {m: d for m, d in zip(METHODS, (-0.21, -0.07, 0.07, 0.21))}   # rows offset per method


def _legend_top(ax_or_fig, handles=None, labels=None, ncol=4, y=1.02):
    target = ax_or_fig
    kwargs = dict(loc="lower left", bbox_to_anchor=(0, y), ncol=ncol, handletextpad=0.4,
                  columnspacing=1.2, borderaxespad=0)
    if handles is not None:
        return target.legend(handles, labels, **kwargs)
    return target.legend(**kwargs)


def _log_axis(ax, which="x"):
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    axis.set_major_locator(LogLocator(base=10))
    axis.set_minor_formatter(NullFormatter())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1000 else f"{v:g}"))


# ------------------------------------------------------------------- E1
def fig_e1_static() -> None:
    s = _read("e1_summary.csv")
    if s.empty or "mean_excess" not in s:
        return
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    x = s["mean_time_s"] * 1000
    _dot(ax, x, s["mean_excess"], METHOD_COLOUR[MAIN])
    offsets = {"Greedy (natural order)": (6, 4), "Greedy (random order)": (6, -10),
               "Welsh-Powell": (6, 3), "Smallest-last": (6, -9), "DSATUR": (6, 3), "RLF": (6, 3),
               "TabuCol-minimise": (-6, 6)}
    for xi, yi, name in zip(x, s["mean_excess"], s["algorithm"]):
        dx, dy = offsets.get(name, (6, 3))
        ax.annotate(name, (xi, yi), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK2,
                    ha="right" if dx < 0 else "left")
    _log_axis(ax, "x")
    ax.set_xlabel("Mean run time per instance (ms, log scale)")
    ax.set_ylabel("Mean colours above\nthe known chromatic number")
    ax.set_ylim(bottom=0)
    save(fig, "fig_e1_static_quality_time")


# ------------------------------------------------------------------- E2
def _instance_order(df: pd.DataFrame) -> list:
    return [i for i in C.E2_INSTANCES if i in set(df["instance"])][::-1]


def fig_e2_colours() -> None:
    rel = _read("e2_relative.csv")
    if rel.empty:
        return
    order = _instance_order(rel)
    rel = rel.set_index("instance").loc[order]
    fig, ax = plt.subplots(figsize=(WIDTH, 4.2))
    y = np.arange(len(order))
    for m in METHODS:
        col = f"colours_{m}"
        if col in rel:
            _dot(ax, rel[col] / rel["k0"], y + DODGE[m], METHOD_COLOUR[m], METHOD_LABEL[m], size=5.5)
    ax.axvline(1.0, color=BASELINE, lw=0.8, zorder=1)
    ax.set_yticks(y, order)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean colours during the stream / colours of the shared start colouring")
    _legend_top(ax)
    save(fig, "fig_e2_colours_by_instance")


def fig_e2_recolouring() -> None:
    bi = _read("e2_by_instance.csv")
    if bi.empty:
        return
    bi = bi[bi["algorithm"].isin(METHODS)]
    order = _instance_order(bi)
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 4.2), sharey=True)
    for ax, metric, xlabel in ((axes[0], "mean_recoloured", "Mean recoloured vertices per update"),
                               (axes[1], "max_recoloured", "Largest recolouring in one update")):
        piv = bi.pivot(index="instance", columns="algorithm", values=metric).reindex(order)
        for m in METHODS:
            if m in piv:
                vals = piv[m].clip(lower=1e-3)
                _dot(ax, vals, y + DODGE[m], METHOD_COLOUR[m], METHOD_LABEL[m], size=5.5)
        _log_axis(ax, "x")
        ax.set_xlabel(xlabel + "\n(log scale)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, order)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False,
               handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout()
    save(fig, "fig_e2_recolouring_by_instance")


def fig_tradeoff() -> None:
    e4 = _read("e4_tradeoff.csv")
    bi = _read("e2_by_instance.csv")
    if e4.empty or bi.empty:
        return
    tws = bi[bi["algorithm"] == "Tabu-WarmStart"].set_index("instance")["mean_colours"]
    def rel(df):
        return 100 * (df["mean_colours"] - df["instance"].map(tws)) / df["instance"].map(tws)
    sweep = e4.assign(rel=rel(e4)).groupby("R").agg(rel=("rel", "mean"), rec=("mean_recoloured", "mean"))
    base = bi[bi["algorithm"].isin(METHODS[1:])]
    base = base.assign(rel=rel(base)).groupby("algorithm").agg(rel=("rel", "mean"), rec=("mean_recoloured", "mean"))
    fig, ax = plt.subplots(figsize=(WIDTH, 3.2))
    ax.plot(sweep["rec"], sweep["rel"], "-", color=METHOD_COLOUR[MAIN], lw=LW, zorder=2)
    _dot(ax, sweep["rec"], sweep["rel"], METHOD_COLOUR[MAIN], "KERB, R = 1 ... 25")
    # fixed label positions (offset in points, alignment): the points for R = 5, 10 and 25 lie close together
    place = {1: (-6, 5, "right"), 2: (6, 5, "left"), 3: (7, 2, "left"), 5: (7, -9, "left"),
             10: (-7, 4, "right"), 25: (-7, -10, "right")}
    for R, row in sweep.iterrows():
        ox, oy, ha = place.get(int(R), (6, 5, "left"))
        ax.annotate(f"R={R}", (row["rec"], row["rel"]), xytext=(ox, oy), textcoords="offset points",
                    fontsize=7.5, color=INK2, ha=ha)
    for m in METHODS[1:]:
        if m in base.index:
            _dot(ax, base.loc[m, "rec"], base.loc[m, "rel"], METHOD_COLOUR[m], METHOD_LABEL[m])
    ax.axhline(0, color=BASELINE, lw=0.8, zorder=1)
    _log_axis(ax, "x")
    ax.set_xlabel("Mean recoloured vertices per update (log scale)")
    ax.set_ylabel("Mean colours above\nTabu-WarmStart (%)")
    _legend_top(ax)
    save(fig, "fig_tradeoff_colours_recolouring")


def fig_e4_budget() -> None:
    e4 = _read("e4_tradeoff.csv")
    if e4.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
    for ax, metric, ylabel in ((axes[0], "vs_tws_pct", "Mean colours above\nTabu-WarmStart (%)"),
                               (axes[1], "mean_recoloured", "Mean recoloured\nvertices per update")):
        for inst, grp in e4.groupby("instance"):
            ax.plot(grp["R"], grp[metric], "-", color=GRID, lw=1.0, zorder=1)
        med = e4.groupby("R")[metric].median()
        ax.plot(med.index, med.values, "-", color=METHOD_COLOUR[MAIN], lw=LW, zorder=3)
        _dot(ax, med.index, med.values, METHOD_COLOUR[MAIN], zorder=4)
        _log_axis(ax, "x")
        ax.set_xticks(C.BUDGET_SWEEP, [str(r) for r in C.BUDGET_SWEEP])
        ax.set_xlabel("Recolouring budget R (log scale)")
        ax.set_ylabel(ylabel)
    axes[0].axhline(0, color=BASELINE, lw=0.8, zorder=1)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=GRID, lw=1.0), Line2D([], [], color=METHOD_COLOUR[MAIN], lw=LW, marker="o",
                                                          ms=MS, mec=SURFACE, mew=RING)]
    fig.legend(handles, ["One DIMACS instance", "Median over the 14 instances"], loc="lower left",
               bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "fig_e4_budget_sweep")


def fig_e7_ablation() -> None:
    ab = _read("e7_ablation.csv")
    if ab.empty:
        return
    names = {"KERB-kempe": "Kempe-chain repair\ninstead of ejection chains",
             "KERB-direct": "Direct moves only\n(no chains)",
             "KERB-noElim": "Without class\nelimination (C3)",
             "KERB-noMig": "Without migration\nplans (C4)",
             "KERB-tabu": "With budgeted\nTabuCol repair"}
    ab = ab.set_index("variant").reindex([v for v in names if v in set(ab["variant"])])
    y = np.arange(len(ab))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
    axes[0].barh(y, ab["mean_delta_colours_pct"], height=0.5, color=METHOD_COLOUR[MAIN], zorder=2)
    axes[0].axvline(0, color=BASELINE, lw=0.8)
    axes[0].set_xlabel("Change in mean colours\nversus full KERB (%)")
    for yi, v in zip(y, ab["mean_delta_colours_pct"]):
        axes[0].annotate(f"{v:+.1f}%", (max(v, 0), yi), xytext=(4, 0), textcoords="offset points",
                         va="center", ha="left", fontsize=7.5, color=INK2)
    axes[1].barh(y, ab["p95_time_ratio"], height=0.5, color=METHOD_COLOUR[MAIN], zorder=2)
    axes[1].axvline(1, color=BASELINE, lw=0.8)
    _log_axis(axes[1], "x")
    axes[1].set_xlabel("95th-percentile update time\nrelative to full KERB (log scale)")
    for yi, v in zip(y, ab["p95_time_ratio"]):
        axes[1].annotate(f"\u00d7{v:.2f}", (v, yi), xytext=(4, 0), textcoords="offset points", va="center",
                         fontsize=7.5, color=INK2)
    axes[0].set_yticks(y, [names[v] for v in ab.index])
    for ax in axes:
        ax.grid(axis="y", visible=False)
    lo, hi = axes[0].get_xlim()
    axes[0].set_xlim(lo - 0.05 * (hi - lo), hi + 0.25 * (hi - lo))
    lo, hi = axes[1].get_xlim()
    axes[1].set_xlim(lo, hi * 3)
    fig.tight_layout()
    save(fig, "fig_e7_ablation")


def fig_time_ecdf() -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 3.0))
    drew = False
    for m in METHODS:
        parts = []
        for inst in C.E2_INSTANCES:
            for s in C.E2_SEEDS:
                p = RAW / "e2" / "traces" / f"{inst}__{m}__s{s}.npz"
                if p.exists():
                    parts.append(np.load(p)["time_us"].astype(float))
        if not parts:
            continue
        t = np.sort(np.concatenate(parts))
        t = np.clip(t, 0.1, None)
        frac = np.arange(1, t.size + 1) / t.size
        step = max(1, t.size // 4000)
        ax.plot(t[::step], frac[::step], "-", color=METHOD_COLOUR[m], lw=LW, label=METHOD_LABEL[m])
        drew = True
    if not drew:
        plt.close(fig)
        return
    _log_axis(ax, "x")
    ax.set_xlabel("Update time (microseconds, log scale)")
    ax.set_ylabel("Share of updates\nat or below this time")
    ax.set_ylim(0, 1.01)
    _legend_top(ax)
    save(fig, "fig_e2_update_time_ecdf")


def fig_trace_example(instance: str = "DSJC250.5", seed: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    drew = False
    for m in ["FF-Repair", "DSATUR-Recompute", MAIN, "Tabu-WarmStart"]:
        p = RAW / "e2" / "traces" / f"{instance}__{m}__s{seed}.npz"
        if not p.exists():
            continue
        tr = np.load(p)
        x = tr["t"] if "t" in tr.files else np.arange(1, tr["colours"].size + 1)
        ax.plot(x, tr["colours"], "-", color=METHOD_COLOUR[m], lw=LW if m != "DSATUR-Recompute" else 1.0,
                label=METHOD_LABEL[m] + (" (every 10th update)" if m == "DSATUR-Recompute" else ""),
                drawstyle="steps-post" if m != "DSATUR-Recompute" else "default")
        drew = True
    if not drew:
        plt.close(fig)
        return
    ax.set_xlabel("Update")
    ax.set_ylabel("Colours in use")
    handles, labels = ax.get_legend_handles_labels()
    _legend_top(ax, handles, labels, ncol=2)
    save(fig, f"fig_trace_{instance}_s{seed}")


def fig_repair_steps() -> None:
    rs = _read("e2_repair_steps.csv")
    if rs.empty:
        return
    steps = [("n_direct", "Direct move"), ("n_chain", "Ejection chain"),
             ("n_direct_victim", "Move into the class being removed (C4)"), ("n_new_colour", "New colour")]
    order = [i for i in C.E2_INSTANCES if i in set(rs["instance"])][::-1]
    rs = rs.set_index("instance").loc[order]
    tot = rs[[c for c, _ in steps]].sum(axis=1).replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
    y = np.arange(len(order))
    left = np.zeros(len(order))
    for (col, label), colour in zip(steps, REPAIR_RAMP):
        share = (100 * rs[col] / tot).fillna(0).values
        ax.barh(y, share, left=left, height=0.6, color=colour, edgecolor=SURFACE, linewidth=1.5,
                label=label, zorder=2)
        left += share
    labels = [f"{i}  ({int(c)})" for i, c in zip(order, rs["conflicts"])]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of conflicting insertions repaired by each step (%); conflicts in brackets")
    _legend_top(ax, ncol=4)
    save(fig, "fig_e2_repair_steps")


def fig_pilot() -> None:
    pl = _read("../pilot/pilot_summary.csv") if not (C.RESULTS / "pilot" / "pilot_summary.csv").exists() \
        else pd.read_csv(C.RESULTS / "pilot" / "pilot_summary.csv")
    if pl.empty:
        return
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    other = pl[~pl["chosen"]]
    chosen = pl[pl["chosen"]]
    _dot(ax, other["mean_time_us"], other["mean_colours"], MUTED, "Other configurations")
    _dot(ax, chosen["mean_time_us"], chosen["mean_colours"], METHOD_COLOUR[MAIN], "Chosen configuration")
    for _, r in pl.iterrows():
        ax.annotate(f"{int(r.search_nodes)}/{int(r.plan_iters)}/{int(r.tabu_iters)}",
                    (r.mean_time_us, r.mean_colours), xytext=(5, 3), textcoords="offset points",
                    fontsize=7, color=INK2)
    ax.set_xlabel("Mean update time (microseconds)")
    ax.set_ylabel("Mean colours")
    _legend_top(ax, ncol=2)
    save(fig, "fig_pilot_configurations")


# ------------------------------------------------------------------- E3
def fig_e3() -> None:
    bs = _read("e3_by_stream.csv")
    if bs.empty:
        return
    bs = bs[bs["algorithm"].isin(METHODS)]
    cps = _read("e3_checkpoints.csv")
    order = [s for s in C.E3_STREAMS if s in set(bs["instance"])][::-1]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6), sharey=True)
    piv = bs.pivot(index="instance", columns="algorithm", values="mean_colours").reindex(order)
    for m in METHODS:
        if m in piv:
            _dot(axes[0], piv[m], y + DODGE[m], METHOD_COLOUR[m], METHOD_LABEL[m], size=5.5)
    axes[0].set_xlabel("Mean colours in use")
    pivr = bs.pivot(index="instance", columns="algorithm", values="mean_recoloured").reindex(order)
    for m in METHODS:
        if m in pivr:
            _dot(axes[1], pivr[m].clip(lower=1e-3), y + DODGE[m], METHOD_COLOUR[m], METHOD_LABEL[m], size=5.5)
    _log_axis(axes[1], "x")
    axes[1].set_xlabel("Mean recoloured vertices\nper update (log scale)")
    axes[0].set_yticks(y, order)
    for ax in axes:
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=4, frameon=False,
               handletextpad=0.4, columnspacing=1.2)
    fig.tight_layout()
    save(fig, "fig_e3_temporal")


# ------------------------------------------------------------------- E5
def fig_e5() -> None:
    rgg = _read("e5_rgg.csv")
    if rgg.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
    for m in (MAIN, "FF-Repair"):
        g = rgg[rgg["algorithm"] == m].sort_values("n")
        if g.empty:
            continue
        axes[0].plot(g["n"], g["median_time_us"], "-", color=METHOD_COLOUR[m], lw=LW, zorder=2)
        _dot(axes[0], g["n"], g["median_time_us"], METHOD_COLOUR[m], METHOD_LABEL[m])
        axes[1].plot(g["n"], g["rss_end_mb"], "-", color=METHOD_COLOUR[m], lw=LW, zorder=2)
        _dot(axes[1], g["n"], g["rss_end_mb"], METHOD_COLOUR[m], METHOD_LABEL[m])
    for ax in axes:
        _log_axis(ax, "x")
        _log_axis(ax, "y")
        ax.set_xlabel("Vertices n (log scale)")
    axes[0].yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axes[0].set_ylim(1, 12)
    axes[0].set_ylabel("Median update time\n(microseconds, log scale)")
    axes[1].set_ylabel("Resident memory at the end\n(MB, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "fig_e5_scalability")


def main() -> None:
    style()
    for fn in (fig_e1_static, fig_e2_colours, fig_e2_recolouring, fig_tradeoff, fig_e4_budget,
               fig_e7_ablation, fig_time_ecdf, fig_trace_example, fig_repair_steps, fig_pilot, fig_e3, fig_e5):
        try:
            fn()
        except Exception as exc:  # a missing table must not stop the other figures
            print(f"{fn.__name__} skipped: {exc!r}")


if __name__ == "__main__":
    main()

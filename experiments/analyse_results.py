"""Merges the raw run files into summary tables, statistics and the coverage score.

    python experiments/analyse_results.py [--no-figures] [--no-tests]

Reads results/raw/ (written by run_experiments.py), results/pilot/ and
results/summary/test_cases.csv, and writes CSV tables to results/summary/.
Figures are drawn by plot_results.py from those tables.  Every number comes
from a stored run; nothing is estimated or filled in.  Missing runs are listed
in results/summary/missing_runs.csv and make the affected criteria fail.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import config as C  # noqa: E402

RAW = C.RESULTS / "raw"
OUT = C.RESULTS / "summary"
MAIN = f"KERB-R{C.MAIN_BUDGET}"
BASELINES = ["FF-Repair", "Tabu-WarmStart", "DSATUR-Recompute"]
METHODS = [MAIN] + BASELINES
RUN_METRICS = ["mean_colours", "max_colours", "final_colours", "mean_recoloured", "max_recoloured",
               "share_updates_recolouring", "conflict_rate", "new_colour_rate", "eliminations",
               "mean_time_us", "median_time_us", "p95_time_us", "max_time_us",
               "max_recolourings_per_vertex", "gini_recolourings"]
REPAIR_NAMES = {0: "none", 1: "direct", 2: "chain", 3: "kempe", 4: "tabu", 5: "direct_victim",
                6: "new_colour", 7: "recompute"}


# ------------------------------------------------------------------ helpers
def _save(df: pd.DataFrame, name: str, index: bool = False) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=index, float_format="%.10g")
    return df


def _load_runs(folder: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob("*.json")):
        with open(path) as fh:
            row = json.load(fh)
        row.pop("checkpoints", None)
        for key, value in (row.pop("kerb_stats", None) or {}).items():
            row[f"ks_{key}"] = value
        row["file"] = path.name
        rows.append(row)
    return pd.DataFrame(rows)


def _load_checkpoints(folder: Path, label: str) -> pd.DataFrame:
    rows = []
    for path in sorted(folder.glob(f"*__{label}__*.json")):
        with open(path) as fh:
            row = json.load(fh)
        for cp in row.get("checkpoints", []):
            rows.append({"instance": row["instance"], "seed": row["seed"], **cp})
    return pd.DataFrame(rows)


def _checkpoint_summary(cps: pd.DataFrame) -> pd.DataFrame:
    """Per instance: KERB colours, DSATUR-from-scratch colours and a clique lower bound,
    all measured on the same graph G_t at the same checkpoints (means over checkpoints)."""
    cps = cps.assign(kerb_minus_lb=cps["colours"] - cps["clique_lb"],
                     kerb_minus_dsatur=cps["colours"] - cps["dsatur_colours"],
                     kerb_le_dsatur=(cps["colours"] <= cps["dsatur_colours"]).astype(float),
                     lb_violations=(cps["clique_lb"] > cps["colours"]).astype(int))
    return cps.groupby("instance").agg(checkpoints=("t", "size"), kerb_colours=("colours", "mean"),
                                       dsatur_colours=("dsatur_colours", "mean"),
                                       clique_lb_mean=("clique_lb", "mean"), clique_lb_max=("clique_lb", "max"),
                                       kerb_minus_lb=("kerb_minus_lb", "mean"),
                                       kerb_minus_dsatur=("kerb_minus_dsatur", "mean"),
                                       share_kerb_le_dsatur=("kerb_le_dsatur", "mean"),
                                       lb_violations=("lb_violations", "sum"),
                                       edges_mean=("edges", "mean"), max_degree=("max_degree", "max"),
                                       dsatur_time_s=("dsatur_time_s", "mean")).reset_index()


def _trace_path(experiment: str, instance: str, label: str, seed: int) -> Path:
    return RAW / experiment / "traces" / f"{instance}__{label}__s{seed}.npz"


def holm(pvalues: Sequence[float]) -> List[float]:
    """Holm step-down adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def paired_test(a: pd.Series, b: pd.Series, alternative: str) -> Dict[str, float]:
    """Wilcoxon signed-rank test on paired values (d = a - b) with effect sizes.

    Zero differences are discarded (SciPy's default ``zero_method='wilcox'``).
    The rank-biserial correlation is (R+ - R-) / (R+ + R-) over the non-zero
    differences, so a negative value means ``a`` tends to be smaller.
    """
    d = (a - b).dropna()
    nz = d[d != 0]
    out = {"n_pairs": int(d.size), "n_nonzero": int(nz.size), "a_lower": int((d < 0).sum()),
           "a_higher": int((d > 0).sum()), "ties": int((d == 0).sum()),
           "median_diff": float(d.median()) if d.size else np.nan,
           "mean_diff": float(d.mean()) if d.size else np.nan}
    if nz.size == 0:
        out.update(statistic=np.nan, p_value=np.nan, method="all differences zero", rank_biserial=0.0)
        return out
    ranks = stats.rankdata(nz.abs())
    r_plus = float(ranks[nz.values > 0].sum())
    r_minus = float(ranks[nz.values < 0].sum())
    res = stats.wilcoxon(a.loc[d.index], b.loc[d.index], zero_method="wilcox", alternative=alternative)
    # Name the method SciPy's default ('auto') actually used: the exact null distribution for small
    # samples without zero differences, the normal approximation otherwise.
    exact = (stats.wilcoxon(a.loc[d.index], b.loc[d.index], zero_method="wilcox", alternative=alternative,
                            method="exact").pvalue if nz.size <= 50 else np.nan)
    method = "exact" if np.isclose(res.pvalue, exact, rtol=1e-12, atol=0) else "normal approximation"
    if d.size > nz.size:
        method += f" ({d.size - nz.size} zero difference{'s' if d.size - nz.size > 1 else ''} discarded)"
    out.update(statistic=float(res.statistic), p_value=float(res.pvalue), method=method,
               rank_biserial=(r_plus - r_minus) / (r_plus + r_minus))
    return out


def loglog_slope(x: Sequence[float], y: Sequence[float]) -> float:
    lx, ly = np.log10(np.asarray(x, float)), np.log10(np.asarray(y, float))
    return float(np.polyfit(lx, ly, 1)[0])


# ======================================================================= E1
def analyse_e1() -> pd.DataFrame:
    rows = []
    trick = pd.read_csv(C.DATA / "dimacs" / "trick_table.csv", dtype=str)
    trick = trick.set_index("instance")
    for path in sorted((RAW / "e1").glob("*.json")):
        with open(path) as fh:
            d = json.load(fh)
        chi = trick.loc[d["instance"], "page_chromatic_number"] if d["instance"] in trick.index else "?"
        base = {k: d[k] for k in ("instance", "n", "m", "density", "max_degree", "degeneracy", "clique_lb")}
        base["family"] = trick.loc[d["instance"], "page_source_code"] if d["instance"] in trick.index else ""
        base["chi_page"] = float(chi) if chi not in ("?", "", None) else np.nan
        for alg, res in d["algorithms"].items():
            rows.append({**base, "algorithm": alg, "colours": res["colours"], "time_s": res["time_s"]})
    long = pd.DataFrame(rows)
    if long.empty:
        return long
    _save(long, "e1_static_long.csv")
    attrs = long.drop_duplicates("instance").set_index("instance")[
        ["family", "n", "m", "density", "max_degree", "degeneracy", "clique_lb", "chi_page"]]
    wide = attrs.join(long.pivot(index="instance", columns="algorithm", values="colours")).reset_index()
    _save(wide, "e1_static.csv")
    known = long.dropna(subset=["chi_page"])
    summary = long.groupby("algorithm").agg(mean_time_s=("time_s", "mean"),
                                            median_time_s=("time_s", "median")).reset_index()
    best = long.groupby("instance")["colours"].transform("min")
    long = long.assign(at_best=(long["colours"] <= best + 1e-9))
    summary = summary.merge(long.groupby("algorithm")["at_best"].sum().rename("instances_at_best_of_seven")
                            .reset_index(), on="algorithm")
    if not known.empty:
        k = known.assign(excess=known["colours"] - known["chi_page"],
                         rel_excess_pct=100 * (known["colours"] - known["chi_page"]) / known["chi_page"])
        agg = k.groupby("algorithm").agg(instances_with_chi=("instance", "nunique"),
                                         mean_excess=("excess", "mean"),
                                         mean_rel_excess_pct=("rel_excess_pct", "mean"),
                                         optimal_hits=("excess", lambda s: int((s <= 0).sum()))).reset_index()
        summary = summary.merge(agg, on="algorithm", how="left")
    _save(summary.sort_values("mean_time_s"), "e1_summary.csv")
    return long


# ============================================================ E2 / E4 / E7
def expected_e2() -> List[Tuple[str, int, str]]:
    labels = [f"KERB-R{R}" for R in C.BUDGET_SWEEP] + list(C.ABLATIONS) + BASELINES
    return [(i, s, l) for i in C.E2_INSTANCES for s in C.E2_SEEDS for l in labels]


def analyse_e2() -> Dict[str, pd.DataFrame]:
    runs = _load_runs(RAW / "e2" / "runs")
    if runs.empty:
        return {}
    _save(runs, "e2_runs.csv")
    have = set(zip(runs["instance"], runs["seed"], runs["algorithm"]))
    missing = [t for t in expected_e2() if t not in have]
    metrics = [m for m in RUN_METRICS if m in runs.columns]
    by_inst = runs.groupby(["instance", "algorithm"])[metrics + ["initial_colours", "n"]].mean().reset_index()
    sd = runs.groupby(["instance", "algorithm"])["mean_colours"].std().rename("sd_mean_colours")
    nseeds = runs.groupby(["instance", "algorithm"])["seed"].nunique().rename("seeds")
    by_inst = by_inst.merge(sd.reset_index(), on=["instance", "algorithm"]).merge(
        nseeds.reset_index(), on=["instance", "algorithm"])
    _save(by_inst, "e2_by_instance.csv")

    main = by_inst[by_inst["algorithm"].isin(METHODS)]
    overall = main.groupby("algorithm")[metrics].mean().reindex(METHODS).reset_index()
    overall.insert(1, "instances", main.groupby("algorithm")["instance"].nunique().reindex(METHODS).values)
    _save(overall, "e2_overall.csv")

    # colours relative to the shared start and to Tabu-WarmStart, per instance
    piv_k = main.pivot(index="instance", columns="algorithm", values="mean_colours")
    piv_r = main.pivot(index="instance", columns="algorithm", values="mean_recoloured")
    k0 = main.groupby("instance")["initial_colours"].mean()
    rel = pd.DataFrame({"k0": k0})
    for m in METHODS:
        if m in piv_k:
            rel[f"colours_{m}"] = piv_k[m]
            rel[f"recoloured_{m}"] = piv_r[m]
    if "Tabu-WarmStart" in piv_k and MAIN in piv_k:
        rel["kerb_vs_tws_pct"] = 100 * (piv_k[MAIN] - piv_k["Tabu-WarmStart"]) / piv_k["Tabu-WarmStart"]
    if "FF-Repair" in piv_k and MAIN in piv_k:
        rel["kerb_vs_ff_pct"] = 100 * (piv_k[MAIN] - piv_k["FF-Repair"]) / piv_k["FF-Repair"]
    if "DSATUR-Recompute" in piv_k and MAIN in piv_k:
        rel["kerb_vs_dsr_pct"] = 100 * (piv_k[MAIN] - piv_k["DSATUR-Recompute"]) / piv_k["DSATUR-Recompute"]
    _save(rel.reset_index(), "e2_relative.csv")

    # run-level paired counts (same instance and seed)
    pair_rows = []
    wide_run = runs.pivot_table(index=["instance", "seed"], columns="algorithm",
                                values=["mean_colours", "mean_recoloured", "max_recoloured"])
    for metric in ("mean_colours", "mean_recoloured", "max_recoloured"):
        for other in BASELINES:
            if (metric, MAIN) not in wide_run or (metric, other) not in wide_run:
                continue
            a, b = wide_run[(metric, MAIN)], wide_run[(metric, other)]
            ok = a.notna() & b.notna()
            d = a[ok] - b[ok]
            pair_rows.append({"metric": metric, "other": other, "runs": int(ok.sum()),
                              "kerb_lower": int((d < 0).sum()), "equal": int((d == 0).sum()),
                              "kerb_higher": int((d > 0).sum()),
                              "share_kerb_lower": float((d < 0).mean()) if ok.any() else np.nan})
    pairs = _save(pd.DataFrame(pair_rows), "e2_run_pairs.csv")

    # pre-registered tests on instance means
    test_rows = []
    for family, comps in (("primary", C.PRIMARY_COMPARISONS), ("secondary", C.SECONDARY_COMPARISONS)):
        for metric, other, alternative in comps:
            piv = main.pivot(index="instance", columns="algorithm", values=metric)
            if MAIN not in piv or other not in piv:
                continue
            res = paired_test(piv[MAIN], piv[other], alternative)
            test_rows.append({"family": family, "metric": metric, "kerb": MAIN, "other": other,
                              "alternative": alternative, **res})
    tests = pd.DataFrame(test_rows)
    if not tests.empty:
        prim = tests["family"] == "primary"
        tests.loc[prim, "p_holm"] = holm(tests.loc[prim, "p_value"].fillna(1.0).tolist())
        tests["significant"] = tests["p_holm"] < C.ALPHA
    _save(tests, "e2_tests.csv")

    # checkpoints: DSATUR from scratch and clique lower bound on G_t (KERB-R3 runs)
    cps = _load_checkpoints(RAW / "e2" / "runs", MAIN)
    if not cps.empty:
        _save(_checkpoint_summary(cps), "e2_checkpoints.csv")

    # E4: budget sweep
    sweep_labels = [f"KERB-R{R}" for R in C.BUDGET_SWEEP]
    sweep = by_inst[by_inst["algorithm"].isin(sweep_labels)].copy()
    sweep["R"] = sweep["algorithm"].str.replace("KERB-R", "", regex=False).astype(int)
    tws = by_inst[by_inst["algorithm"] == "Tabu-WarmStart"].set_index("instance")["mean_colours"]
    sweep["vs_tws_pct"] = 100 * (sweep["mean_colours"] - sweep["instance"].map(tws)) / sweep["instance"].map(tws)
    _save(sweep.sort_values(["instance", "R"]), "e4_tradeoff.csv")
    sp_rows = []
    for inst, grp in sweep.groupby("instance"):
        if grp["R"].nunique() >= 3:
            rho, p = stats.spearmanr(grp["R"], grp["mean_colours"])
            rho_r, _ = stats.spearmanr(grp["R"], grp["mean_recoloured"])
            sp_rows.append({"instance": inst, "budgets": int(grp["R"].nunique()),
                            "spearman_R_colours": rho, "p_value": p, "spearman_R_recoloured": rho_r,
                            "colours_R1": float(grp.loc[grp["R"] == grp["R"].min(), "mean_colours"].iloc[0]),
                            "colours_Rmax": float(grp.loc[grp["R"] == grp["R"].max(), "mean_colours"].iloc[0])})
    _save(pd.DataFrame(sp_rows), "e4_spearman.csv")
    sweep_overall = sweep.groupby("R")[["mean_colours", "mean_recoloured", "max_recoloured", "median_time_us",
                                        "p95_time_us", "vs_tws_pct"]].mean().reset_index()
    sweep_overall["median_vs_tws_pct"] = sweep.groupby("R")["vs_tws_pct"].median().values
    _save(sweep_overall, "e4_overall.csv")

    # E7: ablations at R = MAIN_BUDGET
    abl_labels = [MAIN] + list(C.ABLATIONS)
    abl = by_inst[by_inst["algorithm"].isin(abl_labels)]
    full = abl[abl["algorithm"] == MAIN].set_index("instance")
    abl_rows = []
    for label in C.ABLATIONS:
        part = abl[abl["algorithm"] == label].set_index("instance")
        common = part.index.intersection(full.index)
        if common.empty:
            continue
        dk = part.loc[common, "mean_colours"] - full.loc[common, "mean_colours"]
        dr = part.loc[common, "mean_recoloured"] - full.loc[common, "mean_recoloured"]
        dt = part.loc[common, "median_time_us"] / full.loc[common, "median_time_us"]
        dt95 = part.loc[common, "p95_time_us"] / full.loc[common, "p95_time_us"]
        abl_rows.append({"variant": label, "instances": len(common),
                         "mean_delta_colours": dk.mean(), "median_delta_colours": dk.median(),
                         "mean_delta_colours_pct": (100 * dk / full.loc[common, "mean_colours"]).mean(),
                         "instances_more_colours": int((dk > 1e-9).sum()),
                         "instances_fewer_colours": int((dk < -1e-9).sum()),
                         "mean_delta_recoloured": dr.mean(),
                         "median_time_ratio": float(np.median(dt)), "p95_time_ratio": float(np.median(dt95))})
    _save(pd.DataFrame(abl_rows), "e7_ablation.csv")
    _save(abl.pivot(index="instance", columns="algorithm", values="mean_colours").reset_index(),
          "e7_by_instance.csv")

    # repair steps used by KERB-R3 (from the per-update traces)
    rep_rows = []
    for inst in C.E2_INSTANCES:
        counts = {name: 0 for name in REPAIR_NAMES.values()}
        found = 0
        for s in C.E2_SEEDS:
            path = _trace_path("e2", inst, MAIN, s)
            if not path.exists():
                continue
            found += 1
            codes = np.load(path)["repair"]
            for code, cnt in zip(*np.unique(codes, return_counts=True)):
                counts[REPAIR_NAMES.get(int(code), "other")] = counts.get(REPAIR_NAMES.get(int(code), "other"), 0) + int(cnt)
        if found:
            total_conf = sum(v for k, v in counts.items() if k not in ("none",))
            rep_rows.append({"instance": inst, "runs": found, "updates": sum(counts.values()),
                             "conflicts": total_conf, **{f"n_{k}": v for k, v in counts.items()}})
    _save(pd.DataFrame(rep_rows), "e2_repair_steps.csv")

    _save(pd.DataFrame(missing, columns=["instance", "seed", "algorithm"]), "e2_missing_runs.csv")
    return {"runs": runs, "by_inst": by_inst, "tests": tests, "pairs": pairs, "missing": missing}


def pooled_times(experiment: str, instances: Iterable[str], seeds: Iterable[int], label: str) -> np.ndarray:
    parts = []
    for inst in instances:
        for s in seeds:
            path = _trace_path(experiment, inst, label, s)
            if path.exists():
                parts.append(np.load(path)["time_us"].astype(float))
    return np.concatenate(parts) if parts else np.array([])


def deviation_d1_check() -> pd.DataFrame:
    """Largest n / min_t k_t over the stored E2 KERB runs (D1 can only act above 100 R)."""
    rows = []
    for path in sorted((RAW / "e2" / "traces").glob("*__KERB-*__s*.npz")):
        inst, label, seed = path.stem.split("__")
        n = None
        js = RAW / "e2" / "runs" / f"{path.stem}.json"
        if js.exists():
            with open(js) as fh:
                d = json.load(fh)
            n = d["n"]
            has_new_counter = "plans_skipped" in (d.get("kerb_stats") or {})
        else:
            continue
        k = np.load(path)["colours"]
        budget = int(label.split("-R")[1]) if "-R" in label else C.MAIN_BUDGET
        rows.append({"instance": inst, "algorithm": label, "seed": int(seed[1:]), "n": n, "budget": budget,
                     "min_colours": int(k.min()), "n_over_min_colours": n / int(k.min()),
                     "skip_threshold": budget * C.KERB_PARAMS["max_stall"],
                     "run_has_plans_skipped_counter": has_new_counter})
    df = pd.DataFrame(rows)
    if not df.empty:
        df["rule_could_apply"] = df["n_over_min_colours"] > df["skip_threshold"]
        _save(df, "deviation_D1_check.csv")
    return df


# ======================================================================= E3
def expected_e3() -> List[Tuple[str, int, str]]:
    labels = [f"KERB-R{R}" for R in C.BUDGET_SWEEP_TEMPORAL] + BASELINES
    return [(i, s, l) for i in C.E3_STREAMS for s in C.E3_SEEDS for l in labels]


def analyse_e3() -> Dict[str, pd.DataFrame]:
    runs = _load_runs(RAW / "e3" / "runs")
    if runs.empty:
        return {}
    _save(runs, "e3_runs.csv")
    have = set(zip(runs["instance"], runs["seed"], runs["algorithm"]))
    missing = [t for t in expected_e3() if t not in have]
    metrics = [m for m in RUN_METRICS if m in runs.columns]
    by_stream = runs.groupby(["instance", "algorithm"])[metrics + ["n", "updates"]].mean().reset_index()
    _save(by_stream, "e3_by_stream.csv")
    cps = _load_checkpoints(RAW / "e3" / "runs", MAIN)
    if not cps.empty:
        _save(_checkpoint_summary(cps), "e3_checkpoints.csv")
    main = by_stream[by_stream["algorithm"].isin(METHODS)]
    test_rows = []
    for metric, other, alternative in C.PRIMARY_COMPARISONS + C.SECONDARY_COMPARISONS:
        piv = main.pivot(index="instance", columns="algorithm", values=metric)
        if MAIN in piv and other in piv:
            test_rows.append({"metric": metric, "other": other, "alternative": alternative,
                              "note": "descriptive only (5 streams)",
                              **paired_test(piv[MAIN], piv[other], alternative)})
    _save(pd.DataFrame(test_rows), "e3_tests.csv")
    sweep = by_stream[by_stream["algorithm"].str.startswith("KERB-R")].copy()
    sweep["R"] = sweep["algorithm"].str.replace("KERB-R", "", regex=False).astype(int)
    _save(sweep.sort_values(["instance", "R"]), "e3_tradeoff.csv")
    _save(pd.DataFrame(missing, columns=["instance", "seed", "algorithm"]), "e3_missing_runs.csv")
    return {"runs": runs, "by_stream": by_stream, "missing": missing}


# ======================================================================= E5
def analyse_e5() -> Dict[str, pd.DataFrame]:
    rows = []
    for path in sorted((RAW / "e5").glob("*.json")):
        with open(path) as fh:
            row = json.load(fh)
        for key, value in (row.pop("kerb_stats", None) or {}).items():
            row[f"ks_{key}"] = value
        row["file"] = path.name
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return {}
    rgg = df[df["file"].str.startswith("rgg_")].sort_values(["algorithm", "n"])
    real = df[~df["file"].str.startswith("rgg_")]
    _save(rgg, "e5_rgg.csv")
    _save(real, "e5_real.csv")
    slope_rows = []
    for alg, grp in rgg.groupby("algorithm"):
        if grp["n"].nunique() >= 2:
            slope_rows.append({"algorithm": alg, "sizes": int(grp["n"].nunique()),
                               "slope_median_time": loglog_slope(grp["n"], grp["median_time_us"]),
                               "slope_mean_time": loglog_slope(grp["n"], grp["mean_time_us"]),
                               "slope_rss_graph": loglog_slope(grp["n"], grp["rss_graph_mb"]),
                               "median_time_ratio_max_min_n": float(
                                   grp.loc[grp["n"].idxmax(), "median_time_us"] /
                                   grp.loc[grp["n"].idxmin(), "median_time_us"])})
    _save(pd.DataFrame(slope_rows), "e5_slopes.csv")
    return {"rgg": rgg, "real": real, "slopes": pd.DataFrame(slope_rows)}


# ================================================================ tests (C12)
def run_pytest() -> Dict[str, int]:
    junit = OUT / "pytest_junit.xml"
    subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={junit}"], cwd=ROOT,
                   capture_output=True, text=True)
    root = ET.parse(junit).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    res = {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}
    with open(OUT / "pytest_summary.json", "w") as fh:
        json.dump(res, fh, indent=1)
    return res


# ============================================================ coverage C1-C12
def coverage(e2: Dict, e3: Dict, e5: Dict, pytest_res: Optional[Dict[str, int]]) -> pd.DataFrame:
    errors = sorted((RAW / "errors").glob("*.json"))
    err_funcs = []
    for p in errors:
        with open(p) as fh:
            err_funcs.append(json.load(fh).get("function", "?"))
    tc = pd.read_csv(OUT / "test_cases.csv") if (OUT / "test_cases.csv").exists() else pd.DataFrame()
    tc_pass = dict(zip(tc.get("Test ID", []), tc.get("Status", [])))
    empty = pd.DataFrame(columns=["instance", "seed", "algorithm", "budget", "max_recoloured"])
    runs2 = e2.get("runs", empty)
    runs3 = e3.get("runs", empty)
    kerb2 = runs2[runs2["algorithm"].str.startswith("KERB")]
    kerb3 = runs3[runs3["algorithm"].str.startswith("KERB")]
    n_missing = len(e2.get("missing", expected_e2())) + len(e3.get("missing", expected_e3()))
    rows = []

    def add(cid, score, evidence):
        weight, name, rule = next((w, n, r) for c, w, n, r in C.COVERAGE_CRITERIA if c == cid)
        rows.append({"id": cid, "weight": weight, "criterion": name, "rule": rule, "evidence": evidence,
                     "score": score, "weighted": weight * score})

    validated = len(kerb2) + len(kerb3)
    run_errors = sum(f.startswith(("e2_task", "e3_task", "e5_")) for f in err_funcs)
    ok = run_errors == 0 and n_missing == 0 and validated > 0
    add("C1", 1.0 if ok else 0.0,
        f"{validated} KERB runs (E2, E3, E4, E7) validated after every update by an independent Hamming and "
        f"properness check; {run_errors} run errors; {n_missing} expected E2/E3 runs missing")
    over = 0
    if validated:
        both = pd.concat([kerb2, kerb3])
        over = int((both["max_recoloured"] > both["budget"]).sum())
    add("C2", 1.0 if ok and over == 0 else 0.0,
        f"max r_t <= R in all {validated} KERB runs ({over} exceed R); harness raises on any r_t > R")

    pairs = e2.get("pairs", pd.DataFrame())
    share_ff = share_tws = share_dsr = np.nan
    if not pairs.empty:
        def share(metric, other):
            sel = pairs[(pairs["metric"] == metric) & (pairs["other"] == other)]
            return float(sel["share_kerb_lower"].iloc[0]) if len(sel) else np.nan
        share_ff = share("mean_colours", "FF-Repair")
        share_tws = share("mean_recoloured", "Tabu-WarmStart")
        share_dsr = share("mean_recoloured", "DSATUR-Recompute")
    by_inst = e2.get("by_inst", pd.DataFrame())
    ratio = np.nan
    if not by_inst.empty:
        piv = by_inst.pivot(index="instance", columns="algorithm", values="mean_colours")
        if MAIN in piv and "Tabu-WarmStart" in piv:
            ratio = float(piv[MAIN].mean() / piv["Tabu-WarmStart"].mean())
    part1 = share_ff >= 0.8
    part2 = ratio <= 1.10
    add("C3", 1.0 if (part1 and part2) else (0.5 if part1 else 0.0),
        f"KERB-R3 below FF-Repair in {share_ff:.1%} of E2 runs; mean colours over instances "
        f"{ratio:.3f} x Tabu-WarmStart (rule: <= 1.10)")
    c4 = (share_tws >= 0.8) + (share_dsr >= 0.8)
    add("C4", {2: 1.0, 1: 0.5, 0: 0.0}[int(c4)],
        f"mean recolourings below Tabu-WarmStart in {share_tws:.1%} and below DSATUR-Recompute in "
        f"{share_dsr:.1%} of E2 runs")
    ins_ok = all(tc_pass.get(t) == "PASS" for t in ("T01", "T02"))
    del_ok = all(tc_pass.get(t) == "PASS" for t in ("T04", "T18"))
    add("C5", 1.0 if (ins_ok and del_ok) else (0.5 if (ins_ok or del_ok) else 0.0),
        f"insertion tests T01/T02 {'PASS' if ins_ok else 'not all PASS'}; deletion tests T04/T18 "
        f"{'PASS' if del_ok else 'not all PASS'}; E2 churn mixes both with probability 0.5")
    times = pooled_times("e2", C.E2_INSTANCES, C.E2_SEEDS, MAIN)
    med = float(np.median(times)) if times.size else np.nan
    p95 = float(np.percentile(times, 95)) if times.size else np.nan
    add("C6", 1.0 if (med <= 1000 and p95 <= 10000) else (0.5 if med <= 10000 else 0.0),
        f"pooled over {times.size} KERB-R3 E2 updates: median {med:.0f} us, p95 {p95:.0f} us")
    rgg = e5.get("rgg", pd.DataFrame())
    c7_score, c7_ev = 0.0, "E5 RGG runs not available"
    if not rgg.empty:
        k = rgg[rgg["algorithm"] == MAIN].set_index("n")
        if 10**6 in k.index and 10**3 in k.index:
            growth = float(k.loc[10**6, "median_time_us"] / k.loc[10**3, "median_time_us"])
            c7_score = 1.0 if growth < 3 else 0.5
            c7_ev = f"median update time x{growth:.2f} from n = 10^3 to 10^6; 10^6 run completed"
        else:
            c7_ev = f"RGG sizes completed: {sorted(k.index.tolist())}"
    add("C7", c7_score, c7_ev)
    det_ok = tc_pass.get("T16") == "PASS"
    package = all((ROOT / p).exists() for p in ("experiments/config.py", "experiments/run_experiments.py",
                                                 "experiments/analyse_results.py", "requirements.txt",
                                                 "README.md", "docs/preregistration.md"))
    add("C8", 1.0 if (det_ok and package) else (0.5 if (det_ok or package) else 0.0),
        f"determinism test T16 {'PASS' if det_ok else 'not PASS'}; scripts, configs and pre-registration in "
        f"the repository package {'complete' if package else 'incomplete'} (public URL inserted after upload)")
    r1 = pd.concat([kerb2[kerb2["algorithm"] == "KERB-R1"], kerb3[kerb3["algorithm"] == "KERB-R1"]])
    r1_expected = len(C.E2_INSTANCES) * len(C.E2_SEEDS) + len(C.E3_STREAMS) * len(C.E3_SEEDS)
    r1_ok = len(r1) == r1_expected and (r1["max_recoloured"] <= 1).all() and run_errors == 0
    add("C9", 1.0 if r1_ok else 0.0, f"{len(r1)} of {r1_expected} R = 1 runs completed, "
        f"max r_t = {int(r1['max_recoloured'].max()) if len(r1) else 'n/a'}")
    pilot = pd.read_csv(C.RESULTS / "pilot" / "pilot_summary.csv")
    spread = float(pilot["mean_colours"].max() / pilot["mean_colours"].min() - 1)
    add("C10", 1.0 if spread <= 0.05 else (0.5 if spread <= 0.10 else 0.0),
        f"{len(pilot)} pilot configurations; worst mean colours {100 * spread:.2f}% above the best")
    streams = set()
    if len(runs3):
        cnt = runs3.groupby("instance")["algorithm"].nunique()
        streams = set(cnt[cnt == len(C.BUDGET_SWEEP_TEMPORAL) + len(BASELINES)].index)
    add("C11", 1.0 if len(streams) >= 3 else (0.5 if streams else 0.0),
        f"{len(streams)} real temporal streams evaluated with every method (E3)")
    if pytest_res:
        failed = pytest_res["failures"] + pytest_res["errors"]
        passed = pytest_res["tests"] - failed - pytest_res["skipped"]
        frac = passed / max(1, pytest_res["tests"])
        add("C12", 1.0 if failed == 0 and pytest_res["tests"] > 0 else (0.5 if frac >= 0.9 else 0.0),
            f"pytest: {passed} of {pytest_res['tests']} tests passed")
    else:
        add("C12", 0.0, "pytest not run")
    df = pd.DataFrame(rows)
    total_w = sum(w for _, w, _, _ in C.COVERAGE_CRITERIA)
    df.loc[len(df)] = {"id": "Total", "weight": total_w, "criterion": "Coverage", "rule": "",
                       "evidence": f"target {C.COVERAGE_TARGET:.0%}", "score": df["weighted"].sum() / total_w,
                       "weighted": df["weighted"].sum()}
    _save(df, "coverage.csv")
    return df


# ===================================================================== main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-tests", action="store_true", help="skip running pytest (C12 then scores 0)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    analyse_e1()
    e2 = analyse_e2()
    d1 = deviation_d1_check()
    e3 = analyse_e3()
    e5 = analyse_e5()
    pytest_res = None if args.no_tests else run_pytest()
    cov = coverage(e2, e3, e5, pytest_res)
    print(cov[["id", "weight", "score", "evidence"]].to_string(index=False))
    if not d1.empty:
        print(f"D1 check: max n/min k_t = {d1['n_over_min_colours'].max():.1f} "
              f"(rule could apply in {int(d1['rule_could_apply'].sum())} runs)")
    if not args.no_figures:
        import plot_results
        plot_results.main()


if __name__ == "__main__":
    main()

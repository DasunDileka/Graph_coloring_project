"""Diagnostic measurement (not pre-registered, deviation D5): wasted fallback work.

The theoretical analysis (Chapter 10 of the report) found that the evaluated
implementation builds the input of the budgeted TabuCol fallback before it
checks whether the fallback is switched off.  With the pre-registered
``tabu_iters = 0`` the fallback never runs, but ``KERB._tabu_repair`` still
copies the colouring (Theta(n)) when the repair search runs out of nodes, and
``KERB._tabu_empty_class`` copies it and scans the class when a class
elimination fails with at least two recolourings left.  Neither call changes
the colouring, the random state or any statistic.

This script repeats every KERB run of E2 (including the budget sweep E4 and the
ablations E7), E3 and E5 whose fallback is switched off, with the same inputs,
parameters and seeds, unchanged code, and the same harness, and times every
call of those two methods.  KERB-tabu, the ablation with the fallback switched
on, is excluded, because there the fallback does real work.  The script checks
that each repeated run reproduces the stored run (colours, recolouring and
KERB's own counters) and reports, per run, how often the wasted work happened
and which share of KERB's update time it took.  The repeated runs are timed again, so their absolute times
differ slightly from the stored ones; only the shares are used.

Writes results/summary/fallback_overhead.csv and fallback_overhead.json.
Run: python3 experiments/fallback_overhead.py [--workers 3]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

import config as C  # noqa: E402
from benchmark_runner import make_algorithm, run_stream  # noqa: E402
from dynamic_graph import DynamicGraph  # noqa: E402
from graph_loader import read_dimacs, read_temporal  # noqa: E402
from run_experiments import e2_variants, e3_stream, e3_variants, rgg_edges  # noqa: E402
from static_coloring import dsatur  # noqa: E402
from workloads import random_churn, sliding_window  # noqa: E402

RAW = C.RESULTS / "raw"
OUT = C.RESULTS / "summary"
LABEL = f"KERB-R{C.MAIN_BUDGET}"
COMPARED = ("mean_colours", "max_colours", "final_colours", "total_recoloured", "max_recoloured",
            "conflict_rate", "new_colour_rate", "eliminations")


def instrument(alg):
    """Time every call of the two fallback methods; record the update index."""
    calls = []

    def wrap(name):
        method = getattr(alg, name)

        def timed(*args, **kwargs):
            t0 = time.perf_counter_ns()
            ok = method(*args, **kwargs)
            calls.append((alg.t, name, time.perf_counter_ns() - t0, bool(ok)))
            return ok
        setattr(alg, name, timed)

    wrap("_tabu_repair")
    wrap("_tabu_empty_class")
    return calls


def measure(alg, g, ops, budget, validate):
    calls = instrument(alg)
    summary, traces, _ = run_stream(alg, g, ops, budget=budget, validate=validate)
    time_us = traces["time_us"].astype(np.float64)
    waste_us = np.zeros_like(time_us)
    n_repair = n_empty = 0
    for t, name, ns, ok in calls:
        if ok:
            raise AssertionError("a fallback call succeeded although tabu_iters = 0")
        waste_us[t] += ns / 1000.0
        if name == "_tabu_repair":
            n_repair += 1
        else:
            n_empty += 1
    corrected = np.maximum(time_us - waste_us, 0.0)
    return summary, {
        "updates": int(time_us.size),
        "calls_repair": n_repair,
        "calls_empty_class": n_empty,
        "updates_with_waste": int((waste_us > 0).sum()),
        "waste_s": float(waste_us.sum() / 1e6),
        "total_s": float(time_us.sum() / 1e6),
        "waste_share": float(waste_us.sum() / time_us.sum()) if time_us.sum() else 0.0,
        "mean_us": float(time_us.mean()),
        "median_us": float(np.median(time_us)),
        "p95_us": float(np.percentile(time_us, 95)),
        "mean_us_without_waste": float(corrected.mean()),
        "median_us_without_waste": float(np.median(corrected)),
        "p95_us_without_waste": float(np.percentile(corrected, 95)),
        "mean_waste_per_call_us": float(waste_us.sum() / max(1, n_repair + n_empty)),
    }, alg.stats


def check(stored, summary, stats):
    """Differences between the stored run and the repeated run (should be none)."""
    diffs = [k for k in COMPARED if k in stored and abs(stored[k] - summary[k]) > 1e-9]
    for k, v in stored.get("kerb_stats", {}).items():
        if stats.get(k) != v:
            diffs.append(f"kerb_stats.{k}")
    return diffs


def e2_task(task):
    name, seed, label, budget, params = task
    inst = read_dimacs(C.DATA / "dimacs" / f"{name}.col")
    initial_edges, ops = random_churn(inst.edges, rho=C.E2_RHO, updates=C.E2_UPDATES, seed=seed,
                                      p_insert=C.E2_P_INSERT)
    with open(RAW / "e2" / "initial" / f"{name}__s{seed}.json") as fh:
        initial = json.load(fh)["colouring"]
    g = DynamicGraph(inst.n, initial_edges)
    kerb_params = dict(C.KERB_PARAMS)
    kerb_params.update(params or {})
    alg = make_algorithm(label, g, initial, seed, budget=budget, kerb_params=kerb_params)
    summary, row, stats = measure(alg, g, ops, budget, "local")
    with open(RAW / "e2" / "runs" / f"{name}__{label}__s{seed}.json") as fh:
        stored = json.load(fh)
    return {"workload": "E2", "algorithm": label, "instance": name, "seed": seed, "n": inst.n,
            "differences": check(stored, summary, stats), **row}


def e3_task(task):
    name, seed, label, budget = task
    inst, ops = e3_stream(name, seed)
    g = DynamicGraph(inst.n)
    alg = make_algorithm(label, g, None, seed, budget=budget, kerb_params=C.KERB_PARAMS)
    summary, row, stats = measure(alg, g, ops, budget, "local")
    with open(RAW / "e3" / "runs" / f"{name}__{label}__s{seed}.json") as fh:
        stored = json.load(fh)
    return {"workload": "E3", "algorithm": label, "instance": name, "seed": seed, "n": inst.n,
            "differences": check(stored, summary, stats), **row}


def e5_rgg_task(n):
    edges = rgg_edges(n, C.E5_RGG_MEAN_DEGREE, C.E5_SEED)
    initial_edges, ops = random_churn(edges, rho=C.E2_RHO, updates=C.E5_RGG_UPDATES, seed=C.E5_SEED)
    del edges
    g = DynamicGraph(n, initial_edges)
    initial = dsatur(g.adj)
    alg = make_algorithm(LABEL, g, initial, C.E5_SEED, budget=C.MAIN_BUDGET, kerb_params=C.KERB_PARAMS)
    summary, row, stats = measure(alg, g, ops, C.MAIN_BUDGET, "sample")
    with open(RAW / "e5" / f"rgg_{n}__{LABEL}.json") as fh:
        stored = json.load(fh)
    return {"workload": "E5 RGG", "algorithm": LABEL, "instance": f"rgg_{n}", "seed": C.E5_SEED, "n": n,
            "differences": check(stored, summary, stats), **row}


def e5_real_task(name):
    fn, window = C.E5_REAL[name]
    inst = read_temporal(C.DATA / "temporal" / fn, name)
    ops = [(op, u, v) for op, u, v, _ in sliding_window(inst.events, window)]
    del inst.events[:]
    g = DynamicGraph(inst.n)
    alg = make_algorithm(LABEL, g, None, C.E5_SEED, budget=C.MAIN_BUDGET, kerb_params=C.KERB_PARAMS)
    summary, row, stats = measure(alg, g, ops, C.MAIN_BUDGET, "sample")
    with open(RAW / "e5" / f"{name}__{LABEL}.json") as fh:
        stored = json.load(fh)
    return {"workload": "E5 real", "algorithm": LABEL, "instance": name, "seed": C.E5_SEED, "n": inst.n,
            "differences": check(stored, summary, stats), **row}


def run(task):
    kind, arg = task
    t0 = time.perf_counter()
    if kind == "e2":
        row = e2_task(arg)
    elif kind == "e3":
        row = e3_task(arg)
    elif kind == "rgg":
        row = e5_rgg_task(arg)
    else:
        row = e5_real_task(arg)
    row["wall_s"] = time.perf_counter() - t0
    print(f"done {kind} {arg[:3] if kind in ('e2', 'e3') else arg}: "
          f"{row['calls_repair'] + row['calls_empty_class']} calls, "
          f"share {row['waste_share']:.4f}, differences {row['differences']}", flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    assert C.KERB_PARAMS["tabu_iters"] == 0
    kerb2 = [(label, budget, params) for label, budget, params in e2_variants()
             if label.startswith("KERB") and (params or {}).get("tabu_iters", 0) == 0]
    kerb3 = [(label, budget) for label, budget in e3_variants() if label.startswith("KERB")]
    tasks = ([("real", name) for name in C.E5_REAL] + [("rgg", n) for n in reversed(C.E5_RGG_SIZES)]
             + [("e3", (name, seed, label, budget)) for name in C.E3_STREAMS for seed in C.E3_SEEDS
                for label, budget in kerb3]
             + [("e2", (name, seed, label, budget, params)) for name in C.E2_INSTANCES for seed in C.E2_SEEDS
                for label, budget, params in kerb2])
    with Pool(args.workers, maxtasksperchild=1) as pool:
        rows = pool.map(run, tasks, chunksize=1)
    fields = ["workload", "algorithm", "instance", "seed", "n", "updates", "calls_repair", "calls_empty_class",
              "updates_with_waste", "waste_s", "total_s", "waste_share", "mean_us", "median_us", "p95_us",
              "mean_us_without_waste", "median_us_without_waste", "p95_us_without_waste",
              "mean_waste_per_call_us", "wall_s",
              "differences"]
    with open(OUT / "fallback_overhead.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "differences": ";".join(r["differences"])})
    groups = {}
    for r in rows:
        groups.setdefault(f"{r['workload']} {r['algorithm']}", []).append(r)
    agg = {
        "groups": {key: {"runs": len(g), "calls": sum(r["calls_repair"] + r["calls_empty_class"] for r in g),
                         "updates_with_waste": sum(r["updates_with_waste"] for r in g),
                         "updates": sum(r["updates"] for r in g),
                         "waste_share": sum(r["waste_s"] for r in g) / sum(r["total_s"] for r in g)}
                   for key, g in groups.items()},
        "runs": rows,
        "all_reproduced": all(not r["differences"] for r in rows),
    }
    with open(OUT / "fallback_overhead.json", "w") as fh:
        json.dump(agg, fh, indent=1)
    print("all runs reproduced the stored results:", agg["all_reproduced"])


if __name__ == "__main__":
    main()

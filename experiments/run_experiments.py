"""Runs the pre-registered experiments (docs/preregistration.md).

    python experiments/run_experiments.py e1 [--workers 4]   static DIMACS comparison
    python experiments/run_experiments.py e2 [--workers 4]   DIMACS churn: E2 main, E4 budget sweep, E7 ablations
    python experiments/run_experiments.py e3 [--workers 4]   real temporal streams (sliding window)
    python experiments/run_experiments.py e5 [--workers 1]   scalability (large real streams, RGG up to 10^6)

Every task writes its own JSON summary (and an .npz trace) under
results/raw/<experiment>/, so an interrupted run can be resumed: tasks whose
JSON already exists are skipped.  experiments/analyse_results.py merges the
JSON files into the CSV tables in results/summary/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402

import config as C  # noqa: E402
from benchmark_runner import (dsatur_checkpoint, make_algorithm, run_dsatur_recompute_sampled,  # noqa: E402
                              run_stream)
from dynamic_graph import DynamicGraph  # noqa: E402
from graph_features import graph_features  # noqa: E402
from graph_loader import read_dimacs, read_temporal  # noqa: E402
from static_coloring import (STATIC_ALGORITHMS, dsatur, greedy_clique_lower_bound,  # noqa: E402
                             random_order_greedy, tabucol_minimise)
from validation import is_proper  # noqa: E402
from workloads import random_churn, relabel_vertices, sliding_window  # noqa: E402

RAW = C.RESULTS / "raw"


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, path)


def _environment() -> dict:
    import networkx
    import scipy
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(), "cpus": os.cpu_count(),
            "numpy": np.__version__, "scipy": scipy.__version__, "networkx": networkx.__version__}


def _call(job):
    """Run one task; record a failure instead of stopping the whole experiment."""
    fn_name, task = job
    try:
        return globals()[fn_name](task)
    except Exception as exc:  # reported by the analysis step
        err = {"function": fn_name, "task": repr(task), "error": repr(exc),
               "traceback": traceback.format_exc()}
        _write_json(RAW / "errors" / f"{fn_name}_{abs(hash(repr(task)))}.json", err)
        return None


def _pool_map(fn_name: str, tasks, workers: int):
    jobs = [(fn_name, t) for t in tasks]
    if workers <= 1:
        return [_call(j) for j in jobs]
    with Pool(workers) as pool:
        return pool.map(_call, jobs, chunksize=1)


# ===================================================================== E1
def e1_task(name: str):
    out_path = RAW / "e1" / f"{name}.json"
    if out_path.exists():
        return None
    inst = read_dimacs(C.DATA / "dimacs" / f"{name}.col")
    g = DynamicGraph(inst.n, inst.edges)
    adj = g.adj
    row = {"instance": name, "sha256": inst.sha256, **graph_features(adj)}
    results = {}
    for alg_name, fn in STATIC_ALGORITHMS.items():
        t0 = time.perf_counter()
        col = fn(adj)
        dt = time.perf_counter() - t0
        assert is_proper(adj, col), alg_name
        results[alg_name] = {"colours": len(set(col)), "time_s": dt}
    ks, ts = [], []
    for s in C.E1_RANDOM_GREEDY_SEEDS:
        t0 = time.perf_counter()
        col = random_order_greedy(adj, seed=s)
        ts.append(time.perf_counter() - t0)
        assert is_proper(adj, col)
        ks.append(len(set(col)))
    results["Greedy (random order)"] = {"colours": float(np.mean(ks)), "colours_min": int(min(ks)),
                                        "colours_max": int(max(ks)), "time_s": float(np.mean(ts))}
    t0 = time.perf_counter()
    start = dsatur(adj)
    col, used = tabucol_minimise(adj, start, C.E1_TABUCOL_ITERS, seed=0)
    dt = time.perf_counter() - t0
    assert is_proper(adj, col)
    results["TabuCol-minimise"] = {"colours": len(set(col)), "time_s": dt, "iterations": used}
    row["algorithms"] = results
    _write_json(out_path, row)
    return name


# ============================================================ E2 / E4 / E7
def e2_initial(task):
    """Common start for every method: DSATUR improved offline by TabuCol."""
    name, seed = task
    path = RAW / "e2" / "initial" / f"{name}__s{seed}.json"
    if path.exists():
        return None
    inst = read_dimacs(C.DATA / "dimacs" / f"{name}.col")
    initial_edges, _ = random_churn(inst.edges, rho=C.E2_RHO, updates=C.E2_UPDATES, seed=seed,
                                    p_insert=C.E2_P_INSERT)
    g = DynamicGraph(inst.n, initial_edges)
    t0 = time.perf_counter()
    d = dsatur(g.adj)
    col, used = tabucol_minimise(g.adj, d, C.E2_INITIAL_TABU_ITERS, seed=seed)
    dt = time.perf_counter() - t0
    assert is_proper(g.adj, col)
    _write_json(path, {"instance": name, "seed": seed, "dsatur_colours": len(set(d)),
                       "initial_colours": len(set(col)), "tabucol_iterations": used, "time_s": dt,
                       "m0": g.m, "colouring": col})
    return path


def e2_variants():
    """(algorithm label, budget, extra KERB params) for E2, E4 and E7."""
    out = []
    for R in C.BUDGET_SWEEP:
        out.append((f"KERB-R{R}", R, {}))
    for label, params in C.ABLATIONS.items():
        out.append((label, C.MAIN_BUDGET, params))
    out += [("FF-Repair", None, None), ("Tabu-WarmStart", None, None), ("DSATUR-Recompute", None, None)]
    return out


def e2_task(task):
    name, seed, label, budget, params = task
    out_path = RAW / "e2" / "runs" / f"{name}__{label}__s{seed}.json"
    if out_path.exists():
        return None
    inst = read_dimacs(C.DATA / "dimacs" / f"{name}.col")
    initial_edges, ops = random_churn(inst.edges, rho=C.E2_RHO, updates=C.E2_UPDATES, seed=seed,
                                      p_insert=C.E2_P_INSERT)
    with open(RAW / "e2" / "initial" / f"{name}__s{seed}.json") as fh:
        start = json.load(fh)
    initial = start["colouring"]
    row = {"experiment": "E2", "instance": name, "seed": seed, "algorithm": label, "budget": budget,
           "n": inst.n, "m0": len(initial_edges), "initial_colours": start["initial_colours"]}
    t0 = time.perf_counter()
    if label == "DSATUR-Recompute":
        summary, traces = run_dsatur_recompute_sampled(inst.n, initial_edges, ops,
                                                       C.DSR_SAMPLE_EVERY["churn"])
        checkpoints = []
    else:
        g = DynamicGraph(inst.n, initial_edges)
        kerb_params = dict(C.KERB_PARAMS)
        if params:
            kerb_params.update(params)
        alg = make_algorithm(label, g, initial, seed, budget=budget, kerb_params=kerb_params,
                             tws_params=C.TABU_WARMSTART)
        cp_fn = dsatur_checkpoint if label == f"KERB-R{C.MAIN_BUDGET}" else None
        summary, traces, checkpoints = run_stream(alg, g, ops, budget=budget, validate="local",
                                                  checkpoint_every=C.E2_CHECKPOINT_EVERY,
                                                  checkpoint_fn=cp_fn)
        if hasattr(alg, "stats"):
            row["kerb_stats"] = alg.stats
        if hasattr(alg, "tabu_iterations"):
            row["tabu_iterations"] = alg.tabu_iterations
    row.update(summary)
    row["wall_s"] = time.perf_counter() - t0
    row["checkpoints"] = checkpoints
    np.savez_compressed(RAW / "e2" / "traces" / f"{name}__{label}__s{seed}.npz", **traces)
    _write_json(out_path, row)
    return out_path


# ===================================================================== E3
def e3_stream(name: str, seed: int):
    fn, window = C.E3_STREAMS[name]
    inst = read_temporal(C.DATA / "temporal" / fn, name)
    ops = [(op, u, v) for op, u, v, _ in sliding_window(inst.events, window)]
    _, ops, _ = relabel_vertices(inst.n, [], ops, seed)
    return inst, ops


def e3_variants():
    out = [(f"KERB-R{R}", R) for R in C.BUDGET_SWEEP_TEMPORAL]
    return out + [("FF-Repair", None), ("Tabu-WarmStart", None), ("DSATUR-Recompute", None)]


def e3_task(task):
    name, seed, label, budget = task
    out_path = RAW / "e3" / "runs" / f"{name}__{label}__s{seed}.json"
    if out_path.exists():
        return None
    inst, ops = e3_stream(name, seed)
    row = {"experiment": "E3", "instance": name, "seed": seed, "algorithm": label, "budget": budget,
           "n": inst.n, "window_s": C.E3_STREAMS[name][1], "updates_total": len(ops)}
    t0 = time.perf_counter()
    if label == "DSATUR-Recompute":
        summary, traces = run_dsatur_recompute_sampled(inst.n, [], ops, C.DSR_SAMPLE_EVERY["temporal"])
        checkpoints = []
    else:
        g = DynamicGraph(inst.n)
        alg = make_algorithm(label, g, None, seed, budget=budget, kerb_params=C.KERB_PARAMS,
                             tws_params=C.TABU_WARMSTART)
        cp_fn = dsatur_checkpoint if label == f"KERB-R{C.MAIN_BUDGET}" else None
        summary, traces, checkpoints = run_stream(alg, g, ops, budget=budget, validate="local",
                                                  checkpoint_every=max(1, len(ops) // 10),
                                                  checkpoint_fn=cp_fn)
        if hasattr(alg, "stats"):
            row["kerb_stats"] = alg.stats
    row.update(summary)
    row["wall_s"] = time.perf_counter() - t0
    row["checkpoints"] = checkpoints
    np.savez_compressed(RAW / "e3" / "traces" / f"{name}__{label}__s{seed}.npz", **traces)
    _write_json(out_path, row)
    return out_path


# ===================================================================== E5
def _rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2**20


def rgg_edges(n: int, mean_degree: float, seed: int):
    """Random geometric graph in the unit square, built with a cell grid (O(n + m))."""
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2))
    r = math.sqrt(mean_degree / (math.pi * (n - 1)))
    cells = max(1, int(1.0 / r))
    cell_of = np.minimum((pts * cells).astype(np.int64), cells - 1)
    buckets = {}
    for i, (cx, cy) in enumerate(cell_of):
        buckets.setdefault((int(cx), int(cy)), []).append(i)
    edges = []
    r2 = r * r
    for (cx, cy), members in buckets.items():
        cand = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand.extend(buckets.get((cx + dx, cy + dy), ()))
        cand_arr = np.asarray(cand)
        cp = pts[cand_arr]
        for i in members:
            d2 = ((cp - pts[i]) ** 2).sum(axis=1)
            for j in cand_arr[d2 <= r2]:
                if i < j:
                    edges.append((i, int(j)))
    return edges


def e5_rgg_task(task):
    n, label = task
    out_path = RAW / "e5" / f"rgg_{n}__{label}.json"
    if out_path.exists():
        return None
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    edges = rgg_edges(n, C.E5_RGG_MEAN_DEGREE, C.E5_SEED)
    t_gen = time.perf_counter() - t0
    initial_edges, ops = random_churn(edges, rho=C.E2_RHO, updates=C.E5_RGG_UPDATES, seed=C.E5_SEED)
    del edges
    t0 = time.perf_counter()
    g = DynamicGraph(n, initial_edges)
    t_build = time.perf_counter() - t0
    rss_graph = _rss_mb()
    t0 = time.perf_counter()
    initial = dsatur(g.adj)
    t_dsatur = time.perf_counter() - t0
    budget = C.MAIN_BUDGET if label.startswith("KERB") else None
    alg = make_algorithm(label, g, initial, C.E5_SEED, budget=budget, kerb_params=C.KERB_PARAMS,
                         tws_params=C.TABU_WARMSTART)
    summary, traces, _ = run_stream(alg, g, ops, budget=budget, validate="sample")
    row = {"experiment": "E5", "workload": "RGG churn (synthetic)", "n": n, "algorithm": label,
           "m0": len(initial_edges), "mean_degree_target": C.E5_RGG_MEAN_DEGREE,
           "generate_s": t_gen, "build_s": t_build, "initial_dsatur_s": t_dsatur,
           "initial_colours": len(set(initial)), "rss_start_mb": rss0, "rss_graph_mb": rss_graph,
           "rss_end_mb": _rss_mb(), **summary}
    if hasattr(alg, "stats"):
        row["kerb_stats"] = alg.stats
    np.savez_compressed(RAW / "e5" / f"rgg_{n}__{label}.npz", **traces)
    _write_json(out_path, row)
    return out_path


def e5_real_task(task):
    name, label = task
    out_path = RAW / "e5" / f"{name}__{label}.json"
    if out_path.exists():
        return None
    fn, window = C.E5_REAL[name]
    rss0 = _rss_mb()
    inst = read_temporal(C.DATA / "temporal" / fn, name)
    ops = [(op, u, v) for op, u, v, _ in sliding_window(inst.events, window)]
    del inst.events[:]
    row = {"experiment": "E5", "workload": f"{name} sliding window", "n": inst.n, "algorithm": label,
           "window_s": window, "updates_total": len(ops), "rss_start_mb": rss0}
    if label == "DSATUR-Recompute":
        step = max(1, len(ops) // C.E5_DSR_SAMPLES)
        summary, traces = run_dsatur_recompute_sampled(inst.n, [], ops, step, C.E5_DSR_SAMPLES)
    else:
        if label == "Tabu-WarmStart":
            ops = ops[: C.E5_TWS_PREFIX]
        g = DynamicGraph(inst.n)
        budget = C.MAIN_BUDGET if label.startswith("KERB") else None
        alg = make_algorithm(label, g, None, C.E5_SEED, budget=budget, kerb_params=C.KERB_PARAMS,
                             tws_params=C.TABU_WARMSTART)
        summary, traces, _ = run_stream(alg, g, ops, budget=budget, validate="sample",
                                        checkpoint_every=max(1, len(ops) // 10))
        if hasattr(alg, "stats"):
            row["kerb_stats"] = alg.stats
    row.update(updates_run=len(ops), rss_end_mb=_rss_mb(), **summary)
    np.savez_compressed(RAW / "e5" / f"{name}__{label}.npz", **traces)
    _write_json(out_path, row)
    return out_path


# ================================================================== driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["e1", "e2", "e3", "e5"])
    ap.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    args = ap.parse_args()
    for sub in ("e1", "e2/initial", "e2/runs", "e2/traces", "e3/runs", "e3/traces", "e5", "errors"):
        (RAW / sub).mkdir(parents=True, exist_ok=True)
    _write_json(RAW / f"environment_{args.experiment}.json", _environment())
    t0 = time.time()
    if args.experiment == "e1":
        _pool_map("e1_task", sorted(C.E1_INSTANCES, key=lambda x: -os.path.getsize(
            C.DATA / "dimacs" / f"{x}.col")), args.workers)
    elif args.experiment == "e2":
        starts = [(name, s) for name in C.E2_INSTANCES for s in C.E2_SEEDS]
        _pool_map("e2_initial", starts, args.workers)
        tasks = [(name, s, label, budget, params) for name in C.E2_INSTANCES for s in C.E2_SEEDS
                 for label, budget, params in e2_variants()]
        random.Random(0).shuffle(tasks)  # mix long and short tasks across workers
        _pool_map("e2_task", tasks, args.workers)
    elif args.experiment == "e3":
        tasks = [(name, s, label, budget) for name in C.E3_STREAMS for s in C.E3_SEEDS
                 for label, budget in e3_variants()]
        random.Random(0).shuffle(tasks)
        _pool_map("e3_task", tasks, args.workers)
    elif args.experiment == "e5":
        real = [(name, label) for name in C.E5_REAL
                for label in (f"KERB-R{C.MAIN_BUDGET}", "FF-Repair", "Tabu-WarmStart", "DSATUR-Recompute")]
        rgg = [(n, label) for n in C.E5_RGG_SIZES for label in (f"KERB-R{C.MAIN_BUDGET}", "FF-Repair")]
        # one task at a time by default so that timings and memory are not disturbed
        _pool_map("e5_rgg_task", rgg, args.workers)
        _pool_map("e5_real_task", real, args.workers)
    print(f"{args.experiment} finished in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

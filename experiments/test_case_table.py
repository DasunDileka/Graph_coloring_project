"""E6: executes the test cases listed in the report and records the actual results.

Each case states its purpose, input and expected behaviour; the script runs it
and writes what actually happened to results/summary/test_cases.csv.  The
same behaviours are also asserted by the pytest suites in tests/.
"""

from __future__ import annotations

import csv
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import networkx as nx  # noqa: E402

from baseline_algorithms import FirstFitRepair, TabuWarmStart  # noqa: E402
from dynamic_graph import DynamicGraph, GraphError  # noqa: E402
from graph_loader import InputFormatError, read_dimacs  # noqa: E402
from proposed_algorithm import DELETE, INSERT, KERB, KerbConfig  # noqa: E402
from static_coloring import dsatur  # noqa: E402
from validation import is_proper, recoloured  # noqa: E402
from workloads import random_churn, sliding_window  # noqa: E402

CASES = []


def case(cid, category, purpose, inp, expected):
    def deco(fn):
        CASES.append((cid, category, purpose, inp, expected, fn))
        return fn
    return deco


def replay(alg, g, ops, budget):
    worst_r, bad, k0, maxdeg, bound_ok = 0, 0, alg.num_colours, g.max_degree(), True
    for op in ops:
        before = list(alg.colouring)
        res = alg.apply(*op)
        maxdeg = max(maxdeg, g.max_degree())
        r = recoloured(before, alg.colouring)
        worst_r = max(worst_r, r)
        bad += (not is_proper(g.adj, alg.colouring)) or r != res.recoloured
        bound_ok &= res.colours <= max(k0, maxdeg + 1)
    return worst_r, bad, bound_ok


@case("T01", "Normal", "Conflict repaired by a direct move",
      "Path 0-1-2 plus vertex 3; colours [0,1,0,2]; insert {0,2}", "Proper; exactly 1 vertex recoloured; still 3 colours")
def t01():
    g = DynamicGraph(4, [(0, 1), (1, 2)])
    a = KERB(g, KerbConfig(budget=3, use_elimination=False, use_migration=False), initial_colouring=[0, 1, 0, 2])
    r = a.insert_edge(0, 2)
    return f"proper={is_proper(g.adj, a.colouring)}, r={r.recoloured}, k={r.colours}, repair={r.repair}", \
        is_proper(g.adj, a.colouring) and r.recoloured == 1 and r.colours == 3 and r.repair == "direct"


@case("T02", "Normal", "Ejection chain avoids a new colour",
      "6 vertices; u, v share colour 0 and each sees colours 1, 2; R=2", "Repair by a 2-vertex chain; still 3 colours")
def t02():
    g = DynamicGraph(6, [(0, 2), (0, 3), (1, 4), (1, 5)])
    a = KERB(g, KerbConfig(budget=2, use_elimination=False, use_migration=False), initial_colouring=[0, 0, 1, 2, 1, 2])
    r = a.insert_edge(0, 1)
    return f"repair={r.repair}, r={r.recoloured}, k={r.colours}", r.repair == "chain" and r.recoloured == 2 and r.colours == 3


@case("T03", "Boundary", "Same conflict with R=1 needs a new colour", "As T02 with R=1",
      "New colour; r=1; 4 colours")
def t03():
    g = DynamicGraph(6, [(0, 2), (0, 3), (1, 4), (1, 5)])
    a = KERB(g, KerbConfig(budget=1, use_elimination=False, use_migration=False), initial_colouring=[0, 0, 1, 2, 1, 2])
    r = a.insert_edge(0, 1)
    return f"repair={r.repair}, r={r.recoloured}, k={r.colours}", r.repair == "new_colour" and r.recoloured == 1 and r.colours == 4


@case("T04", "Normal", "Spare budget removes a small class after a deletion",
      "Path 0-1-2-3-4 coloured [0,1,2,0,1]; delete {3,4}; R=2", "Class {2} eliminated; 2 colours; r<=2")
def t04():
    g = DynamicGraph(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    a = KERB(g, KerbConfig(budget=2, use_migration=False), initial_colouring=[0, 1, 2, 0, 1])
    r = a.delete_edge(3, 4)
    return f"k={r.colours}, eliminated={r.eliminated}, r={r.recoloured}", r.colours == 2 and r.recoloured <= 2


@case("T05", "Edge", "Graph with no vertices", "n=0", "Object created; 0 colours; no error")
def t05():
    a = KERB(DynamicGraph(0), KerbConfig())
    return f"colours={a.num_colours}", a.num_colours == 0


@case("T06", "Edge", "Isolated vertices only", "n=5, no edges", "1 colour")
def t06():
    a = KERB(DynamicGraph(5), KerbConfig())
    return f"colours={a.num_colours}", a.num_colours == 1


@case("T07", "Edge", "Insert an edge that already exists", "Edge {0,1} present; insert {0,1}",
      "applied=False; nothing recoloured")
def t07():
    g = DynamicGraph(2, [(0, 1)])
    a = KERB(g, KerbConfig())
    r = a.insert_edge(0, 1)
    return f"applied={r.applied}, r={r.recoloured}", (not r.applied) and r.recoloured == 0


@case("T08", "Edge", "Delete an edge that is absent", "No edges; delete {0,1}", "applied=False; nothing recoloured")
def t08():
    a = KERB(DynamicGraph(2), KerbConfig())
    r = a.delete_edge(0, 1)
    return f"applied={r.applied}, r={r.recoloured}", (not r.applied) and r.recoloured == 0


@case("T09", "Invalid", "Self-loop rejected", "insert {1,1}", "GraphError raised; graph unchanged")
def t09():
    g = DynamicGraph(3)
    a = KERB(g, KerbConfig())
    try:
        a.insert_edge(1, 1)
    except GraphError:
        return "GraphError raised", g.m == 0
    return "no error", False


@case("T10", "Invalid", "Vertex id out of range", "n=3; insert {0,7}", "GraphError raised")
def t10():
    try:
        KERB(DynamicGraph(3), KerbConfig()).insert_edge(0, 7)
    except GraphError:
        return "GraphError raised", True
    return "no error", False


@case("T11", "Invalid", "Budget R=0 rejected", "KerbConfig(budget=0)", "ValueError raised")
def t11():
    try:
        KERB(DynamicGraph(3), KerbConfig(budget=0))
    except ValueError:
        return "ValueError raised", True
    return "no error", False


@case("T12", "Invalid", "Malformed DIMACS file", "edge line before the 'p' line", "InputFormatError raised")
def t12():
    tmp = ROOT / "results" / "summary" / "_bad.col"
    tmp.write_text("e 1 2\n")
    try:
        read_dimacs(tmp)
    except InputFormatError:
        return "InputFormatError raised", True
    finally:
        tmp.unlink()
    return "no error", False


@case("T13", "Boundary", "Complete graph built edge by edge", "K9, 36 insertions in random order, R=3",
      "Always proper; r<=3; ends with 9 colours")
def t13():
    g = DynamicGraph(9)
    a = KERB(g, KerbConfig(budget=3))
    edges = [(u, v) for u in range(9) for v in range(u + 1, 9)]
    random.Random(1).shuffle(edges)
    worst, bad, bound = replay(a, g, [(INSERT, u, v) for u, v in edges], 3)
    return f"invalid={bad}, max r={worst}, k={a.num_colours}", bad == 0 and worst <= 3 and a.num_colours == 9


@case("T14", "Boundary", "R=1 on a dense random graph", "G(60,0.5), 500 churn updates, R=1",
      "Always proper; r<=1 every update")
def t14():
    G = nx.gnp_random_graph(60, 0.5, seed=3)
    init, ops = random_churn(list(G.edges()), rho=0.3, updates=500, seed=3)
    g = DynamicGraph(60, init)
    worst, bad, bound = replay(KERB(g, KerbConfig(budget=1)), g, ops, 1)
    return f"invalid={bad}, max r={worst}", bad == 0 and worst <= 1


@case("T15", "Normal", "Budget and colour bound under random churn", "G(80,0.2), 1000 updates, R=3",
      "Proper; r<=3; k<=max(k0, maxdeg+1) after every update")
def t15():
    G = nx.gnp_random_graph(80, 0.2, seed=5)
    init, ops = random_churn(list(G.edges()), rho=0.3, updates=1000, seed=5)
    g = DynamicGraph(80, init)
    worst, bad, bound = replay(KERB(g, KerbConfig(budget=3)), g, ops, 3)
    return f"invalid={bad}, max r={worst}, bound held={bound}", bad == 0 and worst <= 3 and bound


@case("T16", "Normal", "Deterministic for a fixed seed", "Same stream and seed run twice",
      "Identical colourings and colour counts")
def t16():
    G = nx.gnp_random_graph(50, 0.2, seed=8)
    init, ops = random_churn(list(G.edges()), rho=0.3, updates=400, seed=8)
    out = []
    for _ in range(2):
        g = DynamicGraph(50, init)
        a = KERB(g, KerbConfig(budget=3, seed=4))
        out.append(([a.apply(*op).colours for op in ops], list(a.colouring)))
    return f"identical={out[0] == out[1]}", out[0] == out[1]


@case("T17", "Normal", "Insert then delete the same edge", "Path 0-1-2; insert/delete {0,2} twice",
      "Every step proper and within budget")
def t17():
    g = DynamicGraph(3, [(0, 1), (1, 2)])
    worst, bad, _ = replay(KERB(g, KerbConfig(budget=2)), g,
                           [(INSERT, 0, 2), (DELETE, 0, 2), (INSERT, 0, 2), (DELETE, 0, 2)], 2)
    return f"invalid={bad}, max r={worst}", bad == 0 and worst <= 2


@case("T18", "Normal", "Deleting every edge lets classes disappear", "Cycle C7 (3 colours); delete all 7 edges",
      "Colour count falls below 3")
def t18():
    G = nx.cycle_graph(7)
    g = DynamicGraph(7, G.edges())
    a = KERB(g, KerbConfig(budget=3))
    for u, v in list(G.edges()):
        a.delete_edge(u, v)
    return f"k={a.num_colours}", a.num_colours < 3


@case("T19", "Edge", "Sliding window with equal times and expiry", "Events (0,1,0),(1,2,5),(0,1,8),(2,2,9),(2,3,30); W=10",
      "Self-interaction ignored; {1,2} expires at 15, {0,1} at 18")
def t19():
    ops = sliding_window([(0, 1, 0.0), (1, 2, 5.0), (0, 1, 8.0), (2, 2, 9.0), (2, 3, 30.0)], 10.0)
    want = [("insert", 0, 1, 0.0), ("insert", 1, 2, 5.0), ("delete", 1, 2, 15.0), ("delete", 0, 1, 18.0),
            ("insert", 2, 3, 30.0)]
    return f"{len(ops)} operations, as expected={ops == want}", ops == want


@case("T20", "Baseline", "FF-Repair recolours at most one vertex", "G(60,0.3), 1000 churn updates",
      "Proper; r<=1")
def t20():
    G = nx.gnp_random_graph(60, 0.3, seed=9)
    init, ops = random_churn(list(G.edges()), rho=0.3, updates=1000, seed=9)
    g = DynamicGraph(60, init)
    worst, bad, _ = replay(FirstFitRepair(g), g, ops, None)
    return f"invalid={bad}, max r={worst}", bad == 0 and worst <= 1


@case("T21", "Baseline", "Tabu-WarmStart has no recolouring cap", "G(60,0.5), 500 churn updates",
      "Always proper; r may exceed 3")
def t21():
    G = nx.gnp_random_graph(60, 0.5, seed=10)
    init, ops = random_churn(list(G.edges()), rho=0.3, updates=500, seed=10)
    g = DynamicGraph(60, init)
    worst, bad, _ = replay(TabuWarmStart(g), g, ops, None)
    return f"invalid={bad}, max r={worst}", bad == 0


@case("T22", "Benchmark", "Real benchmark loads and DSATUR colours it", "DIMACS le450_15c",
      "n=450, 16,680 edges; proper colouring")
def t22():
    inst = read_dimacs(ROOT / "datasets" / "dimacs" / "le450_15c.col")
    g = DynamicGraph(inst.n, inst.edges)
    col = dsatur(g.adj)
    return f"n={inst.n}, m={len(inst.edges)}, DSATUR k={len(set(col))}, proper={is_proper(g.adj, col)}", \
        inst.n == 450 and len(inst.edges) == 16680 and is_proper(g.adj, col)


@case("T23", "Large", "Large sparse graph runs within budget", "Sparse random graph G(n, p) with n=20000, p=8/n (seed 11), 2000 churn updates, R=3",
      "Proper; r<=3; mean update time recorded")
def t23():
    G = nx.fast_gnp_random_graph(20000, 8 / 20000, seed=11)
    init, ops = random_churn(list(G.edges()), rho=0.2, updates=2000, seed=11)
    g = DynamicGraph(20000, init)
    a = KERB(g, KerbConfig(budget=3))
    t0 = time.perf_counter()
    worst = 0
    for op in ops:
        worst = max(worst, a.apply(*op).recoloured)
    dt = (time.perf_counter() - t0) / len(ops) * 1e6
    ok = is_proper(g.adj, a.colouring) and worst <= 3
    return f"proper={is_proper(g.adj, a.colouring)}, max r={worst}, mean {dt:.0f} us/update", ok



@case("T24", "Normal", "Planned migration removes a class over several updates",
      "10 vertices; class {7,8,9} free to join colour 1; R=1; six updates that change no edge",
      "C4 plans the move; one vertex per update; 3 colours become 2; without C4 still 3")
def t24():
    def build():
        g = DynamicGraph(10, [(7, 0), (8, 1), (9, 2), (0, 3), (1, 4), (2, 5)])
        return g, [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    g, init = build()
    a = KERB(g, KerbConfig(budget=1, plan_every=1), initial_colouring=init)
    worst, bad, trace = 0, 0, []
    for _ in range(6):
        before = list(a.colouring)
        res = a.delete_edge(3, 6)
        r = recoloured(before, a.colouring)
        worst = max(worst, r)
        bad += (not is_proper(g.adj, a.colouring)) or r != res.recoloured
        trace.append(res.colours)
    g2, init2 = build()
    b = KERB(g2, KerbConfig(budget=1, plan_every=1, use_migration=False), initial_colouring=init2)
    for _ in range(6):
        b.delete_edge(3, 6)
    ok = (bad == 0 and worst <= 1 and a.num_colours == 2 and a.stats["planned_moves"] >= 2
          and b.num_colours == 3)
    return (f"colours per update {trace}, max r={worst}, planned moves={a.stats['planned_moves']}; "
            f"without C4: k={b.num_colours}"), ok

def main() -> None:
    out = ROOT / "results" / "summary"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for cid, cat, purpose, inp, expected, fn in CASES:
        try:
            actual, ok = fn()
        except Exception as exc:  # a crash is a failed test, recorded as such
            actual, ok = f"exception {exc!r}", False
        rows.append({"Test ID": cid, "Category": cat, "Purpose": purpose, "Input": inp,
                     "Expected Behaviour": expected, "Actual Result": actual, "Status": "PASS" if ok else "FAIL"})
        print(cid, "PASS" if ok else "FAIL", actual)
    with open(out / "test_cases.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()

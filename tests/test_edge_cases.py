"""Edge cases, invalid input and the supporting modules (loaders, workloads, metrics)."""

import random

import networkx as nx
import pytest

from dynamic_graph import DynamicGraph, GraphError
from graph_features import connected_components, graph_features
from graph_loader import InputFormatError, read_dimacs, read_temporal
from metrics import RunRecorder, gini
from proposed_algorithm import DELETE, INSERT, KERB, KerbConfig
from validation import assert_budget, assert_proper, is_proper, recoloured
from workloads import random_churn, relabel_vertices, sliding_window


# ------------------------------------------------------------- degenerate graphs
def test_empty_graph_with_no_vertices():
    g = DynamicGraph(0)
    alg = KERB(g, KerbConfig(budget=1))
    assert alg.colouring == [] and alg.num_colours == 0


def test_single_vertex_and_isolated_vertices():
    g = DynamicGraph(5)
    alg = KERB(g, KerbConfig(budget=1))
    assert alg.num_colours == 1 and is_proper(g.adj, alg.colouring)


def test_insert_existing_and_delete_missing_edges_do_nothing():
    g = DynamicGraph(3, [(0, 1)])
    alg = KERB(g, KerbConfig(budget=2))
    before = list(alg.colouring)
    r1 = alg.insert_edge(0, 1)
    r2 = alg.delete_edge(1, 2)
    assert not r1.applied and not r2.applied
    assert r1.recoloured == 0 and r2.recoloured == 0 and alg.colouring == before


def test_insert_then_delete_same_edge_keeps_colouring_proper():
    g = DynamicGraph(4, [(0, 1), (1, 2)])
    alg = KERB(g, KerbConfig(budget=2))
    for op in [(INSERT, 0, 2), (DELETE, 0, 2), (INSERT, 0, 2), (DELETE, 0, 2)]:
        res = alg.apply(*op)
        assert res.applied and is_proper(g.adj, alg.colouring) and res.recoloured <= 2


def test_complete_graph_built_edge_by_edge_needs_n_colours():
    n = 9
    g = DynamicGraph(n)
    alg = KERB(g, KerbConfig(budget=3))
    edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.Random(1).shuffle(edges)
    for u, v in edges:
        res = alg.insert_edge(u, v)
        assert res.recoloured <= 3 and is_proper(g.adj, alg.colouring)
    assert alg.num_colours == n


def test_star_and_disconnected_union():
    g = DynamicGraph(12)
    alg = KERB(g, KerbConfig(budget=2))
    for leaf in range(1, 6):                      # star centred at 0
        alg.insert_edge(0, leaf)
    for u, v in [(6, 7), (7, 8), (8, 6), (9, 10)]:  # triangle plus an edge
        alg.insert_edge(u, v)
    assert is_proper(g.adj, alg.colouring)
    assert alg.num_colours == 3                   # the triangle needs 3, and 3 suffice
    assert connected_components(g.adj) == 4       # star, triangle, edge, vertex 11


def test_deleting_all_edges_returns_towards_one_colour():
    G = nx.cycle_graph(7)
    g = DynamicGraph(7, G.edges())
    alg = KERB(g, KerbConfig(budget=3))
    for u, v in list(G.edges()):
        alg.delete_edge(u, v)
    assert is_proper(g.adj, alg.colouring)
    assert alg.num_colours < 3                    # classes were eliminated with spare budget


# ---------------------------------------------------------------- invalid input
@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_invalid_budget_rejected(bad):
    with pytest.raises(ValueError):
        KERB(DynamicGraph(3), KerbConfig(budget=bad))


def test_invalid_repair_mode_and_operation_rejected():
    with pytest.raises(ValueError):
        KERB(DynamicGraph(3), KerbConfig(repair_mode="magic"))
    alg = KERB(DynamicGraph(3), KerbConfig())
    with pytest.raises(ValueError):
        alg.apply("swap", 0, 1)


def test_self_loop_and_out_of_range_vertices_rejected():
    g = DynamicGraph(3)
    with pytest.raises(GraphError):
        g.add_edge(1, 1)
    with pytest.raises(GraphError):
        g.add_edge(0, 3)
    with pytest.raises(GraphError):
        DynamicGraph(-2)


def test_initial_colouring_of_wrong_length_rejected():
    with pytest.raises(ValueError):
        KERB(DynamicGraph(3), KerbConfig(), initial_colouring=[0, 1])


# ---------------------------------------------------------------------- loaders
def test_read_dimacs_merges_duplicates_and_drops_self_loops(tmp_path):
    f = tmp_path / "toy.col"
    f.write_text("c comment\np edge 4 5\ne 1 2\ne 2 1\ne 2 3\ne 3 3\ne 3 4\n")
    inst = read_dimacs(f)
    assert inst.n == 4 and inst.edges == [(0, 1), (1, 2), (2, 3)]
    assert inst.self_loops == 1 and inst.duplicate_lines == 1 and inst.header_edges == 5
    assert len(inst.sha256) == 64


@pytest.mark.parametrize("text", ["e 1 2\n", "p edge 3 1\ne 1 9\n", "p edge\n", "c only comments\n"])
def test_read_dimacs_rejects_malformed_files(tmp_path, text):
    f = tmp_path / "bad.col"
    f.write_text(text)
    with pytest.raises(InputFormatError):
        read_dimacs(f)


def test_read_temporal_maps_ids_and_sorts_by_time(tmp_path):
    f = tmp_path / "stream.txt"
    f.write_text("% header\n10 20 5\n20 30 1\n30 30 2\n")
    inst = read_temporal(f)
    assert inst.n == 3 and inst.raw_ids == ["10", "20", "30"]
    assert [e[2] for e in inst.events] == [1.0, 2.0, 5.0] and inst.self_loops == 1


def test_read_temporal_rejects_bad_lines(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("1 2 x\n")
    with pytest.raises(InputFormatError):
        read_temporal(f)


# -------------------------------------------------------------------- workloads
def test_random_churn_is_reproducible_and_consistent():
    edges = list(nx.gnp_random_graph(30, 0.3, seed=2).edges())
    a = random_churn(edges, rho=0.25, updates=500, seed=4)
    b = random_churn(edges, rho=0.25, updates=500, seed=4)
    assert a == b
    initial, ops = a
    present = set(initial)
    for op, u, v in ops:
        e = (u, v)
        if op == INSERT:
            assert e not in present
            present.add(e)
        else:
            assert e in present
            present.remove(e)
    assert present <= {(min(x, y), max(x, y)) for x, y in edges}


def test_sliding_window_inserts_and_expires():
    events = [(0, 1, 0.0), (1, 2, 5.0), (0, 1, 8.0), (2, 2, 9.0), (2, 3, 30.0)]
    ops = sliding_window(events, 10.0)
    assert ops == [("insert", 0, 1, 0.0), ("insert", 1, 2, 5.0), ("delete", 1, 2, 15.0),
                   ("delete", 0, 1, 18.0), ("insert", 2, 3, 30.0)]
    with pytest.raises(ValueError):
        sliding_window([(0, 1, 2.0), (0, 1, 1.0)], 5.0)


def test_relabel_vertices_preserves_structure():
    initial = [(0, 1), (1, 2)]
    ops = [("insert", 0, 2), ("delete", 0, 1)]
    new_initial, new_ops, perm = relabel_vertices(3, initial, ops, seed=7)
    assert sorted(perm) == [0, 1, 2]
    assert new_initial == [(perm[0], perm[1]), (perm[1], perm[2])]
    assert new_ops[0] == ("insert", perm[0], perm[2])


# ---------------------------------------------------------- metrics, validation
def test_gini_extremes():
    assert gini([3, 3, 3, 3]) == 0.0
    assert gini([0, 0, 0, 4]) == pytest.approx(0.75)
    assert gini([]) == 0.0


def test_run_recorder_summary():
    g = DynamicGraph(4, [(0, 1)])
    alg = KERB(g, KerbConfig(budget=2))
    rec = RunRecorder()
    for op in [(INSERT, 1, 2), (INSERT, 2, 0), (DELETE, 0, 1)]:
        rec.run_update(alg, *op)
    out = rec.summary(alg.recolour_count)
    assert out["updates"] == 3 and out["max_recoloured"] <= 2
    assert {"mean_colours", "p95_time_us", "gini_recolourings"} <= set(out)


def test_validation_helpers():
    adj = [{1}, {0}]
    with pytest.raises(AssertionError):
        assert_proper(adj, [0, 0])
    with pytest.raises(AssertionError):
        assert_budget([0, 1, 2], [1, 2, 0], 2)
    assert recoloured([0, 1, 2], [0, 2, 2]) == 1


def test_graph_features():
    G = nx.petersen_graph()
    f = graph_features([set(G[v]) for v in range(10)])
    assert f["n"] == 10 and f["m"] == 15 and f["max_degree"] == 3 and f["degeneracy"] == 3
    assert f["clique_lb"] == 2 and f["components"] == 1

"""Correctness tests for KERB, its data structures and the baselines.

The graphs below are small, hand-built or generated with fixed seeds; they
serve as test oracles (known chromatic numbers or exhaustive checks), not
as experimental data.
"""

import itertools
import random

import networkx as nx
import pytest

from baseline_algorithms import (DsaturRecompute, FirstFitRepair, TabuWarmStart,
                                 dsatur_recompute_cost, hamming, match_labels)
from coloring_state import ColouringState
from dynamic_graph import DynamicGraph
from proposed_algorithm import DELETE, INSERT, KERB, KerbConfig
from static_coloring import (STATIC_ALGORITHMS, count_conflicts, dsatur, greedy_clique_lower_bound,
                             rlf, smallest_last, tabucol, tabucol_minimise, welsh_powell)
from validation import is_proper, number_of_colours, recoloured
from workloads import random_churn


def churn_case(n=40, p=0.2, updates=400, seed=0):
    G = nx.gnp_random_graph(n, p, seed=seed)
    return n, random_churn(list(G.edges()), rho=0.3, updates=updates, seed=seed)


def run_and_check(alg, g, ops, budget=None):
    """Apply ops; after every update check properness, exact r_t and k_t,
    the budget, and the colour bound k_t <= max(k_0, max Delta + 1)."""
    k0 = alg.num_colours
    max_deg = g.max_degree()
    for op, u, v in ops:
        before = list(alg.colouring)
        res = alg.apply(op, u, v)
        max_deg = max(max_deg, g.max_degree())
        col = alg.colouring
        assert is_proper(g.adj, col), (op, u, v)
        assert res.recoloured == recoloured(before, col)
        assert res.colours == number_of_colours(col)
        if budget is not None:
            assert res.recoloured <= budget
            assert res.colours <= max(k0, max_deg + 1)
    return alg


# ------------------------------------------------------------- data structures
def test_state_tracks_exact_hamming_distance_and_rollback():
    st = ColouringState([0, 1, 2, 0])
    st.begin_update()
    sp = st.savepoint()
    st.set_colour(0, 1)
    st.set_colour(1, 2)
    assert st.changed == 2
    st.set_colour(0, 2)            # moved twice: still counts once
    assert st.changed == 2
    st.set_colour(1, 1)            # back to its original colour: counts zero
    assert st.changed == 1
    assert st.cost_if_set(0, 0) == -1 and st.cost_if_set(3, 1) == 1 and st.cost_if_set(0, 1) == 0
    st.rollback(sp)
    assert st.col == [0, 1, 2, 0] and st.changed == 0
    assert {c: sorted(m) for c, m in st.classes.items()} == {0: [0, 3], 1: [1], 2: [2]}


def test_state_classes_disappear_and_new_label_is_smallest_unused():
    st = ColouringState([0, 1, 2])
    st.begin_update()
    st.set_colour(1, 0)
    assert 1 not in st.classes and st.num_colours == 2
    assert st.new_label() == 1


def test_dynamic_graph_insert_delete_and_degree():
    g = DynamicGraph(4, [(0, 1), (1, 2)])
    assert g.m == 2 and g.degree(1) == 2
    assert g.add_edge(2, 3) and not g.add_edge(3, 2)
    assert g.remove_edge(0, 1) and not g.remove_edge(0, 1)
    assert g.m == 2 and g.max_degree() == 2 and sorted(g.edges()) == [(1, 2), (2, 3)]


# --------------------------------------------------------- static algorithms
@pytest.mark.parametrize("name", sorted(STATIC_ALGORITHMS))
def test_static_algorithms_are_proper(name):
    G = nx.gnp_random_graph(60, 0.3, seed=3)
    adj = [set(G[v]) for v in range(60)]
    col = STATIC_ALGORITHMS[name](adj)
    assert is_proper(adj, col)


@pytest.mark.parametrize("graph, chi", [
    (nx.complete_graph(7), 7), (nx.cycle_graph(9), 3), (nx.cycle_graph(10), 2),
    (nx.complete_bipartite_graph(5, 6), 2), (nx.star_graph(8), 2), (nx.path_graph(12), 2),
    (nx.petersen_graph(), 3), (nx.empty_graph(5), 1),
])
def test_dsatur_on_graphs_with_known_chromatic_number(graph, chi):
    # DSATUR is exact on bipartite graphs, cycles and complete graphs (Brelaz, 1979)
    adj = [set(graph[v]) for v in range(graph.number_of_nodes())]
    col = dsatur(adj)
    assert is_proper(adj, col) and number_of_colours(col) == chi


def test_clique_bound_and_tabucol_minimise():
    G = nx.gnp_random_graph(50, 0.5, seed=7)
    adj = [set(G[v]) for v in range(50)]
    lb = greedy_clique_lower_bound(adj)
    start = dsatur(adj)
    col, _ = tabucol_minimise(adj, start, 5000, seed=1)
    assert is_proper(adj, col)
    assert lb <= number_of_colours(col) <= number_of_colours(start)
    assert number_of_colours(rlf(adj)) >= lb and number_of_colours(welsh_powell(adj)) >= lb
    assert number_of_colours(smallest_last(adj)) >= lb


def test_tabucol_repairs_a_single_conflict():
    G = nx.cycle_graph(8)
    adj = [set(G[v]) for v in range(8)]
    col = [i % 2 for i in range(8)]
    col[3] = col[4]  # create a conflict with two labels available
    col, conflicts, _ = tabucol(adj, col, [0, 1], 1000, random.Random(0))
    assert conflicts == 0 and count_conflicts(adj, col) == 0


# --------------------------------------------------------------------- KERB
@pytest.mark.parametrize("budget", [1, 2, 3, 5, 10])
@pytest.mark.parametrize("mode", ["ejection", "kempe", "direct"])
def test_kerb_valid_and_within_budget(budget, mode):
    n, (init, ops) = churn_case(seed=budget)
    g = DynamicGraph(n, init)
    alg = KERB(g, KerbConfig(budget=budget, repair_mode=mode, seed=1))
    run_and_check(alg, g, ops, budget)


@pytest.mark.parametrize("elim, mig", [(False, False), (True, False), (False, True)])
def test_kerb_ablations_valid(elim, mig):
    n, (init, ops) = churn_case(seed=11)
    g = DynamicGraph(n, init)
    alg = KERB(g, KerbConfig(budget=3, use_elimination=elim, use_migration=mig))
    run_and_check(alg, g, ops, 3)


@pytest.mark.parametrize("budget", [2, 6, 12])
def test_kerb_budgeted_tabu_fallback_valid(budget):
    # a tiny node limit forces the exact search to give up, so the budgeted
    # TabuCol fallback of C2 and C3 is exercised
    n, (init, ops) = churn_case(n=60, p=0.5, updates=300, seed=budget)
    g = DynamicGraph(n, init)
    alg = KERB(g, KerbConfig(budget=budget, search_nodes=5, tabu_iters=200, seed=2))
    run_and_check(alg, g, ops, budget)


def test_kerb_dense_graph_valid():
    n, (init, ops) = churn_case(n=60, p=0.6, updates=300, seed=5)
    g = DynamicGraph(n, init)
    run_and_check(KERB(g, KerbConfig(budget=4)), g, ops, 4)


def test_kerb_is_deterministic_for_a_seed():
    n, (init, ops) = churn_case(seed=21)
    runs = []
    for _ in range(2):
        g = DynamicGraph(n, init)
        alg = KERB(g, KerbConfig(budget=3, seed=4))
        runs.append([alg.apply(*op).colours for op in ops] + list(alg.colouring))
    assert runs[0] == runs[1]


def test_chain_repair_when_no_direct_move_exists():
    # u = 0 and v = 1 share colour 0 and each sees colours 1 and 2, so no single
    # move resolves the new edge.  Moving u to colour 1 displaces only a = 2,
    # which can then take a free colour: a chain of cost 2 with no new colour.
    g = DynamicGraph(6, [(0, 2), (0, 3), (1, 4), (1, 5)])
    alg = KERB(g, KerbConfig(budget=2, use_elimination=False, use_migration=False),
               initial_colouring=[0, 0, 1, 2, 1, 2])
    res = alg.insert_edge(0, 1)
    assert res.conflict and res.repair == "chain" and res.recoloured == 2
    assert is_proper(g.adj, alg.colouring) and res.colours == 3
    # with R = 1 the same conflict needs a new colour
    g1 = DynamicGraph(6, [(0, 2), (0, 3), (1, 4), (1, 5)])
    alg1 = KERB(g1, KerbConfig(budget=1, use_elimination=False, use_migration=False),
                initial_colouring=[0, 0, 1, 2, 1, 2])
    res1 = alg1.insert_edge(0, 1)
    assert res1.repair == "new_colour" and res1.recoloured == 1 and res1.colours == 4


def test_new_colour_only_when_all_colours_are_blocked():
    # K4 minus edge {0,1}, coloured with 3 colours; inserting {0,1} forces a 4th colour
    g = DynamicGraph(4, [(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    alg = KERB(g, KerbConfig(budget=3), initial_colouring=[0, 0, 1, 2])
    res = alg.insert_edge(0, 1)
    assert res.repair == "new_colour" and res.colours == 4 and res.recoloured == 1


def test_budget_one_is_always_feasible():
    G = nx.complete_graph(12)
    edges = list(G.edges())
    g = DynamicGraph(12)
    alg = KERB(g, KerbConfig(budget=1))
    random.Random(0).shuffle(edges)
    run_and_check(alg, g, [(INSERT, u, v) for u, v in edges], 1)
    assert alg.num_colours == 12  # K12 needs 12 colours


def test_class_elimination_after_deletions():
    # a path coloured with three colours can be reduced to two by deleting nothing;
    # here we delete edges so that a small class becomes removable
    g = DynamicGraph(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    alg = KERB(g, KerbConfig(budget=2, use_migration=False), initial_colouring=[0, 1, 2, 0, 1])
    res = alg.delete_edge(3, 4)
    assert is_proper(g.adj, alg.colouring) and res.recoloured <= 2
    assert res.colours == 2 and res.eliminated == 1



def _migration_case():
    # colour 2 = {7, 8, 9} can only be removed by moving its three vertices into
    # colour 1; with R = 1 that needs three updates, which only C4 can plan
    edges = [(7, 0), (8, 1), (9, 2), (0, 3), (1, 4), (2, 5)]
    init = [0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    return DynamicGraph(10, edges), init


def test_planned_migration_removes_a_class_over_several_updates():
    g, init = _migration_case()
    alg = KERB(g, KerbConfig(budget=1, plan_every=1), initial_colouring=init)
    for _ in range(6):
        before = list(alg.colouring)
        res = alg.delete_edge(3, 6)            # absent edge: the update only offers the budget
        assert not res.applied
        assert is_proper(g.adj, alg.colouring)
        assert recoloured(before, alg.colouring) == res.recoloured <= 1
    assert alg.num_colours == 2
    assert alg.stats["plans_found"] >= 1 and alg.stats["planned_moves"] >= 2
    # without C4 the class cannot be removed: C3 alone never fits it into R = 1
    g2, init2 = _migration_case()
    alg2 = KERB(g2, KerbConfig(budget=1, plan_every=1, use_migration=False), initial_colouring=init2)
    for _ in range(6):
        alg2.delete_edge(3, 6)
    assert alg2.num_colours == 3

# ---------------------------------------------------------------- baselines
@pytest.mark.parametrize("cls", [FirstFitRepair, TabuWarmStart, DsaturRecompute])
def test_baselines_valid(cls):
    n, (init, ops) = churn_case(updates=200, seed=9)
    g = DynamicGraph(n, init)
    run_and_check(cls(g), g, ops)


def test_first_fit_recolours_at_most_one_vertex():
    n, (init, ops) = churn_case(seed=13)
    g = DynamicGraph(n, init)
    alg = FirstFitRepair(g)
    for op in ops:
        assert alg.apply(*op).recoloured <= 1


def test_match_labels_maximises_agreement():
    prev = [0, 0, 1, 1, 2]
    new = [5, 5, 7, 7, 9]           # same partition, different labels
    assert match_labels(prev, new) == prev
    swapped = [1, 1, 0, 0, 2]
    assert hamming(prev, match_labels(prev, swapped)) == 0


def test_dsatur_recompute_cost_matches_the_class():
    n, (init, ops) = churn_case(updates=60, seed=17)
    g = DynamicGraph(n, init)
    alg = DsaturRecompute(g)
    for op, u, v in ops:
        before = [set(a) for a in g.adj]
        res = alg.apply(op, u, v)
        cost, k = dsatur_recompute_cost(before, g.adj)
        assert k == res.colours
        # the class relabels against its own previous colouring (DSATUR of G_{t-1}),
        # so its exact cost equals the stateless computation
        assert cost == res.recoloured

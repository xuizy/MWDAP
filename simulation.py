"""
MWDAP Simulation: Comparing Distributed Algorithm 1, Frederickson-Ja'Ja'
2-Approximation, and Exact ILP.

This script generates random weakly connected digraphs, solves the Minimum
Weight Digraph Augmentation Problem (MWDAP) using three methods, and
compares solution quality (cost, edge count, approximation ratio).

Accompanies the paper:
  "Distributed graph augmentation protocols for weighted strong connectivity
   in multi-agent systems" (Ramos, Poças, Pequito)

Requirements: networkx, numpy, scipy
"""

import itertools
import random
import time
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


# ============================================================
# Graph generation
# ============================================================

def generate_er_digraph(n, p, weight_range=(1, 10), seed=None, connect=True):
    """Erdos-Renyi random digraph, made weakly connected, with random weights.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Edge probability for each directed pair.
    weight_range : tuple
        (min, max) for uniform random weights of non-existing edges.
    seed : int or None
        Random seed for reproducibility.
    connect : bool
        If True, ensure the graph is weakly connected.

    Returns
    -------
    G : nx.DiGraph
        The generated digraph.
    W : np.ndarray
        Weight matrix (n x n); W[u][v] = 0 if (u,v) in E, else random.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for u, v in itertools.permutations(range(n), 2):
        if rng.random() < p:
            G.add_edge(u, v)

    if connect:
        _ensure_weakly_connected(G, rng)

    W = np_rng.uniform(weight_range[0], weight_range[1], size=(n, n))
    np.fill_diagonal(W, np.inf)
    for u, v in G.edges():
        W[u][v] = 0.0

    return G, W


def generate_euclidean_digraph(n, radius=0.35, seed=None, connect=True):
    """Digraph from random points in [0,1]^2; edges within radius, weights = distance.

    Parameters
    ----------
    n : int
        Number of nodes.
    radius : float
        Communication radius; edges possible only between nodes closer than this.
    seed : int or None
        Random seed for reproducibility.
    connect : bool
        If True, ensure the graph is weakly connected.

    Returns
    -------
    G : nx.DiGraph
    W : np.ndarray
        Weight matrix; W[u][v] = Euclidean distance if (u,v) not in E, else 0.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    positions = np_rng.uniform(0, 1, size=(n, 2))

    W = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(n):
            if i != j:
                W[i][j] = np.linalg.norm(positions[i] - positions[j])

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for u in range(n):
        for v in range(n):
            if u != v and W[u][v] < radius:
                if rng.random() < 0.5:
                    G.add_edge(u, v)
                    W[u][v] = 0.0

    if connect:
        _ensure_weakly_connected(G, rng)

    for u, v in G.edges():
        W[u][v] = 0.0

    return G, W


def generate_dag_digraph(n, p=0.3, weight_range=(1, 10), seed=None, connect=True):
    """Random DAG (topological order = node indices), made weakly connected.

    Parameters
    ----------
    n : int
        Number of nodes.
    p : float
        Edge probability for each forward pair (u < v).
    weight_range : tuple
        (min, max) for uniform random weights of non-existing edges.
    seed : int or None
        Random seed for reproducibility.
    connect : bool
        If True, ensure the graph is weakly connected.

    Returns
    -------
    G : nx.DiGraph
    W : np.ndarray
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(n))

    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                G.add_edge(u, v)

    if connect:
        _ensure_weakly_connected(G, rng)

    W = np_rng.uniform(weight_range[0], weight_range[1], size=(n, n))
    np.fill_diagonal(W, np.inf)
    for u, v in G.edges():
        W[u][v] = 0.0

    return G, W


def _ensure_weakly_connected(G, rng):
    """Add edges to make G weakly connected by chaining components."""
    undirected = G.to_undirected()
    components = list(nx.connected_components(undirected))
    if len(components) <= 1:
        return
    for i in range(len(components) - 1):
        u = rng.choice(list(components[i]))
        v = rng.choice(list(components[i + 1]))
        if rng.random() < 0.5:
            G.add_edge(u, v)
        else:
            G.add_edge(v, u)


# ============================================================
# SCC utilities
# ============================================================

def classify_sccs(G):
    """Classify SCCs into source, target, mixed, and isolated.

    Returns
    -------
    source, target, mixed, isolated : lists of sets
        Each list contains the SCCs of the corresponding type.
    scc_map : dict
        Maps each node to its SCC (as a frozenset).
    """
    sccs = list(nx.strongly_connected_components(G))
    scc_map = {}
    for scc in sccs:
        for node in scc:
            scc_map[node] = frozenset(scc)

    source, target, mixed, isolated = [], [], [], []
    for scc in sccs:
        has_in = any(u not in scc for v in scc for u in G.predecessors(v))
        has_out = any(w not in scc for v in scc for w in G.successors(v))
        if has_in and has_out:
            mixed.append(scc)
        elif has_in and not has_out:
            target.append(scc)
        elif not has_in and has_out:
            source.append(scc)
        else:
            isolated.append(scc)

    return source, target, mixed, isolated, scc_map


# ============================================================
# Method 1: Distributed Algorithm (Algorithm 1 simulation)
# ============================================================

def distributed_algorithm(G_orig, W):
    """Simulate Algorithm 1 from the paper.

    Each round:
      Phase 1: Compute SCCs of current graph.
      Phase 2: Each t-SCC representative finds shortest path (via Dijkstra
               on its *known subgraph*) to nearest s-SCC representative.
      Phase 3: Each s-SCC representative selects cheapest incoming proposal.

    The known subgraph of agent t is determined by reverse reachability:
    t knows about all edges originating from nodes that can reach t.

    Parameters
    ----------
    G_orig : nx.DiGraph
        The input digraph.
    W : np.ndarray
        Weight matrix (n x n).

    Returns
    -------
    added_edges : list of (u, v, cost) tuples
    total_cost : float
    rounds : int
    """
    G = G_orig.copy()
    n = len(G.nodes())
    added_edges = []
    total_cost = 0.0
    rounds = 0

    while True:
        source_sccs, target_sccs, _, isolated, _ = classify_sccs(G)

        if not source_sccs and not target_sccs:
            break

        rounds += 1
        source_reps = {min(scc): scc for scc in source_sccs}
        target_reps = {min(scc): scc for scc in target_sccs}

        # Phase 2: proposals from target representatives
        proposals = defaultdict(list)  # s_rep -> [(cost, path, t_rep)]

        for t_rep in target_reps:
            # Nodes that can reach t_rep via directed paths
            reachable_to_t = nx.ancestors(G, t_rep) | {t_rep}

            # Build known subgraph: each node u in reachable_to_t knows
            # its own ancestors and their out-neighbors
            known_graph = nx.DiGraph()
            for u in reachable_to_t:
                reachable_to_u = nx.ancestors(G, u) | {u}
                known_to_u = reachable_to_u.copy()
                for v in reachable_to_u:
                    known_to_u |= set(G.successors(v))
                for v in known_to_u:
                    if u != v:
                        known_graph.add_edge(u, v, weight=W[u][v])

            # Dijkstra from t_rep on known graph
            try:
                distances, paths = nx.single_source_dijkstra(
                    known_graph, t_rep
                )
            except nx.NetworkXError:
                continue

            # Find nearest s-SCC representative
            best_s = None
            best_d = float("inf")
            best_path = None
            for s_rep in source_reps:
                if s_rep in distances and distances[s_rep] < best_d:
                    best_d = distances[s_rep]
                    best_s = s_rep
                    best_path = paths[s_rep]

            if best_s is not None:
                proposals[best_s].append((best_d, best_path, t_rep))

        # Phase 3: each source representative selects cheapest proposal
        for s_rep, props in proposals.items():
            best_cost, best_path, best_t = min(props, key=lambda x: x[0])
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                if not G.has_edge(u, v):
                    G.add_edge(u, v)
                    edge_cost = W[u][v]
                    added_edges.append((u, v, edge_cost))
                    total_cost += edge_cost

        if rounds > 2 * n:  # safety bound
            break

    return added_edges, total_cost, rounds


# ============================================================
# Method 2: Frederickson-Ja'Ja' 2-Approximation
# ============================================================

def frederickson_jaja(G_orig, W):
    """Frederickson-Ja'Ja' 2-approximation for MWDAP.

    1. Compute SCCs and condensation DAG.
    2. Build complete weighted digraph on condensation nodes.
    3. Find minimum-weight arborescence (backward) to select a root.
    4. Find minimum-weight arborescence (forward) from the same root,
       reusing edges from step 3 at zero cost.
    5. Union of branchings gives augmentation with cost <= 2 * OPT.

    Parameters
    ----------
    G_orig : nx.DiGraph
    W : np.ndarray

    Returns
    -------
    added_edges : list of (u, v, cost) tuples
    total_cost : float
    """
    G = G_orig.copy()

    sccs = list(nx.strongly_connected_components(G))
    k = len(sccs)
    if k == 1:
        return [], 0.0

    scc_list = [frozenset(s) for s in sccs]
    node_to_scc = {}
    for idx, scc in enumerate(scc_list):
        for node in scc:
            node_to_scc[node] = idx

    # Build complete weighted digraph on condensation nodes
    # w_H(i, j) = min_{u in S_i, v in S_j} c(u, v)
    w_H = np.full((k, k), np.inf)
    witness = {}
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            best_w = np.inf
            best_edge = None
            for u in scc_list[i]:
                for v in scc_list[j]:
                    if W[u][v] < best_w:
                        best_w = W[u][v]
                        best_edge = (u, v)
            w_H[i][j] = best_w
            witness[(i, j)] = best_edge

    # Backward arborescence: find root and edges to reach all nodes
    backwards_K = nx.DiGraph()
    for i in range(k):
        backwards_K.add_node(i)
    for i in range(k):
        for j in range(k):
            if i != j and w_H[j][i] < np.inf:
                backwards_K.add_edge(i, j, weight=w_H[j][i])

    backwards_B = nx.minimum_spanning_arborescence(backwards_K, attr="weight")
    root = [n for n, d in backwards_B.in_degree() if d == 0][0]
    new_edges = [(j, i) for (i, j) in backwards_B.edges() if w_H[j][i] > 0]

    # Forward arborescence: reuse backward edges at zero cost
    forwards_K = nx.DiGraph()
    for i in range(k):
        forwards_K.add_node(i)
    for i in range(k):
        for j in range(k):
            if i != j and j != root and w_H[i][j] < np.inf:
                forwards_K.add_edge(
                    i, j,
                    weight=0 if (i, j) in new_edges else w_H[i][j]
                )

    forwards_B = nx.minimum_spanning_arborescence(forwards_K, attr="weight")
    new_new_edges = [
        (i, j) for (i, j) in forwards_B.edges()
        if w_H[i][j] > 0 and (i, j) not in new_edges
    ]

    # Collect augmenting edges from both branchings
    added_edges = []
    total_cost = 0.0
    for (i, j) in new_edges + new_new_edges:
        orig_edge = witness[(i, j)]
        added_edges.append(
            (orig_edge[0], orig_edge[1], W[orig_edge[0]][orig_edge[1]])
        )
        total_cost += W[orig_edge[0]][orig_edge[1]]

    return added_edges, total_cost


# ============================================================
# Method 3: Exact ILP via scipy.optimize.milp
# ============================================================

def exact_ilp(G_orig, W, time_limit=120):
    """Exact ILP for MWDAP using multi-commodity flow formulation.

    Variables:
      x_{uv} in {0,1} for (u,v) not in E  (edge selection)
      f^t_{uv} >= 0  for each commodity t, arc (u,v)  (forward flow)
      g^s_{uv} >= 0  for each commodity s, arc (u,v)  (backward flow)

    Parameters
    ----------
    G_orig : nx.DiGraph
    W : np.ndarray
    time_limit : int
        Solver time limit in seconds.

    Returns
    -------
    added_edges : list of (u, v, cost) tuples, or None if infeasible/timeout.
    total_cost : float (inf if infeasible/timeout).
    """
    n = len(G_orig.nodes())
    nodes = sorted(G_orig.nodes())
    existing_edges = set(G_orig.edges())

    candidate_edges = [
        (u, v) for u in nodes for v in nodes
        if u != v and (u, v) not in existing_edges
    ]
    all_arcs = [(u, v) for u in nodes for v in nodes if u != v]
    arc_index = {arc: i for i, arc in enumerate(all_arcs)}

    M = len(candidate_edges)
    A = len(all_arcs)
    cand_index = {e: i for i, e in enumerate(candidate_edges)}

    root = nodes[0]
    commodities = [t for t in nodes if t != root]
    n_comm = len(commodities)

    # Variable layout: [x (M) | f^t (n_comm * A) | g^s (n_comm * A)]
    n_vars = M + 2 * n_comm * A

    # Objective: minimize sum c(u,v) * x_{uv}
    obj = np.zeros(n_vars)
    for (u, v), idx in cand_index.items():
        obj[idx] = W[u][v]

    # Integrality: x binary, flows continuous
    integrality = np.zeros(n_vars, dtype=int)
    integrality[:M] = 1

    # Bounds
    lb = np.zeros(n_vars)
    ub = np.full(n_vars, np.inf)
    ub[:M] = 1.0

    # Constraints
    eq_rows = []
    ineq_rows = []

    def x_idx(u, v):
        if (u, v) in cand_index:
            return cand_index[(u, v)]
        return None

    def f_idx(t_pos, u, v):
        return M + t_pos * A + arc_index[(u, v)]

    def g_idx(s_pos, u, v):
        return M + n_comm * A + s_pos * A + arc_index[(u, v)]

    # Forward flow conservation
    for t_pos, t in enumerate(commodities):
        for u in nodes:
            row = {}
            for v in nodes:
                if v != u:
                    row[f_idx(t_pos, u, v)] = 1.0
                    row[f_idx(t_pos, v, u)] = row.get(f_idx(t_pos, v, u), 0) - 1.0
            if u == root:
                rhs = 1.0
            elif u == t:
                rhs = -1.0
            else:
                rhs = 0.0
            eq_rows.append((row, rhs))

    # Backward flow conservation
    for s_pos, s in enumerate(commodities):
        for u in nodes:
            row = {}
            for v in nodes:
                if v != u:
                    row[g_idx(s_pos, u, v)] = 1.0
                    row[g_idx(s_pos, v, u)] = row.get(g_idx(s_pos, v, u), 0) - 1.0
            if u == s:
                rhs = 1.0
            elif u == root:
                rhs = -1.0
            else:
                rhs = 0.0
            eq_rows.append((row, rhs))

    # Capacity constraints: flow <= 1 for existing edges, <= x for candidates
    for t_pos, t in enumerate(commodities):
        for u, v in all_arcs:
            fi = f_idx(t_pos, u, v)
            if (u, v) in existing_edges:
                ineq_rows.append(({fi: 1.0}, 1.0))
            elif (u, v) in cand_index:
                xi = x_idx(u, v)
                ineq_rows.append(({fi: 1.0, xi: -1.0}, 0.0))

    for s_pos, s in enumerate(commodities):
        for u, v in all_arcs:
            gi = g_idx(s_pos, u, v)
            if (u, v) in existing_edges:
                ineq_rows.append(({gi: 1.0}, 1.0))
            elif (u, v) in cand_index:
                xi = x_idx(u, v)
                ineq_rows.append(({gi: 1.0, xi: -1.0}, 0.0))

    # Build sparse constraint matrices
    n_eq = len(eq_rows)
    n_ineq = len(ineq_rows)

    constraints = []

    if n_eq > 0:
        A_eq = lil_matrix((n_eq, n_vars))
        b_eq = np.zeros(n_eq)
        for i, (row, rhs) in enumerate(eq_rows):
            for j, coeff in row.items():
                A_eq[i, j] = coeff
            b_eq[i] = rhs
        constraints.append(LinearConstraint(A_eq.tocsc(), b_eq, b_eq))

    if n_ineq > 0:
        A_ub = lil_matrix((n_ineq, n_vars))
        b_ub = np.zeros(n_ineq)
        for i, (row, rhs) in enumerate(ineq_rows):
            for j, coeff in row.items():
                A_ub[i, j] = coeff
            b_ub[i] = rhs
        constraints.append(LinearConstraint(A_ub.tocsc(), -np.inf, b_ub))

    bounds = Bounds(lb, ub)

    try:
        result = milp(
            c=obj,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds,
            options={"time_limit": time_limit, "presolve": True},
        )
    except Exception as e:
        print(f"  ILP solver error: {e}")
        return None, float("inf")

    if not result.success:
        print(f"  ILP solver status: {result.message}")
        return None, float("inf")

    x_sol = result.x[:M]
    added_edges = []
    total_cost = 0.0
    for i, (u, v) in enumerate(candidate_edges):
        if x_sol[i] > 0.5:
            added_edges.append((u, v, W[u][v]))
            total_cost += W[u][v]

    return added_edges, total_cost


# ============================================================
# Verification utility
# ============================================================

def verify_augmentation(G_orig, added_edges):
    """Check that adding edges makes the graph strongly connected."""
    G = G_orig.copy()
    for e in added_edges:
        G.add_edge(e[0], e[1])
    return nx.is_strongly_connected(G)


# ============================================================
# Single experiment
# ============================================================

def run_single(G, W, run_ilp=True, ilp_time_limit=600):
    """Run all three methods on a single instance and return results."""
    results = {}

    # Distributed Algorithm
    t0 = time.time()
    da_edges, da_cost, da_rounds = distributed_algorithm(G, W)
    da_time = time.time() - t0
    da_valid = verify_augmentation(G, da_edges)
    results["distributed"] = {
        "cost": da_cost,
        "n_edges": len(da_edges),
        "rounds": da_rounds,
        "time": da_time,
        "valid": da_valid,
    }

    # Frederickson-Ja'Ja'
    t0 = time.time()
    fj_edges, fj_cost = frederickson_jaja(G, W)
    fj_time = time.time() - t0
    fj_valid = verify_augmentation(G, fj_edges) if fj_edges is not None else False
    results["fj"] = {
        "cost": fj_cost,
        "n_edges": len(fj_edges) if fj_edges else 0,
        "time": fj_time,
        "valid": fj_valid,
    }

    # Exact ILP
    if run_ilp:
        t0 = time.time()
        ilp_edges, ilp_cost = exact_ilp(G, W, time_limit=ilp_time_limit)
        ilp_time = time.time() - t0
        ilp_valid = (
            verify_augmentation(G, ilp_edges) if ilp_edges is not None else False
        )
        results["ilp"] = {
            "cost": ilp_cost,
            "n_edges": len(ilp_edges) if ilp_edges else 0,
            "time": ilp_time,
            "valid": ilp_valid,
        }
    else:
        results["ilp"] = None

    return results


# ============================================================
# Batch experiments
# ============================================================

def run_experiments(
    node_sizes,
    params,
    samples_per_size=30,
    graph_model="er",
    ilp_max_n=20,
    seed_base=42,
):
    """Run comparative experiments across node sizes.

    Parameters
    ----------
    node_sizes : list of int
        Network sizes to test.
    params : list of callables
        Each callable maps n -> parameter value (p or radius).
    samples_per_size : int
        Number of random instances per (n, parameter) pair.
    graph_model : str
        One of 'er', 'euclidean', 'dag'.
    ilp_max_n : int
        Maximum n for which to run the exact ILP.
    seed_base : int
        Base random seed for reproducibility.

    Returns
    -------
    all_results : dict
        Mapping n -> list of result dicts.
    """
    all_results = {}

    for n in node_sizes:
        print(f"\n{'='*60}")
        print(f"  Graph model: {graph_model.upper()}, n = {n}, "
              f"samples = {samples_per_size}")
        print(f"{'='*60}")
        results_n = []
        for par_f in params:
            for s in range(samples_per_size):
                seed = seed_base + n * 1000 + s

                if graph_model == "er":
                    G, W = generate_er_digraph(n, p=par_f(n), seed=seed)
                elif graph_model == "euclidean":
                    G, W = generate_euclidean_digraph(n, radius=par_f(n), seed=seed)
                elif graph_model == "dag":
                    G, W = generate_dag_digraph(n, p=par_f(n), seed=seed)
                else:
                    raise ValueError(f"Unknown model: {graph_model}")

                run_ilp_flag = n <= ilp_max_n
                res = run_single(G, W, run_ilp=run_ilp_flag)

                res["n"] = n
                res["sample"] = s

                source_sccs, target_sccs, _, _, _ = classify_sccs(G)
                res["alpha"] = len(source_sccs)
                res["beta"] = len(target_sccs)
                res["d_in_max"] = max(dict(G.in_degree()).values())

                # Compute ratios
                if res["ilp"] is not None and res["ilp"]["cost"] > 1e-9:
                    res["ratio_da_ilp"] = (
                        res["distributed"]["cost"] / res["ilp"]["cost"]
                    )
                    res["ratio_fj_ilp"] = (
                        res["fj"]["cost"] / res["ilp"]["cost"]
                    )
                else:
                    res["ratio_da_ilp"] = None
                    res["ratio_fj_ilp"] = None

                if res["fj"]["cost"] > 1e-9:
                    res["ratio_da_fj"] = (
                        res["distributed"]["cost"] / res["fj"]["cost"]
                    )
                else:
                    res["ratio_da_fj"] = None

                results_n.append(res)

                if (s + 1) % 10 == 0:
                    print(f"  Completed {s+1}/{samples_per_size} samples")

        all_results[n] = results_n
        _print_summary(n, results_n)

    return all_results


def _print_summary(n, results):
    """Print a summary table for one value of n."""
    if not results:
        print("  No valid instances.")
        return

    da_costs = [r["distributed"]["cost"] for r in results]
    fj_costs = [r["fj"]["cost"] for r in results]
    da_times = [r["distributed"]["time"] for r in results]
    fj_times = [r["fj"]["time"] for r in results]

    print(f"\n  --- Summary for n = {n} ({len(results)} instances) ---")
    print(f"  {'Method':<15} {'Avg Cost':>10} {'Avg Time':>10}")
    print(f"  {'-'*40}")
    print(f"  {'Distributed':<15} {np.mean(da_costs):>10.2f} "
          f"{np.mean(da_times):>10.4f}")
    print(f"  {'F-J 2-approx':<15} {np.mean(fj_costs):>10.2f} "
          f"{np.mean(fj_times):>10.4f}")

    ratios_da_ilp = [r["ratio_da_ilp"] for r in results
                     if r["ratio_da_ilp"] is not None]
    ratios_fj_ilp = [r["ratio_fj_ilp"] for r in results
                     if r["ratio_fj_ilp"] is not None]
    ratios_da_fj = [r["ratio_da_fj"] for r in results
                    if r["ratio_da_fj"] is not None]

    if ratios_da_ilp:
        ilp_times = [r["ilp"]["time"] for r in results if r["ilp"] is not None]
        ilp_costs = [r["ilp"]["cost"] for r in results
                     if r["ilp"] is not None and r["ilp"]["cost"] < float("inf")]
        print(f"  {'ILP (exact)':<15} {np.mean(ilp_costs):>10.2f} "
              f"{np.mean(ilp_times):>10.4f}")
        print()
        print(f"  {'Ratio':<25} {'Mean':>8} {'Max':>8} {'Min':>8}")
        print(f"  {'-'*55}")
        print(f"  {'Dist / OPT':<25} {np.mean(ratios_da_ilp):>8.2f} "
              f"{np.max(ratios_da_ilp):>8.2f} {np.min(ratios_da_ilp):>8.2f}")
        print(f"  {'F-J / OPT':<25} {np.mean(ratios_fj_ilp):>8.2f} "
              f"{np.max(ratios_fj_ilp):>8.2f} {np.min(ratios_fj_ilp):>8.2f}")
    if ratios_da_fj:
        print(f"  {'Dist / F-J':<25} {np.mean(ratios_da_fj):>8.2f} "
              f"{np.max(ratios_da_fj):>8.2f} {np.min(ratios_da_fj):>8.2f}")

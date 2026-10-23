"""Minimum dominating set tools and the two-branch IS-to-mDS heuristic."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Hashable, Iterable

import networkx as nx

from .graph_io import ordered_nodes
from .samples import is_independent_set, is_maximal_independent_set, parse_sample_to_nodeset


@dataclass(frozen=True)
class BranchResult:
    branch: str
    best_size: int
    best_set: list[str]
    n_input_samples: int
    n_unique_mis: int
    selected_mis_size: int
    n_candidates_refined: int


@dataclass(frozen=True)
class PipelineResult:
    mIS_branch: BranchResult
    complement_branch: BranchResult
    best_branch: str
    best_size: int
    best_set: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mIS_branch": asdict(self.mIS_branch),
            "complement_branch": asdict(self.complement_branch),
            "best_branch": self.best_branch,
            "best_size": self.best_size,
            "best_set": self.best_set,
        }


def exact_mds_docplex(G: nx.Graph, *, time_limit_sec: float | None = None) -> tuple[set[Hashable], str]:
    """Return an MDS only if CPLEX certifies optimality (not an incumbent)."""
    D, audit = exact_mds_verified(G, solver="cplex", time_limit_sec=time_limit_sec)
    return D, str(audit["status"])



def greedy_dominating_set(G: nx.Graph) -> set[Hashable]:
    """Simple deterministic greedy baseline."""
    uncovered = set(G.nodes())
    D: set[Hashable] = set()
    neigh = {v: set(G.neighbors(v)) | {v} for v in G.nodes()}
    while uncovered:
        v = max(G.nodes(), key=lambda u: (len(neigh[u] & uncovered), G.degree[u], str(u)))
        D.add(v)
        uncovered -= neigh[v]
    return prune_redundant_nodes(G, D)


def random_maximal_independent_set(G: nx.Graph, rng: random.Random) -> set[Hashable]:
    """Generate one random maximal independent set."""
    remaining = list(G.nodes())
    rng.shuffle(remaining)
    I: set[Hashable] = set()
    blocked: set[Hashable] = set()
    for v in remaining:
        if v not in blocked:
            I.add(v)
            blocked.add(v)
            blocked.update(G.neighbors(v))
    return I


def sample_classical_mis(G: nx.Graph, n_samples: int, *, seed: int = 0) -> list[set[Hashable]]:
    rng = random.Random(seed)
    return [random_maximal_independent_set(G, rng) for _ in range(int(n_samples))]


def repair_to_targeted_maximal_is(
    G: nx.Graph,
    S: Iterable[Hashable],
    *,
    removal_tiebreak: str,
    extend_order: str,
    seed: int | None = None,
) -> set[Hashable]:
    """Repair any node set into a maximal independent set."""
    rng = random.Random(seed) if seed is not None else random.Random(0)
    S = set(S)
    neighbors = {v: set(G.neighbors(v)) for v in G.nodes()}
    degrees = dict(G.degree())

    def tie(v: Hashable) -> str:
        return str(v)

    while True:
        conflict_degree = {v: sum((u in S) for u in neighbors[v]) for v in S}
        conflicted = [v for v in S if conflict_degree[v] > 0]
        if not conflicted:
            break
        if removal_tiebreak == "max_degree":
            victim = max(conflicted, key=lambda v: (conflict_degree[v], degrees[v], tie(v)))
        elif removal_tiebreak == "min_degree":
            victim = max(conflicted, key=lambda v: (conflict_degree[v], -degrees[v], tie(v)))
        elif removal_tiebreak == "random":
            max_conf = max(conflict_degree[v] for v in conflicted)
            victim = rng.choice([v for v in conflicted if conflict_degree[v] == max_conf])
        else:
            raise ValueError("removal_tiebreak must be max_degree, min_degree, or random")
        S.remove(victim)

    I = set(S)
    blocked = set(I)
    for v in I:
        blocked.update(neighbors[v])

    while True:
        feasible = [v for v in G.nodes() if v not in blocked]
        if not feasible:
            break
        if extend_order == "min_degree":
            v = min(feasible, key=lambda u: (degrees[u], tie(u)))
        elif extend_order == "max_degree":
            v = max(feasible, key=lambda u: (degrees[u], tie(u)))
        elif extend_order == "random":
            v = rng.choice(feasible)
        else:
            raise ValueError("extend_order must be min_degree, max_degree, or random")
        I.add(v)
        blocked.add(v)
        blocked.update(neighbors[v])

    if not is_maximal_independent_set(G, I):
        raise AssertionError("Repair failed to create a maximal independent set.")
    return I


def repair_quantum_samples_to_mis(
    G: nx.Graph,
    raw_samples: Iterable[Any],
    *,
    mode: str,
    max_samples: int | None = None,
    seed: int = 0,
) -> list[set[Hashable]]:
    nodes = ordered_nodes(G)
    out: list[set[Hashable]] = []
    for i, raw in enumerate(raw_samples):
        if max_samples is not None and i >= int(max_samples):
            break
        S = parse_sample_to_nodeset(raw, nodes)
        if mode == "small":
            I = repair_to_targeted_maximal_is(
                G, S, removal_tiebreak="min_degree", extend_order="max_degree", seed=seed + i
            )
        elif mode == "large":
            I = repair_to_targeted_maximal_is(
                G, S, removal_tiebreak="max_degree", extend_order="min_degree", seed=seed + i
            )
        else:
            raise ValueError("mode must be small or large")
        out.append(I)
    return out


def prune_redundant_nodes(G: nx.Graph, D: Iterable[Hashable]) -> set[Hashable]:
    D = set(D)
    for v in sorted(list(D), key=lambda u: (G.degree[u], str(u))):
        trial = set(D)
        trial.remove(v)
        if nx.is_dominating_set(G, trial):
            D = trial
    return D


def two_neighborhood_sweep(G: nx.Graph, D: Iterable[Hashable]) -> set[Hashable]:
    D = set(D)
    improved = True
    while improved:
        improved = False
        for v in sorted(G.nodes(), key=str):
            if v in D:
                continue
            covered_by_D_neighbors = set(G.neighbors(v)) & D
            if len(covered_by_D_neighbors) == 2:
                trial = (D - covered_by_D_neighbors) | {v}
                if nx.is_dominating_set(G, trial) and len(trial) < len(D):
                    D = trial
                    improved = True
                    break
    return D


def refine_dominating_sets(G: nx.Graph, candidates: Iterable[Iterable[Hashable]]) -> list[set[Hashable]]:
    refined: list[set[Hashable]] = []
    for D0 in candidates:
        D = set(D0)
        if not nx.is_dominating_set(G, D):
            continue
        D = prune_redundant_nodes(G, D)
        D = two_neighborhood_sweep(G, D)
        if not nx.is_dominating_set(G, D):
            raise AssertionError("Refinement produced a non-dominating set.")
        refined.append(D)
    return refined


def _deduplicate_sets(sets: Iterable[set[Hashable]]) -> list[set[Hashable]]:
    seen: set[frozenset[Hashable]] = set()
    out: list[set[Hashable]] = []
    for S in sets:
        fs = frozenset(S)
        if fs not in seen:
            seen.add(fs)
            out.append(set(S))
    return out


def _best_set(sets: Iterable[set[Hashable]]) -> set[Hashable]:
    items = list(sets)
    if not items:
        raise ValueError("No candidate sets available.")
    return min(items, key=lambda S: (len(S), [str(v) for v in sorted(S, key=str)]))


def branch_from_smallest_mis(G: nx.Graph, mis_list: list[set[Hashable]]) -> BranchResult:
    if not mis_list:
        raise ValueError("Empty MIS list.")
    lmin = min(len(I) for I in mis_list)
    selected = _deduplicate_sets([I for I in mis_list if len(I) == lmin])
    refined = refine_dominating_sets(G, selected)
    best = _best_set(refined)
    return BranchResult(
        branch="smallest_mIS",
        best_size=len(best),
        best_set=[str(v) for v in sorted(best, key=str)],
        n_input_samples=len(mis_list),
        n_unique_mis=len(_deduplicate_sets(mis_list)),
        selected_mis_size=lmin,
        n_candidates_refined=len(refined),
    )


def branch_from_largest_mis_complements(G: nx.Graph, mis_list: list[set[Hashable]]) -> BranchResult:
    if not mis_list:
        raise ValueError("Empty MIS list.")
    lmax = max(len(I) for I in mis_list)
    selected = _deduplicate_sets([I for I in mis_list if len(I) == lmax])
    V = set(G.nodes())
    complements = [V - I for I in selected]
    refined = refine_dominating_sets(G, complements)
    best = _best_set(refined)
    return BranchResult(
        branch="largest_mIS_complement",
        best_size=len(best),
        best_set=[str(v) for v in sorted(best, key=str)],
        n_input_samples=len(mis_list),
        n_unique_mis=len(_deduplicate_sets(mis_list)),
        selected_mis_size=lmax,
        n_candidates_refined=len(refined),
    )


def run_two_branch_pipeline(
    G: nx.Graph,
    *,
    raw_quantum_samples: Iterable[Any] | None = None,
    n_classical_samples: int = 1000,
    max_quantum_samples: int | None = None,
    seed: int = 0,
    protocol: str = "clean-v1",
) -> PipelineResult:
    """Run the paper's two-branch IS-to-mDS workflow."""
    if protocol == "paper20260505":
        return _run_paper_pipeline(G, raw_quantum_samples=raw_quantum_samples,
            n_classical_samples=n_classical_samples, max_quantum_samples=max_quantum_samples, seed=seed)
    if protocol != "clean-v1":
        raise ValueError("Unknown protocol: " + protocol)
    if raw_quantum_samples is None:
        mis_small = sample_classical_mis(G, n_classical_samples, seed=seed)
        mis_large = list(mis_small)
    else:
        raw = list(raw_quantum_samples)
        mis_small = repair_quantum_samples_to_mis(
            G, raw, mode="small", max_samples=max_quantum_samples, seed=seed
        )
        mis_large = repair_quantum_samples_to_mis(
            G, raw, mode="large", max_samples=max_quantum_samples, seed=seed
        )

    b1 = branch_from_smallest_mis(G, mis_small)
    b2 = branch_from_largest_mis_complements(G, mis_large)
    best = b1 if b1.best_size <= b2.best_size else b2
    return PipelineResult(
        mIS_branch=b1,
        complement_branch=b2,
        best_branch=best.branch,
        best_size=best.best_size,
        best_set=best.best_set,
    )


# Historical Figure-3 protocol. This is NOT the iterative clean-v1 variant.
# Source: statistical_utils.py -> mDS_solvers.py, functions
# generate_mIS_list(), refine_DS_solution(), repair_nodeset_to_targeted_maximal_is().
# Historical refinement uses graph insertion order for equal degrees and one
# (not repeated-until-convergence) 2-neighbourhood sweep. No lookup of target
# cardinalities or historical results is performed here.
def _paper_refine(G, D0):
    D = set(D0)
    if not nx.is_dominating_set(G, D):
        raise ValueError("Input to historical DS refinement is not dominating")
    for v, _ in sorted(G.degree(), key=lambda item: item[1]):
        if v in D and nx.is_dominating_set(G, D - {v}):
            D.remove(v)
    for v in G.nodes():
        if v not in D:
            adjacent = set(G.neighbors(v)) & D
            if len(adjacent) == 2:
                trial = (D - adjacent) | {v}
                if nx.is_dominating_set(G, trial):
                    D = trial
    assert nx.is_dominating_set(G, D)
    return D


def _run_paper_pipeline(G, *, raw_quantum_samples=None,
        n_classical_samples=500, max_quantum_samples=None, seed=0):
    if not G or any(G.degree(v) == 0 for v in G):
        raise ValueError("The paper two-branch benchmark requires nonempty graphs without isolated vertices")
    if raw_quantum_samples is None:
        # Historical generate_mIS_list(): seed=i for shot i, NOT a shuffled
        # random-greedy ordering, and NOT the seed schedule used in Figure 4.
        small = [set(nx.maximal_independent_set(G, seed=seed+i))
                 for i in range(n_classical_samples)]
        large = small
    else:
        raw = list(raw_quantum_samples)
        if max_quantum_samples is not None:
            raw = raw[:max_quantum_samples]
        small = repair_quantum_samples_to_mis(G, raw, mode="small", seed=seed)
        large = repair_quantum_samples_to_mis(G, raw, mode="large", seed=seed)
    if not small or not large:
        raise ValueError("No samples supplied")
    def branch(pool, complement):
        optimum = (max if complement else min)(map(len, pool))
        chosen = _deduplicate_sets([I for I in pool if len(I) == optimum])
        candidates = [_paper_refine(G, set(G)-I if complement else I) for I in chosen]
        # Cardinalities are invariant under how ties between final sets are
        # printed. Preserve the historical natural ordering when possible.
        best = min(candidates, key=lambda D: (len(D), sorted(D)))
        return BranchResult(
            branch="largest_mIS_complement" if complement else "smallest_mIS",
            best_size=len(best), best_set=[str(v) for v in sorted(best)],
            n_input_samples=len(pool), n_unique_mis=len(_deduplicate_sets(pool)),
            selected_mis_size=optimum, n_candidates_refined=len(chosen))
    b1, b2 = branch(small, False), branch(large, True)
    best = b1 if b1.best_size <= b2.best_size else b2
    return PipelineResult(b1, b2, best.branch, best.best_size, best.best_set)


def exact_mds_verified(G, *, solver="cplex", time_limit_sec=None, audit_dir=None):
    """Recompute the ILP optimum; return a feasible set and a solver certificate.

    HiGHS is an independent re-solve, NOT the historical CPLEX execution.
    Never substitute an archived optimum or accept an uncertified incumbent.
    """
    import math
    nodes = list(G)
    if not nodes:
        raise ValueError("Empty graph")
    if solver == "highs":
        import numpy as np
        import scipy
        from scipy.optimize import milp, Bounds, LinearConstraint
        A = nx.to_numpy_array(G, nodelist=nodes, dtype=float) + np.eye(len(nodes))
        options = {"mip_rel_gap": 0.0, "disp": False}
        if time_limit_sec is not None:
            options["time_limit"] = float(time_limit_sec)
        result = milp(c=np.ones(len(nodes)), integrality=np.ones(len(nodes)),
            bounds=Bounds(np.zeros(len(nodes)), np.ones(len(nodes))),
            constraints=LinearConstraint(A, np.ones(len(nodes)), np.full(len(nodes), np.inf)),
            options=options)
        if result.status != 0 or result.x is None:
            raise RuntimeError("Uncertified MDS solve: " + str(result.message))
        D = {v for v, x in zip(nodes, result.x) if x > .5}
        bound = float(result.mip_dual_bound)
        if not nx.is_dominating_set(G, D) or not math.isclose(bound, len(D), abs_tol=1e-6):
            raise RuntimeError("MDS optimality/feasibility check failed")
        return D, {"solver":"scipy.optimize.milp / HiGHS", "scipy_version": scipy.__version__,
            "historical_cplex_run": False, "status":int(result.status), "message":str(result.message),
            "objective": len(D), "best_bound":bound, "relative_gap":float(result.mip_gap),
            "certified_optimal":True, "options":options}
    if solver == "cplex":
        import docplex
        from docplex.mp.model import Model
        with Model(name="minimum_dominating_set_reproduction") as model:
            # The original DOcplex solver did not override CPLEX's
            # MIP tolerances, thread count, or random seed here.
            # Keep the solver's version-dependent defaults.
            if time_limit_sec is not None:
                model.parameters.timelimit = float(time_limit_sec)
            x = model.binary_var_list(len(nodes))
            index = {v:i for i,v in enumerate(nodes)}
            model.minimize(model.sum(x))
            for v in nodes:
                model.add_constraint(x[index[v]] + model.sum(x[index[u]] for u in G[v]) >= 1)
            from tools.cplex_audit import effective_configuration, before_solve, after_solve
            configuration = effective_configuration(model)
            if audit_dir is not None:
                before_solve(model, audit_dir)
            solution = model.solve(log_output=False)
            if audit_dir is not None:
                after_solve(model, solution, audit_dir)
            if solution is None:
                raise RuntimeError("CPLEX returned no solution")
            D = {v for v in nodes if solution.get_value(x[index[v]]) > .5}
            status = str(model.solve_details.status)
            bound = float(model.solve_details.best_bound)
            if ("optimal" not in status.lower() or
                not nx.is_dominating_set(G, D) or
                math.ceil(bound - 1e-7) < len(D)):
                raise RuntimeError("CPLEX has not certified the MDS optimum: " + status)
            return D, {"solver":"CPLEX", "docplex_version":docplex.__version__,
                "cplex_version":model.get_cplex().get_version(),
                "historical_cplex_run": False, "status":status, "objective":len(D),
                "best_bound":bound, "relative_gap":float(model.solve_details.mip_relative_gap),
                "certified_optimal":True,
                "effective_parameters":configuration,
                "options":{"parameter_policy":"CPLEX defaults unless explicitly overridden",
                           "time_limit":time_limit_sec}}
    raise ValueError("solver must be highs or cplex")

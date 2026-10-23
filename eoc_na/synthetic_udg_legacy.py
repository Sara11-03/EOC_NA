"""Legacy-faithful UDG sampler for the mDS paper benchmark."""
from __future__ import annotations
import math
from itertools import combinations
from typing import Any, Tuple, List, Dict
import numpy as np
import networkx as nx


def positions_array(G: nx.Graph) -> Tuple[np.ndarray, List[Any]]:
    nodes = sorted(G.nodes())
    coords = np.array([G.nodes[u]["pos"] for u in nodes], dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Expected node attribute 'pos' to be (x,y).")
    return coords, nodes


def max_radial_distance(G: nx.Graph) -> float:
    coords, _ = positions_array(G)
    return float(np.linalg.norm(coords, axis=1).max())


def min_pairwise_distance(G: nx.Graph) -> float:
    coords, _ = positions_array(G)
    diffs = coords[:, None, :] - coords[None, :, :]
    D = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(D, np.inf)
    return float(D.min())


def scale_graph_positions(G: nx.Graph, s: float) -> nx.Graph:
    """Return a copy with all node positions scaled by s."""
    H = G.copy()
    for u in H.nodes():
        x, y = H.nodes[u]["pos"]
        H.nodes[u]["pos"] = (float(s * x), float(s * y))
    H.graph["rescale_factor_applied"] = float(s)
    return H


def default_placement_radius(r_conn: float, n: int, target_avg_degree: float, *, min_dist: float, max_radius: float, margin: float) -> float:
    R_max = float(max_radius) - float(margin)
    R_place = r_conn * math.sqrt(max(n - 1, 1) / max(target_avg_degree, 1e-6))
    return float(np.clip(R_place, min_dist, R_max))


def verify_udg_exact(G: nx.Graph, *, r_conn: float, tol: float = 1e-9):
    """Verify the UDG definition exactly:"""
    n = G.number_of_nodes()
    pts = np.array([G.nodes[i]["pos"] for i in range(n)], dtype=float)

    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    thr2 = (r_conn + tol) ** 2

    # Build adjacency matrix from G
    A = np.zeros((n, n), dtype=bool)
    for u, v in G.edges():
        A[u, v] = True
        A[v, u] = True
    np.fill_diagonal(A, False)

    # Edge direction: all edges must be within radius
    bad_edges = np.argwhere(A & (d2 > thr2))
    if bad_edges.size > 0:
        u, v = bad_edges[0]
        dist = float(np.sqrt(d2[u, v]))
        raise AssertionError(f"UDG violated: edge ({u},{v}) has dist {dist:.6f} > r_conn={r_conn:.6f}")

    # Non-edge direction: all non-edges must be outside radius
    # Use strict > r_conn - tol
    thr2_nonedge = (r_conn - tol) ** 2
    bad_nonedges = np.argwhere((~A) & (d2 <= thr2_nonedge) & (~np.eye(n, dtype=bool)))
    if bad_nonedges.size > 0:
        u, v = bad_nonedges[0]
        dist = float(np.sqrt(d2[u, v]))
        raise AssertionError(f"UDG violated: non-edge ({u},{v}) has dist {dist:.6f} <= r_conn={r_conn:.6f}")

    return True


def pulser_unit_disk_connected_graph(
    R_ryd,
    n: int,
    radius: float,
    seed: int | None = None,
    *,
    min_dist: float = 5.0,
    max_radius: float = 46.0,
    radius_is_normalized: bool | None = None,
    placement_radius: float | None = None,
    target_avg_degree: float = 2.5,
    require_connected: bool = True,
    min_degree: int = 1,
    max_graph_tries: int = 500000,
    max_tries_per_node: int = 5000,
    margin: float = 0.5,
) -> nx.Graph:

    rng = np.random.default_rng(seed)

    if radius_is_normalized is None:
        radius_is_normalized = (radius <= 1.0)
        
    r_conn = R_ryd

    if r_conn <= min_dist:
        raise ValueError(
            f"Connection radius r_conn={r_conn:.3f}µm <= min_dist={min_dist:.3f}µm. "
            "With a hard minimum separation, you will get (almost) no edges."
        )

    # --- Placement disk ---
    R_max = float(max_radius) - float(margin)
    if R_max <= 0:
        raise ValueError("max_radius must be > margin.")

    if placement_radius is None:
        R_place = r_conn * math.sqrt(max(n - 1, 1) / max(target_avg_degree, 1e-6))
        R_place = float(np.clip(R_place, min_dist, R_max))
    else:
        R_place = float(placement_radius)
        if R_place > R_max:
            raise ValueError(f"placement_radius={R_place} exceeds allowed max_radius-margin={R_max}.")
        if R_place <= 0:
            raise ValueError("placement_radius must be > 0.")

    # --- Helpers ---
    def sample_uniform_in_disk(R: float) -> np.ndarray:
        rho = R * math.sqrt(rng.random())
        theta = 2 * math.pi * rng.random()
        return np.array([rho * math.cos(theta), rho * math.sin(theta)], dtype=float)

    def far_enough(candidate: np.ndarray, pts: np.ndarray) -> bool:
        if pts.size == 0:
            return True
        d2 = np.sum((pts - candidate[None, :]) ** 2, axis=1)
        return np.all(d2 >= (min_dist - 1e-12) ** 2)

    def build_graph(pts: np.ndarray) -> nx.Graph:
        G = nx.Graph()
        for i, (x, y) in enumerate(pts):
            G.add_node(i, pos=(float(x), float(y)))

        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        thr2 = (r_conn + 1e-12) ** 2

        for i in range(n):
            for j in range(i + 1, n):
                if d2[i, j] <= thr2:
                    G.add_edge(i, j)

        G.graph["radius_um"] = float(r_conn)
        G.graph["min_dist_um"] = float(min_dist)
        G.graph["max_radius_um"] = float(max_radius)
        G.graph["placement_radius_um"] = float(R_place)
        G.graph["R_ryd_um"] = float(R_ryd)
        return G

    def graph_ok(G: nx.Graph) -> bool:
        if require_connected and n > 1 and not nx.is_connected(G):
            return False
        if min_degree > 0 and any(d < min_degree for _, d in G.degree()):
            return False
        return True

    # --- Main sampling loop ---
    for attempt in range(1, max_graph_tries + 1):
        
        pts = [sample_uniform_in_disk(R_place)]

        for k in range(1, n):
            placed = False
            pts_arr = np.array(pts, dtype=float)

            radii = np.linalg.norm(pts_arr, axis=1)
            safe = np.where(radii <= (R_place - r_conn))[0]
            anchor_pool = safe if safe.size > 0 else np.arange(len(pts))

            # debug counters for WHY we reject
            rej_boundary = 0
            rej_mindist = 0

            for t in range(1, max_tries_per_node + 1):
                anchor_idx = int(rng.choice(anchor_pool))
                anchor = pts_arr[anchor_idx]

                u = rng.random()
                dist = min_dist + (r_conn - min_dist) * (u ** 0.5)
                ang = 2 * math.pi * rng.random()
                cand = anchor + dist * np.array([math.cos(ang), math.sin(ang)], dtype=float)

                # inside disk?
                if (cand[0] * cand[0] + cand[1] * cand[1]) > (R_place ** 2):
                    rej_boundary += 1
                    continue

                # min distance?
                if not far_enough(cand, pts_arr):
                    rej_mindist += 1
                    continue

                pts.append(cand)
                placed = True
                break

            if not placed:
                break  # restart whole graph attempt

        if len(pts) != n:
            continue

        G = build_graph(np.array(pts, dtype=float))

        if graph_ok(G):
            return G

    raise RuntimeError(
        f"Failed after {max_graph_tries} restarts. "
        "Try increasing target_avg_degree (denser), increasing max_radius, "
        "or decreasing min_dist."
    )


def udg_distance_window(G: nx.Graph):
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0, float("inf"), True

    pts = np.array([G.nodes[i]["pos"] for i in range(n)], dtype=float)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)

    A = np.zeros((n, n), dtype=bool)
    for u, v in G.edges():
        A[u, v] = True
        A[v, u] = True
    np.fill_diagonal(A, False)

    if np.any(A):
        R_min = float(np.sqrt(np.max(d2[A])))
    else:
        R_min = 0.0

    nonA = (~A) & (~np.eye(n, dtype=bool))
    if np.any(nonA):
        R_max = float(np.sqrt(np.min(d2[nonA])))
    else:
        R_max = float("inf")

    return R_min, R_max, (R_min < R_max)


def pulser_params_from_graph(G: nx.Graph, *, interaction_coeff: float):
    R_min, R_max, ok = udg_distance_window(G)
    if not ok:
        raise ValueError(f"UDG distance window invalid: R_min={R_min}, R_max={R_max} (need R_min < R_max).")
    if R_min <= 0:
        raise ValueError(f"Invalid R_min={R_min}; graph has no edges or zero distances.")

    R_opt = math.sqrt(R_min * R_max) if math.isfinite(R_max) else float("inf")
    omega_max = (interaction_coeff / R_opt) ** (1/6) if math.isfinite(R_opt) and R_opt > 0 else 0.0

    delta_f_max = interaction_coeff / (R_min ** 6)
    if math.isfinite(R_max) and R_max > 0:
        delta_f_min = interaction_coeff / (R_max ** 6)
        delta_f = math.sqrt(delta_f_min * delta_f_max)
    else:
        delta_f_min = 0.0
        delta_f = 0.0

    return {
        "R_min_um": float(R_min),
        "R_max_um": float(R_max),
        "R_opt_um": float(R_opt),
        "omega_max_rad_per_us": float(omega_max),
        "delta_f_rad_per_us": float(delta_f),
        "delta_f_min_rad_per_us": float(delta_f_min),
        "delta_f_max_rad_per_us": float(delta_f_max),
    }

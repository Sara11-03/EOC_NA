"""Selected functions from the author-provided graph_ops(1).py."""
import numpy as np
import networkx as nx
import math
from typing import Dict, List, Tuple, Any, Optional

def _is_connected_or_trivial(G):
    n = G.number_of_nodes()
    return (n <= 1) or nx.is_connected(G)

def _labels_for_rep(rep, G_rep, members_map=None, G_source=None):
    """
    Try to gather any useful text labels that might identify the municipality.
    - rep itself
    - attributes on rep
    - attributes on any original member nodes (if members_map + G_source provided)
    """
    labs = set()
    labs.add(str(rep))

    # attrs on rep (if any)
    for k in ("name", "label", "municipality"):
        if k in G_rep.nodes[rep]:
            labs.add(str(G_rep.nodes[rep][k]))

    # attrs on member nodes in original graph
    if members_map is not None and G_source is not None:
        for m in members_map.get(rep, []):
            if m in G_source:
                labs.add(str(m))
                for k in ("name", "label", "municipality"):
                    if k in G_source.nodes[m]:
                        labs.add(str(G_source.nodes[m][k]))
                # sometimes stored as lonlat only; nothing else to add

    return labs

def _is_protected(KEEP_NAMES, rep, G_rep, members_map=None, G_source=None):
    text = " | ".join(s.strip().lower() for s in _labels_for_rep(rep, G_rep, members_map, G_source))
    return any(k in text for k in KEEP_NAMES)

def _try_remove_node_keep_connected(G, node):
    """Return True and remove node if connectivity preserved; else do nothing and return False."""
    if node not in G:
        return False
    if G.number_of_nodes() <= 1:
        return False

    H = G.copy()
    H.remove_node(node)
    if _is_connected_or_trivial(H):
        G.remove_node(node)
        return True
    return False

def remove_k_highest_degree_keep_connected(KEEP_NAMES, G, k, members_map=None, G_source=None):
    removed = []
    for _ in range(k):
        # recompute degrees each iteration (since graph changes)
        degs = sorted(G.degree(), key=lambda x: (-x[1], str(x[0])))

        picked = None
        for n, d in degs:
            if _is_protected(KEEP_NAMES, n, G, members_map, G_source):
                continue
            if _try_remove_node_keep_connected(G, n):
                picked = n
                removed.append(n)
                break

        if picked is None:
            raise RuntimeError(
                f"Could not remove {k} high-degree nodes without disconnecting. "
                f"Removed so far: {list(map(str, removed))}"
            )
    return removed

def remove_k_farthest_keep_connected(KEEP_NAMES, G, k, pos_key="pos", members_map=None, G_source=None):
    removed = []
    for _ in range(k):
        radii = []
        for n in G.nodes():
            if _is_protected(KEEP_NAMES, n, G, members_map, G_source):
                continue
            p = np.asarray(G.nodes[n][pos_key][:2], dtype=float)
            r = float(np.hypot(p[0], p[1]))  # distance from center (after your recenter)
            radii.append((n, r))

        radii.sort(key=lambda x: (-x[1], str(x[0])))

        picked = None
        for n, r in radii:
            if _try_remove_node_keep_connected(G, n):
                picked = n
                removed.append(n)
                break

        if picked is None:
            raise RuntimeError(
                f"Could not remove {k} farthest nodes without disconnecting. "
                f"Removed so far: {list(map(str, removed))}"
            )
    return removed

def pulser_style_center(G: nx.Graph, pos_key: str = "pos", method: str = "mean") -> Tuple[float, float]:
    P = np.array([G.nodes[v][pos_key][:2] for v in G.nodes()], dtype=float)
    if P.size == 0:
        return 0.0, 0.0
    if method == "mean":
        c = P.mean(axis=0)
    elif method == "median":
        c = np.median(P, axis=0)
    else:
        raise ValueError("method must be 'mean' or 'median'")
    return float(c[0]), float(c[1])

def remove_singletons_inplace(G):
    """Remove isolated nodes (degree 0). Returns list of removed nodes."""
    singles = [v for v in list(G.nodes()) if G.degree(v) == 0]
    if singles:
        G.remove_nodes_from(singles)
    return singles

def offenders_after_centering(
    G: nx.Graph,
    *,
    R: float = 46.0,
    pos_key: str = "pos",
    center_method: str = "mean",
    eps: float = 1e-9
) -> Tuple[List[Any], Dict[Any, float], Tuple[float, float]]:
    cx, cy = pulser_style_center(G, pos_key=pos_key, method=center_method)
    dist: Dict[Any, float] = {}
    outside: List[Any] = []
    for v in G.nodes():
        x, y = map(float, G.nodes[v][pos_key][:2])
        d = float(np.hypot(x - cx, y - cy))
        dist[v] = d
        if d > (R - eps):
            outside.append(v)
    outside.sort(key=lambda v: dist[v], reverse=True)
    return outside, dist, (cx, cy)

def clip_to_device_radius(
    G: nx.Graph,
    *,
    R: float = 46.0,
    pos_key: str = "pos",
    center_method: str = "mean",
    eps: float = 1e-9,
    max_remove: int = 10_000
) -> Tuple[nx.Graph, List[Tuple[Any, float]], Tuple[float, float]]:
    Gc = G.copy()
    removed: List[Tuple[Any, float]] = []
    for _ in range(max_remove):
        outside, dist, center = offenders_after_centering(
            Gc, R=R, pos_key=pos_key, center_method=center_method, eps=eps
        )
        if not outside:
            return Gc, removed, center
        worst = outside[0]
        removed.append((worst, dist[worst]))
        Gc.remove_node(worst)
    raise RuntimeError(f"Removed {max_remove} nodes and still have offenders. Something is wrong.")

def pairwise_dist(P: np.ndarray) -> np.ndarray:
    diff = P[:, None, :] - P[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))

def positions_array(G: nx.Graph, pos_key: str = "pos") -> Tuple[List[Any], np.ndarray]:
    nodes = list(G.nodes())
    P = np.array([G.nodes[v][pos_key][:2] for v in nodes], dtype=float)
    return nodes, P

def min_pair_distance(G: nx.Graph, pos_key: str = "pos") -> Tuple[float, Optional[Tuple[Any, Any]]]:
    nodes, P = positions_array(G, pos_key=pos_key)
    if len(nodes) < 2:
        return float("inf"), None
    D = pairwise_dist(P)
    iu = np.triu_indices(len(nodes), 1)
    k = int(np.argmin(D[iu]))
    i = int(iu[0][k]); j = int(iu[1][k])
    return float(D[i, j]), (nodes[i], nodes[j])

def union_find_components(nodes: List[Any], P: np.ndarray, thr: float) -> List[List[Any]]:
    n = len(nodes)
    parent = list(range(n))
    rank = [0] * n

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(P[i] - P[j]) < thr:
                union(i, j)

    comp: Dict[int, List[Any]] = {}
    for i in range(n):
        r = find(i)
        comp.setdefault(r, []).append(nodes[i])

    return list(comp.values())

def choose_medoid(members: List[Any], pos_dict: Dict[Any, np.ndarray]) -> Any:
    if len(members) == 1:
        return members[0]
    P = np.array([pos_dict[v] for v in members], dtype=float)
    D = pairwise_dist(P)
    ecc = D.max(axis=1)
    k = int(np.argmin(ecc))
    return members[k]

def build_atomic_graph_from_km(G_km: nx.Graph, pos_km: Dict[Any, Tuple[float, float]], scale_um_per_km: float) -> nx.Graph:
    G = nx.Graph()
    for v in G_km.nodes():
        attrs = dict(G_km.nodes[v])
        x_km, y_km = pos_km[v]
        attrs["pos"] = [float(x_km * scale_um_per_km), float(y_km * scale_um_per_km)]
        G.add_node(v, **attrs)
    for u, v, ed in G_km.edges(data=True):
        G.add_edge(u, v, **dict(ed))
    return G

def compress_by_clusters(G: nx.Graph, clusters: List[List[Any]], R_udg_um: float, *, strict_udg: bool = False):
    node_to_c: Dict[Any, int] = {}
    for ci, members in enumerate(clusters):
        for v in members:
            node_to_c[v] = ci

    pos = {v: np.array(G.nodes[v]["pos"][:2], float) for v in G.nodes()}
    rep = {ci: choose_medoid(members, pos) for ci, members in enumerate(clusters)}

    def cname_of_cluster(ci: int) -> str:
        members = clusters[ci]
        return str(members[0]) if len(members) == 1 else f"CL_{ci:03d}"

    Gc = nx.Graph()

    for ci, members in enumerate(clusters):
        rnode = rep[ci]
        cname = cname_of_cluster(ci)
        Gc.add_node(
            cname,
            pos=list(map(float, pos[rnode])),
            rep=str(rnode),
            members=list(map(str, members)),
            size=int(len(members)),
        )

    edge_agg: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for u, v, ed in G.edges(data=True):
        cu = node_to_c[u]
        cv = node_to_c[v]
        if cu == cv:
            continue
        a, b = (cu, cv) if cu < cv else (cv, cu)
        duv = float(np.linalg.norm(pos[u] - pos[v]))
        info = edge_agg.get((a, b))
        if info is None:
            info = {"count_orig_edges": 0, "min_member_dist_um": duv}
        info["count_orig_edges"] += 1
        info["min_member_dist_um"] = min(info["min_member_dist_um"], duv)
        edge_agg[(a, b)] = info

    cname_pos = {n: np.array(Gc.nodes[n]["pos"][:2], float) for n in Gc.nodes()}
    for (a, b), info in edge_agg.items():
        na = cname_of_cluster(a)
        nb = cname_of_cluster(b)
        d_rep = float(np.linalg.norm(cname_pos[na] - cname_pos[nb]))
        if strict_udg and d_rep > R_udg_um:
            continue
        Gc.add_edge(na, nb, **info, rep_dist_um=d_rep)

    rep_map = {cname_of_cluster(ci): str(rep[ci]) for ci in range(len(clusters))}
    cluster_members = {int(ci): list(map(str, members)) for ci, members in enumerate(clusters)}
    return Gc, rep_map, cluster_members

def recenter_graph_inplace(G: nx.Graph, center_xy: Tuple[float, float], pos_key: str = "pos") -> None:
    cx, cy = center_xy
    for v in G.nodes():
        x, y = map(float, G.nodes[v][pos_key][:2])
        G.nodes[v][pos_key] = [x - cx, y - cy]

def scale_graph_inplace(G: nx.Graph, s: float, pos_key: str = "pos") -> None:
    for v in G.nodes():
        x, y = map(float, G.nodes[v][pos_key][:2])
        G.nodes[v][pos_key] = [s * x, s * y]

def haversine_km(lon1, lat1, lon2, lat2) -> float:
    """Haversine distance in km."""
    R = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

def build_subset_graph_with_haversine_edges(G_full: nx.Graph, subset_nodes, threshold_km: float) -> nx.Graph:
    """Copy subset nodes (with all attributes) and rebuild edges by Haversine distance <= threshold_km."""
    Gs = nx.Graph()
    # add nodes with attributes
    for n in subset_nodes:
        if n not in G_full:
            raise KeyError(f"Subset node '{n}' not found in full graph. (Check id/pretty mismatch.)")
        Gs.add_node(n, **dict(G_full.nodes[n]))

    nodes = list(Gs.nodes())
    # edges by haversine
    for i in range(len(nodes)):
        ni = nodes[i]
        lon_i = float(Gs.nodes[ni]["lon"])
        lat_i = float(Gs.nodes[ni]["lat"])
        for j in range(i + 1, len(nodes)):
            nj = nodes[j]
            lon_j = float(Gs.nodes[nj]["lon"])
            lat_j = float(Gs.nodes[nj]["lat"])

            d = haversine_km(lon_i, lat_i, lon_j, lat_j)
            if d <= float(threshold_km):
                Gs.add_edge(ni, nj, dist_km=float(d))
    return Gs

"""Embedding geographic unit-disk graphs into neutral-atom registers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class EmbeddingReport:
    n_input_nodes: int
    n_output_atoms: int
    n_edges: int
    scale_um_per_km: float
    rydberg_radius_um: float
    min_pair_distance_um: float | None
    max_radius_um: float
    max_edge_distance_um: float | None
    min_nonedge_distance_um: float | None
    suggested_rydberg_radius_um: float | None
    unit_disk_gap_um: float | None
    n_clusters: int
    n_removed_singletons: int
    constraints_ok: bool
    warnings: list[str]


def local_xy_km_from_lonlat(G: nx.Graph) -> dict[Any, np.ndarray]:
    """Project lon/lat to a local equirectangular plane in kilometers."""
    nodes = list(G.nodes())
    if not nodes:
        raise ValueError("Cannot embed an empty graph.")
    lon0 = float(np.mean([G.nodes[v]["lon"] for v in nodes]))
    lat0 = float(np.mean([G.nodes[v]["lat"] for v in nodes]))
    lat0_rad = math.radians(lat0)
    out: dict[Any, np.ndarray] = {}
    for v in nodes:
        lon = float(G.nodes[v]["lon"])
        lat = float(G.nodes[v]["lat"])
        x = (lon - lon0) * 111.320 * math.cos(lat0_rad)
        y = (lat - lat0) * 110.574
        out[v] = np.array([x, y], dtype=float)
    return out


def center_positions(pos: dict[Any, np.ndarray]) -> dict[Any, np.ndarray]:
    nodes = list(pos)
    arr = np.array([pos[v] for v in nodes], dtype=float)
    center = arr.mean(axis=0)
    return {v: np.array(pos[v], dtype=float) - center for v in nodes}


def max_radius(pos: dict[Any, np.ndarray]) -> float:
    if not pos:
        return 0.0
    return float(max(np.linalg.norm(p) for p in pos.values()))


def pair_distance_stats(pos: dict[Any, np.ndarray]) -> tuple[float | None, float | None]:
    nodes = list(pos)
    if len(nodes) < 2:
        return None, None
    dmin = float("inf")
    dmax = 0.0
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            d = float(np.linalg.norm(pos[u] - pos[v]))
            dmin = min(dmin, d)
            dmax = max(dmax, d)
    return dmin, dmax


def _union_find(nodes: list[Any]) -> tuple[dict[Any, Any], Any, Any]:
    parent = {v: v for v in nodes}

    def find(x: Any) -> Any:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Any, b: Any) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    return parent, find, union


def cluster_close_atoms(
    G: nx.Graph,
    pos_um: dict[Any, np.ndarray],
    *,
    min_distance_um: float,
) -> tuple[nx.Graph, dict[Any, np.ndarray], list[list[Any]]]:
    """Merge connected components of the graph induced by too-close atoms."""
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return G.copy(), dict(pos_um), [[v] for v in nodes]

    _, find, union = _union_find(nodes)
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            d = float(np.linalg.norm(pos_um[u] - pos_um[v]))
            if d < float(min_distance_um):
                union(u, v)

    components: dict[Any, list[Any]] = {}
    for v in nodes:
        components.setdefault(find(v), []).append(v)
    clusters = list(components.values())

    if all(len(c) == 1 for c in clusters):
        return G.copy(), dict(pos_um), clusters

    H = nx.Graph()
    new_pos: dict[Any, np.ndarray] = {}
    for k, members in enumerate(clusters):
        label = "cluster_" + str(k) if len(members) > 1 else str(members[0])
        coords = np.array([pos_um[v] for v in members], dtype=float)
        p = coords.mean(axis=0)
        new_pos[label] = p
        original_members: list[str] = []
        names: list[str] = []
        lons: list[float] = []
        lats: list[float] = []
        for v in members:
            original_members.extend([str(x) for x in G.nodes[v].get("original_members", [v])])
            names.append(str(G.nodes[v].get("name", v)))
            if "lon" in G.nodes[v]:
                lons.append(float(G.nodes[v]["lon"]))
            if "lat" in G.nodes[v]:
                lats.append(float(G.nodes[v]["lat"]))
        H.add_node(
            label,
            original_members=original_members,
            name=" + ".join(names),
            lon=float(np.mean(lons)) if lons else None,
            lat=float(np.mean(lats)) if lats else None,
        )
    return H, new_pos, clusters


def make_atomic_unit_disk_graph(
    template_nodes: nx.Graph,
    pos_um: dict[Any, np.ndarray],
    *,
    rydberg_radius_um: float,
) -> nx.Graph:
    H = nx.Graph()
    nodes = list(pos_um)
    for idx, v in enumerate(nodes):
        attrs = dict(template_nodes.nodes[v]) if v in template_nodes.nodes else {}
        attrs["idx"] = idx
        attrs["pos"] = [float(pos_um[v][0]), float(pos_um[v][1])]
        H.add_node(v, **attrs)

    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            d = float(np.linalg.norm(pos_um[u] - pos_um[v]))
            if d <= float(rydberg_radius_um):
                H.add_edge(u, v, distance_um=d)
    H.graph["rydberg_radius_um"] = float(rydberg_radius_um)
    return H


def remove_singletons(G: nx.Graph) -> tuple[nx.Graph, list[Any]]:
    removed = [v for v, deg in G.degree() if deg == 0]
    H = G.copy()
    H.remove_nodes_from(removed)
    for idx, v in enumerate(H.nodes()):
        H.nodes[v]["idx"] = idx
    return H, removed


def unit_disk_gap_report(G: nx.Graph, *, pos_key: str = "pos") -> tuple[float | None, float | None, float | None, float | None]:
    nodes = list(G.nodes())
    if len(nodes) < 2:
        return None, None, None, None
    edge_dist: list[float] = []
    nonedge_dist: list[float] = []
    edge_set = {frozenset((u, v)) for u, v in G.edges()}
    for i, u in enumerate(nodes):
        pu = np.array(G.nodes[u][pos_key], dtype=float)
        for v in nodes[i + 1 :]:
            pv = np.array(G.nodes[v][pos_key], dtype=float)
            d = float(np.linalg.norm(pu - pv))
            if frozenset((u, v)) in edge_set:
                edge_dist.append(d)
            else:
                nonedge_dist.append(d)
    max_edge = max(edge_dist) if edge_dist else None
    min_nonedge = min(nonedge_dist) if nonedge_dist else None
    if max_edge is not None and min_nonedge is not None:
        gap = min_nonedge - max_edge
        suggested = 0.5 * (max_edge + min_nonedge) if gap > 0 else None
    else:
        gap = None
        suggested = max_edge
    return max_edge, min_nonedge, suggested, gap


def embed_graph_for_device(
    G_geo: nx.Graph,
    *,
    edge_radius_km: float,
    desired_edge_radius_um: float = 10.0,
    device_radius_um: float = 46.0,
    min_atom_distance_um: float = 5.0,
    max_atoms: int | None = 100,
    cluster_close: bool = True,
    drop_singletons: bool = False,
) -> tuple[nx.Graph, EmbeddingReport]:
    """Create a hardware-compatible unit-disk register from a geographic graph."""
    if edge_radius_km <= 0:
        raise ValueError("edge_radius_km must be positive.")

    pos_km = center_positions(local_xy_km_from_lonlat(G_geo))
    scale = float(desired_edge_radius_um) / float(edge_radius_km)
    pos_um = {v: p * scale for v, p in pos_km.items()}

    rmax = max_radius(pos_um)
    if rmax > device_radius_um and rmax > 0:
        shrink = float(device_radius_um) / rmax
        scale *= shrink
        pos_um = {v: p * shrink for v, p in pos_um.items()}

    rydberg_radius_um = float(edge_radius_km) * scale

    n_clusters = G_geo.number_of_nodes()
    if cluster_close:
        template, pos_um, clusters = cluster_close_atoms(G_geo, pos_um, min_distance_um=min_atom_distance_um)
        n_clusters = len(clusters)
        pos_um = center_positions(pos_um)
    else:
        template = G_geo.copy()

    rmax = max_radius(pos_um)
    if rmax > device_radius_um and rmax > 0:
        shrink = float(device_radius_um) / rmax
        scale *= shrink
        rydberg_radius_um *= shrink
        pos_um = {v: p * shrink for v, p in pos_um.items()}

    H = make_atomic_unit_disk_graph(template, pos_um, rydberg_radius_um=rydberg_radius_um)
    removed_singletons: list[Any] = []
    if drop_singletons:
        H, removed_singletons = remove_singletons(H)

    dmin, _ = pair_distance_stats({v: np.array(H.nodes[v]["pos"], dtype=float) for v in H.nodes()})
    max_edge, min_nonedge, suggested_R, gap = unit_disk_gap_report(H)
    warnings: list[str] = []
    if dmin is not None and dmin < min_atom_distance_um - 1e-9:
        warnings.append(f"minimum distance {dmin:.6g} um is below requested {min_atom_distance_um:.6g} um")
    if max_radius({v: np.array(H.nodes[v]["pos"], dtype=float) for v in H.nodes()}) > device_radius_um + 1e-9:
        warnings.append("register exceeds device radius")
    if max_atoms is not None and H.number_of_nodes() > int(max_atoms):
        warnings.append(f"register has {H.number_of_nodes()} atoms, above max_atoms={max_atoms}")
    if gap is not None and gap <= 0:
        warnings.append("no positive unit-disk gap between edges and non-edges")

    H.graph.update(
        {
            "source_graph_nodes": int(G_geo.number_of_nodes()),
            "edge_radius_km": float(edge_radius_km),
            "scale_um_per_km": float(scale),
            "device_radius_um": float(device_radius_um),
            "min_atom_distance_um": float(min_atom_distance_um),
            "rydberg_radius_um": float(rydberg_radius_um),
            "dropped_singletons": [str(x) for x in removed_singletons],
        }
    )

    report = EmbeddingReport(
        n_input_nodes=G_geo.number_of_nodes(),
        n_output_atoms=H.number_of_nodes(),
        n_edges=H.number_of_edges(),
        scale_um_per_km=float(scale),
        rydberg_radius_um=float(rydberg_radius_um),
        min_pair_distance_um=dmin,
        max_radius_um=max_radius({v: np.array(H.nodes[v]["pos"], dtype=float) for v in H.nodes()}),
        max_edge_distance_um=max_edge,
        min_nonedge_distance_um=min_nonedge,
        suggested_rydberg_radius_um=suggested_R,
        unit_disk_gap_um=gap,
        n_clusters=n_clusters,
        n_removed_singletons=len(removed_singletons),
        constraints_ok=(len(warnings) == 0),
        warnings=warnings,
    )
    return H, report

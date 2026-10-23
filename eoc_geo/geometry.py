"""Historical geometry workflow, with explicit stages and validation.

The fixed candidate subsets are author-provided data. This module does not
reconstruct the colleague's upstream selection algorithm or optimize pulses.
"""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
import math
import networkx as nx
import numpy as np
from pyproj import Transformer
from . import legacy_geometry as old
from .io import ROOT, read_json, write_json, save_graph, sha256, environment, write_csv


def load_selected(case: str, selected_path: Path | None = None, coordinates_path: Path | None = None) -> tuple[nx.Graph, dict]:
    path = selected_path or ROOT / "data" / "inputs" / f"{case}_selected_nodes.json"
    data = read_json(path)
    rows = data["nodes"] if isinstance(data, dict) else data
    g = nx.Graph()
    for row in rows:
        row = dict(row)
        name = row.pop("id")
        if name in g:
            raise ValueError(f"Duplicate selected municipality: {name}")
        lon, lat = float(row["lon"]), float(row["lat"])
        if not (math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f"Invalid geographic coordinates for {name}")
        row["lon"], row["lat"] = lon, lat
        g.add_node(name, **row)
    if coordinates_path is not None:
        points = read_json(coordinates_path)
        index = {}
        for v, a in g.nodes(data=True):
            key = (round(a['lon'], 10), round(a['lat'], 10))
            index.setdefault(key, []).append(v)
        chosen = set()
        for lon, lat in points:
            ids = index.get((round(float(lon), 10), round(float(lat), 10)), [])
            if len(ids) != 1:
                raise ValueError(f"Coordinate cannot be matched uniquely in fixed data: {lon}, {lat}")
            if ids[0] in chosen:
                raise ValueError(f"Repeated selected point: {ids[0]}")
            chosen.add(ids[0])
        # Keep the historical graph order, not the order of the email.
        g = g.subgraph([v for v in g if v in chosen]).copy()
    if len(g) < 2:
        raise ValueError("At least two selected positions are required.")
    provenance = {"input_file": path.name, "input_sha256": sha256(path), "initial_selection": "fixed author-supplied subset; upstream algorithm not supplied", "randomness": "none in this pipeline"}
    return g, provenance


def make_geographical_graph(case: str, config: dict, selected_path: Path | None = None, coordinates_path: Path | None = None) -> tuple[nx.Graph, dict]:
    g, report = load_selected(case, selected_path, coordinates_path)
    g = old.build_subset_graph_with_haversine_edges(g, list(g), config['coverage_radius_km'])
    g.graph.update(case=case, coordinates_crs="EPSG:4326", coverage_radius_km=config['coverage_radius_km'], source_selection="archived subset")
    report.update(nodes=len(g), edges=g.number_of_edges(), threshold_km=config['coverage_radius_km'], environment=environment())
    return g, report


def project_centered(g: nx.Graph, epsg: int) -> tuple[dict, tuple[float, float]]:
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    raw = {}
    for v in g:
        x, y = transformer.transform(float(g.nodes[v]['lon']), float(g.nodes[v]['lat']))
        raw[v] = (float(x) / 1000, float(y) / 1000)
    center = (float(np.mean([p[0] for p in raw.values()])), float(np.mean([p[1] for p in raw.values()])))
    return {v: (p[0] - center[0], p[1] - center[1]) for v, p in raw.items()}, center


def geometry_metrics(g: nx.Graph) -> dict:
    nodes, points = old.positions_array(g)
    if len(nodes) < 2 or points.shape != (len(nodes), 2) or not np.isfinite(points).all():
        raise ValueError("Need at least two finite 2D positions.")
    distances = old.pairwise_dist(points)
    edge_dist, nonedge_dist = [], []
    for i, u in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            (edge_dist if g.has_edge(u, nodes[j]) else nonedge_dist).append((float(distances[i, j]), [u, nodes[j]]))
    longest_edge = max(edge_dist, key=lambda a: a[0]) if edge_dist else (None, None)
    shortest_nonedge = min(nonedge_dist, key=lambda a: a[0]) if nonedge_dist else (None, None)
    dmin, pair = old.min_pair_distance(g)
    radii = np.linalg.norm(points, axis=1)
    edge_max, nonedge_min = longest_edge[0], shortest_nonedge[0]
    gap = nonedge_min - edge_max if edge_max is not None and nonedge_min is not None else None
    return {"num_nodes": len(g), "num_edges": g.number_of_edges(), "connected": nx.is_connected(g),
            "min_separation_um": dmin, "closest_pair": list(pair), "max_radius_um": float(max(radii)),
            "farthest_vertex": nodes[int(np.argmax(radii))], "centroid_um": points.mean(axis=0).tolist(),
            "max_edge_length_um": edge_max, "min_nonedge_length_um": nonedge_min,
            "edge_nonedge_gap_um": gap, "suggested_radius_um": (edge_max + nonedge_min) / 2 if gap is not None and gap > 0 else None,
            "max_edge_pair": longest_edge[1], "min_nonedge_pair": shortest_nonedge[1]}


def _pruning_details(g: nx.Graph, removed: list, kind: str) -> list[dict]:
    copy = g.copy()
    out = []
    for v in removed:
        p = copy.nodes[v]['pos']
        row = {"vertex": v, "reason": kind, "degree_at_removal": int(copy.degree[v]), "radius_um": float(np.hypot(*p)), "nodes_before": len(copy)}
        copy.remove_node(v)
        row['connected_after'] = old._is_connected_or_trivial(copy)
        out.append(row)
    return out


def build_register(geographical: nx.Graph, config: dict) -> tuple[nx.Graph, dict, dict[str, nx.Graph]]:
    src = geographical.copy()
    stages = {"01_geographical": src.copy()}
    positions, geo_center = project_centered(src, int(config['projection_epsg']))
    scale = config['initial_radius_um'] / config['coverage_radius_km']
    initial = old.build_atomic_graph_from_km(src, positions, scale)
    stages['02_initial_atomic'] = deepcopy(initial)
    nodes, points = old.positions_array(initial)
    clusters = sorted(old.union_find_components(nodes, points, config['min_separation_um']), key=len, reverse=True)
    compressed, rep_map, members = old.compress_by_clusters(initial, clusters, config['initial_radius_um'], strict_udg=config['strict_edge_filter'])
    stages['03_clustered'] = deepcopy(compressed)
    clipped, removed, center = old.clip_to_device_radius(compressed, R=config['max_radius_um'], eps=config['clipping_epsilon_um'], center_method=config['center_method'])
    stages['04_clipped_uncentered'] = deepcopy(clipped)
    isolates = old.remove_singletons_inplace(clipped) if config['remove_isolates'] else []
    old.recenter_graph_inplace(clipped, center)
    dmin, _ = old.min_pair_distance(clipped)
    if not math.isfinite(dmin) or dmin <= 0:
        raise ValueError(f"Invalid pre-scaling minimum distance: {dmin}")
    factor = config['target_min_separation_um'] / dmin * (1 + config['scaling_guard'])
    old.scale_graph_inplace(clipped, factor)
    before_prune = geometry_metrics(clipped)
    stages['05_rescaled_before_capacity_pruning'] = deepcopy(clipped)
    component_drop = []
    if config['keep_largest_component'] and len(clipped) > 0 and not nx.is_connected(clipped):
        components = sorted(nx.connected_components(clipped), key=len, reverse=True)
        keep = set(components[0])
        component_drop = [v for v in clipped if v not in keep]
        clipped.remove_nodes_from(component_drop)
    keep_names = set(config['historical_protected_substrings'])
    before_deg = deepcopy(clipped)
    deleted_deg = old.remove_k_highest_degree_keep_connected(keep_names, clipped, config['prune_high_degree'], members_map=members, G_source=src)
    before_border = deepcopy(clipped)
    deleted_border = old.remove_k_farthest_keep_connected(keep_names, clipped, config['prune_periphery'], members_map=members, G_source=src)
    # Historical workflow does NOT re-center after the final five Modena deletions.
    final = clipped
    for v, a in final.nodes(data=True):
        representative = a['rep']
        if representative in src:
            copied = dict(src.nodes[representative])
            if 'idx' in copied:
                copied['source_idx'] = copied.pop('idx')
            copied.update(a)
            a.update(copied)
    # Explicit analysis order. Source municipality idx is not a bit index.
    order = sorted(final, key=str)
    for i, v in enumerate(order):
        final.nodes[v]['idx'] = i
        final.nodes[v]['qubit_id'] = f"q{i:03d}"
    # Refresh stale physical distance attributes after scaling.
    for u, v, a in final.edges(data=True):
        length = float(np.linalg.norm(np.asarray(final.nodes[u]['pos']) - np.asarray(final.nodes[v]['pos'])))
        a['distance_um'] = length
        if 'rep_dist_um' in a:
            a['rep_dist_um'] = length
        if 'min_member_dist_um' in a:
            a['min_member_dist_um'] *= factor
        a['dist_km'] = old.haversine_km(final.nodes[u]['lon'], final.nodes[u]['lat'], final.nodes[v]['lon'], final.nodes[v]['lat'])
    metrics = geometry_metrics(final)
    final.graph.update(case=config['case'], projection_epsg=config['projection_epsg'], position_units="um",
                       coverage_radius_km=config['coverage_radius_km'], initial_scale_um_per_km=scale,
                       final_scale_factor=factor, total_scale_um_per_km=scale * factor,
                       rydberg_radius_um=metrics['suggested_radius_um'], bitstring_node_order=order,
                       bit_order_convention="leftmost character is node_order[0]",
                       bit_order_provenance="lexicographic analysis convention; cross-checked for bundled cases by geo_06, not automatically certified for new graphs",
                       recentered_after_capacity_pruning=False)
    if metrics['min_separation_um'] < config['min_separation_um'] - 1e-9:
        raise AssertionError("Final separation violates configured constraint.")
    if metrics['max_radius_um'] > config['max_radius_um'] + 1e-9:
        raise AssertionError("Final register exceeds configured chamber radius.")
    if metrics['suggested_radius_um'] is None:
        raise AssertionError("No strictly separating unit-disk threshold; graph not silently repaired.")
    if len(final) != config['expected_nodes'] or final.number_of_edges() != config['expected_edges']:
        raise AssertionError(f"Different instance than archived reference: {metrics}")
    stages['06_final'] = deepcopy(final)
    initial_edges = {frozenset(e) for e in src.edges()}
    final_edges = {frozenset(e) for e in final.edges()}
    surviving_original_edges = {e for e in initial_edges if e <= set(final)}
    report = {"case": config['case'], "configuration": config, "initial": geometry_metrics(initial),
              "projected_initial_center_km": list(geo_center), "initial_scale_um_per_km": scale,
              "clusters": clusters, "multi_node_clusters": sum(len(c) > 1 for c in clusters),
              "radial_clipping": [{"vertex": v, "radius_at_removal_um": r} for v, r in removed],
              "post_clip_center_in_initial_um": list(center), "isolated_vertices_removed": isolates,
              "minimum_before_second_scaling_um": dmin, "second_scale_factor": factor,
              "total_scale_um_per_km": scale * factor, "before_capacity_pruning": before_prune,
              "component_vertices_removed": component_drop,
              "high_degree_pruning": _pruning_details(before_deg, deleted_deg, 'capacity_pruning_high_degree'),
              "peripheral_pruning": _pruning_details(before_border, deleted_border, 'capacity_pruning_peripheral'),
              "final": metrics, "surviving_haversine_edges_lost": [sorted(e) for e in surviving_original_edges - final_edges],
              "final_edges_not_in_haversine_graph": [sorted(e) for e in final_edges - initial_edges],
              "node_order": order, "environment": environment(),
              "limitations": ["Does not reconstruct upstream candidate selection.", "Reference geometry is reconstructed from archived scripts, not a fresh QPU download.", "Uses EPSG:32632 also for Sicily to preserve historical behavior.", "Historical protected substring 'reggio emilia' does not match 'Reggio nell\u0027Emilia'; retained for parity, and that vertex survives.", "A geometric blockade threshold is not a complete model of the Rydberg dynamics."]}
    return final, report, stages


def save_register_outputs(final: nx.Graph, report: dict, stages: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_graph(final, out_dir / 'register.json')
    write_json(report, out_dir / 'geometry_report.json')
    write_json({'node_order': report['node_order'], 'bitstring_index0': 'left', 'provenance': final.graph['bit_order_provenance']}, out_dir / 'node_order.json')
    rows=[]
    for v in report['node_order']:
        a=final.nodes[v]
        rows.append({'bit_index':a['idx'],'qubit_id':a['qubit_id'],'municipality':v,'lon':a['lon'],'lat':a['lat'],'x_um':a['pos'][0],'y_um':a['pos'][1]})
    write_csv(rows,out_dir/'register_coordinates.csv')
    for name, graph in stages.items():
        save_graph(graph,out_dir/'stages'/f'{name}.json')


def compare_reference(g: nx.Graph, reference: nx.Graph, tolerance_um: float = 1e-8) -> dict:
    nodes_equal = set(g) == set(reference)
    edges_equal = {frozenset(e) for e in g.edges()} == {frozenset(e) for e in reference.edges()}
    error = max((float(np.linalg.norm(np.asarray(g.nodes[v]['pos']) - np.asarray(reference.nodes[v]['pos']))) for v in set(g)&set(reference)), default=float('inf'))
    return {'same_node_set': nodes_equal, 'same_edge_set': edges_equal, 'max_coordinate_error_um': error if math.isfinite(error) else None,
            'tolerance_um': tolerance_um, 'passed': nodes_equal and edges_equal and error <= tolerance_um}

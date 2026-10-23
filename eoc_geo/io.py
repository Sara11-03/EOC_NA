"""Plain JSON I/O, checksums and explicit graph ordering.

No reliance on a NetworkX-version-dependent links/edges default.
"""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(data: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def guard_outputs(paths: list[Path], overwrite: bool = False) -> None:
    exists = [str(p) for p in paths if p.exists()]
    if exists and not overwrite:
        raise FileExistsError("Output already exists; use --overwrite explicitly: " + ", ".join(exists))


def save_graph(graph: nx.Graph, path: str | Path) -> None:
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("This release supports simple undirected graphs only.")
    nodes = [{**dict(a), "id": v} for v, a in graph.nodes(data=True)]
    edges = [{**dict(a), "source": u, "target": v} for u, v, a in graph.edges(data=True)]
    write_json({"directed": False, "multigraph": False, "graph": dict(graph.graph), "nodes": nodes, "edges": edges}, path)


def load_graph(path: str | Path) -> nx.Graph:
    d = read_json(path)
    if not isinstance(d, dict) or "nodes" not in d:
        raise ValueError(f"Not a node-link graph: {path}")
    if d.get("directed") or d.get("multigraph"):
        raise ValueError("Expected a simple undirected graph.")
    if "edges" in d and "links" in d and d["edges"] != d["links"]:
        raise ValueError("Ambiguous JSON: edges and links differ.")
    key = "edges" if "edges" in d else "links" if "links" in d else None
    if key is None:
        raise ValueError("Missing edges/links field (use an empty list for a nodes-only graph).")
    g = nx.Graph()
    g.graph.update(d.get("graph", {}))
    for row in d["nodes"]:
        v = row["id"]
        if v in g:
            raise ValueError(f"Duplicate node identifier: {v!r}")
        g.add_node(v, **{k: v0 for k, v0 in row.items() if k != "id"})
    for row in d[key]:
        u, v = row["source"], row["target"]
        if u not in g or v not in g or u == v:
            raise ValueError(f"Invalid edge: {u!r}, {v!r}")
        if g.has_edge(u, v):
            raise ValueError(f"Duplicate edge: {u!r}, {v!r}")
        g.add_edge(u, v, **{k: val for k, val in row.items() if k not in ("source", "target")})
    return g


def environment() -> dict:
    versions = {}
    for p in ("numpy", "networkx", "pyproj", "shapely", "matplotlib", "docplex", "cplex", "pulser", "pulser-core"):
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": versions}


def case_config(case: str, path: Path | None = None) -> dict:
    return read_json(path or ROOT / "configs" / f"{case}.json")


def node_order_from_file(path: str | Path) -> list:
    d = read_json(path)
    return d if isinstance(d, list) else d["node_order"]


def write_csv(rows: list[dict], path: Path, fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fields:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

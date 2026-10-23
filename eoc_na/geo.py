"""Geographic data handling and graph construction."""

from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx

from .graph_io import save_graph, write_json

OPENPOLIS_COMUNI_URL = (
    "https://raw.githubusercontent.com/openpolis/geojson-italy/master/comuni.geojson"
)


@dataclass(frozen=True)
class Municipality:
    node_id: str
    name: str
    lon: float
    lat: float
    properties: dict[str, Any]


def normalize_text(x: Any) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    s = re.sub(r"[^a-z0-9àèéìòóùç\s_-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def download_file(url: str, out_path: str | Path, *, overwrite: bool = False) -> Path:
    out_path = Path(out_path)
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        out_path.write_bytes(response.read())
    return out_path


def _iter_coordinates(geometry: dict[str, Any]) -> Iterable[tuple[float, float]]:
    """Yield all lon/lat coordinate pairs from any GeoJSON geometry."""
    if not geometry:
        return
    gtype = geometry.get("type")
    coords = geometry.get("coordinates")

    def walk(obj: Any) -> Iterable[tuple[float, float]]:
        if isinstance(obj, (list, tuple)) and len(obj) >= 2:
            if isinstance(obj[0], (int, float)) and isinstance(obj[1], (int, float)):
                yield (float(obj[0]), float(obj[1]))
            else:
                for item in obj:
                    yield from walk(item)

    if gtype in {"Point", "MultiPoint", "LineString", "MultiLineString", "Polygon", "MultiPolygon"}:
        yield from walk(coords)


def representative_lonlat(feature: dict[str, Any]) -> tuple[float, float]:
    """Return a representative lon/lat point for a municipality geometry."""
    try:
        from shapely.geometry import shape  # type: ignore

        geom = shape(feature["geometry"])
        pt = geom.representative_point()
        return float(pt.x), float(pt.y)
    except Exception:
        pts = list(_iter_coordinates(feature.get("geometry", {})))
        if not pts:
            raise ValueError("Feature has no readable coordinates.")
        lon = sum(p[0] for p in pts) / len(pts)
        lat = sum(p[1] for p in pts) / len(pts)
        return float(lon), float(lat)


def _prop(props: dict[str, Any], aliases: Sequence[str], default: str = "") -> str:
    lower = {str(k).lower(): v for k, v in props.items()}
    for key in aliases:
        if key.lower() in lower and lower[key.lower()] not in (None, ""):
            return str(lower[key.lower()])
    return default


def load_municipalities(
    geojson_path: str | Path,
    *,
    name_field: str | None = None,
    province_field: str | None = None,
    region_field: str | None = None,
) -> list[Municipality]:
    path = Path(geojson_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    municipalities: list[Municipality] = []

    name_aliases = [name_field] if name_field else [
        "name", "nome", "comune", "denominazione", "denominazione_ita", "com_name", "comune_nome"
    ]
    province_aliases = [province_field] if province_field else [
        "prov_name", "provincia", "provincia_nome", "prov_acr", "sigla", "sigla_provincia"
    ]
    region_aliases = [region_field] if region_field else [
        "reg_name", "regione", "regione_nome", "region", "nome_reg"
    ]

    for i, feature in enumerate(features):
        props = dict(feature.get("properties") or {})
        name = _prop(props, name_aliases, default=f"municipality_{i}")
        province = _prop(props, province_aliases, default="")
        region = _prop(props, region_aliases, default="")
        lon, lat = representative_lonlat(feature)
        node_id = f"{name} ({province})" if province else name
        props.setdefault("name", name)
        props.setdefault("province", province)
        props.setdefault("region", region)
        municipalities.append(Municipality(node_id=node_id, name=name, lon=lon, lat=lat, properties=props))

    return municipalities


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in kilometers."""
    radius_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(a))


def find_municipality(municipalities: Sequence[Municipality], name: str) -> Municipality:
    target = normalize_text(name)
    exact = [m for m in municipalities if normalize_text(m.name) == target or normalize_text(m.node_id) == target]
    if exact:
        return exact[0]
    contains = [m for m in municipalities if target in normalize_text(m.name) or target in normalize_text(m.node_id)]
    if contains:
        return contains[0]
    raise ValueError(f"Cannot find municipality matching {name!r}.")


def select_region(
    municipalities: Sequence[Municipality],
    *,
    center_name: str | None = None,
    center_lon: float | None = None,
    center_lat: float | None = None,
    radius_km: float | None = None,
    provinces: Sequence[str] | None = None,
    regions: Sequence[str] | None = None,
    max_nodes: int | None = None,
) -> list[Municipality]:
    selected = list(municipalities)

    if provinces:
        wanted = {normalize_text(p) for p in provinces}
        selected = [m for m in selected if normalize_text(m.properties.get("province", "")) in wanted]

    if regions:
        wanted = {normalize_text(r) for r in regions}
        selected = [m for m in selected if normalize_text(m.properties.get("region", "")) in wanted]

    if center_name is not None:
        center = find_municipality(municipalities, center_name)
        center_lon = center.lon
        center_lat = center.lat

    if radius_km is not None:
        if center_lon is None or center_lat is None:
            raise ValueError("radius_km requires either center_name or center_lon/center_lat.")
        rows = [
            (haversine_km(float(center_lon), float(center_lat), m.lon, m.lat), m)
            for m in selected
        ]
        rows = [row for row in rows if row[0] <= float(radius_km)]
        rows.sort(key=lambda row: (row[0], normalize_text(row[1].name)))
        if max_nodes is not None:
            rows = rows[: int(max_nodes)]
        selected = [m for _, m in rows]
    elif max_nodes is not None:
        selected = selected[: int(max_nodes)]

    if not selected:
        raise ValueError("The selection produced no municipalities.")
    return selected


def build_coverage_graph(municipalities: Sequence[Municipality], edge_radius_km: float) -> nx.Graph:
    """Build the EOC coverage graph by thresholding Haversine distances."""
    G = nx.Graph(edge_radius_km=float(edge_radius_km))
    for idx, m in enumerate(municipalities):
        G.add_node(
            m.node_id,
            idx=idx,
            name=m.name,
            lon=float(m.lon),
            lat=float(m.lat),
            original_members=[m.node_id],
            **{f"geo_{k}": v for k, v in m.properties.items() if isinstance(k, str)},
        )

    nodes = list(G.nodes())
    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            duv = haversine_km(G.nodes[u]["lon"], G.nodes[u]["lat"], G.nodes[v]["lon"], G.nodes[v]["lat"])
            if duv <= float(edge_radius_km):
                G.add_edge(u, v, distance_km=float(duv))
    return G


def build_and_save_region_graph(
    geojson_path: str | Path,
    out_graph: str | Path,
    out_selection: str | Path,
    *,
    edge_radius_km: float,
    center_name: str | None = None,
    center_lon: float | None = None,
    center_lat: float | None = None,
    radius_km: float | None = None,
    provinces: Sequence[str] | None = None,
    regions: Sequence[str] | None = None,
    max_nodes: int | None = None,
) -> nx.Graph:
    municipalities = load_municipalities(geojson_path)
    selected = select_region(
        municipalities,
        center_name=center_name,
        center_lon=center_lon,
        center_lat=center_lat,
        radius_km=radius_km,
        provinces=provinces,
        regions=regions,
        max_nodes=max_nodes,
    )
    G = build_coverage_graph(selected, edge_radius_km=edge_radius_km)
    save_graph(G, out_graph)
    write_json(
        {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "edge_radius_km": float(edge_radius_km),
            "selected_nodes": [m.node_id for m in selected],
        },
        out_selection,
    )
    return G

#!/usr/bin/env python3
"""Build a municipality coverage graph from GeoJSON coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow execution as `python scripts/...py` without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eoc_na.geo import build_and_save_region_graph


def _csv_or_none(x: str | None) -> list[str] | None:
    if not x:
        return None
    return [p.strip() for p in x.split(",") if p.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--geojson", type=Path, default=Path("data/raw/comuni.geojson"))
    p.add_argument("--out-graph", type=Path, required=True)
    p.add_argument("--out-selection", type=Path, default=None)
    p.add_argument("--edge-radius-km", type=float, default=13.0)
    p.add_argument("--center-name", default=None)
    p.add_argument("--center-lon", type=float, default=None)
    p.add_argument("--center-lat", type=float, default=None)
    p.add_argument("--radius-km", type=float, default=None)
    p.add_argument("--provinces", default=None, help="Comma-separated province names/acronyms, if available in GeoJSON properties.")
    p.add_argument("--regions", default=None, help="Comma-separated region names, if available in GeoJSON properties.")
    p.add_argument("--max-nodes", type=int, default=None)
    args = p.parse_args()

    out_selection = args.out_selection or args.out_graph.with_suffix(".selection.json")
    G = build_and_save_region_graph(
        args.geojson,
        args.out_graph,
        out_selection,
        edge_radius_km=args.edge_radius_km,
        center_name=args.center_name,
        center_lon=args.center_lon,
        center_lat=args.center_lat,
        radius_km=args.radius_km,
        provinces=_csv_or_none(args.provinces),
        regions=_csv_or_none(args.regions),
        max_nodes=args.max_nodes,
    )
    print(f"Saved {args.out_graph}")
    print(f"n_nodes={G.number_of_nodes()} n_edges={G.number_of_edges()} edge_radius_km={args.edge_radius_km}")


if __name__ == "__main__":
    main()

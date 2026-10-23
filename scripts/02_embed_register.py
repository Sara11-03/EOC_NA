#!/usr/bin/env python3
"""Embed a geographic coverage graph into a neutral-atom register."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

# Allow execution as `python scripts/...py` without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eoc_na.embedding import embed_graph_for_device
from eoc_na.graph_io import load_graph, save_graph, write_json
from eoc_na.plotting import plot_register


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--out-register", type=Path, required=True)
    p.add_argument("--out-report", type=Path, default=None)
    p.add_argument("--out-plot", type=Path, default=None)
    p.add_argument("--edge-radius-km", type=float, default=None)
    p.add_argument("--desired-edge-radius-um", type=float, default=10.0)
    p.add_argument("--device-radius-um", type=float, default=46.0)
    p.add_argument("--min-atom-distance-um", type=float, default=5.0)
    p.add_argument("--max-atoms", type=int, default=100)
    p.add_argument("--no-clustering", action="store_true")
    p.add_argument("--drop-singletons", action="store_true")
    args = p.parse_args()

    G = load_graph(args.graph)
    edge_radius_km = args.edge_radius_km or float(G.graph.get("edge_radius_km", 13.0))
    H, report = embed_graph_for_device(
        G,
        edge_radius_km=edge_radius_km,
        desired_edge_radius_um=args.desired_edge_radius_um,
        device_radius_um=args.device_radius_um,
        min_atom_distance_um=args.min_atom_distance_um,
        max_atoms=args.max_atoms,
        cluster_close=(not args.no_clustering),
        drop_singletons=args.drop_singletons,
    )
    save_graph(H, args.out_register)
    out_report = args.out_report or args.out_register.with_suffix(".embedding_report.json")
    write_json(asdict(report), out_report)
    if args.out_plot:
        plot_register(H, args.out_plot)
    print(f"Saved register: {args.out_register}")
    print(f"Saved report:   {out_report}")
    print(f"n_atoms={H.number_of_nodes()} n_edges={H.number_of_edges()} constraints_ok={report.constraints_ok}")
    if report.warnings:
        print("Warnings:")
        for w in report.warnings:
            print(f"  - {w}")


if __name__ == "__main__":
    main()

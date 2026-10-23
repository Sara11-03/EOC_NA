#!/usr/bin/env python3
"""Analyze raw bitstring samples: IS fraction, size distribution, violations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow execution as `python scripts/...py` without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eoc_na.graph_io import load_graph, write_json
from eoc_na.plotting import plot_sample_size_distribution
from eoc_na.samples import edge_violation_frequencies, load_samples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register", type=Path, required=True)
    p.add_argument("--samples", type=Path, required=True, help="samples.json list or counts.json dictionary")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    G = load_graph(args.register)
    samples = load_samples(args.samples)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats = plot_sample_size_distribution(G, samples, args.out_dir / "sample_size_distribution.pdf")
    write_json(stats, args.out_dir / "sample_statistics.json")
    write_json(edge_violation_frequencies(G, samples), args.out_dir / "edge_violation_frequencies.json")
    print(f"Saved analysis in {args.out_dir}")
    print(f"n_samples={stats['n_samples_valid_length']} IS_fraction={stats['independent_fraction']:.4f}")


if __name__ == "__main__":
    main()

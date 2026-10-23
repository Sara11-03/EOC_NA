#!/usr/bin/env python3
"""Post-process classical or quantum samples into candidate mDS solutions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow execution as `python scripts/...py` without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eoc_na.graph_io import load_graph, write_json
from eoc_na.mds import exact_mds_docplex, greedy_dominating_set, run_two_branch_pipeline
from eoc_na.samples import load_samples


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register", type=Path, required=True)
    p.add_argument("--samples", type=Path, default=None, help="Optional quantum samples/counts. If omitted, use classical random mIS sampler.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-classical-samples", type=int, default=1000)
    p.add_argument("--max-quantum-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--protocol", choices=["clean-v1", "paper20260505"], default="paper20260505",
                   help="paper20260505 reproduces the synthetic historical workflow; clean-v1 is the earlier refactored variant.")
    p.add_argument("--exact", action="store_true", help="Also try exact mDS with DOcplex/CPLEX.")
    args = p.parse_args()

    G = load_graph(args.register)
    raw_samples = load_samples(args.samples) if args.samples is not None else None
    result = run_two_branch_pipeline(
        G,
        raw_quantum_samples=raw_samples,
        n_classical_samples=args.n_classical_samples,
        max_quantum_samples=args.max_quantum_samples,
        seed=args.seed,
        protocol=args.protocol,
    ).to_dict()

    greedy = greedy_dominating_set(G)
    result["greedy_baseline"] = {"size": len(greedy), "set": [str(v) for v in sorted(greedy, key=str)]}

    if args.exact:
        D, status = exact_mds_docplex(G)
        result["exact_docplex"] = {"status": status, "size": len(D), "set": [str(v) for v in sorted(D, key=str)]}

    write_json(result, args.out)
    print(f"Saved {args.out}")
    print(f"best_branch={result['best_branch']} best_size={result['best_size']}")


if __name__ == "__main__":
    main()

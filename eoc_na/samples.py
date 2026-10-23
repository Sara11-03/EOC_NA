"""Quantum/classical sample parsing and IS diagnostics."""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path
from typing import Any, Hashable, Iterable

import networkx as nx

from .graph_io import ordered_nodes


def load_samples(path: str | Path) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        out: list[str] = []
        for bitstring, count in data.items():
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError("Counts must have nonnegative integer multiplicities")
            out.extend([str(bitstring)] * int(count))
        return out
    raise ValueError("Samples file must contain either a list of bitstrings or a counts dictionary.")


def save_counts_and_samples(samples: Iterable[str], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = [str(s) for s in samples]
    (out_dir / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    counts = Counter(samples)
    (out_dir / "counts.json").write_text(json.dumps(dict(counts), indent=2, sort_keys=True), encoding="utf-8")


def normalize_bitstring(x: Any, n: int) -> str | None:
    s = str(x).strip()
    if not s:
        return None
    clean = "".join(ch for ch in s if ch in "01")
    return clean if len(clean) == n else None


def parse_sample_to_nodeset(sample: Any, nodes: list[Hashable]) -> set[Hashable]:
    """Convert a bitstring or an explicit node list into a node set."""
    if isinstance(sample, (list, tuple, set)):
        return set(sample)

    if isinstance(sample, str):
        s = sample.strip()
        clean = normalize_bitstring(s, len(nodes))
        if clean is not None:
            return {nodes[i] for i, bit in enumerate(clean) if bit == "1"}
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple, set)):
                return set(obj)
    raise ValueError(f"Unsupported sample format: {sample!r}")


def nodeset_to_bitstring(G: nx.Graph, S: Iterable[Hashable]) -> str:
    S = set(S)
    return "".join("1" if v in S else "0" for v in ordered_nodes(G))


def is_independent_set(G: nx.Graph, S: Iterable[Hashable]) -> bool:
    S = set(S)
    return all(not (u in S and v in S) for u, v in G.edges())


def conflict_stats(G: nx.Graph, S: Iterable[Hashable]) -> tuple[int, int]:
    """Return (number of violating edges, number of conflicted nodes)."""
    S = set(S)
    bad_edges = 0
    bad_nodes: set[Hashable] = set()
    for u, v in G.edges():
        if u in S and v in S:
            bad_edges += 1
            bad_nodes.add(u)
            bad_nodes.add(v)
    return bad_edges, len(bad_nodes)


def is_maximal_independent_set(G: nx.Graph, S: Iterable[Hashable]) -> bool:
    S = set(S)
    if not is_independent_set(G, S):
        return False
    for v in G.nodes():
        if v not in S and all(u not in S for u in G.neighbors(v)):
            return False
    return True


def analyze_samples_by_size(G: nx.Graph, raw_samples: Iterable[Any]) -> dict[str, Any]:
    nodes = ordered_nodes(G)
    all_counts: Counter[int] = Counter()
    is_counts: Counter[int] = Counter()
    notis_counts: Counter[int] = Counter()
    violation_counts: Counter[int] = Counter()
    n_bad_length = 0
    n_total = 0
    sum_viol_edges_non_is = 0
    n_non_is = 0

    for raw in raw_samples:
        bitstring = normalize_bitstring(raw, len(nodes))
        if bitstring is None:
            n_bad_length += 1
            continue
        S = {nodes[i] for i, bit in enumerate(bitstring) if bit == "1"}
        size = len(S)
        n_total += 1
        all_counts[size] += 1
        n_viol_edges, _ = conflict_stats(G, S)
        if n_viol_edges == 0:
            is_counts[size] += 1
        else:
            notis_counts[size] += 1
            violation_counts[n_viol_edges] += 1
            sum_viol_edges_non_is += n_viol_edges
            n_non_is += 1

    sizes = sorted(all_counts)
    return {
        "n_samples_valid_length": n_total,
        "n_samples_bad_length": n_bad_length,
        "n_independent": int(sum(is_counts.values())),
        "n_not_independent": int(sum(notis_counts.values())),
        "independent_fraction": float(sum(is_counts.values()) / n_total) if n_total else 0.0,
        "mean_violating_edges_given_not_is": float(sum_viol_edges_non_is / n_non_is) if n_non_is else 0.0,
        "size_counts_all": {str(k): int(all_counts[k]) for k in sizes},
        "size_counts_is": {str(k): int(is_counts[k]) for k in sorted(is_counts)},
        "size_counts_not_is": {str(k): int(notis_counts[k]) for k in sorted(notis_counts)},
        "violation_edge_counts": {str(k): int(v) for k, v in sorted(violation_counts.items())},
    }


def edge_violation_frequencies(G: nx.Graph, raw_samples: Iterable[Any]) -> list[dict[str, Any]]:
    nodes = ordered_nodes(G)
    counts: Counter[tuple[Hashable, Hashable]] = Counter()
    n_valid = 0
    for raw in raw_samples:
        bitstring = normalize_bitstring(raw, len(nodes))
        if bitstring is None:
            continue
        n_valid += 1
        S = {nodes[i] for i, bit in enumerate(bitstring) if bit == "1"}
        for u, v in G.edges():
            if u in S and v in S:
                counts[(u, v)] += 1
    rows = []
    for u, v in G.edges():
        pu = G.nodes[u].get("pos")
        pv = G.nodes[v].get("pos")
        dist = None
        if pu is not None and pv is not None:
            dx = float(pu[0]) - float(pv[0])
            dy = float(pu[1]) - float(pv[1])
            dist = (dx * dx + dy * dy) ** 0.5
        c = counts[(u, v)]
        rows.append(
            {
                "u": str(u),
                "v": str(v),
                "violation_count": int(c),
                "violation_frequency": float(c / n_valid) if n_valid else 0.0,
                "distance_um": dist,
            }
        )
    return rows

"""Small plotting utilities used by the scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx

from .samples import analyze_samples_by_size


def plot_register(G: nx.Graph, out_path: str | Path, *, highlight: set[Any] | None = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = {v: (float(G.nodes[v]["pos"][0]), float(G.nodes[v]["pos"][1])) for v in G.nodes()}
    highlight = set() if highlight is None else set(highlight)
    node_colors = ["tab:red" if v in highlight else "tab:blue" for v in G.nodes()]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    nx.draw_networkx_edges(G, pos=pos, ax=ax, alpha=0.5, width=0.8)
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_size=45, node_color=node_colors, edgecolors="black", linewidths=0.3)
    R = float(G.graph.get("device_radius_um", 46.0))
    circle = plt.Circle((0, 0), R, fill=False, linestyle="--", linewidth=1.0)
    ax.add_patch(circle)
    ax.set_xlabel("x (um)")
    ax.set_ylabel("y (um)")
    ax.set_title(f"{G.number_of_nodes()} atoms, {G.number_of_edges()} edges")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_sample_size_distribution(G: nx.Graph, samples: list[str], out_path: str | Path) -> dict[str, Any]:
    stats = analyze_samples_by_size(G, samples)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_sizes = sorted({int(k) for k in stats["size_counts_all"].keys()})
    is_counts = [stats["size_counts_is"].get(str(k), 0) for k in all_sizes]
    not_counts = [stats["size_counts_not_is"].get(str(k), 0) for k in all_sizes]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(all_sizes, is_counts, label="IS")
    ax.bar(all_sizes, not_counts, bottom=is_counts, label="not-IS")
    ax.set_xlabel("|S|")
    ax.set_ylabel("counts")
    ax.legend(frameon=False)
    ax.set_title(f"IS fraction = {stats['independent_fraction']:.3f}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return stats

"""Synthetic unit-disk benchmark utilities."""

from __future__ import annotations

import csv
import json
import math
import pickle
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np

from .graph_io import load_graph, save_graph, write_json
from .mds import exact_mds_docplex, greedy_dominating_set, run_two_branch_pipeline
from .samples import analyze_samples_by_size, load_samples


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    lower: int = 10
    upper: int = 100
    step: int = 5
    instances_per_size: int = 30
    rydberg_radius_um: float = 7.186760677748386
    device_radius_um: float = 46.0
    min_atom_distance_um: float = 5.0
    seed: int = 12345


@dataclass(frozen=True)
class GraphStats:
    n_nodes: int
    n_edges: int
    min_degree: int
    max_degree: int
    mean_degree: float
    is_connected: bool
    min_pair_distance_um: float | None
    max_radius_um: float | None


@dataclass(frozen=True)
class BenchmarkCopyReport:
    source: str
    destination: str
    copied_graphs: bool
    n_sizes: int
    n_instances: int
    copied_optional_items: list[str]
    skipped_optional_items: list[str]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json_safe(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def sample_points_in_disk(
    n: int,
    *,
    device_radius_um: float,
    min_distance_um: float,
    rng: np.random.Generator,
    max_attempts: int = 200_000,
) -> np.ndarray:
    """Sample ``n`` points in a disk with a hard minimum separation."""
    points: list[tuple[float, float]] = []
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        r = device_radius_um * math.sqrt(float(rng.random()))
        theta = 2.0 * math.pi * float(rng.random())
        p = np.array([r * math.cos(theta), r * math.sin(theta)], dtype=float)
        if not points:
            points.append((float(p[0]), float(p[1])))
            continue
        arr = np.array(points, dtype=float)
        dists = np.linalg.norm(arr - p, axis=1)
        if float(np.min(dists)) >= min_distance_um:
            points.append((float(p[0]), float(p[1])))
    if len(points) != n:
        raise RuntimeError(
            f"Could not sample {n} points with min distance {min_distance_um} um "
            f"inside radius {device_radius_um} um after {max_attempts} attempts."
        )
    return np.array(points, dtype=float)




def sample_connected_points_in_disk(
    n: int,
    *,
    edge_radius_um: float,
    device_radius_um: float,
    min_distance_um: float,
    rng: np.random.Generator,
    max_restarts: int = 2_000,
    attempts_per_node: int = 2_000,
) -> np.ndarray:
    """Sample points so that every new atom is connected to an older one."""
    if min_distance_um >= edge_radius_um:
        raise ValueError("min_distance_um must be smaller than edge_radius_um")

    for _restart in range(max_restarts):
        points: list[tuple[float, float]] = [(0.0, 0.0)]
        failed = False
        while len(points) < n:
            placed = False
            arr = np.array(points, dtype=float)
            # Prefer lower-degree expansion by choosing parents uniformly.
            for _ in range(attempts_per_node):
                parent = arr[int(rng.integers(0, len(points)))]
                radius = float(rng.uniform(1.02 * min_distance_um, 0.98 * edge_radius_um))
                theta = float(rng.uniform(0.0, 2.0 * math.pi))
                p = parent + radius * np.array([math.cos(theta), math.sin(theta)])
                if float(np.linalg.norm(p)) > device_radius_um:
                    continue
                dists = np.linalg.norm(arr - p, axis=1)
                if float(np.min(dists)) < min_distance_um:
                    continue
                # Ensure at least one unit-disk edge to previous points.
                if float(np.min(dists)) > edge_radius_um:
                    continue
                points.append((float(p[0]), float(p[1])))
                placed = True
                break
            if not placed:
                failed = True
                break
        if not failed and len(points) == n:
            return np.array(points, dtype=float)
    raise RuntimeError(
        f"Could not construct {n} connected points inside radius {device_radius_um} um."
    )


def build_unit_disk_graph(points: np.ndarray, *, edge_radius_um: float) -> nx.Graph:
    """Build the unit-disk graph induced by Euclidean distances."""
    G = nx.Graph()
    for i, (x, y) in enumerate(np.asarray(points, dtype=float)):
        G.add_node(int(i), idx=int(i), pos=[float(x), float(y)])
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points[i] - points[j]))
            if d <= edge_radius_um:
                G.add_edge(int(i), int(j), distance_um=d)
    return G


def graph_stats(G: nx.Graph) -> GraphStats:
    degrees = [int(d) for _, d in G.degree()]
    positions = [G.nodes[v].get("pos") for v in G.nodes()]
    min_dist = None
    max_radius = None
    if positions and all(p is not None for p in positions):
        arr = np.array(positions, dtype=float)
        radii = np.linalg.norm(arr[:, :2], axis=1)
        max_radius = float(np.max(radii)) if radii.size else None
        if len(arr) >= 2:
            vals = []
            for i in range(len(arr)):
                for j in range(i + 1, len(arr)):
                    vals.append(float(np.linalg.norm(arr[i, :2] - arr[j, :2])))
            min_dist = float(np.min(vals)) if vals else None
    return GraphStats(
        n_nodes=int(G.number_of_nodes()),
        n_edges=int(G.number_of_edges()),
        min_degree=int(min(degrees)) if degrees else 0,
        max_degree=int(max(degrees)) if degrees else 0,
        mean_degree=float(np.mean(degrees)) if degrees else 0.0,
        is_connected=bool(nx.is_connected(G)) if G.number_of_nodes() > 0 else True,
        min_pair_distance_um=min_dist,
        max_radius_um=max_radius,
    )


def generate_feasible_udg(
    n: int,
    *,
    rydberg_radius_um: float,
    device_radius_um: float,
    min_atom_distance_um: float,
    rng: np.random.Generator,
    require_connected: bool = True,
    require_no_singletons: bool = True,
    max_graph_attempts: int = 5_000,
) -> nx.Graph:
    """Generate one hardware-feasible unit-disk graph."""
    last_stats: GraphStats | None = None
    for _ in range(max_graph_attempts):
        points = sample_connected_points_in_disk(
            n,
            edge_radius_um=rydberg_radius_um,
            device_radius_um=device_radius_um,
            min_distance_um=min_atom_distance_um,
            rng=rng,
        )
        G = build_unit_disk_graph(points, edge_radius_um=rydberg_radius_um)
        G.graph.update(
            {
                "benchmark_type": "synthetic_unit_disk",
                "rydberg_radius_um": float(rydberg_radius_um),
                "device_radius_um": float(device_radius_um),
                "min_atom_distance_um": float(min_atom_distance_um),
            }
        )
        st = graph_stats(G)
        last_stats = st
        if require_connected and not st.is_connected:
            continue
        if require_no_singletons and st.min_degree <= 0:
            continue
        return G
    raise RuntimeError(
        f"Could not generate a feasible UDG for n={n}. Last stats: "
        f"{asdict(last_stats) if last_stats else None}"
    )


def save_benchmark_instance(G: nx.Graph, inst_dir: str | Path) -> None:
    """Save one synthetic instance in both pickle and JSON format."""
    inst_dir = Path(inst_dir)
    inst_dir.mkdir(parents=True, exist_ok=True)
    with (inst_dir / "graph_target.pkl").open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    save_graph(G, inst_dir / "graph_target.json")
    write_json_safe(asdict(graph_stats(G)), inst_dir / "graph_stats.json")
    with (inst_dir / "positions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "x_um", "y_um"])
        for v in sorted(G.nodes(), key=lambda u: int(G.nodes[u].get("idx", u))):
            x, y = G.nodes[v]["pos"][:2]
            writer.writerow([v, float(x), float(y)])


def load_benchmark_graph(inst_dir: str | Path) -> nx.Graph:
    """Load a benchmark graph from an instance folder."""
    inst_dir = Path(inst_dir)
    pkl_path = inst_dir / "graph_target.pkl"
    json_path = inst_dir / "graph_target.json"
    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            return pickle.load(f)
    if json_path.exists():
        return load_graph(json_path)
    raise FileNotFoundError(f"No graph_target.pkl or graph_target.json in {inst_dir}")


def generate_udg_benchmark(
    out_dir: str | Path,
    *,
    config: SyntheticBenchmarkConfig = SyntheticBenchmarkConfig(),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate a full synthetic benchmark folder ``nXX/instYYY``."""
    out_dir = Path(out_dir)
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    manifest: dict[str, Any] = {"config": asdict(config), "sizes": {}}
    for n in range(config.lower, config.upper + 1, config.step):
        n_dir = out_dir / f"n{n:02d}"
        n_dir.mkdir(parents=True, exist_ok=True)
        size_rows = []
        for inst in range(config.instances_per_size):
            G = generate_feasible_udg(
                n,
                rydberg_radius_um=config.rydberg_radius_um,
                device_radius_um=config.device_radius_um,
                min_atom_distance_um=config.min_atom_distance_um,
                rng=rng,
            )
            inst_dir = n_dir / f"inst{inst:03d}"
            save_benchmark_instance(G, inst_dir)
            size_rows.append(asdict(graph_stats(G)))
        manifest["sizes"][str(n)] = size_rows
        print(f"[OK] generated n={n}, instances={config.instances_per_size}", flush=True)
    write_json_safe(manifest, out_dir / "benchmark_manifest.json")
    return manifest


def iter_benchmark_instance_dirs(bench_dir: str | Path) -> list[tuple[int, int, Path]]:
    """Return ``[(n, inst, inst_dir), ...]`` for a benchmark folder."""
    bench_dir = Path(bench_dir)
    rows: list[tuple[int, int, Path]] = []
    for n_dir in sorted(bench_dir.glob("n*")):
        if not n_dir.is_dir() or not n_dir.name[1:].isdigit():
            continue
        n = int(n_dir.name[1:])
        for inst_dir in sorted(n_dir.glob("inst*")):
            if not inst_dir.is_dir() or not inst_dir.name[4:].isdigit():
                continue
            rows.append((n, int(inst_dir.name[4:]), inst_dir))
    return rows


def summarize_benchmark(bench_dir: str | Path, out_dir: str | Path | None = None) -> dict[str, Any]:
    """Summarize graph counts and structural statistics."""
    bench_dir = Path(bench_dir)
    rows = []
    by_size: dict[int, list[dict[str, Any]]] = {}
    for n, inst, inst_dir in iter_benchmark_instance_dirs(bench_dir):
        G = load_benchmark_graph(inst_dir)
        st = asdict(graph_stats(G))
        st.update({"n": int(n), "inst": int(inst), "path": str(inst_dir)})
        rows.append(st)
        by_size.setdefault(n, []).append(st)
    summary = {
        "benchmark_dir": str(bench_dir),
        "n_instances_total": len(rows),
        "sizes": {},
    }
    for n, vals in sorted(by_size.items()):
        summary["sizes"][str(n)] = {
            "instances": len(vals),
            "mean_edges": float(np.mean([v["n_edges"] for v in vals])),
            "mean_degree": float(np.mean([v["mean_degree"] for v in vals])),
            "all_connected": bool(all(v["is_connected"] for v in vals)),
            "min_pair_distance_um_min": float(np.min([v["min_pair_distance_um"] for v in vals])),
            "max_radius_um_max": float(np.max([v["max_radius_um"] for v in vals])),
        }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json_safe(summary, out_dir / "synthetic_benchmark_summary.json")
        if rows:
            with (out_dir / "synthetic_benchmark_summary.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
    return summary


def copy_synthetic_benchmark_data(
    *,
    source_scripts_dir: str | Path,
    project_root: str | Path,
    benchmark_name: str = "udg_benchmark_Rryd_7_18_rescaled",
    copy_result_caches: bool = True,
    copy_evaluation_results: bool = True,
    copy_plots: bool = False,
    copy_submissions: bool = False,
    overwrite: bool = False,
) -> BenchmarkCopyReport:
    """Copy the old synthetic benchmark into the clean repository layout.

    Parameters are intentionally explicit because the original scripts used many
    absolute paths and one-off folder names.
    """
    source_scripts_dir = Path(source_scripts_dir)
    project_root = Path(project_root)
    src_bench = source_scripts_dir / benchmark_name
    if not src_bench.exists():
        raise FileNotFoundError(f"Benchmark folder not found: {src_bench}")

    data_root = project_root / "data" / "synthetic"
    dst_bench = data_root / benchmark_name
    if dst_bench.exists() and overwrite:
        shutil.rmtree(dst_bench)
    if not dst_bench.exists():
        shutil.copytree(src_bench, dst_bench)
    copied = [benchmark_name]
    skipped = []

    def copy_dir_or_file(src: Path, dst: Path) -> None:
        if not src.exists():
            skipped.append(src.name)
            return
        if dst.exists() and overwrite:
            if dst.is_dir():
                shutil.rmtree(dst)
            else:
                dst.unlink()
        if dst.exists():
            skipped.append(f"{src.name} (already exists)")
            return
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        copied.append(src.name)

    if copy_result_caches:
        caches_root = data_root / "result_caches"
        for src in sorted(source_scripts_dir.glob("pasqal_results_cache*")):
            copy_dir_or_file(src, caches_root / src.name)
        for src in [source_scripts_dir / "quantum_samples_map.pkl", source_scripts_dir / "job_map.pkl"]:
            copy_dir_or_file(src, caches_root / src.name)

    if copy_evaluation_results:
        eval_root = data_root / "evaluation"
        for name in ["results_not_corr", "results", "exact_mis_checkpoint.pkl"]:
            copy_dir_or_file(source_scripts_dir / name, eval_root / name)

    if copy_plots:
        copy_dir_or_file(source_scripts_dir / "plots_evaluation", data_root / "plots_evaluation")

    if copy_submissions:
        sub_root = data_root / "pasqal_submissions"
        submissions = source_scripts_dir / "pasqal_submissions"
        if submissions.exists():
            for src in sorted(submissions.glob("submit_EMU*")):
                copy_dir_or_file(src, sub_root / src.name)
        else:
            skipped.append("pasqal_submissions")

    summary = summarize_benchmark(dst_bench, out_dir=data_root)
    report = BenchmarkCopyReport(
        source=str(src_bench),
        destination=str(dst_bench),
        copied_graphs=True,
        n_sizes=len(summary.get("sizes", {})),
        n_instances=int(summary.get("n_instances_total", 0)),
        copied_optional_items=copied,
        skipped_optional_items=skipped,
    )
    write_json_safe(asdict(report), data_root / "copy_report.json")
    return report


def _sample_path_from_cache(cache_root: str | Path, n: int, inst: int) -> Path | None:
    cache_root = Path(cache_root)
    for candidate in [
        cache_root / f"n{n:02d}" / f"inst{inst:03d}_samples.json",
        cache_root / f"n{n:02d}" / f"inst{inst:03d}_counts.json",
        cache_root / f"n{n}" / f"inst{inst:03d}_samples.json",
        cache_root / f"n{n}" / f"inst{inst:03d}_counts.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def evaluate_benchmark_from_cache(
    *,
    bench_dir: str | Path,
    out_dir: str | Path,
    cache_root: str | Path | None = None,
    lower: int = 10,
    upper: int = 100,
    step: int = 5,
    instances_per_size: int = 30,
    classical_samples: int = 1000,
    quantum_samples_cap: int | None = None,
    exact_time_limit_sec: float | None = None,
    skip_exact: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate classical and optional quantum two-branch mDS results."""
    bench_dir = Path(bench_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for n in range(lower, upper + 1, step):
        for inst in range(instances_per_size):
            inst_dir = bench_dir / f"n{n:02d}" / f"inst{inst:03d}"
            if not inst_dir.exists():
                continue
            G = load_benchmark_graph(inst_dir)
            row: dict[str, Any] = {
                "n": int(n),
                "inst": int(inst),
                "n_edges": int(G.number_of_edges()),
            }
            if not skip_exact:
                try:
                    D_exact, status = exact_mds_docplex(G, time_limit_sec=exact_time_limit_sec)
                    row["mds_exact_size"] = int(len(D_exact))
                    row["mds_exact_status"] = status
                except Exception as exc:
                    D_greedy = greedy_dominating_set(G)
                    row["mds_exact_size"] = None
                    row["mds_exact_status"] = f"not_available: {type(exc).__name__}: {exc}"
                    row["mds_greedy_size"] = int(len(D_greedy))
            classical = run_two_branch_pipeline(G, n_classical_samples=classical_samples, seed=seed + n * 1000 + inst)
            row["classical_best_size"] = int(classical.best_size)
            row["classical_best_branch"] = classical.best_branch
            row["classical_mis_branch_size"] = int(classical.mIS_branch.best_size)
            row["classical_complement_branch_size"] = int(classical.complement_branch.best_size)

            if cache_root is not None:
                sample_path = _sample_path_from_cache(cache_root, n, inst)
                if sample_path is not None:
                    samples = load_samples(sample_path)
                    stats = analyze_samples_by_size(G, samples)
                    quantum = run_two_branch_pipeline(
                        G,
                        raw_quantum_samples=samples,
                        max_quantum_samples=quantum_samples_cap,
                        seed=seed + 10_000 + n * 1000 + inst,
                    )
                    row["quantum_sample_path"] = str(sample_path)
                    row["quantum_n_samples"] = int(stats["n_samples_valid_length"])
                    row["quantum_independent_fraction"] = float(stats["independent_fraction"])
                    row["quantum_mean_violating_edges_given_not_is"] = float(
                        stats["mean_violating_edges_given_not_is"]
                    )
                    row["quantum_best_size"] = int(quantum.best_size)
                    row["quantum_best_branch"] = quantum.best_branch
                    row["quantum_mis_branch_size"] = int(quantum.mIS_branch.best_size)
                    row["quantum_complement_branch_size"] = int(quantum.complement_branch.best_size)
                else:
                    row["quantum_sample_path"] = None
            records.append(row)
        print(f"[OK] evaluated n={n}", flush=True)

    out = {
        "bench_dir": str(bench_dir),
        "cache_root": str(cache_root) if cache_root is not None else None,
        "lower": lower,
        "upper": upper,
        "step": step,
        "instances_per_size": instances_per_size,
        "records": records,
    }
    write_json_safe(out, out_dir / "synthetic_mds_results.json")
    if records:
        keys = sorted({k for r in records for k in r})
        with (out_dir / "synthetic_mds_results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in records:
                writer.writerow(r)
    return out

"""Pasqal/Pulser integration boundary."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import networkx as nx

from .graph_io import ordered_nodes, write_json


def register_coordinates(G: nx.Graph) -> dict[str, tuple[float, float]]:
    coords: dict[str, tuple[float, float]] = {}
    for v in ordered_nodes(G):
        p = G.nodes[v].get("pos")
        if p is None:
            raise ValueError(f"Node {v!r} has no 'pos' attribute.")
        coords[str(G.nodes[v].get("idx", len(coords)))] = (float(p[0]), float(p[1]))
    return coords


def write_submission_manifest(
    *,
    G: nx.Graph,
    schedule: dict[str, Any],
    out_dir: str | Path,
    backend: str,
    runs: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": backend,
        "runs": int(runs),
        "n_atoms": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "node_order": [str(v) for v in ordered_nodes(G)],
        "coordinates_um_by_index": register_coordinates(G),
        "schedule": schedule,
        "graph_metadata": dict(G.graph),
    }
    if extra:
        manifest.update(extra)
    path = out_dir / "submission_manifest.json"
    write_json(manifest, path)
    return path


def build_pulser_sequence(G: nx.Graph, *, device: Any, schedule: dict[str, Any]) -> Any:
    """Build a fixed three-stage linear-ramp Pulser sequence for a register graph.

    This function requires Pulser at runtime.  It is not imported at module
    import time so the rest of the pipeline remains usable without Pulser.
    """
    try:
        from pulser import Register, Sequence
        from pulser.pulse import Pulse
        from pulser.waveforms import ConstantWaveform, RampWaveform
    except Exception as exc:
        raise RuntimeError("Pulser is required to build/submit a sequence.") from exc

    coords = register_coordinates(G)
    register = Register.from_coordinates(list(coords.values()), prefix="q", center=False)
    sequence = Sequence(register, device)
    sequence.declare_channel("rydberg", "rydberg_global")

    omega = float(schedule["omega_max_rad_per_us"])
    delta_0 = float(schedule["delta_0_rad_per_us"])
    delta_f = float(schedule["delta_f_rad_per_us"])
    t_rise = int(schedule["t_rise_ns"])
    t_sweep = int(schedule["t_sweep_ns"])
    t_fall = int(schedule["t_fall_ns"])

    rise = Pulse(RampWaveform(t_rise, 0.0, omega), ConstantWaveform(t_rise, delta_0), 0.0)
    sweep = Pulse(ConstantWaveform(t_sweep, omega), RampWaveform(t_sweep, delta_0, delta_f), 0.0)
    fall = Pulse(RampWaveform(t_fall, omega, 0.0), ConstantWaveform(t_fall, delta_f), 0.0)
    if min(t_rise,t_sweep,t_fall) <= 0 or omega <= 0:
        raise ValueError("Invalid pulse amplitudes or durations")
    sequence.add(rise, "rydberg")
    sequence.add(sweep, "rydberg")
    sequence.add(fall, "rydberg")
    return sequence


def submit_with_pasqal_cloud(
    *,
    G: nx.Graph,
    schedule: dict[str, Any],
    backend_name: str,
    runs: int,
    username: str | None = None,
    project_id: str | None = None,
    wait: bool = False,
) -> Any:
    """Submit a sequence through pasqal-cloud."""
    try:
        try:
            from pasqal_cloud import PasqalCloudConnection
        except ImportError:
            from pulser_pasqal import PasqalCloud as PasqalCloudConnection
        from pulser.backend import QPUBackend
    except Exception as exc:
        raise RuntimeError("pasqal-cloud and pulser are required for submission.") from exc

    username = username or os.environ.get("PASQAL_USERNAME")
    project_id = project_id or os.environ.get("PASQAL_PROJECT_ID")
    if not username or not project_id:
        raise ValueError("Provide username/project_id or set PASQAL_USERNAME/PASQAL_PROJECT_ID.")

    connection = PasqalCloudConnection(username=username, project_id=project_id, password=os.environ.get("PASQAL_PASSWORD"))
    devices = connection.fetch_available_devices()
    device = devices[backend_name]
    sequence = build_pulser_sequence(G, device=device, schedule=schedule)
    backend = QPUBackend(sequence=sequence, connection=connection)
    return backend.run(job_params=[{"runs": int(runs), "variables": {}}], wait=wait)


def save_result_counts(result_obj: Any, out_dir: str | Path) -> Path:
    """Best-effort extraction of bitstring counts from a Pasqal/Pulser result."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] | None = None
    if isinstance(result_obj, dict):
        if "bitstring_counts" in result_obj:
            counts = dict(result_obj["bitstring_counts"])
        elif "counts" in result_obj:
            counts = dict(result_obj["counts"])
    else:
        for attr in ["bitstring_counts", "counts"]:
            if hasattr(result_obj, attr):
                counts = dict(getattr(result_obj, attr))
                break
    if counts is None:
        path = out_dir / "raw_result_repr.txt"
        path.write_text(repr(result_obj), encoding="utf-8")
        return path
    path = out_dir / "counts.json"
    path.write_text(json.dumps(counts, indent=2, sort_keys=True), encoding="utf-8")
    return path

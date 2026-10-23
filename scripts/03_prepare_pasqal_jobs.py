#!/usr/bin/env python3
"""Create a Pasqal submission manifest, and optionally submit a job."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow execution as `python scripts/...py` without an editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eoc_na.graph_io import load_graph, write_json
from eoc_na.pasqal_io import submit_with_pasqal_cloud, write_submission_manifest
from eoc_na.rydberg import make_schedule


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--register", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--backend", default="FRESNEL_CAN1")
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--omega-max", type=float, required=True, help="Omega_max in rad/us. Required for real Pulser sequence submission.")
    p.add_argument("--total-time-ns", type=int, default=3600)
    p.add_argument("--delta-f-multiplier", type=float, default=2.0)
    p.add_argument("--delta-0-multiplier", type=float, default=-6.0)
    p.add_argument("--submit", action="store_true", help="Actually submit through pasqal-cloud instead of dry-run manifest only.")
    p.add_argument("--wait", action="store_true")
    p.add_argument("--username", default=None)
    p.add_argument("--project-id", default=None)
    args = p.parse_args()

    if args.submit and 'EMU' in args.backend.upper():
        p.error('This entrypoint submits QPU jobs. Use 09/11 for remote emulation.')
    G = load_graph(args.register)
    Rb = float(G.graph.get("rydberg_radius_um"))
    schedule = make_schedule(
        rydberg_radius_um=Rb,
        omega_max_rad_per_us=args.omega_max,
        total_time_ns=args.total_time_ns,
        delta_f_multiplier=args.delta_f_multiplier,
        delta_0_multiplier=args.delta_0_multiplier,
    ).to_dict()
    manifest_path = write_submission_manifest(
        G=G,
        schedule=schedule,
        out_dir=args.out_dir,
        backend=args.backend,
        runs=args.runs,
        extra={"submitted": False, "submission_requested": bool(args.submit)},
    )
    print(f"Wrote {manifest_path}")

    if args.submit:
        if args.omega_max is None:
            raise ValueError("--omega-max is required when --submit is used.")
        result = submit_with_pasqal_cloud(
            G=G,
            schedule=schedule,
            backend_name=args.backend,
            runs=args.runs,
            username=args.username,
            project_id=args.project_id,
            wait=args.wait,
        )
        write_json({"result_repr": repr(result)}, args.out_dir / "submission_result.json")
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["submitted"] = True
        if hasattr(result, "batch_id"):
            manifest["batch_id"] = str(result.batch_id)
        write_json(manifest, manifest_path)
        print(f"Submission result saved in {args.out_dir / 'submission_result.json'}")


if __name__ == "__main__":
    main()

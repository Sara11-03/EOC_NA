#!/usr/bin/env python3
"""Reproduce the synthetic-benchmark approximation-ratio boxplot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


RUN_NAME = "run_20260505_113509"

REQUIRED_KEYS = [
    "mDS",
    "notMIS",
    "mDS_from_mIS",
    "notMIS_quantum",
    "mDS_from_mIS_quantum",
]

METHODS = [
    ("notMIS", r"$\rho_{\overline{\mathrm{IS}}}$ classical", "#1f77b4"),
    ("mDS_from_mIS", r"$\rho_{\mathrm{mIS}}$ classical", "#9ecae1"),
    ("notMIS_quantum", r"$\rho_{\overline{\mathrm{IS}}}$ quantum", "#7b2cbf"),
    ("mDS_from_mIS_quantum", r"$\rho_{\mathrm{mIS}}$ quantum", "#ff4da6"),
]


# Path helpers
def default_run_dir(project_root: Path) -> Path:
    return (
        project_root
        / "data"
        / "synthetic_case_studies"
        / "result_caches"
        / "results_not_corr"
        / RUN_NAME
    )


def default_output_dir(project_root: Path) -> Path:
    return (
        project_root
        / "data"
        / "synthetic_case_studies"
        / "plots"
        / "reproduced_boxplot_check"
    )




def resolve_run_dir(project_root: Path, user_run_dir: Optional[str]) -> Path:
    if user_run_dir:
        p = Path(user_run_dir).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"--run-dir does not exist: {p}")

    candidates = [
        default_run_dir(project_root),
    ]

    for p in candidates:
        if p.exists():
            return p

    msg = ["Could not find the synthetic result run directory.", "Tried:"]
    msg += [f"  - {p}" for p in candidates]
    msg += [
        "",
        "Either copy the run cache first, or pass it explicitly, e.g.",
        "  python scripts/06_synthetic_approx_ratio_boxplot.py --run-dir \"outputs/recomputed_fig3\"",
    ]
    raise FileNotFoundError("\n".join(msg))


# Robust cached-result loading
def is_int_like_key(k: Any) -> bool:
    try:
        int(k)
        return True
    except Exception:
        return False


def normalize_res_dict(obj: Any) -> Optional[Dict[int, Dict[str, Any]]]:
    """Return a clean {N: metrics_dict} result dictionary if obj looks right."""
    if not isinstance(obj, dict):
        return None

    # Some containers may store the result dictionary under a wrapper key.
    for wrapper_key in ["res", "results", "result", "data", "res_dict"]:
        if wrapper_key in obj:
            maybe = normalize_res_dict(obj[wrapper_key])
            if maybe is not None:
                return maybe

    if not obj:
        return None

    if not all(is_int_like_key(k) for k in obj.keys()):
        return None

    out: Dict[int, Dict[str, Any]] = {}
    for k, v in obj.items():
        if not isinstance(v, dict):
            return None
        out[int(k)] = v

    # Must contain enough expected keys to be the benchmark result cache.
    n_with_mds = sum("mDS" in v for v in out.values())
    n_with_methods = sum(
        any(method_key in v for method_key, _, _ in METHODS) for v in out.values()
    )
    if n_with_mds == 0 or n_with_methods == 0:
        return None

    return dict(sorted(out.items()))


def try_load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def try_load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def candidate_result_files(run_dir: Path) -> List[Path]:
    preferred_names = [
        "res.pkl",
        "res_dict.pkl",
        "results.pkl",
        "result.pkl",
        "res.pickle",
        "results.pickle",
        "res.json",
        "results.json",
        "result.json",
    ]

    preferred: List[Path] = []
    for name in preferred_names:
        p = run_dir / name
        if p.exists() and p.is_file():
            preferred.append(p)

    all_candidates = sorted(
        [
            p
            for p in run_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {".pkl", ".pickle", ".json"}
        ],
        key=lambda p: (len(p.parts), str(p).lower()),
    )

    # Keep preferred first, then all others without duplicates.
    seen = set()
    out: List[Path] = []
    for p in preferred + all_candidates:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out



def is_number(x: Any) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


ALIASES = {
    "mDS": [
        "mDS", "mds", "mDS_exact", "exact_mDS", "exact_mds", "mDS_size",
        "mds_size", "exact_mDS_size", "exact_mds_size", "docplex_exact",
        "docplex_exact_size", "minimum_dominating_set", "minimum_dominating_set_size",
    ],
    "notMIS": [
        "notMIS", "not_mis", "from_notMIS", "from_notMIS_classical",
        "mDS_from_notMIS", "mDS_from_notMIS_classical", "notMIS_classical",
        "mDS_our_best_notMIS", "rho_notMIS_numerator",
    ],
    "mDS_from_mIS": [
        "mDS_from_mIS", "from_mIS", "from_mIS_classical", "mIS_classical",
        "mDS_from_mIS_classical", "rho_mIS_numerator",
    ],
    "notMIS_quantum": [
        "notMIS_quantum", "not_mis_quantum", "from_notMIS_quantum",
        "mDS_from_notMIS_quantum", "q_notMIS", "notMIS_Q", "mDS_our_best_quantum_notMIS",
    ],
    "mDS_from_mIS_quantum": [
        "mDS_from_mIS_quantum", "from_mIS_quantum", "q_mDS_from_mIS",
        "mIS_quantum", "mDS_from_mIS_Q",
    ],
}


def scalar_from_value(value: Any) -> Optional[float]:
    """Convert a metric value from an instance JSON to a scalar solution size.

    The old cache format can store a solution either as an integer size, a list/set
    of selected nodes, or a small dict containing fields such as size/best_ds_size/nodes.
    """
    if is_number(value):
        return float(value)

    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return None

    if isinstance(value, (list, tuple, set)):
        vals = list(value)
        if not vals:
            return None
        # A singleton numeric list is usually a wrapped scalar.
        if len(vals) == 1 and is_number(vals[0]):
            return float(vals[0])
        # For per-instance JSON files, lists are often node sets.
        return float(len(vals))

    if isinstance(value, dict):
        size_fields = [
            "size", "best_ds_size", "ds_size", "mDS_size", "mds_size",
            "value", "cardinality", "n_nodes", "num_nodes", "length", "len",
        ]
        for field in size_fields:
            if field in value:
                got = scalar_from_value(value[field])
                if got is not None:
                    return got
        node_fields = [
            "nodes", "selected_nodes", "ds", "dominating_set", "solution",
            "best_ds", "mds_nodes", "node_set",
        ]
        for field in node_fields:
            if field in value and isinstance(value[field], (list, tuple, set)):
                return float(len(value[field]))

    return None


def find_exact_key_recursively(obj: Any, key: str) -> Optional[Any]:
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_exact_key_recursively(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_exact_key_recursively(v, key)
            if found is not None:
                return found
    return None


def extract_metric_scalar(instance_data: Dict[str, Any], canonical_key: str) -> Optional[float]:
    # First try aliases exactly, recursively.
    for alias in ALIASES.get(canonical_key, [canonical_key]):
        found = find_exact_key_recursively(instance_data, alias)
        if found is not None:
            scalar = scalar_from_value(found)
            if scalar is not None:
                return scalar

    # Fallback: some files have a list of methods with name/size fields.
    for container_key in ["methods", "results", "solutions", "rows", "records"]:
        container = instance_data.get(container_key)
        if not isinstance(container, list):
            continue
        for item in container:
            if not isinstance(item, dict):
                continue
            name = str(item.get("method", item.get("name", item.get("key", ""))))
            if name in ALIASES.get(canonical_key, [canonical_key]):
                scalar = scalar_from_value(item)
                if scalar is not None:
                    return scalar

    return None


def instance_json_files_for_n(n_dir: Path) -> List[Path]:
    """Return only true per-instance JSON files: inst000.json, inst001.json, ..."""
    import re

    pat = re.compile(r"^inst(\d{3})\.json$")
    return sorted([p for p in n_dir.iterdir() if p.is_file() and pat.match(p.name)])


def load_res_from_instance_json_layout(run_dir: Path) -> Tuple[Optional[Dict[int, Dict[str, Any]]], str]:
    """Build res[N][metric] from a run directory containing nXX/instYYY.json files."""
    import re

    n_pat = re.compile(r"^n(\d+)$")
    n_dirs = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        m = n_pat.match(p.name)
        if m:
            n_dirs.append((int(m.group(1)), p))

    if not n_dirs:
        return None, "no nXX directories found"

    res: Dict[int, Dict[str, List[float]]] = {}
    diagnostics: List[str] = []

    for n, n_dir in sorted(n_dirs):
        files = instance_json_files_for_n(n_dir)
        if not files:
            diagnostics.append(f"n{n:02d}: no instYYY.json files")
            continue

        res[n] = {key: [] for key in REQUIRED_KEYS}
        missing_by_key = {key: 0 for key in REQUIRED_KEYS}

        for path in files:
            try:
                data = try_load_json(path)
            except Exception as exc:
                diagnostics.append(f"{path.name}: cannot read JSON ({type(exc).__name__}: {exc})")
                continue
            if not isinstance(data, dict):
                diagnostics.append(f"{path.name}: JSON root is not a dict")
                continue

            for key in REQUIRED_KEYS:
                scalar = extract_metric_scalar(data, key)
                if scalar is None or not np.isfinite(scalar):
                    missing_by_key[key] += 1
                    scalar = float("nan")
                res[n][key].append(float(scalar))

        counts = {key: len(res[n].get(key, [])) for key in REQUIRED_KEYS}
        diagnostics.append(
            f"n{n:02d}: files={len(files)}, " + ", ".join(f"{k}={v}" for k, v in counts.items())
        )

    # Keep only sizes with at least exact mDS and one method.
    cleaned: Dict[int, Dict[str, Any]] = {}
    for n, metrics in sorted(res.items()):
        if len(metrics.get("mDS", [])) == 0:
            continue
        if not any(len(metrics.get(k, [])) > 0 for k in ["notMIS", "mDS_from_mIS", "notMIS_quantum", "mDS_from_mIS_quantum"]):
            continue
        cleaned[n] = metrics

    if not cleaned:
        return None, "instance layout found, but no usable metrics extracted:\n  " + "\n  ".join(diagnostics[:80])

    return cleaned, "instance-json layout\n  " + "\n  ".join(diagnostics[:80])


def load_res_dict_from_run_dir(run_dir: Path) -> Tuple[Dict[int, Dict[str, Any]], Path]:
    errors: List[str] = []
    candidates = candidate_result_files(run_dir)

    # Prefer paired per-instance records whenever present.
    if any(run_dir.glob("n*/inst000.json")):
        res, diagnostics = load_res_from_instance_json_layout(run_dir)
        if res is None:
            raise ValueError(diagnostics)
        return res, run_dir / "<paired per-instance JSON records>"

    # 1) Compact/aggregate layout: res.pkl, results.json, etc.
    for path in candidates:
        # Avoid treating thousands of per-instance JSON files as aggregate caches.
        if path.name.startswith("inst") and path.suffix.lower() == ".json":
            continue
        try:
            if path.suffix.lower() in {".pkl", ".pickle"}:
                obj = try_load_pickle(path)
            elif path.suffix.lower() == ".json":
                obj = try_load_json(path)
            else:
                continue
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
            continue

        res = normalize_res_dict(obj)
        if res is not None:
            return res, path

    # 2) Then try your current copied layout: n10/inst000.json, ..., n100/inst029.json.
    instance_res, diagnostics = load_res_from_instance_json_layout(run_dir)
    if instance_res is not None:
        marker = run_dir / "<rebuilt from nXX/instYYY.json>"
        print("\n[LOAD] rebuilt result dictionary from per-instance JSON files:")
        print("  " + diagnostics.replace("\n", "\n  "))
        return instance_res, marker

    msg = [
        f"Could not identify a valid synthetic result cache in {run_dir}.",
        "I tried both aggregate files and per-instance nXX/instYYY.json files.",
        "Per-instance diagnostics:",
        "  " + diagnostics.replace("\n", "\n  "),
        "",
        "Scanned aggregate candidate files:",
    ]
    msg += [f"  - {p}" for p in candidates[:50]]
    if len(candidates) > 50:
        msg.append(f"  ... and {len(candidates) - 50} more")
    if errors:
        msg.append("\nSome loading errors:")
        msg += [f"  - {e}" for e in errors[:20]]
    raise RuntimeError("\n".join(msg))


# Plot and summary
def safe_ratio(num_values: Iterable[Any], den_values: Iterable[Any]) -> np.ndarray:
    num = np.array(list(num_values), dtype=float)
    den = np.array(list(den_values), dtype=float)

    if num.size == 0 or den.size == 0:
        return np.array([], dtype=float)

    if num.size != den.size:
        raise ValueError("Unpaired method/exact arrays: refusing to truncate or misalign instances")
    m = num.size

    out = np.full(m, np.nan, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (den != 0)
    out[mask] = num[mask] / den[mask]
    return out[np.isfinite(out)]


def ratio_for(res: Dict[int, Dict[str, Any]], n: int, key: str) -> np.ndarray:
    return safe_ratio(res[n].get(key, []), res[n].get("mDS", []))


def validate_res(res: Dict[int, Dict[str, Any]], strict: bool = False) -> None:
    print("\n[VALIDATE] result dictionary")
    print("  sizes:", sorted(res.keys()))

    missing_messages = []
    for n in sorted(res.keys()):
        metrics = res[n]
        for key in REQUIRED_KEYS:
            if key not in metrics:
                missing_messages.append(f"|V|={n}: missing key {key!r}")
                continue
            try:
                length = len(metrics[key])
            except TypeError:
                length = -1
            if length > 0 and any(not np.isfinite(float(v)) for v in metrics[key]):
                missing_messages.append(f"|V|={n}: non-finite/missing values for {key!r}")
            if length == 0:
                missing_messages.append(f"|V|={n}: empty key {key!r}")

    if missing_messages:
        print("\n[WARNING] Missing/empty fields detected:")
        for msg in missing_messages[:30]:
            print("  -", msg)
        if len(missing_messages) > 30:
            print(f"  ... and {len(missing_messages) - 30} more")
        if strict:
            raise ValueError("Missing/empty required fields; stopping because --strict was used.")

    print("\n[COUNTS] values per size")
    header = ["|V|", "mDS", "notMIS", "mDS_from_mIS", "notMIS_Q", "mIS_Q"]
    print("  " + "  ".join(f"{h:>14}" for h in header))
    for n in sorted(res.keys()):
        metrics = res[n]
        row = [
            n,
            len(metrics.get("mDS", [])),
            len(metrics.get("notMIS", [])),
            len(metrics.get("mDS_from_mIS", [])),
            len(metrics.get("notMIS_quantum", [])),
            len(metrics.get("mDS_from_mIS_quantum", [])),
        ]
        print("  " + "  ".join(f"{str(x):>14}" for x in row))


def write_summary_csv(res: Dict[int, Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "N",
                "method_key",
                "method_label",
                "n_instances",
                "min_rho",
                "q1_rho",
                "median_rho",
                "mean_rho",
                "q3_rho",
                "max_rho",
            ],
        )
        writer.writeheader()

        for n in sorted(res.keys()):
            for key, label, _ in METHODS:
                vals = ratio_for(res, n, key)
                if vals.size == 0:
                    row = {
                        "N": n,
                        "method_key": key,
                        "method_label": label,
                        "n_instances": 0,
                        "min_rho": "",
                        "q1_rho": "",
                        "median_rho": "",
                        "mean_rho": "",
                        "q3_rho": "",
                        "max_rho": "",
                    }
                else:
                    row = {
                        "N": n,
                        "method_key": key,
                        "method_label": label,
                        "n_instances": int(vals.size),
                        "min_rho": float(np.min(vals)),
                        "q1_rho": float(np.quantile(vals, 0.25)),
                        "median_rho": float(np.median(vals)),
                        "mean_rho": float(np.mean(vals)),
                        "q3_rho": float(np.quantile(vals, 0.75)),
                        "max_rho": float(np.max(vals)),
                    }
                writer.writerow(row)


def plot_compact_every10(
    res, output_dir, *, name="udg_pasqal_2_panels", x_step=10, x_max=100,
    figsize=(3.6, 4.4), group_width_factor=0.62, box_width_factor=0.78,
    y_min=None, y_max=None, showfliers=False,
):
    """Original paper plotting layout; only I/O and strict pairing modernized."""
    show_classical = show_quantum = True
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    all_x = sorted(res.keys())

    x = [
        n for n in all_x
        if int(n) % int(x_step) == 0
        and (x_max is None or int(n) <= int(x_max))
    ]

    if not x:
        raise ValueError(
            f"No |V| values found with step {x_step}"
            + ("" if x_max is None else f" and |V| <= {x_max}")
        )

    centers = np.array(x, dtype=float)
    labels = [str(int(d)) for d in x]

    min_gap = float(np.min(np.diff(centers))) if len(centers) > 1 else float(x_step)

    def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        if num.size == 0 or den.size == 0:
            return np.array([np.nan], dtype=float)

        if num.size != den.size:
            raise ValueError("Unpaired method/reference arrays")
        m = num.size
        num_m = num
        den_m = den

        out = np.full(m, np.nan, dtype=float)
        mask = np.isfinite(num_m) & np.isfinite(den_m) & (den_m != 0)
        out[mask] = num_m[mask] / den_m[mask]

        return out

    def ratio_list(d: int, key: str) -> np.ndarray:
        num = np.array(res[d].get(key, []), dtype=float)
        den = np.array(res[d].get("mDS", []), dtype=float)
        return _safe_ratio(num, den)

    C_NOTMIS = "#1f77b4"
    C_MIS    = "#9ecae1"

    Q_NOTMIS = "#7b2cbf"
    Q_MIS    = "#ff4da6"

    classical_series = [
        (
            [ratio_list(d, "notMIS") for d in x],
            C_NOTMIS,
            r"$\rho_{\overline{\mathrm{IS}}}$ classical",
        ),
        (
            [ratio_list(d, "mDS_from_mIS") for d in x],
            C_MIS,
            r"$\rho_{\mathrm{mIS}}$ classical",
        ),
    ]

    quantum_series = [
        (
            [ratio_list(d, "notMIS_quantum") for d in x],
            Q_NOTMIS,
            r"$\rho_{\overline{\mathrm{IS}}}$ quantum",
        ),
        (
            [ratio_list(d, "mDS_from_mIS_quantum") for d in x],
            Q_MIS,
            r"$\rho_{\mathrm{mIS}}$ quantum",
        ),
    ]

    if not show_classical:
        classical_series = []

    if not show_quantum:
        quantum_series = []

    if not classical_series and not quantum_series:
        raise ValueError("Nothing to plot: enable show_classical and/or show_quantum.")

    panels = []
    if classical_series:
        panels.append(classical_series)
    if quantum_series:
        panels.append(quantum_series)

    with plt.rc_context({
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.4,
    }):
        fig, axes = plt.subplots(
            nrows=len(panels),
            ncols=1,
            figsize=figsize,
            sharex=True,
            sharey=True,
        )

        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        group_width = group_width_factor * min_gap
        offsets = np.array([-group_width / 4.0, group_width / 4.0])
        per_series_spacing = float(np.min(np.diff(offsets)))
        box_width = box_width_factor * per_series_spacing

        common_kw = dict(
            patch_artist=True,
            showfliers=showfliers,
            manage_ticks=False,
            medianprops={"color": "yellow", "linewidth": 1.25},
            whiskerprops={"color": "black", "linewidth": 0.75},
            capprops={"color": "black", "linewidth": 0.75},
            boxprops={"edgecolor": "black", "linewidth": 0.75},
        )

        def beautify(ax):
            ax.set_axisbelow(True)
            ax.grid(True, axis="y", linestyle="--", alpha=0.25, linewidth=0.65)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            ax.tick_params(axis="both", which="major", length=3.8, width=0.8)

            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

            ax.axhline(1.0, color="black", linestyle="--", linewidth=0.75)

            ax.set_ylabel("approx. ratio")

        def draw_panel(ax, panel_series):
            beautify(ax)

            handles = []

            for off, (data, color, label) in zip(offsets, panel_series):
                positions = centers + off

                bp = ax.boxplot(
                    data,
                    positions=positions,
                    widths=box_width,
                    **common_kw,
                )

                for b in bp["boxes"]:
                    b.set_facecolor(color)
                    b.set_alpha(0.86)

                handles.append(
                    Patch(
                        facecolor=color,
                        edgecolor="black",
                        label=label,
                    )
                )

            ax.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.16),
                ncol=2,
                frameon=False,
                columnspacing=0.7,
                handlelength=1.2,
                handletextpad=0.4,
                borderaxespad=0.0,
            )

        for ax, panel_series in zip(axes, panels):
            draw_panel(ax, panel_series)

        axes[-1].set_xticks(centers)
        axes[-1].set_xticklabels(labels)
        axes[-1].set_xlabel(r"$|V|$")

        for ax in axes:
            if y_min is not None or y_max is not None:
                ax.set_ylim(bottom=y_min, top=y_max)
        fig.tight_layout(h_pad=1.25)

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        pdf_path = Path(output_dir) / f"mDS_{name}_boxplot_stacked_compact_every10_REPRODUCED.pdf"
        png_path = Path(output_dir) / f"mDS_{name}_boxplot_stacked_compact_every10_REPRODUCED.png"

        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")

        plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")

    return pdf_path, png_path


# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the synthetic benchmark approximation-ratio boxplot."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Clean EOC pipeline root. Default: current working directory.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit results_not_corr/run_... directory. Default: archived Figure-3 per-instance results bundled in the repository.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for plot and summary CSV.",
    )
    parser.add_argument("--x-step", type=int, default=10)
    parser.add_argument("--x-max", type=int, default=100)
    parser.add_argument("--fig-width", type=float, default=3.6)
    parser.add_argument("--fig-height", type=float, default=4.4)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--showfliers", action="store_true", help="Show outlier points in boxplots.")
    parser.add_argument("--strict", action="store_true", help="Fail if expected keys are missing/empty.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    run_dir = resolve_run_dir(project_root, args.run_dir)
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else default_output_dir(project_root)
    output_dir = output_dir.resolve()

    print("[INPUT]")
    print(f"  project_root: {project_root}")
    print(f"  run_dir:      {run_dir}")
    print(f"  output_dir:   {output_dir}")

    res, source_file = load_res_dict_from_run_dir(run_dir)
    print(f"\n[LOAD] result dictionary loaded from:\n  {source_file}")

    validate_res(res, strict=args.strict)

    summary_csv = output_dir / "synthetic_approx_ratio_summary_REPRODUCED.csv"
    write_summary_csv(res, summary_csv)

    pdf_path, png_path = plot_compact_every10(
        res,
        output_dir,
        x_step=args.x_step,
        x_max=args.x_max,
        figsize=(args.fig_width, args.fig_height),
        y_min=args.y_min,
        y_max=args.y_max,
        showfliers=args.showfliers,
    )

    print("\n[OUTPUT]")
    print(f"  summary_csv: {summary_csv}")
    print(f"  pdf:         {pdf_path}")
    print(f"  png:         {png_path}")
    print("\nCompare the generated PDF/PNG with the reference figure.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

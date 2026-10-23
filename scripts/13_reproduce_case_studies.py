#!/usr/bin/env python3
"""Recompute Figures 6-7 and the case-study table from archived raw samples."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eoc_geo.geometry import make_geographical_graph, build_register
from eoc_geo.io import case_config
from eoc_geo.qpu import checked_counts, evaluate_counts, compare_archived
from eoc_na.graph_io import load_graph, write_json, save_graph
from eoc_na.mds import run_two_branch_pipeline, exact_mds_verified


def ordered_integer_graph(graph, order):
    """Preserve the archived bit-to-municipality correspondence."""
    if len(order) != len(graph) or set(order) != set(graph):
        raise ValueError("The bit-to-node mapping does not cover this register")
    index = {v: i for i, v in enumerate(order)}
    result = nx.Graph()
    for v in order:
        attrs = dict(graph.nodes[v]); attrs['idx'] = index[v]
        result.add_node(index[v], **attrs)
    for u, v, attrs in graph.edges(data=True):
        result.add_edge(index[u], index[v], **dict(attrs))
    return result


def read_counts(path, n):
    data = json.loads(path.read_text(encoding="utf8"))
    counts = checked_counts(data["final_counts"], n)
    combined = Counter()
    for job in data['jobs_fetched']:
        combined.update(checked_counts(job['counts'], n))
    if combined != Counter(counts) or sum(counts.values()) != data['final_n_samples']:
        raise ValueError(f"Inconsistent per-job and aggregate counts: {path}")
    samples = [bits for bits, count in counts.items() for _ in range(count)]
    return data, counts, samples


def plot_case(panels, path, case):
    """Plot the original histograms; retain gaps as categorical bins."""
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 8.0))
    for ax, backend in zip(axes, ("emulator", "qpu")):
        stats = panels[backend]
        good = {int(k): v for k, v in stats['size_counts_is'].items()}
        bad = {int(k): v for k, v in stats['size_counts_not'].items()}
        sizes = sorted(set(good) | set(bad)); x = np.arange(len(sizes))
        y = [good.get(k, 0) for k in sizes]
        ax.bar(x, y, label="IS", color="#1B7837", edgecolor="#222222", linewidth=.55)
        ax.bar(x, [bad.get(k, 0) for k in sizes], bottom=y, label="not-IS",
               color="#C0392B", edgecolor="#222222", linewidth=.55)
        ax.set_xticks(x, [str(k) for k in sizes]); ax.tick_params(labelsize=8)
        ax.set_xlabel(r"$|S|$"); ax.set_ylabel("counts")
        ax.set_title("(a) Emulator" if backend == "emulator" else "(b) QPU")
        ax.legend(loc="upper right", frameon=False)
        ax.set_ylim(0, max(good.get(k, 0) + bad.get(k, 0) for k in sizes) * 1.15 + 1)
        ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        severity = {int(k): v for k, v in stats['violating_edge_histogram_on_nonIS'].items()}
        if severity:
            ins = inset_axes(ax, width="30%", height="38%", loc="upper left", borderpad=1)
            keys = sorted(severity)
            ins.bar(np.arange(len(keys)), [severity[k] for k in keys], color="#C0392B", edgecolor="#222222", linewidth=.45)
            ins.set_xticks(np.arange(len(keys)), [str(k) for k in keys]); ins.tick_params(labelsize=6)
            ins.set_xlabel(r"$|E_{\mathrm{viol}}|$", fontsize=9, labelpad=1)
            ins.spines[['top', 'right']].set_visible(False)
            if case == 'modena':
                mean = stats['total_edge_violations'] / stats['notIS_count']
                ins.text(.98, .94, rf"$\mathbb{{E}}[|E_{{\mathrm{{viol}}}}|\mid\neg IS]={mean:.2f}$",
                         transform=ins.transAxes, ha='right', va='top', fontsize=7,
                         bbox=dict(facecolor='white', edgecolor='none', alpha=.8))
    fig.subplots_adjust(hspace=.44, left=.11, right=.98, top=.95, bottom=.07)
    for ext in ('.png', '.pdf'):
        fig.savefig(path.with_suffix(ext), dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', choices=['modena', 'palermo', 'both'], default='both')
    p.add_argument('--out-dir', type=Path, default=ROOT/'outputs/case_studies')
    p.add_argument('--plots-only', action='store_true', help='Recompute raw statistics only; do not produce an MDS table.')
    p.add_argument('--skip-exact', action='store_true', help='Run heuristics without CPLEX; leave the exact column empty.')
    p.add_argument('--time-limit', type=float, default=None)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args(argv)
    if a.time_limit is not None and a.time_limit <= 0:
        p.error('The time limit must be positive')
    if a.out_dir.resolve() == ROOT or (ROOT/'data').resolve() in (a.out_dir.resolve(), *a.out_dir.resolve().parents):
        p.error('Use a separate output directory, not the input data directory')
    if a.out_dir.exists() and any(a.out_dir.iterdir()) and not a.overwrite:
        p.error('Output exists; choose a new directory or add --overwrite')
    a.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []; all_checks = {}
    for case in (['palermo', 'modena'] if a.case == 'both' else [a.case]):
        cfg = case_config(case)
        geo, _ = make_geographical_graph(case, cfg)
        graph, geo_report, _ = build_register(geo, cfg)
        out = a.out_dir/case; out.mkdir(exist_ok=True)
        save_graph(graph, out/'register.json')
        write_json(geo_report, out/'geometry_report.json')
        order = json.loads((ROOT/f'data/qpu/{case}/node_order.json').read_text())['node_order']
        write_json({'node_order': order, 'index0': 'left'}, out/'node_order.json')
        G = ordered_integer_graph(graph, order)
        exact_size = None
        if not a.plots_only and not a.skip_exact:
            D, certificate = exact_mds_verified(G, solver='cplex', time_limit_sec=a.time_limit, audit_dir=out/'cplex')
            exact_size = len(D)
            write_json({'nodes': [order[i] for i in sorted(D)], 'audit': certificate}, out/'exact_mds.json')
        classical = None if a.plots_only else run_two_branch_pipeline(G, n_classical_samples=500, seed=0, protocol='paper20260505')
        if classical is not None:
            write_json(classical.to_dict(), out/'classical.json')
        panels = {}
        for backend in ['emulator', 'qpu']:
            directory = ROOT/f'data/realistic_case_studies/{case}/{backend}'
            source = directory/'fetched_results_summary.json'
            fetched, counts, samples = read_counts(source, len(G))
            if len(samples) != 1000:
                raise ValueError(f'{case}/{backend}: expected 1000 archived samples')
            stats = evaluate_counts(graph, counts, order)
            panels[backend] = stats
            checks = compare_archived(stats, directory/'selected_point_hw_stats.json', directory/'edge_violation_stats.json')
            provenance = {'jobs': [j['job_id'] for j in fetched['jobs_fetched']],
                          'raw_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                          'raw_source': source.relative_to(ROOT).as_posix(),
                          'node_order': order, 'sample_order': 'Histogram expansion, not acquisition chronology.'}
            result = {'stats': stats, 'provenance': provenance, 'comparison': checks}
            row = {'case': case, 'backend': backend, 'shots': len(samples), 'IS_fraction': stats['IS_fraction'],
                   'mean_violations_nonIS': stats['total_edge_violations']/stats['notIS_count'] if stats['notIS_count'] else 0,
                   'exact_mds': exact_size}
            if not a.plots_only:
                Q = run_two_branch_pipeline(G, raw_quantum_samples=samples, max_quantum_samples=1000, seed=0, protocol='paper20260505')
                result['mds'] = Q.to_dict()
                row.update(classical_small=classical.mIS_branch.best_size,
                           classical_complement=classical.complement_branch.best_size,
                           quantum_small=Q.mIS_branch.best_size, quantum_complement=Q.complement_branch.best_size)
                expected = json.loads((directory/'selected_point_mds_results.json').read_text())['mds_sizes']
                checks['heuristic_sizes'] = all([
                    row['classical_small'] == expected['from_mIS_classical'],
                    row['classical_complement'] == expected['from_notMIS_classical'],
                    row['quantum_small'] == expected['from_mIS_quantum'],
                    row['quantum_complement'] == expected['from_notMIS_quantum']])
                if exact_size is not None:
                    checks['exact_size'] = exact_size == expected['docplex_exact']
                checks['passed'] = checks['passed'] and checks['heuristic_sizes'] and checks.get('exact_size', True)
                print(f'{case}/{backend}: IS={stats["IS_count"]}/1000; DS={row["quantum_small"]}/{row["quantum_complement"]}', flush=True)
            else:
                print(f'{case}/{backend}: IS={stats["IS_count"]}/1000', flush=True)
            write_json(result, out/f'{backend}_recomputed.json')
            all_checks[f'{case}/{backend}'] = checks
            rows.append(row)
        plot_case(panels, out/('figure6' if case == 'palermo' else 'figure7'), case)
    with (a.out_dir/'results.csv').open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    if not a.plots_only and a.case == 'both':
        by = {(r['case'], r['backend']): r for r in rows}
        labels = [('Exact MDS', 'exact_mds', 'qpu'), ('Classical, small maximal IS', 'classical_small', 'qpu'),
                  ('Classical, complement', 'classical_complement', 'qpu'), ('Emulation, small maximal IS', 'quantum_small', 'emulator'),
                  ('Emulation, complement', 'quantum_complement', 'emulator'), ('QPU, small maximal IS', 'quantum_small', 'qpu'),
                  ('QPU, complement', 'quantum_complement', 'qpu')]
        tex = ['\\begin{tabular}{lcc}', '\\hline', 'Method & Palermo & Modena \\\\', '\\hline']
        for title, key, backend in labels:
            values = [by[(case, backend)][key] for case in ['palermo', 'modena']]
            tex.append(title + ' & ' + ' & '.join('--' if x is None else str(x) for x in values) + ' \\\\')
        tex.extend(['\\hline', '\\end{tabular}'])
        (a.out_dir/'table_case_studies.tex').write_text('\n'.join(tex)+'\n')
    if a.case == 'both':
        parameter_rows = []
        for case in ['palermo', 'modena']:
            source = ROOT/f'data/parameters/{case}_qpu_submission.json'
            record = json.loads(source.read_text())
            parameter_rows.append({'case':case,'job_id':record['job_id'],
                'blockade_radius_um':record['R_ryd'],'omega_max_rad_per_us':record['Omega_max'],
                'delta_0_rad_per_us':record['delta_0'],'delta_f_rad_per_us':record['delta_f'],
                'rise_ns':record['t_rise'],'sweep_ns':record['t_sweep'],'fall_ns':record['t_fall'],
                'total_ns':sum(record[k] for k in ['t_rise','t_sweep','t_fall']),
                'shots':record['runs']})
        with (a.out_dir/'qpu_parameters.csv').open('w',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(parameter_rows[0]))
            writer.writeheader();writer.writerows(parameter_rows)
    write_json({'checks': all_checks, 'passed': all(x['passed'] for x in all_checks.values()),
                'exact_solver': None if a.plots_only or a.skip_exact else 'CPLEX',
                'paper_table_discrepancy': 'Archived Modena QPU small-maximal-IS branch is 32, not 31; no value is forced.'}, a.out_dir/'verification.json')
    if not all(x['passed'] for x in all_checks.values()):
        raise SystemExit('A comparison failed; inspect verification.json before using these results.')
    print('PASS: raw statistics and computed results agree with the archived analysis.')

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Recompute Figures 4-5 from raw samples and the historical MIS sampler."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eoc_na.graph_io import load_graph, ordered_nodes, write_json
from eoc_na.samples import load_samples
from eoc_na.exact_mis import maximum_independent_set_exact_fast
from eoc_na.paper_plots import plot_is_size_mean_max_lines_shaded_onecolumn, plot_quantum_corrections_side_by_side


def metrics(graph, samples):
    order = ordered_nodes(graph)
    if any(len(z) != len(order) or set(z)-{'0','1'} for z in samples):
        raise ValueError('Invalid raw bitstring')
    bits = np.array([[b == '1' for b in z] for z in samples], dtype=bool)
    idx = {v: i for i, v in enumerate(order)}
    e = np.array([[idx[u],idx[v]] for u,v in graph.edges()], dtype=int)
    violation = (bits[:,e[:,0]] & bits[:,e[:,1]]).sum(axis=1)
    weights = bits.sum(axis=1); valid = weights[violation == 0]
    invalid = violation[violation > 0]
    return {'q_is_mean_size':float(valid.mean()) if len(valid) else None,
            'q_is_max_size':int(valid.max()) if len(valid) else None,
            'q_is_min_size':int(valid.min()) if len(valid) else None,
            'q_is_frac_raw':float((violation == 0).mean()),
            'q_corr_frac_raw_not_independent':float((violation > 0).mean()),
            'q_corr_mean_conflict_edges_raw':float(violation.mean()),
            'q_corr_mean_conflict_edges_given_violation':float(invalid.mean()) if len(invalid) else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench-dir', type=Path, default=ROOT/'data/synthetic_case_studies/udg_benchmark_Rryd_7_18')
    p.add_argument('--samples-dir', type=Path, default=ROOT/'data/synthetic_samples')
    p.add_argument('--out-dir', type=Path, default=ROOT/'outputs/figures45')
    p.add_argument('--classical-samples', type=int, default=500)
    p.add_argument('--base-seed', type=int, default=12345)
    p.add_argument('--mis-time-limit', type=float, default=600)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(argv)
    if a.classical_samples <= 0 or a.mis_time_limit <= 0:p.error('Budgets must be positive')
    if a.out_dir.exists() and any(a.out_dir.iterdir()) and not a.resume:
        p.error('Output exists. Choose another directory or use --resume.')
    if (ROOT/'data').resolve() in (a.out_dir.resolve(), *a.out_dir.resolve().parents):
        p.error('Do not write outputs inside input data.')
    paths = sorted(a.bench_dir.glob('n*/inst*/graph_target.json'), key=lambda x:(int(x.parent.parent.name[1:]),int(x.parent.name[4:])))
    if not paths:p.error('No graph_target.json files')
    a.out_dir.mkdir(parents=True,exist_ok=True);res={};rows=[]
    for gp in paths:
        n = int(gp.parent.parent.name[1:]);inst = int(gp.parent.name[4:])
        sp = a.samples_dir/gp.parent.parent.name/gp.parent.name/'samples.json'
        settings = {'graph':hashlib.sha256(gp.read_bytes()).hexdigest(),'raw':hashlib.sha256(sp.read_bytes()).hexdigest(),
                    'classical_samples':a.classical_samples,'base_seed':a.base_seed,'networkx':nx.__version__}
        signature=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
        out=a.out_dir/gp.parent.parent.name/(gp.parent.name+'.json')
        if a.resume and out.is_file():
            record=json.loads(out.read_text())
            if record.get('signature')!=signature or not record.get('mis_certified_optimal'):
                raise ValueError(f'Changed inputs or missing optimality certificate: {out}')
        else:
            G=load_graph(gp);samples=load_samples(sp)
            if len(samples)!=500:raise ValueError(f'{sp}: expected 500 raw samples')
            record={'n':n,'inst':inst,'signature':signature,'settings':settings,**metrics(G,samples)}
            sizes=[len(nx.maximal_independent_set(G, seed=a.base_seed+1_000_000*n+10_000*inst+s)) for s in range(a.classical_samples)]
            maximum,exact=maximum_independent_set_exact_fast(G,time_limit=a.mis_time_limit)
            if not exact:raise RuntimeError(f'MIS optimum not proved for {gp}; increase --mis-time-limit.')
            if any(u in maximum and v in maximum for u,v in G.edges()):raise AssertionError('Invalid exact IS')
            record.update(c_is_mean_size=float(np.mean(sizes)),c_is_max_size=max(sizes),c_is_min_size=min(sizes),
                          exact_MIS_size=len(maximum),exact_MIS_nodes=sorted(maximum),mis_certified_optimal=exact)
            write_json(record,out)
        for k,v in record.items():
            if k.startswith(('q_', 'c_')) or k=='exact_MIS_size':res.setdefault(n,{}).setdefault(k,[]).append(v)
        rows.append({k:v for k,v in record.items() if not isinstance(v,(dict,list))})
        print(f'[{len(rows)}/{len(paths)}] n={n}, inst={inst}, alpha={record["exact_MIS_size"]}',flush=True)
    with (a.out_dir/'instance_statistics.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    exact_means={n:float(np.mean(d['exact_MIS_size'])) for n,d in res.items()}
    # The historical figure rounded these means to two decimals for N <= 65.
    plotted_means={n:round(value,2) if n<=65 else value for n,value in exact_means.items()}
    plot_is_size_mean_max_lines_shaded_onecolumn(res,name='figure4',dir=str(a.out_dir),exact_mis_mean_from_log=plotted_means)
    plot_quantum_corrections_side_by_side(res,name='figure5',dir=str(a.out_dir))
    write_json({'instances':len(rows),'exact_MIS_mean':exact_means,'plotted_exact_MIS_mean':plotted_means,
                'classical_seed':'base_seed + 1000000*N + 10000*instance + sample_index',
                'note':'Fresh computation. The maximum-IS label corrects the ambiguous original legend.'},a.out_dir/'summary.json')
    print('Saved Figures 4-5 and per-instance statistics.')

if __name__=='__main__':main()

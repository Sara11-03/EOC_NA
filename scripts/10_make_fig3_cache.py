#!/usr/bin/env python3
"""Recompute all Figure-3 metrics from graphs and RAW samples, not from old answers."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import networkx as nx
from eoc_na.graph_io import load_graph, ordered_nodes, write_json
from eoc_na.mds import run_two_branch_pipeline, exact_mds_verified
from eoc_na.samples import load_samples, analyze_samples_by_size


def graph_fingerprint(G):
    order = ordered_nodes(G)
    obj = {"nodes":order, "edges": sorted([sorted(e) for e in G.edges()]),
           "coordinates_um":[[v, *[round(float(x),8) for x in G.nodes[v]['pos']]] for v in order]}
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench-dir', type=Path, required=True)
    p.add_argument('--emu-dir', type=Path, required=True,
        help='nXX/instYYY/samples.json or counts.json; either historical RAW data or new measurements')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--classical-samples', type=int, default=500)
    p.add_argument('--quantum-samples', type=int, default=500)
    p.add_argument('--seed', type=int, default=0, help='Historical classical convention is seed=i, i.e. offset=0')
    p.add_argument('--protocol', choices=['paper20260505','clean-v1'], default='paper20260505')
    p.add_argument('--exact-solver', choices=['cplex','highs'], default='cplex')
    p.add_argument('--time-limit', type=float, default=None, help='Only set a limit when it matches the original experiment; CPLEX defaults otherwise')
    p.add_argument('--skip-missing', action='store_true', help='Explicitly allow a PARTIAL benchmark')
    p.add_argument('--allow-prefix', action='store_true', help='Use the first M samples if more than M are available; NOT recommended for counts-only data')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args(argv)
    if a.resume and a.overwrite:
        p.error('Choose --resume OR --overwrite, not both')
    if min(a.classical_samples, a.quantum_samples) <= 0:
        p.error('Sample budgets must be positive')
    if a.time_limit is not None and a.time_limit <= 0:
        p.error('--time-limit must be positive')
    paths = sorted(a.bench_dir.glob('n*/inst*/graph_target.json'),
        key=lambda q: (int(q.parent.parent.name[1:]), int(q.parent.name[4:])))
    if not paths:
        p.error('No graph_target.json files. Run 07_generate_synthetic_udgs.py first.')
    if a.out_dir.resolve() in [a.bench_dir.resolve(), a.emu_dir.resolve()]:
        p.error('Use a separate output directory; input data must not be overwritten')
    if a.out_dir.exists() and any(a.out_dir.iterdir()) and not (a.resume or a.overwrite):
        p.error('Output already exists. Choose a new directory or use --resume / --overwrite explicitly.')
    entries, missing = [], []
    for gp in paths:
        folder = a.emu_dir / gp.parent.parent.name / gp.parent.name
        sp = folder/'samples.json'
        if not sp.is_file():
            sp = folder/'counts.json'
        if not sp.is_file():
            missing.append(f'{gp.parent.parent.name}/{gp.parent.name}')
        else:
            entries.append((gp, sp))
    if missing and not a.skip_missing:
        p.error(f'{len(missing)} instances have no raw samples (e.g. {missing[:3]}). No metrics computed. Use a complete input or --skip-missing for a clearly labelled partial run.')
    if not entries:
        p.error('No raw samples found')
    if a.overwrite and a.out_dir.exists() and any(a.out_dir.iterdir()):
        import shutil
        marker = a.out_dir / 'manifest.json'
        if not marker.is_file() or json.loads(marker.read_text()).get('historical_results_used_in_calculation') is not False:
            p.error('Refusing to delete a directory not owned by this recomputation script. Choose a new output path.')
        if any(a.out_dir.resolve() in src.resolve().parents for src in [a.bench_dir, a.emu_dir]):
            p.error('Output must not contain an input directory')
        shutil.rmtree(a.out_dir)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    settings = {'protocol':a.protocol, 'classical_samples':a.classical_samples,
        'quantum_samples':a.quantum_samples, 'classical_seed_offset':a.seed,
        'exact_solver':a.exact_solver, 'time_limit_sec':a.time_limit,
        'networkx_version':nx.__version__, 'python_version':platform.python_version(),
        'allow_prefix':a.allow_prefix}
    manifest = {'settings':settings,'expected_graphs':len(paths),'available_samples':len(entries),
                'missing_samples':missing,'completed_instances':0,'complete':False,
                'historical_results_used_in_calculation':False,'records':[]}
    write_json(manifest,a.out_dir/'manifest.json')
    for gp, sp in entries:
        nd, inst = gp.parent.parent.name, gp.parent.name
        G = load_graph(gp)
        order = ordered_nodes(G)
        if list(G) != order:
            # Historical single-pass algorithm depends on insertion order.
            H = nx.Graph(); H.graph.update(G.graph)
            H.add_nodes_from((v,dict(G.nodes[v])) for v in order)
            H.add_edges_from(G.edges(data=True)); G=H
        samples = load_samples(sp)
        if len(samples) < a.quantum_samples:
            raise ValueError(f'{nd}/{inst}: only {len(samples)} samples, requested {a.quantum_samples}')
        if len(samples) > a.quantum_samples and not a.allow_prefix:
            raise ValueError(f'{nd}/{inst}: found {len(samples)} samples, requested {a.quantum_samples}. Specify the intended subset explicitly; do not silently take a sorted-count prefix.')
        raw = samples[:a.quantum_samples]
        if any(len(b) != len(G) or set(b)-{'0','1'} for b in raw):
            raise ValueError(f'{nd}/{inst}: invalid raw bitstring or wrong length')
        fingerprint=graph_fingerprint(G)
        meta_path=sp.parent/'metadata.json'
        metadata=json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        if 'node_order' in metadata and metadata['node_order'] != order:
            raise ValueError(f'{nd}/{inst}: sample node_order differs from graph. Do not guess bit order.')
        if metadata.get('graph_fingerprint_round8_sha256',fingerprint) != fingerprint:
            raise ValueError(f'{nd}/{inst}: samples belong to a different graph geometry/topology')
        source_hash=hashlib.sha256(sp.read_bytes()).hexdigest()
        signature=hashlib.sha256(json.dumps({'settings':settings,'graph':fingerprint,'samples':source_hash},sort_keys=True).encode()).hexdigest()
        op=a.out_dir/nd/(inst+'.json')
        if a.resume and op.is_file():
            previous=json.loads(op.read_text())
            if previous.get('calculation_signature') != signature:
                raise ValueError(f'{op}: resume settings or inputs changed; use a new directory')
            if not previous.get('exact_audit',{}).get('certified_optimal'):
                raise ValueError(f'{op}: no certified optimum in resumed record')
            result=previous
            print(f'[RESUME] {nd}/{inst}',flush=True)
        else:
            t=time.perf_counter()
            D, certificate=exact_mds_verified(G,solver=a.exact_solver,time_limit_sec=a.time_limit,
                audit_dir=a.out_dir/"solver_records"/nd/inst)
            C=run_two_branch_pipeline(G,n_classical_samples=a.classical_samples,seed=a.seed,protocol=a.protocol)
            Q=run_two_branch_pipeline(G,raw_quantum_samples=raw,max_quantum_samples=a.quantum_samples,seed=a.seed,protocol=a.protocol)
            result={'n':len(G),'inst':int(inst[4:]),'num_nodes':len(G),'num_edges':G.number_of_edges(),
                'mDS':len(D),'mDS_from_mIS':C.mIS_branch.best_size,'notMIS':C.complement_branch.best_size,
                'mDS_from_mIS_quantum':Q.mIS_branch.best_size,'notMIS_quantum':Q.complement_branch.best_size,
                'classical_n_samples':a.classical_samples,'quantum_n_samples_used':len(raw),
                'exact_audit':certificate,'exact_nodes':list(sorted(D)),
                'classical_pipeline':C.to_dict(),'quantum_pipeline':Q.to_dict(),
                'raw_statistics':analyze_samples_by_size(G,raw),
                'graph_fingerprint_round8_sha256':fingerprint,'raw_samples_sha256':source_hash,
                'raw_sample_origin':metadata.get('origin','user supplied; provenance in input directory'),
                'source_sample_path':sp.relative_to(a.emu_dir).as_posix(),
                'protocol':a.protocol,'calculation_signature':signature,
                'elapsed_seconds':time.perf_counter()-t,
                'historical_results_used_in_calculation':False}
            if any(result[k] < len(D) for k in ['mDS_from_mIS','notMIS','mDS_from_mIS_quantum','notMIS_quantum']):
                raise AssertionError('Heuristic result smaller than exact optimum')
            write_json(result,op)
            print(f'[OK] {nd}/{inst}: exact={len(D)}, classical={result["mDS_from_mIS"]}/{result["notMIS"]}, quantum={result["mDS_from_mIS_quantum"]}/{result["notMIS_quantum"]}',flush=True)
        manifest['records'].append({'n':len(G),'inst':int(inst[4:]),'relpath':op.relative_to(a.out_dir).as_posix()})
        manifest['completed_instances']+=1
        write_json(manifest,a.out_dir/'manifest.json')
    manifest['complete']=not missing and len(entries)==len(paths)
    write_json(manifest,a.out_dir/'manifest.json')
    print(f'Computed {len(entries)} instances; complete={manifest["complete"]}. Exact denominators solved afresh; no historical solution sizes read.')

if __name__=='__main__':
    main()

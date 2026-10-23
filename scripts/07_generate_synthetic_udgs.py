#!/usr/bin/env python3
"""Generate the paper-style connected, device-feasible UDG synthetic benchmark."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

# Compatible with `python scripts/07_...py` invoked from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eoc_na.synthetic_udg_legacy import (
    default_placement_radius, min_pairwise_distance, max_radial_distance,
    pulser_unit_disk_connected_graph, pulser_params_from_graph, scale_graph_positions,
    udg_distance_window, verify_udg_exact,
)

from eoc_na.graph_io import save_graph

R_RYD = 7.186760677748386
C6 = 868066.9281
PL_SC = (1.0, 1.15, 1.35, 1.6, 1.9, 2.3)


def write_json(obj, p):
    p=Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False),encoding='utf-8')


def hash_geometry(G):
    # Hash numeric geometry/topology, not pickle bytes or versions.
    data={'positions':[[int(v),float(G.nodes[v]['pos'][0]),float(G.nodes[v]['pos'][1])]
                       for v in sorted(G.nodes())],
          'edges':[list(e) for e in sorted(tuple(sorted((int(u),int(v)))) for u,v in G.edges())]}
    b=json.dumps(data,sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(b).hexdigest()


def stat(G):
    d=dict(G.degree())
    Rmin,Rmax,valid=udg_distance_window(G)
    return {'n':len(G),'m':G.number_of_edges(), 'mean_degree':sum(d.values())/len(G),
            'min_degree':min(d.values()),'max_degree':max(d.values()),
            'connected':nx.is_connected(G), 'min_distance_um':min_pairwise_distance(G),
            'max_radial_um':max_radial_distance(G),
            'max_edge_distance_um':Rmin, 'min_nonedge_distance_um':Rmax,
            'udg_gap_um':Rmax-Rmin,'udg_gap_positive':valid,'geometry_sha256':hash_geometry(G)}


def apply_historical_rescaling(G, r_target, min_distance=5.0, chamber=46.0):
    # Historical rescale_summary.csv: no shrinking; enlarge where geometric
    # mean of largest edge / smallest nonedge permits reaching R_ryd.
    r0,r1,ok=udg_distance_window(G)
    if not ok or not math.isfinite(r1):
        raise ValueError('Cannot compute historical scale without valid UDG distance window')
    ropt=math.sqrt(r0*r1)
    starget=r_target/ropt
    s_min=min_distance/min_pairwise_distance(G)
    s_max=chamber/max_radial_distance(G)
    # Original dataset uses no shrink; refuse enlargements that exceed bounds.
    s=max(1.0,starget)
    reasons=[]
    if starget<s_min: reasons.append('min_dist')
    if starget<1.0: reasons.append('no_shrink')
    if s>s_max+1e-12:
        raise RuntimeError(f'Historical enlargement {s:.8f} breaches chamber constraint {s_max:.8f}')
    H=scale_graph_positions(G,s)
    H.graph['historical_uniform_rescale_factor']=s
    return H, {'s_target':starget,'s_min_dist':s_min,'s_max_chamber':s_max,
               's_applied':s,'clamp_reasons':'|'.join(reasons),
               'R_opt_before':ropt,'R_opt_after':math.sqrt(udg_distance_window(H)[0]*udg_distance_window(H)[1]),
               'dmin_before':min_pairwise_distance(G),'dmin_after':min_pairwise_distance(H),
               'max_radial_after':max_radial_distance(H)}


def generate_one(n, i, a):
    base=default_placement_radius(a.R_ryd,n,a.target_avg_degree,min_dist=a.min_dist,
          max_radius=a.max_radius, margin=a.margin)
    radius_cap=a.max_radius-a.margin
    last=None
    for trial in range(a.max_seed_tries):
        seed=int(np.random.SeedSequence([n,i,trial]).generate_state(1,dtype=np.uint32)[0])
        try:
            for scale in PL_SC:
                r_try=min(base*scale,radius_cap)
                try:
                    G=pulser_unit_disk_connected_graph(
                        a.R_ryd,n=n,radius=.2,seed=seed,min_dist=a.min_dist,
                        max_radius=a.max_radius,radius_is_normalized=True,
                        placement_radius=r_try,target_avg_degree=a.target_avg_degree,
                        require_connected=True,min_degree=1,margin=a.margin)
                    break
                except RuntimeError as ex:
                    last=str(ex)
            else:
                continue
            verify_udg_exact(G,r_conn=a.R_ryd)
            if min_pairwise_distance(G)<a.min_dist-1e-9 or max_radial_distance(G)>a.max_radius+1e-9:
                raise ValueError('Geometry constraints not satisfied')
            phys=pulser_params_from_graph(G,interaction_coeff=a.C6)
            if phys['omega_max_rad_per_us']>a.max_amp+1e-9 or phys['delta_f_rad_per_us']>a.max_det+1e-9:
                raise ValueError('Legacy physical parameter checks failed')
            if a.rescale=='historical':
                H,rescale=apply_historical_rescaling(G,a.R_ryd,a.min_dist,a.max_radius)
            else:
                H,rescale=G,{'s_applied':1.0,'clamp_reasons':'none'}
            # Preserve legacy random drawing and node geometry. Explicit idx is
            # added for unambiguous bitstring order in the clean eoc_na pipeline.
            for v in H.nodes():
                H.nodes[v]['idx']=int(v)
            H.graph.update({'rydberg_radius_um':a.R_ryd,'benchmark_type':'connected_growth_udg',
                            'sampling_seed':seed,'sampling_seed_try':trial})
            verify_udg_exact(H,r_conn=a.R_ryd if a.rescale=='none' else a.R_ryd*rescale['s_applied'])
            return H, {'n':n,'inst':i,'seed':seed,'seed_try':trial,
                       'original_generator': 'connected sequential growth with rejection',
                       'target_mean_degree_parameter':a.target_avg_degree,
                       'placement_radius_original_um':G.graph['placement_radius_um'],
                       'geometry_before_rescale':stat(G),
                       'geometry_after_rescale':stat(H),'historical_rescaling':rescale,
                       'legacy_physics_checks_pre_rescale':phys}
        except Exception as ex:
            last=str(ex)
            continue
    raise RuntimeError(f'Failed N={n} instance={i}: {last}')


def load_reference(path):
    if path is None: return {}
    return {(int(r['n']),int(r['inst'])):r for r in csv.DictReader(Path(path).open(encoding='utf-8'))}


def check_reference(meta,ref,tol=1e-8):
    key=(meta['n'],meta['inst']); r=ref.get(key)
    if r is None: raise ValueError(f'No historical reference for {key}')
    for field,actual in [ ('seed',meta['seed']),
                          ('dmin_before',meta['geometry_before_rescale']['min_distance_um']),
                          ('R_opt_before', math.sqrt(meta['geometry_before_rescale']['max_edge_distance_um']*meta['geometry_before_rescale']['min_nonedge_distance_um'])),
                          ('s_applied',meta['historical_rescaling']['s_applied']),
                          ('dmin_after',meta['geometry_after_rescale']['min_distance_um'])]:
        expected=float(r[field])
        if not math.isclose(float(actual),expected,rel_tol=tol,abs_tol=tol):
            raise AssertionError(f'Historical mismatch N={key[0]} i={key[1]} {field}: got {actual} expected {expected}')
    return True


def plot_chamber(G,p):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    fig,ax=plt.subplots(figsize=(6.4,6.4))
    pos={v:G.nodes[v]['pos'] for v in G.nodes()}
    nx.draw_networkx_edges(G,pos,ax=ax,alpha=.25,width=.7)
    nx.draw_networkx_nodes(G,pos,ax=ax,node_size=27)
    ax.add_patch(Circle((0,0),46,fill=False,ls='--',lw=1.2))
    ax.set(xlim=(-48,48),ylim=(-48,48),xlabel='x (um)',ylabel='y (um)')
    ax.set_aspect('equal');ax.set_title(f'Synthetic UDG N={len(G)}  E={G.number_of_edges()}')
    fig.tight_layout();fig.savefig(p,dpi=160);plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--lower',type=int,default=10)
    p.add_argument('--upper',type=int,default=100)
    p.add_argument('--step',type=int,default=5)
    p.add_argument('--instances',type=int,default=30)
    p.add_argument('--R-ryd',type=float,default=R_RYD)
    p.add_argument('--C6',type=float,default=C6)
    p.add_argument('--min-dist',type=float,default=5.0)
    p.add_argument('--max-radius',type=float,default=46.0)
    p.add_argument('--margin',type=float,default=.5)
    p.add_argument('--target-avg-degree',type=float,default=2.5)
    p.add_argument('--max-amp',type=float,default=4*math.pi)
    p.add_argument('--max-det',type=float,default=20*math.pi)
    p.add_argument('--sequence-duration-ns',type=int,default=6000)
    p.add_argument('--max-sequence-duration-ns',type=int,default=6000)
    p.add_argument('--max-seed-tries',type=int,default=200)
    p.add_argument('--rescale',choices=['historical','none'],default='historical')
    p.add_argument('--plot',action='store_true')
    p.add_argument('--reference-csv',type=Path,default=None)
    p.add_argument('--verify-reference',action='store_true', help='Optional legacy diagnostic ONLY; not needed for generation or rescaling.')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args(argv)
    if a.verify_reference and a.rescale!='historical':
        p.error('--verify-reference requires --rescale historical')
    if a.verify_reference:
        if a.reference_csv is None or not a.reference_csv.is_file():
            p.error('Optional reference check requested, but CSV is absent. To generate independently, REMOVE --verify-reference and --reference-csv. No reference data are needed.')
    if a.lower<2 or a.upper<a.lower or a.step<1 or a.instances<1: p.error('Invalid size grid')
    if a.sequence_duration_ns>a.max_sequence_duration_ns:
        p.error('Duration exceeds device upper bound')
    out=a.out_dir
    if out.exists() and any(out.iterdir()):
        if not a.overwrite:
            p.error(f'{out} is non-empty; choose a new directory or --overwrite')
        import shutil
        if not (out/'manifest.json').is_file():
            p.error('Refusing to overwrite a non-empty directory without a generator manifest; choose a new output directory.')
        if json.loads((out/'manifest.json').read_text()).get('generator') != 'sequential connected growth, adapted from legacy build_graph_helpers':
            p.error('The existing directory is not a generated benchmark; choose a new output directory.')
        shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    ref=load_reference(a.reference_csv) if a.verify_reference else {}
    rows=[]
    for n in range(a.lower,a.upper+1,a.step):
        for i in range(a.instances):
            H,meta=generate_one(n,i,a)
            if a.verify_reference: check_reference(meta,ref)
            idir=out/f'n{n:02d}'/f'inst{i:03d}'; idir.mkdir(parents=True,exist_ok=True)
            save_graph(H, idir/'graph_target.json')
            with (idir/'graph_target.pkl').open('wb') as f: pickle.dump(H,f,protocol=4)
            write_json(meta,idir/'meta.json')
            with (idir/'positions_target.csv').open('w',encoding='utf-8',newline='') as f:
                w=csv.writer(f);w.writerow(['node','x_um','y_um'])
                for v in sorted(H.nodes()):w.writerow([v,*H.nodes[v]['pos']])
            if a.plot: plot_chamber(H,idir/'graph.png')
            st=meta['geometry_after_rescale']
            rows.append({'n':n,'inst':i,'seed':meta['seed'],'n_edges':st['m'],
                         'avg_degree':st['mean_degree'],'min_degree':st['min_degree'],
                         'max_degree':st['max_degree'],'min_distance_um':st['min_distance_um'],
                         'max_radial_um':st['max_radial_um'],'udg_gap_um':st['udg_gap_um'],
                         'scale':meta['historical_rescaling']['s_applied'],'sha256':st['geometry_sha256'],
                         'relpath':(idir/'graph_target.json').relative_to(out).as_posix()})
            print(f'[OK] N={n:3d} instance={i:03d} seed={meta["seed"]} edges={st["m"]} '
                  f'dmin={st["min_distance_um"]:.4f}um scale={meta["historical_rescaling"]["s_applied"]:.6f}')
    with (out/'instances.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    config={k:v for k,v in vars(a).items() if k not in {'out_dir','reference_csv'}}
    write_json({'generator':'sequential connected growth, adapted from legacy build_graph_helpers',
                'historical_reference_checked':bool(a.verify_reference),
                'output_instances':len(rows),'config':config,'graphs':rows},out/'manifest.json')
    print(f'PASS: {len(rows)} graphs saved under {out}')

if __name__=='__main__':main()

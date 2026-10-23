#!/usr/bin/env python3
"""Audit and summarize graph_target.json instances from 07_generate_synthetic_udgs.py."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import networkx as nx
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eoc_na.graph_io import load_graph
from eoc_na.synthetic_udg_legacy import min_pairwise_distance,max_radial_distance,udg_distance_window,verify_udg_exact

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--bench-dir',type=Path,required=True)
    a.add_argument('--out-csv',type=Path,default=None)
    v=a.parse_args()
    rows=[]
    for p in sorted(v.bench_dir.glob('n*/inst*/graph_target.json')):
        G=load_graph(p)
        R=float(G.graph['rydberg_radius_um'])*float(G.graph.get('historical_uniform_rescale_factor',1))
        verify_udg_exact(G,r_conn=R)
        assert nx.is_connected(G)
        dmin=min_pairwise_distance(G);rmax=max_radial_distance(G)
        assert dmin >= 5.0-1e-9 and rmax<=46.0+1e-9
        rmin,rnonedge,ok=udg_distance_window(G)
        assert ok
        rows.append({'n':len(G),'inst':int(p.parent.name[4:]),'n_edges':G.number_of_edges(),
                     'mean_degree':float(np.mean([x for _,x in G.degree()])),
                     'max_degree':max(dict(G.degree()).values()),
                     'clustering':nx.average_clustering(G),
                     'diameter':nx.diameter(G),
                     'leaf_fraction':sum(d==1 for _,d in G.degree())/len(G),
                     'min_sep_um':dmin,
                     'max_radial_um':rmax,'longest_edge_um':rmin,'shortest_nonedge_um':rnonedge,
                     'udg_gap_um':rnonedge-rmin,'graph_path':str(p.relative_to(v.bench_dir))})
    if not rows:raise ValueError('No graph_target.json found')
    out=v.out_csv or v.bench_dir/'graph_structure_summary.csv'
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    from collections import defaultdict
    grouped=defaultdict(list)
    for r in rows:grouped[r['n']].append(r)
    print('N  instances  avg(mean-degree)  range(mean-degree)  max(max-degree)  median(UDG gap)')
    for n, rr in sorted(grouped.items()):
        mu=[x['mean_degree'] for x in rr]
        print(f'{n:3d} {len(rr):8d} {np.mean(mu):16.3f} {min(mu):7.3f}..{max(mu):7.3f} {max(x["max_degree"] for x in rr):16d} {np.median([x["udg_gap_um"] for x in rr]):15.5f}')
    print('Saved',out,'validated',len(rows),'graphs')
if __name__=='__main__':main()

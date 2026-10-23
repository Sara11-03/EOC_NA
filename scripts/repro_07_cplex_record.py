#!/usr/bin/env python3
"""Record a NEW CPLEX MDS/MIS run, not the undocumented settings of an old run."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import math
import networkx as nx
from eoc_geo.io import load_graph,write_json,guard_outputs,sha256
from tools.cplex_audit import before_solve,after_solve


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph',type=Path,required=True);p.add_argument('--problem',choices=('mds','mis'),default='mds')
    p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--time-limit',type=float)
    p.add_argument('--threads',type=int);p.add_argument('--random-seed',type=int)
    p.add_argument('--relative-gap',type=float);p.add_argument('--absolute-gap',type=float)
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args();guard_outputs([a.out_dir/'cplex_parameters.json'],a.overwrite)
    try:from docplex.mp.model import Model
    except ImportError:raise SystemExit('Install DOcplex and a working licensed CPLEX runtime in your own environment. They are not required for geographical reconstruction.')
    g=load_graph(a.graph);nodes=list(g);model=Model(name=a.problem)
    assignments={'timelimit':a.time_limit,'threads':a.threads,'randomseed':a.random_seed,'mip.tolerances.mipgap':a.relative_gap,'mip.tolerances.absmipgap':a.absolute_gap}
    for key,value in assignments.items():
        if value is None:continue
        param=model.parameters
        for part in key.split('.'):param=getattr(param,part)
        param.set(value)
    x={v:model.binary_var(name=f'x_{i}') for i,v in enumerate(nodes)}
    if a.problem=='mds':
        model.minimize(model.sum(x.values()))
        for i,v in enumerate(nodes):model.add_constraint(x[v]+model.sum(x[u] for u in g[v])>=1,ctname=f'dom_{i}')
    else:
        model.maximize(model.sum(x.values()))
        for i,(u,v) in enumerate(g.edges()):model.add_constraint(x[u]+x[v]<=1,ctname=f'edge_{i}')
    before_solve(model,a.out_dir)
    write_json({'graph_sha256':sha256(a.graph),'problem':a.problem,'node_order':nodes,'explicit_overrides':{k:v for k,v in assignments.items() if v is not None}},a.out_dir/'run_metadata.json')
    with (a.out_dir/'solver.log').open('w',encoding='utf8') as log:sol=model.solve(log_output=log)
    result=after_solve(model,sol,a.out_dir)
    if sol is None:raise SystemExit('No feasible solution. See solver.log and solve_details.json.')
    chosen=[v for v in nodes if sol.get_value(x[v])>.5]
    valid=nx.is_dominating_set(g,chosen) if a.problem=='mds' else all(not (u in chosen and v in chosen) for u,v in g.edges())
    if not valid:raise AssertionError('Solver solution failed combinatorial validation')
    bound=result.get('best_bound');objective=len(chosen)
    # Certify integer optimality from the integer objective and finite solver bound,
    # allowing a small numerical tolerance. Not a formal exact-arithmetic proof.
    if bound is not None:
        eps=1e-6
        integer_bound=math.ceil(bound-eps) if a.problem=='mds' else math.floor(bound+eps)
        certified=integer_bound==objective
    else:integer_bound=None;certified=False
    write_json({'problem':a.problem,'size':objective,'nodes':chosen,'feasible':valid,'best_bound':bound,'integer_bound':integer_bound,'integer_optimality_certified_from_bound':certified,'solver_status':result['status']},a.out_dir/'solution.json')
    print(f'{a.problem}: size={objective}, status={result["status"]}, bound-certified={certified}')

if __name__=='__main__':main()

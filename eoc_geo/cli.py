"""Argparse entry points. Defaults are safe, offline and non-submitting."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .io import ROOT, read_json, write_json, save_graph, load_graph, case_config, guard_outputs, write_csv, sha256, node_order_from_file


def case_arg(p):
    p.add_argument('--case',choices=('modena','palermo'),default='modena')
    p.add_argument('--out-dir',type=Path,default=None)
    p.add_argument('--overwrite',action='store_true')


def base_out(args):return args.out_dir or ROOT/'artifacts'/args.case


def download_main(argv=None):
    from .sources import download_provinces,representative_catalog,DEFAULT_REF
    p=argparse.ArgumentParser(description='Download/cache municipality boundaries, then extract representative points. Does not select a new graph.')
    case_arg(p);p.add_argument('--ref',default=DEFAULT_REF,help='Upstream tag or commit; recorded with content hashes.')
    p.add_argument('--offline',action='store_true',help='Use previously downloaded province files only.')
    p.add_argument('--provinces',type=int,nargs='+',default=None)
    a=p.parse_args(argv);out=a.out_dir or ROOT/'data'/'downloaded'/a.case
    config=case_config(a.case);codes=a.provinces or config['province_codes']
    download_provinces(codes,out,ref=a.ref,offline=a.offline,overwrite=a.overwrite)
    paths=[out/f'limits_P_{k}_municipalities.geojson' for k in codes]
    catalog=representative_catalog(paths);write_json(catalog,out/'municipalities.json');write_csv(catalog['nodes'],out/'municipalities.csv')
    print(f'Catalogue: {out / "municipalities.json"} ({len(catalog["nodes"])} municipalities)')


def geographic_main(argv=None):
    from .geometry import make_geographical_graph
    from .sources import verify_selected_catalog
    p=argparse.ArgumentParser(description='Rebuild geographical graph from the archived colleague-selected points.')
    case_arg(p);p.add_argument('--selected-nodes',type=Path);p.add_argument('--coordinates',type=Path)
    p.add_argument('--catalog',type=Path,help='Optional downloaded catalogue: checks names and coordinates, but does not replace frozen coordinates.')
    p.add_argument('--match-tolerance-m',type=float,default=30.)
    a=p.parse_args(argv);out=base_out(a);guard_outputs([out/'geographical_graph.json'],a.overwrite)
    g,report=make_geographical_graph(a.case,case_config(a.case),a.selected_nodes,a.coordinates)
    if a.catalog:
        matches=verify_selected_catalog(g,read_json(a.catalog),a.match_tolerance_m)
        write_csv(matches,out/'catalogue_matching.csv')
        report['catalogue_sha256']=sha256(a.catalog)
        report['catalogue_matching']={'all_names_and_tolerances_match':all(r['accepted'] for r in matches),'rounded_exact_matches':sum(r['rounded_exact'] for r in matches)}
        if not all(r['accepted'] for r in matches):
            write_json(report,out/'geographical_input_report.json')
            raise ValueError('Downloaded catalogue does not reproduce all selected municipalities. Inspect catalogue_matching.csv. No partial graph was saved.')
    save_graph(g,out/'geographical_graph.json');write_json(report,out/'geographical_input_report.json')
    write_csv([{'id':v,**a} for v,a in g.nodes(data=True)],out/'selected_municipalities.csv')
    print(f'{a.case}: geographical n={len(g)}, m={g.number_of_edges()}; saved {out / "geographical_graph.json"}')


def embed_main(argv=None):
    from .geometry import build_register,save_register_outputs
    p=argparse.ArgumentParser(description='Historical UTM/scale/clip/prune workflow. No QPU access or automatic resubmission.')
    case_arg(p);p.add_argument('--input',type=Path);p.add_argument('--config',type=Path)
    a=p.parse_args(argv);out=base_out(a);source=a.input or out/'geographical_graph.json'
    guard_outputs([out/'register.json',out/'geometry_report.json'],a.overwrite)
    g=load_graph(source);final,report,stages=build_register(g,case_config(a.case,a.config));report['input_sha256']=sha256(source)
    save_register_outputs(final,report,stages,out)
    print(json.dumps(report['final'],indent=2,ensure_ascii=False))
    print(f'Saved register and all stages: {out}')


def plot_main(argv=None):
    from .plotting import plot_geographical,plot_chamber
    p=argparse.ArgumentParser(description='Plot geographical locations and the atomic register; offline by default.')
    case_arg(p);p.add_argument('--graph',type=Path);p.add_argument('--boundaries-dir',type=Path)
    p.add_argument('--labels',choices=('none','key','all'),default='key')
    p.add_argument('--solution-results',type=Path,help='Saved mds_nodes_results JSON; highlights the recorded docplex_exact set without re-solving.')
    p.add_argument('--show',action='store_true');p.add_argument('--formats',nargs='+',choices=('png','svg','pdf'),default=['png','svg'])
    a=p.parse_args(argv);out=base_out(a);g=load_graph(a.graph or out/'register.json')
    paths=[out/'plots'/f'{base}.{ext}' for base in ('geographical_map','atomic_chamber') for ext in a.formats]
    guard_outputs(paths,a.overwrite)
    boundary=list(a.boundaries_dir.glob('limits_P_*_municipalities.geojson')) if a.boundaries_dir else None
    solution=None
    if a.solution_results:
        d=read_json(a.solution_results);solution=d.get('mds',d)['mds_nodes']['docplex_exact']
        import networkx as nx
        if not nx.is_dominating_set(g,solution):raise ValueError('Recorded solution does not dominate this graph')
    plot_geographical(g,out/'plots'/'geographical_map',boundary,a.labels,solution=solution,formats=a.formats,show=a.show)
    plot_chamber(g,out/'plots'/'atomic_chamber',radius_um=case_config(a.case)['max_radius_um'],solution=solution,formats=a.formats,show=a.show)
    print(f'Figures: {out / "plots"}')


def validate_main(argv=None):
    from .geometry import compare_reference,geometry_metrics
    p=argparse.ArgumentParser(description='Validate node/edge identity and physical coordinates against reconstructed reference.')
    case_arg(p);p.add_argument('--graph',type=Path);p.add_argument('--reference',type=Path);p.add_argument('--atol-um',type=float,default=1e-8)
    a=p.parse_args(argv);out=base_out(a);g=load_graph(a.graph or out/'register.json')
    ref=load_graph(a.reference or ROOT/'data'/'reference'/f'{a.case}_reconstructed_reference.json')
    report=compare_reference(g,ref,a.atol_um);report['geometry']=geometry_metrics(g)
    write_json(report,out/'geometry_validation.json');print(json.dumps(report,indent=2,ensure_ascii=False))
    if not report['passed']:raise SystemExit(2)


def export_qpu_main(argv=None):
    from .qpu import export_fetched
    p=argparse.ArgumentParser(description='Extract uncorrected counts and returned sample lists from the historical fetch output.')
    case_arg(p);p.add_argument('--fetched',type=Path,required=True);p.add_argument('--node-order',type=Path)
    a=p.parse_args(argv);out=a.out_dir or ROOT/'exports'/'qpu'/a.case
    guard_outputs([out/'counts.json'],a.overwrite)
    order=node_order_from_file(a.node_order) if a.node_order else None
    result=export_fetched(a.fetched,out,case_config(a.case)['expected_nodes'],a.case,order)
    print(json.dumps(result,indent=2))


def check_qpu_main(argv=None):
    from .qpu import evaluate_counts,compare_archived
    p=argparse.ArgumentParser(description='Recompute raw IS/cardinality and named-edge violation statistics with an explicit bit-to-node mapping.')
    case_arg(p);p.add_argument('--graph',type=Path);p.add_argument('--node-order',type=Path);p.add_argument('--counts',type=Path)
    p.add_argument('--reported-stats',type=Path);p.add_argument('--reported-edges',type=Path)
    a=p.parse_args(argv);out=base_out(a);data=ROOT/'data'/'qpu'/a.case
    g=load_graph(a.graph or out/'register.json');order=node_order_from_file(a.node_order or data/'node_order.json')
    result=evaluate_counts(g,read_json(a.counts or data/'counts.json'),order)
    comparison=compare_archived(result,a.reported_stats or data/'reported_hw_stats.json',a.reported_edges or data/'reported_edge_stats.json')
    write_json(result,out/'raw_qpu_recomputed.json');write_json(comparison,out/'raw_qpu_validation.json')
    print(f'{a.case}: {result["IS_count"]}/{result["num_shots"]} raw valid independent sets; all diagnostics match: {comparison["passed"]}')
    print(f'Named edges cross-checked: {comparison["edges_cross_checked"]}')
    if not comparison['passed']:raise SystemExit(2)


def demo_main(argv=None):
    p=argparse.ArgumentParser(description='Run geographical reconstruction, plots and archived-data checks. No quantum jobs or CPLEX calls.')
    case_arg(p);p.add_argument('--download',action='store_true',help='Optional remote source check; not needed for archived reconstruction.')
    p.add_argument('--boundaries-dir',type=Path);a=p.parse_args(argv);out=base_out(a)
    common=['--case',a.case,'--out-dir',str(out)]+(['--overwrite'] if a.overwrite else [])
    extra=[];bounds=a.boundaries_dir
    if a.download:
        download_main(['--case',a.case]);bounds=ROOT/'data'/'downloaded'/a.case
        extra=['--catalog',str(bounds/'municipalities.json')]
    geographic_main(common+extra);embed_main(common)
    plots=common+(['--boundaries-dir',str(bounds)] if bounds else [])
    plot_main(plots);validate_main(common);check_qpu_main(common)
    print('PASS: geometry and every archived raw-QPU diagnostic reproduced.')


def main_guard(function):
    try:function()
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);raise SystemExit(2)

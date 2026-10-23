"""Release checks for the archived case studies and solver records."""
import importlib.util
import json
from pathlib import Path
import pytest
from eoc_geo.geometry import make_geographical_graph, build_register
from eoc_geo.io import case_config
from eoc_na.mds import run_two_branch_pipeline, exact_mds_verified
ROOT=Path(__file__).resolve().parents[1]


def cases_module():
    spec=importlib.util.spec_from_file_location('case_replay',ROOT/'scripts/13_reproduce_case_studies.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('case,backend,small,large',[
    ('palermo','emulator',13,12),('palermo','qpu',13,12),
    ('modena','emulator',31,29),('modena','qpu',32,29)])
def test_case_study_heuristics(case,backend,small,large):
    m=cases_module();geo,_=make_geographical_graph(case,case_config(case))
    G,_,_=build_register(geo,case_config(case))
    order=json.loads((ROOT/f'data/qpu/{case}/node_order.json').read_text())['node_order']
    G=m.ordered_integer_graph(G,order)
    _,_,samples=m.read_counts(ROOT/f'data/realistic_case_studies/{case}/{backend}/fetched_results_summary.json',len(G))
    result=run_two_branch_pipeline(G,raw_quantum_samples=samples,seed=0,protocol='paper20260505')
    assert (result.mIS_branch.best_size,result.complement_branch.best_size)==(small,large)


def test_cplex_records_when_installed(tmp_path):
    pytest.importorskip('cplex');pytest.importorskip('docplex')
    import networkx as nx
    D,audit=exact_mds_verified(nx.path_graph(10),solver='cplex',audit_dir=tmp_path)
    assert len(D)==4 and audit['certified_optimal']
    for name in ['cplex_parameters.json','solve_details.json','model.lp','solution_values.json']:
        assert (tmp_path/name).is_file()
    assert audit['effective_parameters']['threads']['is_default']


def test_publication_default_is_cplex():
    import inspect
    assert inspect.signature(exact_mds_verified).parameters['solver'].default=='cplex'

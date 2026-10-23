"""Regression tests for input integrity and the verified Figure-3 replay.

Comparisons to expected outputs are TESTS only; calculation code does not read them.
Cloud/Pulser tests below are mocks, not execution tests on those backends.
"""
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from eoc_na.graph_io import load_graph, save_graph
from eoc_na.mds import exact_mds_verified, run_two_branch_pipeline
from eoc_na.samples import load_samples

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/synthetic_samples'
HIST = ROOT / 'data/synthetic_case_studies/result_caches/results_not_corr/run_20260505_113509'
GRAPHS = ROOT / 'data/synthetic_case_studies/udg_benchmark_Rryd_7_18'


def load_script(name):
    path = ROOT / 'scripts' / name
    spec = importlib.util.spec_from_file_location('_test_'+path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command(script, *args):
    return subprocess.run([sys.executable, str(ROOT/'scripts'/script), *map(str,args)],
        cwd=ROOT, text=True, capture_output=True, timeout=90)


def legacy_graph(n, inst=0):
    return load_graph(GRAPHS / f'n{n:02d}' / f'inst{inst:03d}' / 'graph_target.json')


def test_generation_without_any_reference(tmp_path):
    result=command('07_generate_synthetic_udgs.py','--out-dir',tmp_path/'bench',
                   '--lower',10,'--upper',10,'--instances',1)
    assert result.returncode == 0, result.stdout+result.stderr
    generated=load_graph(tmp_path/'bench/n10/inst000/graph_target.json')
    original=legacy_graph(10)
    assert set(generated)==set(original)
    assert {frozenset(e) for e in generated.edges()}=={frozenset(e) for e in original.edges()}
    assert np.allclose([generated.nodes[v]['pos'] for v in generated],
                       [original.nodes[v]['pos'] for v in generated],rtol=0,atol=1e-10)


def test_missing_optional_reference_reports_fix_before_writing(tmp_path):
    out=tmp_path/'must_not_exist'
    result=command('07_generate_synthetic_udgs.py','--out-dir',out,
        '--verify-reference','--reference-csv',tmp_path/'absent.csv')
    assert result.returncode != 0
    assert 'REMOVE --verify-reference' in result.stderr
    assert not out.exists()


@pytest.mark.parametrize('n,inst',[(10,0),(20,1),(50,0),(100,0),(100,1)])
def test_paper_protocol_on_raw_archived_samples(n,inst):
    G=legacy_graph(n,inst)
    # Generated JSON preserves insertion order through explicit idx fields.
    for i,v in enumerate(G): G.nodes[v]['idx']=i
    raw=load_samples(RAW/f'n{n:02d}'/f'inst{inst:03d}'/'samples.json')
    expected=json.loads((HIST/f'n{n:02d}'/f'inst{inst:03d}.json').read_text())
    classical=run_two_branch_pipeline(G,n_classical_samples=500,seed=0,protocol='paper20260505')
    quantum=run_two_branch_pipeline(G,raw_quantum_samples=raw,seed=0,protocol='paper20260505')
    assert classical.mIS_branch.best_size==expected['mDS_from_mIS']
    assert classical.complement_branch.best_size==expected['notMIS']
    assert quantum.mIS_branch.best_size==expected['mDS_from_mIS_quantum']
    assert quantum.complement_branch.best_size==expected['notMIS_quantum']
    D,certificate=exact_mds_verified(G,solver='highs',time_limit_sec=120)
    assert len(D)==expected['mDS']
    assert certificate['certified_optimal']
    assert certificate['historical_cplex_run'] is False


def test_graph_roundtrip_numeric_node_order(tmp_path):
    G=nx.path_graph(12)
    for i,v in enumerate(G):G.nodes[v].update(pos=[float(i),0.0],idx=i)
    save_graph(G,tmp_path/'graph.json')
    from eoc_na.graph_io import ordered_nodes
    assert ordered_nodes(load_graph(tmp_path/'graph.json')) == list(range(12))


def test_counts_are_not_silently_coerced(tmp_path):
    p=tmp_path/'counts.json'
    p.write_text(json.dumps({'00':-1,'01':1}))
    with pytest.raises(ValueError):load_samples(p)


def test_ratio_loader_rejects_unpaired_arrays():
    module=load_script('06_synthetic_approx_ratio_boxplot.py')
    with pytest.raises(ValueError):module.safe_ratio([3,4],[2])


def test_missing_metric_keeps_instance_position(tmp_path):
    module=load_script('06_synthetic_approx_ratio_boxplot.py')
    d=tmp_path/'n10';d.mkdir()
    base={'mDS':3,'mDS_from_mIS':3,'notMIS':3,'mDS_from_mIS_quantum':3,'notMIS_quantum':3}
    first=base.copy();first.pop('notMIS_quantum')
    (d/'inst000.json').write_text(json.dumps(first))
    second=base.copy();second['mDS']=4;second['notMIS_quantum']=5
    (d/'inst001.json').write_text(json.dumps(second))
    results,_=module.load_res_from_instance_json_layout(tmp_path)
    assert len(results[10]['notMIS_quantum'])==2
    assert np.isnan(results[10]['notMIS_quantum'][0])
    assert module.ratio_for(results,10,'notMIS_quantum').tolist()==[1.25]


def test_dry_run_never_needs_pulser_or_cloud(tmp_path):
    bench=tmp_path/'bench';place=bench/'n10/inst000';place.mkdir(parents=True)
    G=legacy_graph(10)
    for i,v in enumerate(G):G.nodes[v]['idx']=i
    save_graph(G,place/'graph_target.json')
    result=command('11_run_emu_grid.py','--bench-dir',bench,'--out-dir',tmp_path/'emu','--max-jobs',1)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'DRY RUN' in result.stdout
    report=json.loads((tmp_path/'emu/n10/inst000/requested_protocol.json').read_text())
    assert report['runs']==500
    assert report['schedule']['total_time_ns']==5400
    assert 'NOT established' in report['provenance_status']
    assert not list((tmp_path/'emu').rglob('submission_ids.json'))


def test_cloud_submission_needs_explicit_acknowledgment(tmp_path):
    result=command('09_run_emu_mps_smoke.py','--bench-dir',tmp_path,'--n',10,
                   '--out-dir',tmp_path/'out','--submit')
    assert result.returncode != 0
    assert '--confirm-new-protocol' in result.stderr


def test_pulse_builder_linear_endpoints_mock(monkeypatch):
    # Checks endpoint wiring, NOT Pulser's validity checks or physical execution.
    class Ramp:
        def __init__(self,duration,start,stop):self.duration,self.start,self.stop=duration,start,stop
    class Constant:
        def __init__(self,duration,value):self.duration,self.value=duration,value
    class Pulse:
        def __init__(self,amp,det,phase):self.amplitude,self.detuning,self.phase=amp,det,phase
    class Register:
        @staticmethod
        def from_coordinates(coords,**kw):return coords
    class Sequence:
        def __init__(self,register,device):self.pulses=[]
        def declare_channel(self,*a):pass
        def add(self,pulse,channel):self.pulses.append(pulse)
    module=types.ModuleType('pulser');module.Register=Register;module.Sequence=Sequence
    pulse_module=types.ModuleType('pulser.pulse');pulse_module.Pulse=Pulse
    wave_module=types.ModuleType('pulser.waveforms');wave_module.RampWaveform=Ramp;wave_module.ConstantWaveform=Constant
    monkeypatch.setitem(sys.modules,'pulser',module)
    monkeypatch.setitem(sys.modules,'pulser.pulse',pulse_module)
    monkeypatch.setitem(sys.modules,'pulser.waveforms',wave_module)
    G=nx.path_graph(2)
    for i in G:G.nodes[i].update(pos=[i*6.,0.],idx=i)
    from eoc_na.pasqal_io import build_pulser_sequence
    seq=build_pulser_sequence(G,device=object(),schedule={
        'omega_max_rad_per_us':6.,'delta_0_rad_per_us':-18.,'delta_f_rad_per_us':5.,
        't_rise_ns':900,'t_sweep_ns':2700,'t_fall_ns':1800})
    a,b,c=seq.pulses
    assert (a.amplitude.start,a.amplitude.stop)==(0.,6.)
    assert (b.detuning.start,b.detuning.stop)==(-18.,5.)
    assert (c.amplitude.start,c.amplitude.stop)==(6.,0.)
    assert sum(x.amplitude.duration for x in seq.pulses)==5400

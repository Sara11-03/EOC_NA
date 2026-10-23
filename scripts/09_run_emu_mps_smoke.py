#!/usr/bin/env python3
"""Prepare ONE synthetic sequence, or explicitly submit it to EMU-MPS."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eoc_na.graph_io import load_graph, ordered_nodes, write_json
from eoc_na.pasqal_io import build_pulser_sequence


def fingerprint(G):
    order=ordered_nodes(G)
    data={'nodes':order,'edges':sorted([sorted(e) for e in G.edges()]),
          'coordinates_um':[[v,*[round(float(x),8) for x in G.nodes[v]['pos']]] for v in order]}
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def connection_and_backend():
    # Support the two officially documented interfaces, without guessing kwargs.
    try:
        from pasqal_cloud import PasqalCloudConnection, RemoteMPSBackend
        return PasqalCloudConnection, RemoteMPSBackend
    except ImportError:
        try:
            from pulser_pasqal import PasqalCloud, EmuMPSBackend
            return PasqalCloud, EmuMPSBackend
        except ImportError as exc:
            raise RuntimeError('Install requirements-emulation.txt, or use a compatible pasqal-cloud with RemoteMPSBackend. No job has been submitted.') from exc


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench-dir',type=Path,required=True)
    p.add_argument('--n',type=int,required=True)
    p.add_argument('--inst',type=int,default=0)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--backend-device',default='FRESNEL')
    p.add_argument('--omega-max',type=float,default=2*math.pi)
    p.add_argument('--delta-0',type=float,default=-18.997567657715674)
    p.add_argument('--delta-f-over-U',type=float,default=1.9)
    p.add_argument('--total-time-ns',type=int,default=5400)
    p.add_argument('--runs',type=int,default=500)
    p.add_argument('--max-bond-dim',type=int,default=300)
    p.add_argument('--dt-ns',type=float,default=10.)
    p.add_argument('--precision',type=float,default=1e-5)
    p.add_argument('--submit',action='store_true',help='Consumes remote emulator resources')
    p.add_argument('--fetch-only',action='store_true',help='Retrieve a saved batch without submitting a new one')
    p.add_argument('--wait',action='store_true')
    p.add_argument('--timeout-seconds',type=float,default=3600.)
    p.add_argument('--confirm-new-protocol',action='store_true',help='Acknowledge that this is a new calculation, not a certified historical pulse replay')
    a=p.parse_args(argv)
    if a.submit and a.fetch_only:p.error('Choose --submit OR --fetch-only')
    if a.submit and not a.confirm_new_protocol:p.error('This protocol is not certified against the historical job sequences. Add --confirm-new-protocol only for a NEW calculation.')
    if a.total_time_ns<=0 or a.total_time_ns>6000 or a.total_time_ns%24:
        p.error('Use a total time divisible by 24 ns (all 1/6, 1/2, 1/3 segments on a 4-ns clock), at most 6000 ns')
    if min(a.runs,a.max_bond_dim,a.dt_ns,a.precision,a.timeout_seconds,a.omega_max)<=0:p.error('All budgets, durations and tolerances must be positive')
    gp=a.bench_dir/f'n{a.n:02d}'/f'inst{a.inst:03d}'/'graph_target.json'
    G=load_graph(gp);nodes=ordered_nodes(G)
    if len(G)!=a.n:raise ValueError('Graph size mismatch')
    schedule={'omega_max_rad_per_us':a.omega_max,'delta_0_rad_per_us':a.delta_0,
              'delta_f_rad_per_us':a.delta_f_over_U*a.omega_max/2,
              't_rise_ns':a.total_time_ns//6,'t_sweep_ns':a.total_time_ns//2,
              't_fall_ns':a.total_time_ns//3,'total_time_ns':a.total_time_ns}
    params={'schedule':schedule,'node_order':nodes,'num_edges':G.number_of_edges(),
            'graph_fingerprint_round8_sha256':fingerprint(G),'device_key':a.backend_device,
            'runs':a.runs,'max_bond_dim':a.max_bond_dim,'dt_ns':a.dt_ns,'precision':a.precision,
            'with_modulation':False,'prefer_device_noise_model':False,'noise_types':[],
            'waveform':'linear amplitude ramps; linear detuning sweep',
            'provenance_status':'candidate protocol; historical job-sequence identity NOT established'}
    sig=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()
    a.out_dir.mkdir(parents=True,exist_ok=True)
    request=a.out_dir/'requested_protocol.json';ids=a.out_dir/'submission_ids.json'
    if request.exists() and json.loads(request.read_text()).get('request_sha256')!=sig:
        raise ValueError('Existing output directory belongs to different settings. Use a new directory.')
    write_json({**params,'request_sha256':sig},request)
    print('Protocol recorded:',request,flush=True)
    if not(a.submit or a.fetch_only):
        print('DRY RUN: no emulator execution or cloud connection.');return
    if a.submit and ids.exists():
        p.error('A submission is already recorded here. Use --fetch-only, not another submission.')
    if a.fetch_only and not ids.exists():p.error('No submission_ids.json to retrieve')
    if (a.out_dir/'counts.json').exists():
        print('Counts already saved; not submitting or overwriting.');return
    factory,backend_class=connection_and_backend()
    username=os.environ.get('PASQAL_USERNAME');project=os.environ.get('PASQAL_PROJECT_ID')
    if not username or not project:p.error('Set PASQAL_USERNAME and PASQAL_PROJECT_ID locally; do not store credentials in the repository.')
    connection=factory(username=username,project_id=project,password=os.environ.get('PASQAL_PASSWORD'))
    if a.fetch_only:
        from pulser.backend import RemoteResults
        remote=RemoteResults(batch_id=json.loads(ids.read_text())['batch_id'],connection=connection)
    else:
        from pulser.backend import EmulationConfig, BitStrings
        from pulser.noise_model import NoiseModel
        devices=connection.fetch_available_devices()
        if a.backend_device not in devices:
            raise ValueError(f'Device {a.backend_device!r} unavailable. Available keys: {list(devices)}. Do not silently substitute a different device.')
        device=devices[a.backend_device]
        seq=build_pulser_sequence(G,device=device,schedule=schedule)
        config=EmulationConfig(observables=(BitStrings(num_shots=a.runs,evaluation_times=[1.]),),
            with_modulation=False,prefer_device_noise_model=False,noise_model=NoiseModel(),
            max_bond_dim=a.max_bond_dim,dt=a.dt_ns,precision=a.precision)
        (a.out_dir/'sequence_abstract.json').write_text(seq.to_abstract_repr(),encoding='utf-8')
        (a.out_dir/'emulator_config.json').write_text(config.to_abstract_repr(),encoding='utf-8')
        versions={}
        for package in ['pulser-core','pulser-pasqal','pasqal-cloud']:
            try:versions[package]=importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:pass
        write_json(versions,a.out_dir/'versions.json')
        backend=backend_class(seq,connection=connection,config=config)
        # Both documented emulator interfaces accept run() and return RemoteResults.
        remote=backend.run()
        write_json({'batch_id':str(remote.batch_id),'request_sha256':sig,'status':'submitted',
                    'backend_class':backend_class.__name__},ids)
        print('Submitted batch:',remote.batch_id,flush=True)
    started=time.monotonic()
    while True:
        available=remote.get_available_results()
        if available:break
        status=str(remote.get_batch_status())
        if any(s in status.upper() for s in ['ERROR','CANCELED','CANCELLED','FAILED']):
            raise RuntimeError(f'Batch ended without results: {status}; see saved batch ID')
        if not a.wait:
            print('Not completed yet. Re-run with --fetch-only --wait to fetch without resubmission.');return
        if time.monotonic()-started>a.timeout_seconds:
            raise TimeoutError('Local waiting limit reached; the batch may still run. Resume with --fetch-only --wait.')
        time.sleep(5)
    if len(available)!=1:raise ValueError('Expected exactly one job; refusing to merge unknown results')
    jid,result=next(iter(available.items()))
    counts=getattr(result,'final_bitstrings',None)
    if counts is None:raise ValueError('No final bitstring observable in the result')
    counts={str(k):int(v) for k,v in counts.items()}
    if sum(counts.values())!=a.runs or any(len(s)!=len(G) or set(s)-{'0','1'} for s in counts):
        raise ValueError('Unexpected number/format of shots')
    expected_order=tuple(f'q{i}' for i in range(len(G)))
    if tuple(result.atom_order)!=expected_order:
        raise ValueError(f'Unexpected backend atom order {result.atom_order}; mapping must be resolved before analysis')
    write_json(counts,a.out_dir/'counts.json')
    write_json({'origin':'fresh remote MPS emulation','node_order':nodes,
        'atom_order':list(result.atom_order),'graph_fingerprint_round8_sha256':fingerprint(G),
        'batch_id':str(remote.batch_id),'job_id':str(jid),'request_sha256':sig,
        'n_samples':sum(counts.values()),'sample_order':'counts only; chronological order unavailable',
        'no_postselection':True,'no_correction':True},a.out_dir/'metadata.json')
    print('Saved raw counts for job',jid,flush=True)

if __name__=='__main__':main()

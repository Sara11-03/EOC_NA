#!/usr/bin/env python3
"""Apply existing 09 runner to a bounded set of graphs. No cloud calls by default."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench-dir',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--max-jobs',type=int,default=1)
    p.add_argument('--only-n',type=int,default=None)
    p.add_argument('--submit',action='store_true')
    p.add_argument('--fetch-only',action='store_true')
    p.add_argument('--wait',action='store_true')
    p.add_argument('--confirm-new-protocol',action='store_true')
    p.add_argument('--backend-device',default='FRESNEL')
    p.add_argument('--omega-max',type=float,default=6.283185307179586)
    p.add_argument('--delta-0',type=float,default=-18.997567657715674)
    p.add_argument('--total-time-ns',type=int,default=5400)
    p.add_argument('--delta-f-over-U',type=float,default=1.9)
    p.add_argument('--runs',type=int,default=500)
    p.add_argument('--max-bond-dim',type=int,default=300)
    p.add_argument('--dt-ns',type=float,default=10.)
    p.add_argument('--precision',type=float,default=1e-5)
    a=p.parse_args(argv)
    if a.max_jobs<1:p.error('--max-jobs must be positive')
    if a.submit and a.fetch_only:p.error('Choose --submit OR --fetch-only')
    if a.submit and not a.confirm_new_protocol:p.error('Fresh simulations require --confirm-new-protocol; historical sequence identity is not certified.')
    paths=sorted(a.bench_dir.glob('n*/inst*/graph_target.json'),key=lambda q:(int(q.parent.parent.name[1:]),int(q.parent.name[4:])))
    if a.only_n is not None:paths=[q for q in paths if int(q.parent.parent.name[1:])==a.only_n]
    if not paths:p.error('No graph_target.json found')
    script=Path(__file__).with_name('09_run_emu_mps_smoke.py')
    done=0
    for gp in paths:
        if done>=a.max_jobs:break
        od=a.out_dir/gp.parent.parent.name/gp.parent.name
        if (od/'counts.json').exists():
            print('[PRESENT]',od,flush=True);continue
        if a.fetch_only and not (od/'submission_ids.json').exists():continue
        cmd=[sys.executable,str(script),'--bench-dir',str(a.bench_dir),'--n',gp.parent.parent.name[1:],
             '--inst',gp.parent.name[4:],'--out-dir',str(od)]
        for attr in ['backend_device','omega_max','delta_0','total_time_ns','delta_f_over_U','runs','max_bond_dim','dt_ns','precision']:
            cmd+=['--'+attr.replace('_','-'),str(getattr(a,attr))]
        if a.submit and (od/'submission_ids.json').exists():
            cmd+=['--fetch-only']
        elif a.submit:cmd+=['--submit','--confirm-new-protocol']
        elif a.fetch_only:cmd+=['--fetch-only']
        if a.wait:cmd+=['--wait']
        print('[RUN]',gp.parent.parent.name,gp.parent.name,flush=True)
        subprocess.run(cmd,check=True)
        done+=1
    print('Processed',done,'instances; selected graph count',len(paths))
if __name__=='__main__':main()

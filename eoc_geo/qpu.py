"""Export and validate returned bitstrings without repairing or postselecting."""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import math
from .io import read_json, write_json, sha256, write_csv


def checked_counts(counts, n: int) -> dict[str,int]:
    if not isinstance(counts,dict):raise ValueError('Counts must be a dictionary')
    out={}
    for z,k in counts.items():
        if not isinstance(z,str) or len(z)!=n or set(z)-{'0','1'}:raise ValueError(f'Invalid raw bitstring: {z!r}')
        if isinstance(k,bool) or not isinstance(k,int) or k<=0:raise ValueError(f'Invalid multiplicity: {k!r}')
        out[z]=k
    return out


def export_fetched(path: Path, out_dir: Path, n: int, case: str, node_order: list | None=None) -> dict:
    d=read_json(path)
    if d.get('backend_kind')!='qpu':raise ValueError('This exporter expects QPU results, not emulator output')
    aggregate=Counter();metadata=[]
    out_dir.mkdir(parents=True,exist_ok=True)
    for job in d['jobs_fetched']:
        if job['status'] not in ('DONE','JobStatus.DONE'):raise ValueError(f'Unfinished job: {job["status"]}')
        counts=checked_counts(job['counts'],n)
        aggregate.update(counts)
        jdir=out_dir/'jobs'/str(job['job_id']);jdir.mkdir(parents=True,exist_ok=True)
        write_json(counts,jdir/'counts.json')
        if 'samples' in job:
            samples=job['samples']
            if Counter(samples)!=Counter(counts):raise ValueError('Returned samples disagree with counts')
            write_json(samples,jdir/'samples_returned.json')
        metadata.append({k:job.get(k) for k in ('job_id','status','n_samples','runs_requested','seed')})
    if dict(aggregate)!=checked_counts(d['final_counts'],n):raise ValueError('Per-job counts disagree with final_counts')
    total=sum(aggregate.values())
    if total!=d['final_n_samples']:raise ValueError('Shot total mismatch')
    write_json(dict(aggregate),out_dir/'counts.json')
    write_csv([{'bitstring':z,'count':k} for z,k in aggregate.items()],out_dir/'counts.csv')
    if node_order is not None:
        if len(node_order)!=n or len(set(node_order))!=n:raise ValueError('Invalid bit-to-node mapping')
        write_json({'node_order':node_order,'index0':'left','note':'Analysis mapping; verify against submission register when available.'},out_dir/'node_order.json')
    meta={'case':case,'backend_kind':'qpu','num_bits':n,'shots':total,'unique_bitstrings':len(aggregate),'jobs':metadata,
          'source_file':path.name,'source_sha256':sha256(path),
          'source_submission_basename':str(d.get('submission_dir','')).replace('\\','/').rsplit('/',1)[-1],
          'processing':'No correction, no postselection, no relabeling of bits; exact counts preserved.',
          'sample_order':'The original samples list is retained if present. Acquisition order is not established by this export; do not use sorted counts as a chronological shot sequence.',
          'seed_note':'A recorded software seed does not make physical QPU measurements deterministic.',
          'node_mapping_included':node_order is not None}
    write_json(meta,out_dir/'metadata.json')
    return meta


def evaluate_counts(g, counts: dict, order: list) -> dict:
    if len(order)!=len(g) or len(set(order))!=len(order) or set(order)!=set(g):raise ValueError('Mapping does not exactly cover graph nodes')
    counts=checked_counts(counts,len(g));idx={v:i for i,v in enumerate(order)}
    sizes=Counter();good_sizes=Counter();bad_sizes=Counter();severity=Counter();edge_counts=Counter()
    edges=list(g.edges());good=0;total=0
    for z,number in counts.items():
        violated=[(u,v) for u,v in edges if z[idx[u]]=='1' and z[idx[v]]=='1']
        weight=z.count('1');sizes[weight]+=number;total+=number
        if not violated:good+=number;good_sizes[weight]+=number
        else:bad_sizes[weight]+=number;severity[len(violated)]+=number
        for u,v in violated:edge_counts[frozenset((u,v))]+=number
    return {'num_shots':total,'schema_valid_shots':total,'IS_count':good,'notIS_count':total-good,'IS_fraction':good/total if total else None,
            'size_counts_all':dict(sizes),'size_counts_is':dict(good_sizes),'size_counts_not':dict(bad_sizes),
            'violating_edge_histogram_on_nonIS':dict(severity),'total_edge_violations':sum(k*v for k,v in severity.items()),
            'edge_rows':[{'u':u,'v':v,'violation_count':edge_counts[frozenset((u,v))]} for u,v in edges]}


def compare_archived(computed: dict, stats_path: Path, edge_path: Path) -> dict:
    d=read_json(stats_path);s=d.get('stats',d.get('hw_stats',d))
    tests={k: computed[k]==s[k] for k in ('IS_count','notIS_count','IS_fraction')}
    for name in ('size_counts_all','size_counts_is','size_counts_not'):
        tests[name]={str(k):v for k,v in computed[name].items()}=={str(k):v for k,v in s[name].items()}
    e=read_json(edge_path)
    expected={frozenset((r['u'],r['v'])):r['violation_count'] for r in e['edge_rows']}
    observed={frozenset((r['u'],r['v'])):r['violation_count'] for r in computed['edge_rows']}
    tests['all_named_edge_counts']=observed==expected
    tests['total_edge_violations']=computed['total_edge_violations']==e['total_edge_violations']
    tests['number_of_shots']=computed['num_shots']==s['n_samples_valid']
    return {'checks':tests,'passed':all(tests.values()),'edges_cross_checked':len(expected),
            'mapping_evidence':'Consistency of supplied mapping with all named-edge counts and all IS/cardinality histograms; does not recover unrecorded shot chronology.'}

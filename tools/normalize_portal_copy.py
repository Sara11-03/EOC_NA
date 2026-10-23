#!/usr/bin/env python3
"""Convert a copied typed JSON-tree display to JSON, without evaluating code."""
from pathlib import Path
import argparse
import hashlib
import json
import re


def normalize(text):
    jobs=re.findall(r'^Job ([0-9a-f-]{36})',text,flags=re.M)
    if len(jobs)!=1:raise ValueError('Expected exactly one Job header')
    first=text.find('{')
    if first<0:raise ValueError('No JSON object found')
    s=text[first:]
    s=re.sub(r'(?m)^\s*\d+:','',s)
    s=re.sub(r':\s*(string|float|int|bool)(?=["\-\d]|true|false)',':',s)
    s=re.sub(r'\bNULL\b','null',s)
    s=re.sub(r'(\}|\])(,?)\s*\d+ items?',r'\1\2',s)
    s=re.sub(r'(?m)^\s*(string|float|int|bool)(?=["\-\d]|true|false)','',s)
    d=json.loads(s)
    for field in ('register','operations','device','pulser_version'):
        if field not in d:raise ValueError(f'Missing required field {field}')
    return jobs[0],d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();raw=a.input.read_bytes();job,obj=normalize(raw.decode('utf-8-sig'))
    a.out_dir.mkdir(parents=True,exist_ok=True)
    out=a.out_dir/'sequence_reconstituted_from_portal.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    provenance={'job_id':job,'source_file':a.input.name,'source_sha256':hashlib.sha256(raw).hexdigest(),
                'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
                'origin':'JSON reconstituted from author-provided Pasqal portal tree copy; all displayed numeric values retained.',
                'validation':'JSON syntax checked; not deserialized with Pulser in this environment.'}
    (a.out_dir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf8')
    print(out)

if __name__=='__main__':main()

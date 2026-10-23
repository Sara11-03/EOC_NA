#!/usr/bin/env python3
"""Check release files for credentials, private paths and oversized files."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', '__pycache__', '.pytest_cache', '.venv', 'venv', 'outputs', 'artifacts', 'exports', 'build', 'dist'}
PRIVATE = [re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+', re.I),
           re.compile('/'+'mnt'+'/data/'), re.compile('/'+'home'+'/oai/'),
           re.compile('/'+'Users'+'/[^/]+/'), re.compile('One'+'Drive', re.I)]
TOKEN = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)')
SENSITIVE = {'password', 'passwd', 'access_token', 'refresh_token', 'api_key', 'client_secret', 'authorization'}


def has_secret(obj):
    if isinstance(obj, dict):
        for k,v in obj.items():
            if str(k).lower() in SENSITIVE and isinstance(v,str) and v.strip() and not v.startswith(('<','YOUR_','REPLACE_')):
                return True
            if has_secret(v):return True
    elif isinstance(obj,list):return any(has_secret(v) for v in obj)
    return False


def findings(path, raw):
    problems=[]
    if len(raw)>100*1024*1024:problems.append('file exceeds 100 MiB')
    name=path.name.lower()
    if name.startswith('.env') or ('auth' in name and name.endswith('.json')) or name.endswith(('.pem','.key','.p12','.pfx','.pyc')):
        problems.append('private or generated file name')
    if path.suffix.lower() not in {'.py','.json','.geojson','.csv','.txt','.md','.toml','.yaml','.yml','.bat'} and path.name not in {'.gitignore','.gitattributes'}:
        return problems
    try:text=raw.decode('utf-8-sig')
    except UnicodeDecodeError:return problems+['unexpected text encoding']
    if any(p.search(text) for p in PRIVATE):problems.append('private absolute path')
    if TOKEN.search(text):problems.append('possible access token or private key')
    if path.suffix=='.json':
        try:
            if has_secret(json.loads(text)):problems.append('nonempty credential field')
        except json.JSONDecodeError:problems.append('invalid JSON')
    if path.suffix=='.py':
        try:
            tree=ast.parse(text)
            for n in ast.walk(tree):
                if isinstance(n,(ast.Assign,ast.AnnAssign)):
                    value=n.value
                    names=[n.target] if isinstance(n,ast.AnnAssign) else n.targets
                    if isinstance(value,ast.Constant) and isinstance(value.value,str) and value.value:
                        if any(isinstance(x,ast.Name) and x.id.lower() in SENSITIVE for x in names):
                            problems.append('literal credential assignment')
        except SyntaxError:problems.append('invalid Python syntax')
    return problems


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--staged', action='store_true', help='Inspect the Git index instead of the working tree.')
    a=p.parse_args();items=[]
    if a.staged:
        try:
            names=subprocess.check_output(['git','diff','--cached','--name-only','--diff-filter=ACMR','-z'],cwd=ROOT).decode().split('\0')
            for name in names:
                if name:items.append((Path(name),subprocess.check_output(['git','show',':'+name],cwd=ROOT)))
        except (FileNotFoundError,subprocess.CalledProcessError):
            raise SystemExit('Git index unavailable; run git add first from a cloned repository.')
    else:
        for path in sorted(ROOT.rglob('*')):
            if path.is_file() and not set(path.relative_to(ROOT).parts)&SKIP:
                items.append((path.relative_to(ROOT),path.read_bytes()))
    errors=[]
    for path,raw in items:
        for message in findings(path,raw):errors.append((path.as_posix(),message))
    for path,message in errors:print(f'CHECK: {path}: {message}')
    print(f'Files checked: {len(items)}; findings: {len(errors)}')
    if errors:raise SystemExit(2)
    print('PASS: no credential or private-path patterns found. This is not a security guarantee.')

if __name__=='__main__':main()

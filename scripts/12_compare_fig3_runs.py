#!/usr/bin/env python3
"""Compare Figure-3 outputs by graph identity (N, instance), not list position."""
from __future__ import annotations
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

METHODS = ['mDS_from_mIS', 'notMIS', 'mDS_from_mIS_quantum', 'notMIS_quantum']

def read(folder: Path):
    if not folder.is_dir():
        raise ValueError(f'Result directory does not exist: {folder}')
    out = {}
    for path in sorted(folder.glob('n*/inst*.json')):
        if not re.fullmatch(r'inst[0-9]+', path.stem):
            continue
        record = json.loads(path.read_text(encoding='utf-8'))
        if 'mDS' not in record or any(m not in record for m in METHODS):
            raise ValueError(f'Incomplete per-instance result: {path}')
        key = int(path.parent.name[1:]), int(path.stem[4:])
        for method in ['mDS', *METHODS]:
            value = record[method]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value) or value < 1:
                raise ValueError(f'Invalid size {method} in {path}')
        out[key] = record
    return out

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--historical', type=Path, required=True)
    p.add_argument('--new', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--require-identical', action='store_true')
    p.add_argument('--require-full-coverage', action='store_true')
    a = p.parse_args(argv)
    old, new = read(a.historical), read(a.new)
    keys = sorted(old.keys() & new.keys())
    if not keys:
        p.error('No paired instances')
    rows, mismatches = [], []
    matches = {m: 0 for m in ['mDS', *METHODS]}
    for n, inst in keys:
        x, y = old[n, inst], new[n, inst]
        for method in matches:
            if x[method] == y[method]:
                matches[method] += 1
            else:
                mismatches.append({'N': n, 'inst': inst, 'method': method,
                    'historical_size': x[method], 'new_size': y[method]})
            rows.append({'N': n, 'inst': inst, 'method': method,
                'historical_size': x[method], 'new_size': y[method],
                'historical_rho': x[method] / x['mDS'],
                'new_rho': y[method] / y['mDS'],
                'delta_rho': y[method] / y['mDS'] - x[method] / x['mDS']})
    report = {'historical_instances': len(old), 'new_instances': len(new),
        'paired_instances': len(keys), 'same_instance_set': old.keys() == new.keys(),
        'only_historical': sorted(old.keys() - new.keys()),
        'only_new': sorted(new.keys() - old.keys()),
        'identical_values_by_method': matches,
        'identical_value_count': sum(matches.values()),
        'compared_value_count': len(matches) * len(keys), 'mismatches': mismatches,
        'all_paired_values_identical': not mismatches}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    a.out.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'Historical {len(old)}; new {len(new)}; paired {len(keys)}')
    for method in matches:
        print(f'{method}: {matches[method]}/{len(keys)} identical')
    print(f'Overall: {sum(matches.values())}/{len(matches)*len(keys)} identical values')
    print('Saved:', a.out, 'and', a.out.with_suffix('.json'))
    if matches['mDS'] != len(keys):
        raise SystemExit('FAIL: exact MDS denominators differ; investigate graphs or solver certificates')
    if a.require_full_coverage and old.keys() != new.keys():
        raise SystemExit('FAIL: different instance coverage (see report)')
    if a.require_identical and mismatches:
        raise SystemExit('FAIL: results differ (see report); fresh sampling is not an exact replay')
    print('PASS: requested comparison checks satisfied.')

if __name__ == '__main__':
    main()

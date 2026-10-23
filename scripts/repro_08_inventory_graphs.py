#!/usr/bin/env python3
"""Inventory graph JSON files without regenerating or silently changing them."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from eoc_geo.io import load_graph,write_csv,read_json,sha256,write_json
from eoc_geo.geometry import geometry_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-dir',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=[];errors=[]
    for path in sorted(a.input_dir.rglob('*.json')):
        try:
            obj=read_json(path)
            if not isinstance(obj,dict) or 'nodes' not in obj or not ('edges' in obj or 'links' in obj):continue
            g=load_graph(path)
            row={'file':path.relative_to(a.input_dir).as_posix(),'sha256':sha256(path),'nodes':len(g),'edges':g.number_of_edges(),
                 'seed_recorded':g.graph.get('seed',None),'position_units':g.graph.get('position_units','unspecified')}
            if row['position_units']=='um' and len(g)>1 and all('pos' in d for _,d in g.nodes(data=True)):
                m=geometry_metrics(g);row.update(min_separation_um=m['min_separation_um'],max_radius_um=m['max_radius_um'])
            rows.append(row)
        except (OSError,ValueError,KeyError,TypeError) as exc:errors.append({'file':path.name,'error':str(exc)})
    if not rows:raise SystemExit('No readable node-link graphs. A summary table is not a graph archive. Export pickle graphs to JSON in the original trusted environment first.')
    write_csv(rows,a.out);write_json({'n_graphs':len(rows),'errors':errors},a.out.with_suffix('.report.json'))
    print(f'Indexed {len(rows)} graph files; errors={len(errors)}')

if __name__=='__main__':main()

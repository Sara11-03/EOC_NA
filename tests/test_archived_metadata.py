import unittest
import math
import numpy as np
from eoc_geo.io import ROOT, read_json, case_config, node_order_from_file
from eoc_geo.geometry import make_geographical_graph, build_register


class MetadataTests(unittest.TestCase):
    def test_all_reported_edge_lengths(self):
        for case in ('modena','palermo'):
            geo,_=make_geographical_graph(case,case_config(case));g,_,_=build_register(geo,case_config(case))
            for row in read_json(ROOT/'data/qpu'/case/'reported_edge_stats.json')['edge_rows']:
                length=float(np.linalg.norm(np.asarray(g.nodes[row['u']]['pos'])-g.nodes[row['v']]['pos']))
                self.assertAlmostEqual(length,row['edge_length'],places=8)

    def test_submission_ids_and_final_parameters(self):
        for case,mult in [('modena',2.0),('palermo',1.6)]:
            params=read_json(ROOT/'data/parameters'/f'{case}_qpu_submission.json')
            raw=read_json(ROOT/'data/qpu'/case/'metadata.json')
            self.assertEqual(params['job_id'],raw['jobs'][0]['job_id'])
            self.assertEqual(params['t_rise']+params['t_sweep']+params['t_fall'],3000)
            self.assertAlmostEqual(params['delta_f']/(params['Omega_max']/2),mult)

    def test_palermo_portal_qubit_mapping(self):
        seq=read_json(ROOT/'data/pulser/palermo/qpu/sequence_reconstituted_from_portal.json')
        geo,_=make_geographical_graph('palermo',case_config('palermo'));g,_,_=build_register(geo,case_config('palermo'))
        order=node_order_from_file(ROOT/'data/qpu/palermo/node_order.json')
        for i,q in enumerate(seq['register']):
            self.assertEqual(q['name'],f'q{i:03d}')
            p=g.nodes[order[i]]['pos']
            self.assertLess(math.hypot(q['x']-p[0],q['y']-p[1]),1e-6)
        self.assertEqual(sum(p['amplitude']['duration'] for p in seq['operations']),3000)

if __name__=='__main__':unittest.main()

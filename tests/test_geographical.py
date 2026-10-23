import unittest
from pathlib import Path
import tempfile
import json
import shutil
import networkx as nx
import numpy as np
from eoc_geo.io import ROOT,case_config,read_json,write_json,load_graph,save_graph,node_order_from_file
from eoc_geo.geometry import make_geographical_graph,build_register,compare_reference,geometry_metrics,load_selected
from eoc_geo.qpu import evaluate_counts,compare_archived,checked_counts,export_fetched
from eoc_geo.sources import representative_catalog,verify_selected_catalog,download_provinces
from eoc_geo import legacy_geometry as old


class HistoricalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases={}
        for case in ('modena','palermo'):
            g,_=make_geographical_graph(case,case_config(case))
            final,report,stages=build_register(g,case_config(case))
            cls.cases[case]=(g,final,report,stages)

    def test_modena_nodes_edges(self):
        initial,final,_,_=self.cases['modena']
        self.assertEqual((len(initial),initial.number_of_edges(),len(final),final.number_of_edges()),(112,170,100,145))

    def test_palermo_nodes_edges(self):
        initial,final,_,_=self.cases['palermo']
        self.assertEqual((len(initial),initial.number_of_edges(),len(final),final.number_of_edges()),(50,68,41,56))

    def test_no_actual_clustering(self):
        for case,(_,_,report,_) in self.cases.items():self.assertEqual(report['multi_node_clusters'],0,case)

    def test_clipping_counts(self):
        self.assertEqual(len(self.cases['modena'][2]['radial_clipping']),7)
        self.assertEqual(len(self.cases['palermo'][2]['radial_clipping']),9)

    def test_capacity_pruning_exact_vertices(self):
        report=self.cases['modena'][2]
        self.assertEqual([r['vertex'] for r in report['high_degree_pruning']],['Bozzolo (MN)','Castelnuovo Rangone (MO)'])
        self.assertEqual([r['degree_at_removal'] for r in report['high_degree_pruning']],[6,5])
        self.assertEqual([r['vertex'] for r in report['peripheral_pruning']],['Monte San Pietro (BO)','Bore (PR)','Zola Predosa (BO)'])

    def test_spatial_constraints_and_connectivity(self):
        for case,(_,g,report,_) in self.cases.items():
            self.assertTrue(nx.is_connected(g))
            self.assertGreaterEqual(report['final']['min_separation_um'],5.)
            self.assertLessEqual(report['final']['max_radius_um'],46.)
            self.assertAlmostEqual(report['final']['min_separation_um'],5.0001,places=8)

    def test_graph_preserved_on_survivors(self):
        for case,(_,_,r,_) in self.cases.items():
            self.assertEqual(r['surviving_haversine_edges_lost'],[])
            self.assertEqual(r['final_edges_not_in_haversine_graph'],[])

    def test_reference_coordinates(self):
        for case,(_,g,_,_) in self.cases.items():
            ref=load_graph(ROOT/'data'/'reference'/f'{case}_reconstructed_reference.json')
            self.assertTrue(compare_reference(g,ref)['passed'])

    def test_no_post_pruning_recentering(self):
        report=self.cases['modena'][2]
        self.assertAlmostEqual(report['final']['centroid_um'][0],-0.4143990844,places=8)
        self.assertAlmostEqual(report['final']['centroid_um'][1],0.4721228164,places=8)

    def test_unit_disk_equivalence(self):
        for case,(_,g,r,_) in self.cases.items():
            radius=r['final']['suggested_radius_um'];nodes=list(g)
            for i,u in enumerate(nodes):
                for v in nodes[i+1:]:
                    close=np.linalg.norm(np.asarray(g.nodes[u]['pos'])-g.nodes[v]['pos'])<=radius
                    self.assertEqual(bool(close),g.has_edge(u,v),(case,u,v))

    def test_raw_qpu_all_diagnostics(self):
        for case,(_,g,_,_) in self.cases.items():
            d=ROOT/'data'/'qpu'/case
            results=evaluate_counts(g,read_json(d/'counts.json'),node_order_from_file(d/'node_order.json'))
            validation=compare_archived(results,d/'reported_hw_stats.json',d/'reported_edge_stats.json')
            self.assertTrue(validation['passed'],validation)
            self.assertEqual(results['num_shots'],1000)

    def test_wrong_order_does_not_pass(self):
        case='modena';d=ROOT/'data'/'qpu'/case;g=self.cases[case][1]
        wrong=node_order_from_file(d/'node_order.json')[::-1]
        results=evaluate_counts(g,read_json(d/'counts.json'),wrong)
        self.assertFalse(compare_archived(results,d/'reported_hw_stats.json',d/'reported_edge_stats.json')['passed'])

    def test_recorded_dominating_sets_are_feasible(self):
        for case,(_,g,_,_) in self.cases.items():
            d=read_json(ROOT/'data'/'qpu'/case/'reported_mds_results.json')
            for name,vertices in d['mds_nodes'].items():
                self.assertTrue(set(vertices)<=set(g))
                self.assertTrue(nx.is_dominating_set(g,vertices),(case,name))
                self.assertEqual(len(vertices),d['mds_sizes'][name])

    def test_email_coordinates_same_set(self):
        for case in self.cases:
            g,_=load_selected(case,coordinates_path=ROOT/'data'/'inputs'/f'{case}_colleague_coordinates.json')
            self.assertEqual(set(g),set(self.cases[case][0]))

    def test_geometry_json_roundtrip_and_legacy_links(self):
        g=self.cases['modena'][1]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'g.json';save_graph(g,p)
            self.assertTrue(compare_reference(load_graph(p),g)['passed'])
            data=read_json(p);data['links']=data.pop('edges');write_json(data,p)
            self.assertTrue(compare_reference(load_graph(p),g)['passed'])

    def test_conflicting_json_edge_keys_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'g.json';write_json({'nodes':[{'id':'a'},{'id':'b'}],'edges':[],'links':[{'source':'a','target':'b'}]},p)
            with self.assertRaises(ValueError):load_graph(p)

    def test_raw_counts_reject_invalid_bitstrings(self):
        for bad in ({'0102':1},{'001':1},{'0010':0},{'0010':1.5}):
            with self.assertRaises(ValueError):checked_counts(bad,4)

    def test_pruning_preserves_connectivity(self):
        g=nx.path_graph(5);g=nx.relabel_nodes(g,lambda i:str(i))
        for i,v in enumerate(g):g.nodes[v]['pos']=[i,0]
        removed=old.remove_k_highest_degree_keep_connected(set(),g,1)
        self.assertIn(removed[0],['0','4']);self.assertTrue(nx.is_connected(g))

    def test_polygon_representative_points(self):
        catalog=representative_catalog([ROOT/'tests/fixtures/limits_P_36_municipalities.geojson'])
        g=self.cases['modena'][0]
        selected=g.subgraph([r['id'] for r in catalog['nodes']]).copy()
        matches=verify_selected_catalog(selected,catalog)
        self.assertEqual(len(matches),4)
        self.assertTrue(all(r['accepted'] and r['rounded_exact'] for r in matches))

    def test_download_offline_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);shutil.copyfile(ROOT/'tests/fixtures/limits_P_36_municipalities.geojson',d/'limits_P_36_municipalities.geojson')
            manifest=download_provinces([36],d,offline=True)
            self.assertEqual(manifest['files'][0]['n_features'],4)
            with self.assertRaises(FileNotFoundError):download_provinces([19],d,offline=True)

    def test_download_cache_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);p=d/'limits_P_36_municipalities.geojson'
            shutil.copyfile(ROOT/'tests/fixtures/limits_P_36_municipalities.geojson',p)
            download_provinces([36],d,offline=True)
            p.write_text(p.read_text()+' ',encoding='utf8')
            with self.assertRaises(ValueError):download_provinces([36],d,offline=True)


if __name__=='__main__':unittest.main()

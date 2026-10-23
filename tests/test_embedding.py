import networkx as nx

from eoc_na.embedding import embed_graph_for_device


def test_embedding_produces_positions_and_indices():
    G = nx.Graph(edge_radius_km=13.0)
    G.add_node("a", lon=11.0, lat=44.0, original_members=["a"], name="a")
    G.add_node("b", lon=11.01, lat=44.0, original_members=["b"], name="b")
    G.add_node("c", lon=11.20, lat=44.0, original_members=["c"], name="c")
    G.add_edge("a", "b", distance_km=1.0)
    H, report = embed_graph_for_device(G, edge_radius_km=13.0, min_atom_distance_um=1.0, cluster_close=False)
    assert H.number_of_nodes() == 3
    assert all("pos" in H.nodes[v] for v in H.nodes())
    assert all("idx" in H.nodes[v] for v in H.nodes())
    assert report.n_output_atoms == 3

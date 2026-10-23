import networkx as nx

from eoc_na.mds import run_two_branch_pipeline
from eoc_na.samples import nodeset_to_bitstring


def test_two_branch_pipeline_returns_dominating_set_from_quantum_like_samples():
    G = nx.path_graph(5)
    for i, v in enumerate(G.nodes()):
        G.nodes[v]["idx"] = i
    samples = [nodeset_to_bitstring(G, {0, 2, 4}), nodeset_to_bitstring(G, {1, 3})]
    result = run_two_branch_pipeline(G, raw_quantum_samples=samples, seed=0)
    D = set(int(x) for x in result.best_set)
    assert nx.is_dominating_set(G, D)
    assert result.best_size <= 2

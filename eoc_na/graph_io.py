"""Compatibility replacement for the small eoc_na.graph_io module supplied here."""
from __future__ import annotations
import json
import inspect
from pathlib import Path
from typing import Any
import networkx as nx
from networkx.readwrite import json_graph


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(obj: Any,path: str | Path) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8')


def save_graph(G: nx.Graph,path: str | Path) -> None:
    key='edges' if 'edges' in inspect.signature(json_graph.node_link_data).parameters else 'link'
    write_json(json_graph.node_link_data(G,**{key:'edges'}),path)


def load_graph(path: str | Path) -> nx.Graph:
    data=read_json(path)
    if 'edges' in data and 'links' in data and data['edges']!=data['links']:
        raise ValueError('Conflicting edges and links fields')
    edge_key='edges' if 'edges' in data else 'links' if 'links' in data else None
    if edge_key is None:raise ValueError('Missing edges/links field')
    kw='edges' if 'edges' in inspect.signature(json_graph.node_link_graph).parameters else 'link'
    return json_graph.node_link_graph(data,**{kw:edge_key})


def ordered_nodes(G: nx.Graph) -> list[Any]:
    explicit=G.graph.get('bitstring_node_order')
    if explicit is not None:
        if len(explicit)!=len(G) or len(set(explicit))!=len(G) or set(explicit)!=set(G):
            raise ValueError('Invalid graph bitstring_node_order metadata')
        return list(explicit)
    def key(v):
        idx=G.nodes[v].get('idx')
        return (1,str(v)) if idx is None else (0,int(idx))
    return sorted(G.nodes(),key=key)

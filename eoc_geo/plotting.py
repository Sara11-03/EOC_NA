"""Geographical and chamber plots; no network request is made implicitly."""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
from pyproj import Transformer
from .io import read_json


def plot_geographical(g, out: Path, boundaries: list[Path] | None = None, labels: str='key', show_edges: bool=True, solution: list | None = None, formats=('png','svg'), show: bool=False):
    import matplotlib
    if not show:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import MaxNLocator
    out.parent.mkdir(parents=True,exist_ok=True)
    nodes=list(g)
    xy=np.array([(g.nodes[v]['lon'],g.nodes[v]['lat']) for v in nodes])
    if not len(nodes):raise ValueError('Cannot plot an empty graph')
    fig,ax=plt.subplots(figsize=(10.2,7.6))
    margin=np.maximum(xy.ptp(axis=0) if hasattr(xy,'ptp') else np.ptp(xy,axis=0),0.05)*0.10
    lo=xy.min(axis=0)-margin;hi=xy.max(axis=0)+margin
    n_boundaries=0
    if boundaries:
        segments=[]
        for path in boundaries:
            for feat in read_json(path).get('features',[]):
                geom=feat['geometry'];kind=geom['type']
                polys=[geom['coordinates']] if kind=='Polygon' else geom['coordinates'] if kind=='MultiPolygon' else []
                for poly in polys:
                    arr=np.asarray(poly[0])[:,:2]
                    if np.any(arr.max(axis=0)<lo) or np.any(arr.min(axis=0)>hi):continue
                    segments.append(arr);n_boundaries+=1
        if segments:ax.add_collection(LineCollection(segments,linewidths=.45,alpha=.30,zorder=0))
    pos=dict(zip(nodes,xy))
    if show_edges:
        ax.add_collection(LineCollection([[pos[u],pos[v]] for u,v in g.edges()],linewidths=.8,alpha=.65,zorder=1))
    ax.scatter(xy[:,0],xy[:,1],s=27,zorder=2,label=f'{len(g)} municipalities')
    if solution:
        missing=set(solution)-set(g)
        if missing:raise ValueError(f'Solution has missing nodes: {missing}')
        pp=np.array([pos[v] for v in solution]);ax.scatter(pp[:,0],pp[:,1],s=100,marker='*',zorder=3,label=f'{len(solution)} selected hubs')
    key={'Modena','Reggio nell\u0027Emilia','Bologna','Palermo','Parma','Mantova','Cremona'}
    for v in nodes:
        name=g.nodes[v].get('pretty',str(v).rsplit(' (',1)[0])
        if labels=='all' or labels=='key' and name in key:
            ax.annotate(name,pos[v],xytext=(5,5),textcoords='offset points',fontsize=8 if labels=='all' else 10)
    ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_aspect(1/math.cos(math.radians(float(xy[:,1].mean()))))
    ax.xaxis.set_major_locator(MaxNLocator(6));ax.yaxis.set_major_locator(MaxNLocator(6))
    ax.set_xlabel('Longitude (degrees east)');ax.set_ylabel('Latitude (degrees north)')
    case=g.graph.get('case','Geographical instance').capitalize()
    ax.set_title(f'{case}: {len(g)} vertices, {g.number_of_edges()} edges',pad=14)
    ax.grid(alpha=.18);ax.legend(loc='best',frameon=False)
    note='Municipality points: author-supplied historical selection; source: Openpolis / geojson-italy / ISTAT.'
    if n_boundaries:note+=' Boundaries: supplied GeoJSON files (coverage may be partial).'
    fig.text(.5,.015,note,ha='center',fontsize=7,wrap=True)
    fig.tight_layout(rect=(0,.045,1,1))
    for fmt in formats:fig.savefig(out.with_suffix('.'+fmt),dpi=220)
    if show:plt.show()
    plt.close(fig)


def plot_chamber(g, out: Path, radius_um: float=46., solution: list | None=None, formats=('png','svg'), show: bool=False):
    import matplotlib
    if not show:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle
    from .geometry import geometry_metrics
    metrics=geometry_metrics(g)
    pos={v:np.asarray(a['pos']) for v,a in g.nodes(data=True)}
    fig,ax=plt.subplots(figsize=(8.0,8.0))
    ax.add_patch(Circle((0,0),radius_um,fill=False,linewidth=1.3,linestyle='--',label=f'Configured radius: {radius_um:g} um'))
    ax.add_collection(LineCollection([[pos[u],pos[v]] for u,v in g.edges()],linewidths=.8,alpha=.7,zorder=1))
    pp=np.array(list(pos.values()));ax.scatter(pp[:,0],pp[:,1],s=28,zorder=2,label=f'{len(g)} atoms')
    if solution:
        if not set(solution)<=set(g):raise ValueError('Unknown solution nodes')
        sp=np.array([pos[v] for v in solution]);ax.scatter(sp[:,0],sp[:,1],s=100,marker='*',zorder=3,label='Selected hubs')
    ax.scatter([0],[0],s=35,marker='+',label='Coordinate origin')
    ax.set_aspect('equal');ax.set_xlim(-radius_um-5,radius_um+5);ax.set_ylim(-radius_um-5,radius_um+5)
    ax.set_xlabel('x (um)');ax.set_ylabel('y (um)');ax.grid(alpha=.15)
    ax.set_title(f'{g.graph.get("case","Register").capitalize()} atomic register\n{len(g)} atoms, {g.number_of_edges()} edges',pad=12)
    ax.legend(loc='upper right',fontsize=8,frameon=False)
    fig.text(.5,.025,f'Minimum separation: {metrics["min_separation_um"]:.4f} um; maximum radius: {metrics["max_radius_um"]:.3f} um',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1))
    for fmt in formats:fig.savefig(out.with_suffix('.'+fmt),dpi=220)
    if show:plt.show()
    plt.close(fig)

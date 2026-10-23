"""Download boundary snapshots and recover representative municipality points."""
from __future__ import annotations
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform
from .io import write_json, read_json, sha256, write_csv, environment
from .legacy_geometry import haversine_km

REPOSITORIES = ('guglielmo/geojson-italy', 'openpolis/geojson-italy')
DEFAULT_REF = '2023.1'


def download_provinces(codes: list[int], directory: Path, ref: str = DEFAULT_REF, offline: bool = False, overwrite: bool = False) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    records=[]
    for code in codes:
        filename=f'limits_P_{code}_municipalities.geojson'
        path=directory/filename
        sidecar=path.with_suffix('.source.json')
        downloaded=False
        if path.exists() and not overwrite:
            payload=path.read_bytes()
            source=read_json(sidecar) if sidecar.exists() else {'url':None,'ref':None,'origin':'preexisting local file; source revision unknown'}
            recorded_hash=source.get('sha256')
            if recorded_hash and recorded_hash != hashlib.sha256(payload).hexdigest():
                raise ValueError(f'Cached file checksum differs from provenance: {path}')
            if not offline and source.get('ref') not in (None, ref):
                raise ValueError(f'Cached ref {source.get("ref")} differs from requested {ref}. Use another directory or --overwrite.')
        else:
            if offline:
                raise FileNotFoundError(f'Missing cached province {code}: {path}')
            errors=[]
            for repository in REPOSITORIES:
                url=f'https://raw.githubusercontent.com/{repository}/{ref}/geojson/{filename}'
                for attempt in range(2):
                    try:
                        request=Request(url,headers={'User-Agent':'eoc-geographical-repro/0.1','Accept':'application/json'})
                        with urlopen(request,timeout=60) as response:
                            payload=response.read()
                            final_url=response.geturl()
                        parsed=json.loads(payload)
                        if parsed.get('type')!='FeatureCollection' or not parsed.get('features'):
                            raise ValueError('Not a nonempty GeoJSON FeatureCollection')
                        source={'url':url,'resolved_url':final_url,'ref':ref,'downloaded_utc':datetime.now(timezone.utc).isoformat(),'origin':'download'}
                        downloaded=True
                        break
                    except (URLError,HTTPError,OSError,ValueError) as error:
                        errors.append(f'{url}: {error}')
                        if attempt==0:time.sleep(0.4)
                if downloaded:break
            if not downloaded:
                raise RuntimeError(f'Cannot download province {code}. Offline reconstruction remains available.\n'+'\n'.join(errors))
            tmp=path.with_suffix('.tmp')
            tmp.write_bytes(payload);tmp.replace(path)
        obj=json.loads(payload)
        if obj.get('type')!='FeatureCollection':raise ValueError(f'Invalid cache: {path}')
        source.update(filename=filename,province_code=code,sha256=hashlib.sha256(payload).hexdigest(),n_features=len(obj['features']))
        write_json(source,sidecar)
        records.append(source)
        print(f'{filename}: {len(obj["features"])} features ({"downloaded" if downloaded else "cached"})')
    manifest={'requested_ref':ref,'offline':offline,'files':records,'attribution':'Municipality boundaries: ISTAT; distributed by geojson-italy (formerly Openpolis). See THIRD_PARTY_DATA.md.','environment':environment()}
    write_json(manifest,directory/'download_manifest.json')
    return manifest


def representative_catalog(paths: list[Path]) -> dict:
    to_map=Transformer.from_crs(4326,3857,always_xy=True).transform
    to_ll=Transformer.from_crs(3857,4326,always_xy=True).transform
    catalog=[];seen=set()
    for path in paths:
        obj=read_json(path)
        for feature in obj.get('features',[]):
            props=feature.get('properties',{})
            if not props.get('name') or 'com_istat_code_num' not in props:
                raise ValueError(f'Missing municipality metadata in {path}')
            code=int(props['com_istat_code_num'])
            if code in seen:raise ValueError(f'Duplicate municipality ISTAT code {code}')
            geom=shape(feature['geometry'])
            if geom.is_empty or not geom.is_valid:
                raise ValueError(f'Empty/invalid polygon for {props["name"]}; no automatic geometry repair is applied.')
            planar=transform(to_map,geom)
            point=transform(to_ll,planar.representative_point())
            x,y=to_map(point.x,point.y)
            province=props.get('prov_acr')
            if not province:raise ValueError(f'Missing province abbreviation for {props["name"]}')
            catalog.append({'id':f'{props["name"]} ({province})','pretty':props['name'], 'istat_code':code,'province':province,
                            'lon':float(point.x),'lat':float(point.y),'x_3857':float(x),'y_3857':float(y),
                            'source_file':path.name,'source_sha256':sha256(path)})
            seen.add(code)
    catalog.sort(key=lambda a:a['istat_code'])
    return {'point_method':'EPSG:4326 -> EPSG:3857, representative_point(), -> EPSG:4326 (historical method)',
            'nodes':catalog,'environment':environment()}


def verify_selected_catalog(selected, catalog: dict, tolerance_m: float = 30.) -> list[dict]:
    rows=catalog['nodes']
    if not rows:raise ValueError('Empty downloaded municipality catalogue')
    matches=[]
    for node,a in selected.nodes(data=True):
        lon,lat=a['lon'],a['lat']
        dist,row=min(((1000*haversine_km(lon,lat,r['lon'],r['lat']),r) for r in rows),key=lambda x:x[0])
        same_name=node==row['id']
        rounded_same=round(lon,10)==round(row['lon'],10) and round(lat,10)==round(row['lat'],10)
        matches.append({'selected_id':node,'downloaded_id':row['id'],'distance_m':dist,'same_name':same_name,
                        'rounded_exact':rounded_same,'within_tolerance':dist<=tolerance_m,
                        'accepted':same_name and dist<=tolerance_m})
    return matches

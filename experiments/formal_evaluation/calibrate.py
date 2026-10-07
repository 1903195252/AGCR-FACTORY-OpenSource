"""Freeze exit-region controls using validation references before test scoring."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import read_json,write_json,sha256
from .evaluate import region_key


def fit_regions(index_path,output,padding_m=.25,development=False):
    index=read_json(index_path)
    if index.get('development_only') and not development:raise ValueError('Formal calibration requires formal data')
    points=defaultdict(lambda:defaultdict(list));families=[]
    for family in index['families']:
        if family['split']!='val':continue
        families.append(family['family_id'])
        for record in family['records']:
            if record['camera']!=0:continue
            with np.load(Path(index['root'])/record['path'],allow_pickle=False) as native:
                key=region_key(family,native);points[key][record['world']].append(native['world_state'][-1,family['target_index'],:3])
    if not families:raise ValueError('No validation families')
    regions={}
    for key,worlds in points.items():
        if set(worlds)!={0,1}:raise ValueError('Unpaired calibration worlds')
        regions[key]=[]
        for w in (0,1):
            p=np.stack(worlds[w]);lo=p.min(0)-padding_m;hi=p.max(0)+padding_m
            regions[key].append(dict(min_xyz_m=lo.tolist(),max_xyz_m=hi.tolist(),validation_examples=len(p)))
        a,b=regions[key]
        if np.all(np.maximum(a['min_xyz_m'],b['min_xyz_m'])<=np.minimum(a['max_xyz_m'],b['max_xyz_m'])):
            raise ValueError('Overlapping exit regions: '+key+'; predefine disjoint physical regions, do not resolve with nearest test trajectory')
    artifact=dict(schema='agcr-fixed-exit-regions-1',split='val',frozen=True,development_only=development,
        calibration='axis-aligned canonical exit-center envelope plus fixed %.3fm margin'%padding_m,
        index_sha256=sha256(index_path),families=sorted(families),regions=regions)
    write_json(output,artifact);return artifact


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',required=True);p.add_argument('--output',required=True)
    p.add_argument('--padding-m',type=float,default=.25);p.add_argument('--development',action='store_true');a=p.parse_args()
    if a.padding_m<=0:raise ValueError('Positive prespecified padding required')
    fit_regions(a.index,a.output,a.padding_m,a.development)
if __name__=='__main__':main()

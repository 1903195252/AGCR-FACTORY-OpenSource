"""Fetch primary metric assets, record hashes and execute real CPU forwards."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess
import urllib.request
import numpy as np
from .common import sha256,write_json
from .video import I3D_URL,I3D_SOURCE,FVD_REFERENCE,LPIPS_SOURCE,reference_resize224


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    path=root/'i3d_torchscript.pt'
    if not path.exists():
        req=urllib.request.Request(I3D_URL,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=120) as source,(root/'i3d.download').open('wb') as dest:
            while True:
                data=source.read(8*1024*1024)
                if not data:break
                dest.write(data)
        (root/'i3d.download').replace(path)
    import torch,lpips,trackeval,pycocotools
    torch.set_num_threads(4)
    model=torch.jit.load(str(path),map_location='cpu').eval()
    with torch.no_grad():
        clip=torch.zeros(1,16,32,48,3,dtype=torch.uint8)
        feature=model(reference_resize224(clip),rescale=False,resize=False,return_features=True)
    if tuple(feature.shape)!=(1,400) or not torch.isfinite(feature).all():raise ValueError('Invalid official I3D output')
    perceptual=lpips.LPIPS(net='alex',version='0.1',verbose=False).eval().requires_grad_(False)
    with torch.no_grad():distance=perceptual(torch.zeros(1,3,64,64),torch.zeros(1,3,64,64)).item()
    if abs(distance)>1e-7:raise ValueError('LPIPS self identity failed')
    from .tracking import official_data,evaluate_tracking,combine_tracking
    row=dict(episode='cpu-contract',frame=0,view='air',canvas_size=[64,64],detections=[dict(track_id=7,class_id=1,box_xyxy=[1,1,20,20],score=1.)])
    frames=[row,dict(row,view='ground')];tracking=combine_tracking(list(evaluate_tracking(frames,frames).values()))
    if abs(tracking['HOTA']-1)>1e-8 or abs(tracking['IDF1']-1)>1e-8:raise ValueError('Official TrackEval self match failed')
    source=Path(trackeval.__file__).resolve().parents[1]
    try:commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    except subprocess.CalledProcessError:raise ValueError('TrackEval must retain official git commit provenance')
    weights={}
    for file in (Path(torch.hub.get_dir())/'checkpoints').glob('alexnet*.pth'):weights[str(file)]=sha256(file)
    lpweight=Path(lpips.__file__).parent/'weights'/'v0.1'/'alex.pth';weights[str(lpweight)]=sha256(lpweight)
    evidence=dict(schema='agcr-official-metric-assets-1',i3d=dict(url=I3D_URL,source=I3D_SOURCE,reference=FVD_REFERENCE,sha256=sha256(path),shape=list(feature.shape)),
        lpips=dict(source=LPIPS_SOURCE,version=importlib.metadata.version('lpips'),weights=weights,self_distance=distance),
        trackeval=dict(source='https://github.com/JonathonLuiten/TrackEval',commit=commit,numpy_compatibility='removed float/int aliases mapped to identical float64/int64; no equation changes',HOTA_self=tracking['HOTA'],IDF1_self=tracking['IDF1']),
        pycocotools_version=importlib.metadata.version('pycocotools'),scope='CPU backend validity only; no experiment results')
    write_json(root/'assets.json',evidence);print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()

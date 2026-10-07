"""Actual-weight, actual-resolution acceptance; no replacement toy backbone."""
import argparse
import itertools
import json
from pathlib import Path
import time

import torch
from .data import FamilyDataset,to_device
from .system import JointSystem


def main():
    p=argparse.ArgumentParser();p.add_argument('--paths',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--decode',action='store_true')
    a=p.parse_args();torch.set_num_threads(8);torch.manual_seed(17)
    paths=json.loads(Path(a.paths).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text())
    variant=next(x for x in protocol['essential_variants'] if x['id']=='full')
    model=JointSystem(paths,a.device,'video',variant)
    ds=FamilyDataset(paths['families'],'train');family=ds[0]
    branch=to_device(family['branches'][(0,0)],a.device)
    g=torch.Generator(device=a.device).manual_seed(1701)
    print('Preparing native history',flush=True)
    with torch.no_grad():
        _,history,draw,c=model.prepared(branch,g,False)
    print('Predicted tracks',len(history['result']['tracks']),flush=True)
    report=dict(development_only=True,scientific_results=False,tracks=len(history['result']['tracks']))
    torch.save({k:v.cpu() for k,v in c.items()},out/'predicted_conditions.pt')
    z=torch.randn(2,48,5,30,46,device=a.device,generator=g)
    s=torch.tensor(.5,device=a.device)
    print('Full adapter forward',flush=True)
    with torch.no_grad():
        own=model.wan(z,s,c)
        t=torch.ones(2,1725,device=a.device)*500;t[:,:345]=0
        official=model.wan.backbone(z.to(torch.bfloat16),t,model.wan.empty_text.expand(2,-1,-1)).sample.float()
    error=float((own-official).abs().max());report['zero_adapter_native_max_error']=error
    if error>1e-5: raise AssertionError('Zero adapters changed the official transformer')
    print('Zero adapter equivalence',error,flush=True)
    with torch.no_grad():
        cond=model.condition(c);q=torch.randn(2,4,345,256,device=a.device,generator=g)
        errors=[]
        for adapter in model.wan.adapters:
            ref=adapter.appearance(q,cond,model.wan.gate,False)
            for order in itertools.permutations(range(4)):
                value=adapter.appearance(q,cond,model.wan.gate,False,list(order))
                errors.append(float((value-ref).abs().max()))
    report['mean_gate_10_locations_24_permutations_max_error']=max(errors)
    if max(errors)>1e-5: raise AssertionError('Appearance control leaks frame order')
    print('Full appearance invariance',max(errors),flush=True)
    if a.decode:
        with torch.no_grad():
            rgb=torch.cat([branch['inputs']['rgb'][-1:],branch['supervision']['rgb']],0)
            latent=model.vae.encode(rgb)
            decoded=model.decode(latent,c)
        report['decoded_shapes']={k:list(v.shape) for k,v in decoded.items()}
        report['depth_unique_values_first_frame']=int(torch.unique(decoded['depth_mm'][0,0]).numel())
        if report['depth_unique_values_first_frame']<=1024: raise AssertionError('Metric depth head has been quantized')
        if not all(torch.isfinite(v).all() for v in decoded.values()): raise AssertionError('Nonfinite native RGB-D')
        print('Decoded',report['decoded_shapes'],flush=True)
    report['passed']=True;report['peak_gpu_bytes']=torch.cuda.max_memory_allocated(a.device)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__': main()

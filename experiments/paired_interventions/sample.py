"""Eight paired pure-noise samples per controlled history, with provenance.

Exports generated RGB-D and internal states. Auxiliary reader outputs are
labelled as training-reader diagnostics, never independent benchmark scores.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .data import FamilyDataset,to_device,file_sha256,historical_labels
from .world import predict_world,world_conditions,query_labels
from .system import JointSystem
from .sampling import pure_noise_euler
from .train import import_weights,atomic_json


def perturbed_family(family,config,seed):
    if not config or config.get('kind')=='clean':return family
    from experiments.formal_evaluation.perturbations import degrade_history
    branches={}
    for (world,camera),branch in family['branches'].items():
        alternate=family['branches'][(1-world,camera)]['inputs']
        inputs=degrade_history(branch['inputs'],config,seed,
            partner=1-int(family['metadata']['receiver']),alternate_history=alternate)
        branches[(world,camera)]=dict(branch,inputs=inputs)
    return dict(family,branches=branches)


def stream_seed(seed,family,draw,stream):
    payload=json.dumps([seed,family,draw,stream],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],'big')%(2**63-1)


@torch.no_grad()
def generate_family(model,family,sample_seed,training_seed,device):
    name=family['metadata']['family_id'];outputs={}
    generator=lambda stream:torch.Generator(device=device).manual_seed(stream_seed(training_seed,name,sample_seed,stream))
    for world in range(2):
        branch=family['branches'][(world,0)];inputs=branch['inputs']
        encoded=model.encoders.history(inputs['rgb'])
        history,draw=predict_world(model.upstream,inputs,encoded,generator('state'))
        ids=query_labels(history,historical_labels(branch))
        anchor=encoded[1][-1,:,None]
        noises=torch.stack([torch.randn(4,48,30,46,device=device,generator=generator(stream)) for stream in ('air','ground')])
        for camera in range(2):
            current=family['branches'][(world,camera)]
            c=world_conditions(model.upstream,history,draw,current['inputs'])
            def velocity(z,s,condition): return model.wan(z.permute(0,2,1,3,4),s,condition).permute(0,2,1,3,4)
            z=pure_noise_euler(velocity,noises,anchor,c,50,2).permute(0,2,1,3,4)
            decoded=model.decode(z,c)
            # Re-run each view's image reader; never copy internal state as a
            # generated-image estimate or use it for independent visual scores.
            reader=model.video_readout(decoded,c,current['inputs'])
            result=dict(rgb=((decoded['rgb'].permute(0,1,3,4,2).clamp(-1,1)+1)*127.5).round().byte(),
                depth_mm=decoded['depth_mm'],validity_probability=decoded['validity_logits'].sigmoid(),
                internal_state=draw['state'][0],state_present=draw['present'][0],historical_reference_ids=ids,
                auxiliary_centers=reader['center'],auxiliary_identity=reader['descriptor'],
                auxiliary_masks=reader['mask_logits'].sigmoid(),predicted_visible_layers=c['M'],
                predicted_full_layers=c['P'],predicted_surface_depth_mm=c['Z'],
                initial_state=history['result']['initial_state'],
                class_id=torch.stack([t['class_prob'][1:].argmax()+1 for t in history['result']['tracks']]) if history['result']['tracks'] else ids.new_empty(0),
                time_s=current['inputs']['future_time_s'],
                K=current['inputs']['K'],camera_to_world=current['inputs']['planned_camera_to_world'])
            outputs[(world,camera)]={k:v.cpu().numpy() for k,v in result.items()}
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paths',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--development',action='store_true');p.add_argument('--gpu-hours',type=float,required=True)
    p.add_argument('--perturbation-json',help='Registered history-only degradation; future references stay unchanged')
    p.add_argument('--limit',type=int);p.add_argument('--draws',type=int,default=8);a=p.parse_args()
    if not a.development and (a.draws!=8 or a.limit is not None): raise ValueError('Formal sampling uses every registered family and all eight draws')
    if not 1<=a.draws<=8 or a.gpu_hours<=0 or (a.limit is not None and a.limit<1):
        raise ValueError('Use 1..8 draws, a positive GPU budget and a positive family limit')
    torch.set_num_threads(8);paths=json.loads(Path(a.paths).read_text())
    perturbation=json.loads(Path(a.perturbation_json).read_text()) if a.perturbation_json else {'kind':'clean'}
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if checkpoint['identity']['phase']!='video':
        raise ValueError('Sampling requires a video checkpoint, not a predecessor stage')
    if not a.development and (checkpoint['identity']['development_only'] or not checkpoint['completed']):
        raise ValueError('A completed formal generator is required')
    protocol_path=Path(__file__).with_name('protocol.json')
    source={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')}
    protocol=json.loads(protocol_path.read_text());variant=next(v for v in protocol['essential_variants'] if v['id']==checkpoint['identity']['variant'])
    if not a.development:
        identity=checkpoint['identity']
        if source!=identity['source_sha256'] or file_sha256(protocol_path)!=identity['protocol_sha256']:
            raise ValueError('Formal sampling source/protocol differs from its training checkpoint')
        for name,key in [('upstream','parent_sha256'),('statistics','statistics_sha256'),('empty_text','empty_text_sha256'),('families','family_index_sha256')]:
            if file_sha256(paths[name])!=identity[key]:
                raise ValueError('Formal sampling input provenance changed: '+name)
    device=torch.device('cuda:0');model=JointSystem(paths,device,'video',variant)
    required={name for name in model.state_dict() if not name.startswith('wan.backbone.') or 'lora_' in name}
    missing=required-set(checkpoint['model'])
    if missing: raise ValueError('Incomplete generator checkpoint: '+str(sorted(missing)[:8]))
    import_weights(model,checkpoint['model'])
    model.eval().requires_grad_(False)
    dataset=FamilyDataset(paths['families'],a.split)
    if not a.development and dataset.index['development_only']: raise ValueError('Development family index')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    seeds=protocol['paired_sample_seeds'][:a.draws]
    manifest=dict(schema='agcr-intervention-predictions-1',split=a.split,sample_seeds=seeds,
        checkpoint_sha256=file_sha256(a.checkpoint),index_sha256=file_sha256(paths['families']),
        training_seed=checkpoint['identity']['seed'],development_only=a.development,
        source_sha256=source,protocol_sha256=file_sha256(protocol_path),
        conditions='history_inferred_state_only',reader_scope='frozen_training_reader_diagnostics',
        independent_visual_metrics=False,primary_aggregation='mean_of_8' if a.draws==8 else 'development_mean_of_%d'%a.draws,
        family_limit=a.limit,perturbation=perturbation)
    if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text())!=manifest:
        raise ValueError('Refusing to mix prediction provenance')
    atomic_json(root/'manifest.json',manifest);start=time.monotonic();completed=0
    family_count=min(len(dataset),a.limit or len(dataset))
    for index in range(family_count):
        family=to_device(dataset[index],device);name=family['metadata']['family_id']
        family=perturbed_family(family,perturbation,stream_seed(manifest['training_seed'],name,0,'degradation'))
        for draw,seed in enumerate(seeds):
            marker=root/name/('draw_%02d.json'%draw)
            if marker.exists():
                saved=json.loads(marker.read_text())
                if saved['seed']!=seed or any(not (marker.parent/name).is_file() or file_sha256(marker.parent/name)!=sha for name,sha in saved['files'].items()):
                    raise ValueError('Prediction completion marker has missing or altered files: '+str(marker))
                completed+=1;continue
            if (time.monotonic()-start)/3600>=a.gpu_hours: return
            values=generate_family(model,family,seed,manifest['training_seed'],device)
            marker.parent.mkdir(parents=True,exist_ok=True);files={}
            for (world,camera),arrays in values.items():
                path=marker.parent/('world%d_camera%d_sample%02d.npz'%(world,camera,draw))
                tmp=path.with_suffix('.npz.tmp')
                with tmp.open('wb') as stream: np.savez_compressed(stream,**arrays)
                tmp.replace(path);files[path.name]=file_sha256(path)
            atomic_json(marker,dict(seed=seed,files=files,metadata=family['metadata']));completed+=1
            atomic_json(root/'status.json',dict(completed_family_draws=completed,total_family_draws=family_count*len(seeds)))
            print(json.dumps(dict(family=name,draw=draw,completed=completed)),flush=True)
    atomic_json(root/'completion.json',dict(completed=True,development_only=a.development,draws_per_history=len(seeds)))


if __name__=='__main__': main()

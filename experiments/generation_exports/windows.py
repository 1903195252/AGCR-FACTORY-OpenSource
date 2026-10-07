"""History-only joint RGB-D proposals from frozen ordinary fixed windows.

The generated manifest is consumed by independent downstream annotation.
Candidate state/geometry are diagnostic conditions, not image measurements.
"""
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from experiments.paired_interventions.data import OrdinaryDataset,to_device,file_sha256
from experiments.paired_interventions.system import JointSystem
from experiments.paired_interventions.world import predict_world,world_conditions
from experiments.paired_interventions.sample import stream_seed
from experiments.paired_interventions.sampling import pure_noise_euler
from experiments.paired_interventions.train import import_weights,atomic_json
from experiments.paper_experiments.agcr_exp.protocol import DEFAULT


def fixed_rows(dataset,count,seed):
    rows=[r for r in dataset.dataset.rows if r['start'] in DEFAULT['evaluation_starts']]
    if count and len(rows)<count:raise ValueError('Insufficient independent qualified fixed windows: %d < %d'%(len(rows),count))
    from experiments.paper_experiments.agcr_exp.selection import stratified_order
    return stratified_order(rows,seed,limit=count)



@torch.no_grad()
def generate_pair(model,inputs,uid,draw_seed,training_seed,device):
    generator=lambda stream:torch.Generator(device=device).manual_seed(stream_seed(training_seed,uid,draw_seed,stream))
    encoded=model.encoders.history(inputs['rgb'])
    history,draw=predict_world(model.upstream,inputs,encoded,generator('state'))
    c=world_conditions(model.upstream,history,draw,inputs)
    anchor=encoded[1][-1,:,None]
    noises=torch.stack([torch.randn(4,48,30,46,device=device,generator=generator(stream)) for stream in ('air','ground')])
    def velocity(z,s,condition):return model.wan(z.permute(0,2,1,3,4),s,condition).permute(0,2,1,3,4)
    z=pure_noise_euler(velocity,noises,anchor,c,50,2).permute(0,2,1,3,4)
    decoded=model.decode(z,c)
    tracks=history['result']['tracks']
    arrays=dict(rgb=((decoded['rgb'].permute(0,1,3,4,2).clamp(-1,1)+1)*127.5).round().byte(),
        depth_mm=decoded['depth_mm'],validity_probability=decoded['validity_logits'].sigmoid(),
        initial_state=history['result']['initial_state'],internal_state=draw['state'][0],state_present=draw['present'][0],
        class_id=torch.stack([t['class_prob'][1:].argmax()+1 for t in tracks]) if tracks else torch.empty(0,device=device,dtype=torch.long),
        predicted_visible_layers=c['M'],predicted_full_layers=c['P'],predicted_surface_depth_mm=c['Z'],
        K=inputs['K'],camera_to_world=inputs['planned_camera_to_world'],time_s=inputs['future_time_s'])
    return {k:v.cpu().numpy() for k,v in arrays.items()}


def save_npz(path,values):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.npz.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,**values)
    temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('paths','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--split',default='train',choices=['train','augmentation_train','val','id_test','test'])
    p.add_argument('--count',type=int,default=4000);p.add_argument('--draws',type=int,default=8)
    p.add_argument('--gpu-hours',type=float,required=True);p.add_argument('--development',action='store_true')
    p.add_argument('--min-free-gib',type=float,default=25);a=p.parse_args()
    if not 1<=a.draws<=8 or a.count<1 or a.gpu_hours<=0:raise ValueError('Positive budget/count and 1..8 draws required')
    if not a.development and a.draws!=8:raise ValueError('The registered proposal pool uses eight fixed draws')
    start=time.monotonic();torch.set_num_threads(8)
    paths=json.loads(Path(a.paths).read_text());cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if cp['identity']['phase']!='video':raise ValueError('A complete video checkpoint is required')
    if not a.development and (cp['identity']['development_only'] or not cp['completed']):raise ValueError('Development/incomplete generator')
    protocol_path=Path('experiments/paired_interventions/protocol.json');protocol=json.loads(protocol_path.read_text())
    source={q.name:file_sha256(q) for q in protocol_path.parent.glob('*.py')}
    if not a.development:
        if source!=cp['identity']['source_sha256'] or file_sha256(protocol_path)!=cp['identity']['protocol_sha256']:raise ValueError('Changed trained source or protocol')
        for name,key in [('ordinary','ordinary_index_sha256'),('families','family_index_sha256'),('upstream','parent_sha256'),('statistics','statistics_sha256'),('empty_text','empty_text_sha256')]:
            if file_sha256(paths[name])!=cp['identity'][key]:raise ValueError('Changed provenance: '+name)
    ds=OrdinaryDataset(paths['ordinary'],a.split);rows=fixed_rows(ds,a.count,cp['identity']['seed']);ds.dataset.rows=rows
    if not a.development and ds.index['development_only']:raise ValueError('Development source index')
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    meta=dict(schema='agcr-generated-fixed-windows-1',role=a.split,checkpoint_sha256=file_sha256(a.checkpoint),
        split_sha256=ds.index['split_sha256'],ordinary_index_sha256=file_sha256(paths['ordinary']),
        selected_uids=[r['uid'] for r in rows],sample_seeds=protocol['paired_sample_seeds'][:a.draws],
        development_only=a.development,coordinate_frame='ground_anchor_local_m',depth_unit='mm',
        state_source='history_inferred_unfiltered_candidates',geometry_padding=dict(height=480,padded_width=736,crop_left=8,crop_right=8),
        export_source_sha256=file_sha256(__file__))
    if (root/'generation_contract.json').exists() and json.loads((root/'generation_contract.json').read_text())!=meta:raise ValueError('Changed generation cohort/provenance')
    atomic_json(root/'generation_contract.json',meta)
    device=torch.device('cuda:0');variant=next(v for v in protocol['essential_variants'] if v['id']==cp['identity']['variant'])
    model=JointSystem(paths,device,'video',variant);import_weights(model,cp['model']);model.eval().requires_grad_(False)
    clips=[]
    def publish(complete=False,reason=None):
        atomic_json(root/'manifest.json',dict(meta,clips=clips))
        atomic_json(root/'completion.json',dict(completed=complete,reason=reason,proposals=len(clips),expected=len(rows)*a.draws,development_only=a.development))
    for i,row in enumerate(rows):
        name=hashlib.sha256(row['uid'].encode()).hexdigest()[:20];directory=root/name
        inputs=None;metadata=None
        for draw,seed in enumerate(meta['sample_seeds']):
            marker=directory/('sample_%02d.json'%draw)
            if marker.exists():
                item=json.loads(marker.read_text())
                if file_sha256(item['rgbd_path'])!=item['rgbd_sha256'] or file_sha256(item['history_path'])!=item['history_sha256']:raise ValueError('Changed completed proposal')
                clips.append(item);continue
            if (time.monotonic()-start)/3600>=a.gpu_hours:publish(reason='gpu_budget');return
            if shutil.disk_usage(root).free<a.min_free_gib*2**30:publish(reason='disk_floor');return
            if inputs is None:
                record=ds[i];metadata=record['metadata'];inputs=to_device(record['inputs'],device)
                historical={k:v.detach().cpu().numpy() for k,v in inputs.items() if k not in ('planned_camera_to_world','future_time_s')}
                historical['valid']=historical.pop('depth_valid');save_npz(directory/'history.npz',historical)
            values=generate_pair(model,inputs,row['uid'],seed,cp['identity']['seed'],device)
            path=directory/('sample_%02d.npz'%draw);save_npz(path,values)
            item=dict(metadata,conditioning_uid=row['uid'],uid=row['uid']+':sample%02d'%draw,proposal_id=name+'_%02d'%draw,draw=draw,sample_seed=seed,
                split_sha256=meta['split_sha256'],rgbd_path=str(path),rgbd_sha256=file_sha256(path),
                history_path=str(directory/'history.npz'),history_sha256=file_sha256(directory/'history.npz'),
                development_only=a.development,frames=16,views=['air','ground'],includes_history=False,output_modality='rgbd',generator_method=cp['identity']['variant'])
            atomic_json(marker,item);clips.append(item);publish()
            print(json.dumps(dict(uid=row['uid'],draw=draw,proposals=len(clips))),flush=True)
    publish(complete=True)
if __name__=='__main__':main()

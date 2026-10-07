"""Native external RGB proposals for the shared fixed-window downstream cohort."""
import argparse,hashlib,json,shutil,time,signal
from pathlib import Path
import torch
from experiments.paired_interventions.data import OrdinaryDataset,to_device,file_sha256
from experiments.generation_exports.windows import fixed_rows,save_npz
from .run import atomic,load_learned
from .models import create
from .provenance import verify_sampling_implementation


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('method','source','backbone','text','ordinary','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--modality',choices=['rgb','rgbd'],default='rgbd');p.add_argument('--forecaster-paths');p.add_argument('--forecaster-checkpoint')
    p.add_argument('--split',choices=['train','augmentation_train','val','id_test','test'],default='train')
    p.add_argument('--count',type=int,default=4000);p.add_argument('--draws',type=int,default=8);p.add_argument('--gpu-hours',type=float,required=True)
    p.add_argument('--sampling-steps',type=int,default=50)
    p.add_argument('--development',action='store_true');p.add_argument('--min-free-gib',type=float,default=25);a=p.parse_args()
    if a.count<1 or not 1<=a.draws<=8 or a.gpu_hours<=0:raise ValueError('Invalid count/draws/budget')
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False);identity=cp['identity']
    if identity['method']!=a.method or identity['modality']!=a.modality:raise ValueError('Native adaptation identity differs')
    if not a.development and (not cp['completed'] or identity['development_only'] or a.draws!=8 or a.sampling_steps!=50):raise ValueError('Formal proposals require completed native adaptation and eight draws')
    if not a.development and file_sha256(a.ordinary)!=identity['ordinary_sha256']:raise ValueError('Changed ordinary index')
    if not a.development:verify_sampling_implementation(identity,a.method,a.source,a.text,a.modality,a.forecaster_checkpoint)
    ds=OrdinaryDataset(a.ordinary,a.split);rows=fixed_rows(ds,a.count,identity['seed']);ds.dataset.rows=rows
    if not a.development and ds.index['development_only']:raise ValueError('Development source index')
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    meta=dict(schema='agcr-generated-fixed-windows-1',role=a.split,checkpoint_sha256=file_sha256(a.checkpoint),split_sha256=ds.index['split_sha256'],ordinary_index_sha256=file_sha256(a.ordinary),selected_uids=[r['uid'] for r in rows],sample_seeds=[101,211,307,401,503,601,701,809][:a.draws],development_only=a.development,coordinate_frame='ground_anchor_local_m',output_modality='rgb',conditioning_modality=a.modality,sampling_steps=a.sampling_steps,generator_method=a.method,export_source_sha256=file_sha256(__file__),native_adaptation_identity=identity)
    contract=root/'generation_contract.json'
    if contract.exists() and json.loads(contract.read_text())!=meta:raise ValueError('Changed native proposal contract')
    atomic(contract,meta);torch.set_num_threads(8);device=torch.device('cuda:0')
    kwargs=dict(forecaster_paths=a.forecaster_paths,forecaster_checkpoint=a.forecaster_checkpoint,development=a.development,training_condition_source=identity['training_condition_source']) if a.method in ('panacea','magicdrive_v2') else {}
    model=create(a.method,source=a.source,backbone=a.backbone,text_path=a.text,device=device,modality=a.modality,**kwargs)
    if a.method=='panacea':model.set_stage('video')
    load_learned(model,cp['model']);model.eval().requires_grad_(False)
    clips=[];started=time.monotonic();stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    def publish(completed=False,reason=None):
        atomic(root/'manifest.json',dict(meta,clips=clips));atomic(root/'completion.json',dict(completed=completed,reason=reason,proposals=len(clips),expected=len(rows)*a.draws,development_only=a.development))
    for i,row in enumerate(rows):
        name=hashlib.sha256(row['uid'].encode()).hexdigest()[:20];directory=root/name;inputs=None
        for draw,seed in enumerate(meta['sample_seeds']):
            marker=directory/f'sample_{draw:02d}.json'
            if marker.exists():
                item=json.loads(marker.read_text())
                if any(file_sha256(item[k+'_path'])!=item[k+'_sha256'] for k in ('rgbd','history')):raise ValueError('Changed native proposal')
                clips.append(item);continue
            if stop[0]:publish(reason='stop_requested');return
            if (time.monotonic()-started)/3600>=a.gpu_hours:publish(reason='gpu_budget');return
            if shutil.disk_usage(root).free<a.min_free_gib*2**30:publish(reason='disk_floor');return
            if inputs is None:
                record=ds[i];inputs=to_device(record['inputs'],device)
                history={k:v.cpu().numpy() for k,v in inputs.items() if k not in ('planned_camera_to_world','future_time_s')};history['valid']=history.pop('depth_valid');save_npz(directory/'history.npz',history)
            # Common random numbers across native methods, uid and training seed.
            sample_seed=int.from_bytes(hashlib.sha256(f'{identity["seed"]}:{row["uid"]}:{seed}:external_rgb'.encode()).digest()[:8],'little')%(2**63-1)
            with torch.no_grad():rgb=model.sample(inputs,sample_seed,a.sampling_steps)
            if rgb.shape!=(16,2,480,720,3):raise ValueError('Wrong native output dimensions')
            path=directory/f'sample_{draw:02d}.npz';save_npz(path,dict(rgb=rgb.cpu().numpy(),K=inputs['K'].cpu().numpy(),camera_to_world=inputs['planned_camera_to_world'].cpu().numpy(),time_s=inputs['future_time_s'].cpu().numpy()))
            item=dict(record['metadata'],conditioning_uid=row['uid'],uid=row['uid']+f':sample{draw:02d}',proposal_id=f'{name}_{draw:02d}',draw=draw,sample_seed=seed,native_noise_seed=sample_seed,split_sha256=meta['split_sha256'],rgbd_path=str(path),rgbd_sha256=file_sha256(path),history_path=str(directory/'history.npz'),history_sha256=file_sha256(directory/'history.npz'),development_only=a.development,frames=16,views=['air','ground'],includes_history=False,output_modality='rgb',generator_method=a.method)
            atomic(marker,item);clips.append(item);publish();print(json.dumps(dict(uid=row['uid'],draw=draw,proposals=len(clips))),flush=True)
    publish(completed=True)
if __name__=='__main__':main()

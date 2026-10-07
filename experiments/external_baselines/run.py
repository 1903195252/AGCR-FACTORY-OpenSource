"""Native external-model train/sample entry point; no registry-only execution.

python -m torch.distributed.run --nproc_per_node=2 -m experiments.external_baselines.run train ...
"""
import argparse,hashlib,json,os,signal,time
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from experiments.paired_interventions.data import FamilyDataset,OrdinaryDataset,to_device,file_sha256,load_branch
from .models import create
from .official import LOCK
from .provenance import implementation_hashes,verify_sampling_implementation


def atomic(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


class Objective(nn.Module):
    def __init__(self,model):super().__init__();self.model=model
    def forward(self,branch,generator):return self.model.loss(branch,generator)


def learned(model):
    names={n for n,p in model.named_parameters() if p.requires_grad}
    return {k:v.detach().cpu() for k,v in model.state_dict().items() if k in names}


def load_learned(model,state,allow_new_temporal=False):
    expected=set(learned(model));actual=set(state)
    missing=expected-actual
    if actual-expected or any(not(allow_new_temporal and 'temporal' in k) for k in missing):raise ValueError('Adaptation weights mismatch: '+str(sorted(missing)[:5]))
    model.load_state_dict(state,strict=False)


def branch(dataset,index,cell):
    row=dataset.families[index]
    record=next(r for r in row['records'] if (r['world'],r['camera'])==cell)
    return load_branch(dataset.root/record['path']),row


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','sample'])
    p.add_argument('--method',choices=['syncammaster','shareverse'],required=True)
    p.add_argument('--source',required=True);p.add_argument('--backbone',required=True);p.add_argument('--text',required=True)
    p.add_argument('--families',required=True);p.add_argument('--ordinary');p.add_argument('--output',required=True)
    p.add_argument('--checkpoint');p.add_argument('--resume');p.add_argument('--development',action='store_true')
    p.add_argument('--training-condition-source',choices=['annotated','predicted'],default='annotated')
    p.add_argument('--phase',choices=['image','video'],default='video');p.add_argument('--forecaster-paths');p.add_argument('--forecaster-checkpoint')
    p.add_argument('--modality',choices=['rgb','rgbd'],default='rgbd');p.add_argument('--steps',type=int,default=80000);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--lr',type=float);p.add_argument('--gpu-hours',type=float,required=True)
    p.add_argument('--save-every',type=int,default=100);p.add_argument('--split',choices=['val','test'],default='test');p.add_argument('--draws',type=int,default=8)
    p.add_argument('--limit',type=int);p.add_argument('--sampling-steps',type=int,default=50);p.add_argument('--stop-file');p.add_argument('--perturbation-json');a=p.parse_args()
    a.lr=a.lr or {'syncammaster':1e-4,'shareverse':2e-5,'panacea':1e-5,'magicdrive_v2':1e-5}[a.method]
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local);torch.set_num_threads(max(1,16//world));device=torch.device('cuda',local)
    if a.batch%world or a.gpu_hours<=0:raise ValueError('Invalid budget/batch')
    expected_steps=(20000 if a.phase=='image' else 60000) if a.method=='panacea' else 80000
    if not a.development and ((a.action=='train' and a.steps*a.batch!=expected_steps*16) or a.draws!=8 or a.limit is not None or not a.ordinary):raise ValueError('Formal exposure/8-draw/ordinary-data contract violated')
    dataset=FamilyDataset(a.families,'train' if a.action=='train' else a.split)
    ordinary=OrdinaryDataset(a.ordinary) if a.action=='train' and a.ordinary else None
    if not a.development and dataset.index['development_only']:raise ValueError('Formal data audit incomplete')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    identity=dict(method=a.method,training_condition_source=a.training_condition_source if a.method in ('panacea','magicdrive_v2') else 'native_rgb_supervision',phase=a.phase,learning_rate=a.lr,forecaster_checkpoint_sha256=file_sha256(a.forecaster_checkpoint) if a.forecaster_checkpoint else None,source=LOCK[a.method],native_source_sha256={str(p.relative_to(a.source)):file_sha256(p) for p in Path(a.source).rglob('*.py') if '.git' not in p.parts},backbone=str(Path(a.backbone).resolve()),backbone_provenance_sha256=file_sha256(Path(a.backbone)/'source_manifest.json') if (Path(a.backbone)/'source_manifest.json').is_file() else None,modality=a.modality,seed=a.seed,batch=a.batch,steps=a.steps,world_size=world,families_sha256=file_sha256(a.families),ordinary_sha256=file_sha256(a.ordinary) if a.ordinary else None,text_sha256=file_sha256(a.text),source_hashes={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py')},development_only=a.development)
    identity.update(implementation_hashes(a.method,a.source))
    if a.action=='train' and rank==0:
        if (root/'last.pt').exists() and not a.resume:raise FileExistsError('Use --resume; refusing to replace existing native run')
        if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text())!=identity:raise ValueError('Changed native run identity')
        atomic(root/'manifest.json',identity)
    extra={}
    if a.method in ('panacea','magicdrive_v2'):extra=dict(forecaster_paths=a.forecaster_paths,forecaster_checkpoint=a.forecaster_checkpoint,development=a.development,training_condition_source=a.training_condition_source)
    model=create(a.method,source=a.source,backbone=a.backbone,text_path=a.text,device=device,modality=a.modality,**extra)
    if a.method=='panacea':model.set_stage(a.phase)
    if rank==0 and a.action=='train':
        params=sum(v.numel() for v in model.parameters());trainable=sum(v.numel() for v in model.parameters() if v.requires_grad)
        atomic(root/'execution.json',dict(total_parameters=params,trainable_parameters=trainable,planned_updates=a.steps,planned_clip_presentations=a.steps*a.batch,phase=a.phase,matched_video_adaptation_only=True,all_stages_total_cost_report_required=True,paired_fraction=.25 if ordinary is not None else 1.))
    generator=torch.Generator(device=device).manual_seed(a.seed+100003*rank)
    previous=None
    if a.resume or a.checkpoint:
        previous=torch.load(a.resume or a.checkpoint,map_location='cpu',weights_only=False)
        initialize_temporal=a.method=='panacea' and a.action=='train' and a.phase=='video' and not a.resume
        load_learned(model,previous['model'],initialize_temporal)
        if initialize_temporal:model.initialize_temporal_from_spatial()
        if a.resume and previous['identity']!=identity:raise ValueError('Resume identity changed')
        if not a.development and previous['identity']['development_only']:raise ValueError('Development weights cannot supply formal results')
        if a.action=='sample' and not a.development and a.split=='test' and not previous.get('completed'):raise ValueError('Formal test sampling requires completed adaptation training')
        if a.action=='sample' and not a.development:
            verify_sampling_implementation(previous['identity'],a.method,a.source,a.text,a.modality,a.forecaster_checkpoint)
            if previous['identity']['families_sha256']!=file_sha256(a.families):raise ValueError('Changed formal family index')
    if a.action=='sample' and not previous:raise ValueError('Sampling requires trained adaptation checkpoint')
    prior_gpu_hours=float(previous.get('cumulative_gpu_hours',0.)) if a.resume else 0.
    started=time.monotonic();stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    if a.action=='sample':
        if world!=1:raise ValueError('Shard whole-family sampling via independent explicit output shards')
        seeds=[101,211,307,401,503,601,701,809]
        if not 1<=a.draws<=8:raise ValueError('Draws must be between one and eight')
        model.eval()
        sample_manifest=dict(schema='agcr-intervention-predictions-1',method=a.method,sample_seeds=seeds[:a.draws],training_seed=a.seed,checkpoint_sha256=file_sha256(a.checkpoint),index_sha256=file_sha256(a.families),split=a.split,development_only=a.development,conditions='history_only',sampling_steps=a.sampling_steps,conditioning_modality=a.modality,output_modality='rgb',native_upstream=LOCK[a.method],adaptation_identity=previous['identity'],perturbation=json.loads(Path(a.perturbation_json).read_text()) if a.perturbation_json else {'kind':'clean'})
        if (root/'manifest.json').exists() and json.loads((root/'manifest.json').read_text())!=sample_manifest:raise ValueError('Refusing to mix external prediction provenance')
        atomic(root/'manifest.json',sample_manifest)
        for i in range(min(a.limit or len(dataset),len(dataset))):
            row=dataset.families[i];folder=root/row['family_id'];folder.mkdir(exist_ok=True)
            for draw,seed in enumerate(seeds[:a.draws]):
                marker=folder/f'draw_{draw:02d}.json'
                if marker.exists():
                    record=json.loads(marker.read_text())
                    if all(file_sha256(folder/k)==v for k,v in record['files'].items()):continue
                    raise ValueError('Changed generated draw')
                files={}
                for w in range(2):
                    for c in range(2):
                        name=f'world{w}_camera{c}_sample{draw:02d}.npz';path=folder/name
                        sample,_=branch(dataset,i,(w,c));inputs=to_device(sample['inputs'],device)
                        if a.perturbation_json:
                            from experiments.formal_evaluation.perturbations import degrade_history
                            config=json.loads(Path(a.perturbation_json).read_text())
                            alternate=None
                            if config.get('kind')=='partner_mismatch':
                                alternate_branch,_=branch(dataset,i,(1-w,c));alternate=to_device(alternate_branch['inputs'],device)
                            inputs=degrade_history(inputs,config,seed,partner=1-int(row['receiver']),alternate_history=alternate)
                        with torch.no_grad():rgb=model.sample(inputs,seed,a.sampling_steps)
                        if rgb.shape!=(16,2,480,720,3) and not a.development:raise ValueError('Wrong exported shape')
                        np.savez_compressed(path,rgb=rgb.cpu().numpy(),K=inputs['K'].cpu().numpy(),camera_to_world=inputs['planned_camera_to_world'].cpu().numpy())
                        files[name]=file_sha256(path)
                        if stop[0] or time.monotonic()-started>=a.gpu_hours*3600:return
                atomic(marker,dict(seed=seed,files=files))
                if stop[0] or time.monotonic()-started>=a.gpu_hours*3600:return
        atomic(root/'completed.json',dict(completed=True,benchmark_results=False));return
    if world>1:torch.distributed.init_process_group('nccl')
    wrapped=Objective(model)
    if world>1:wrapped=DistributedDataParallel(wrapped,device_ids=[local],find_unused_parameters=False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=.01)
    step=0
    if a.resume:
        optimizer.load_state_dict(previous['optimizer']);step=previous['step'];generator.set_state(previous['rng'][rank]['generator']);torch.set_rng_state(previous['rng'][rank]['cpu']);torch.cuda.set_rng_state(previous['rng'][rank]['cuda'])
    model.train();model.vae.eval();lastloss=None
    while step<a.steps:
        optimizer.zero_grad(set_to_none=True);total=0.
        for slot in range(rank,a.batch,world):
            global_id=step*a.batch+slot;choice=np.random.default_rng(np.random.SeedSequence([a.seed,global_id]))
            if ordinary is not None and global_id%4:
                sample=ordinary[int(choice.integers(len(ordinary)))]
            else:
                sample,_=branch(dataset,int(choice.integers(len(dataset))),(int(choice.integers(2)),int(choice.integers(2))))
            sample=to_device(sample,device)
            context=wrapped.no_sync() if world>1 and slot+world<a.batch else nullcontext()
            with context:
                loss=wrapped(sample,generator)/(a.batch//world)
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite native objective')
                loss.backward()
            total+=float(loss.detach())
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0);optimizer.step();step+=1
        value=torch.tensor(total,device=device)
        if world>1:torch.distributed.all_reduce(value);value/=world
        lastloss=float(value)
        exhausted=time.monotonic()-started>=a.gpu_hours*3600/world
        request=stop[0] or exhausted or (a.stop_file and Path(a.stop_file).exists())
        flag=torch.tensor(int(bool(request)),device=device)
        if world>1:torch.distributed.all_reduce(flag,op=torch.distributed.ReduceOp.MAX)
        if rank==0:
            row=dict(step=step,total_steps=a.steps,clip_presentations=step*a.batch,phase=a.phase,loss=lastloss,elapsed_s=time.monotonic()-started,gpu_hours=(time.monotonic()-started)*world/3600,cumulative_gpu_hours=prior_gpu_hours+(time.monotonic()-started)*world/3600,peak_gib=torch.cuda.max_memory_allocated()/2**30,development_only=a.development)
            atomic(root/'status.json',row);print(json.dumps(row),flush=True)
        if step%a.save_every==0 or step==a.steps or flag.item():
            rng=dict(generator=generator.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
            states=[None]*world
            if world>1:torch.distributed.all_gather_object(states,rng)
            else:states=[rng]
            if rank==0:
                tmp=root/'last.tmp';torch.save(dict(identity=identity,model=learned(model),optimizer=optimizer.state_dict(),step=step,rng=states,cumulative_gpu_hours=prior_gpu_hours+(time.monotonic()-started)*world/3600,completed=step==a.steps),tmp);tmp.replace(root/'last.pt')
                if step==a.steps:atomic(root/'completed.json',dict(completed=True,step=step))
        if flag.item():break
    if rank==0 and step==a.steps:atomic(root/'completed.json',dict(completed=True,step=step))
    if world>1:torch.distributed.destroy_process_group()
if __name__=='__main__':main()

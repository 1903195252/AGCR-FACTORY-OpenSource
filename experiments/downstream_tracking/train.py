"""Independent readers/downstream learners; exact clip budget and restartable DDP."""
import argparse,contextlib,hashlib,json,math,os,random,signal,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .data import ClipDataset,move,pack_images,sha256,native_clip,make_targets
from .model import CooperativeDetector,head_losses


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def index_at(size,seed,cursor):
    epoch,at=divmod(cursor,size)
    return int(np.random.default_rng(np.random.SeedSequence([seed,epoch])).permutation(size)[at])


def batch_indices(dataset,seed,step,batch):
    groups={}
    for i,row in enumerate(dataset.rows):groups.setdefault(row.get('source','rendered'),[]).append(i)
    if len(groups)==1:return [index_at(len(dataset),seed,step*batch+i) for i in range(batch)]
    result=[]
    for n,(source,indices) in enumerate(sorted(groups.items())):
        quota=batch*len(indices)/len(dataset)
        if int(quota)!=quota:raise ValueError('Global batch cannot exactly represent registered source fraction')
        quota=int(quota)
        result.extend(indices[index_at(len(indices),seed+1009*n,step*quota+i)] for i in range(quota))
    return result


@torch.no_grad()
def validate(model,dataset,device,rank,world,development):
    rows=[r for r in dataset.index['windows'] if r['role']=='val' and r['history_covered']]
    rows=sorted(rows,key=lambda r:hashlib.sha256(('validation:17029:'+r['uid']).encode()).hexdigest())[:2 if development else 64]
    if not rows:raise ValueError('No validation episodes')
    was_training=model.training;model.eval();total=count=0.
    for row in rows[rank::world]:
        x,y=native_clip(dataset.root,row['episode'],row['start']+6)
        targets=move(make_targets(x,y,row['episode']),device)
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
            output=model(**move(pack_images(x),device),labels=[{k:t[k] for k in ('boxes','class_labels')} for t in targets])
            loss=head_losses(output,targets)['total']
        total+=float(loss);count+=1
    value=torch.tensor([total,count],device=device,dtype=torch.float64)
    if world>1:dist.all_reduce(value)
    model.train(was_training)
    return float(value[0]/value[1]),[r['uid'] for r in rows]


def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'])
    if s['cuda'] is not None:torch.cuda.set_rng_state(s['cuda'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',required=True);p.add_argument('--assets',required=True)
    p.add_argument('--output',required=True);p.add_argument('--cohort');p.add_argument('--role',choices=['annotation','video_evaluation','downstream'],required=True)
    p.add_argument('--modality',choices=['rgb','rgbd'],default='rgbd');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--steps',type=int,default=40000);p.add_argument('--batch',type=int,default=16);p.add_argument('--gpu-hours',type=float,required=True)
    p.add_argument('--stop-file');p.add_argument('--resume',action='store_true');p.add_argument('--development',action='store_true');p.add_argument('--device',default='cuda')
    p.add_argument('--prefetch-workers',type=int,default=4);p.add_argument('--save-every',type=int,default=200);p.add_argument('--validate-every',type=int,default=2000)
    a=p.parse_args();rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    if a.gpu_hours<=0 or a.steps<1 or a.batch%world or a.batch<world:raise ValueError('Positive budget and global batch divisible by world size required')
    if not a.development and (a.steps!=40000 or a.batch!=16 or a.validate_every!=2000):raise ValueError('Formal downstream protocol is 40000 updates, batch16 clips')
    if a.role=='downstream' and not a.cohort and not a.development:raise ValueError('Formal downstream training needs an immutable matched cohort')
    device=torch.device('cuda',local) if a.device=='cuda' else torch.device(a.device)
    if device.type=='cuda':torch.cuda.set_device(device)
    torch.set_num_threads(max(1,min(8,(os.cpu_count() or 8)//world)))
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    random.seed(a.seed+rank);np.random.seed(a.seed+rank);torch.manual_seed(a.seed+rank)
    dataset=ClipDataset(a.index,a.role,a.cohort,a.modality)
    if not a.development and dataset.index.get('development_only'):raise ValueError('Formal run cannot use pilot data')
    source={x.name:sha256(x) for x in Path(__file__).parent.glob('*.py')}
    identity=dict(schema='agcr-independent-tracker-1',role=a.role,modality=a.modality,seed=a.seed,steps=a.steps,batch=a.batch,
        world_size=world,index_sha256=sha256(a.index),cohort_sha256=sha256(a.cohort) if a.cohort else None,
        source_sha256=source,assets={x.name:sha256(x) for x in Path(a.assets).glob('*') if x.is_file()},
        split_sha256=dataset.index['split_sha256'],training_role=dataset.role,development_only=a.development,
        training_episodes=sorted({r['episode'] for r in dataset.rows}),generator_weights_used=False)
    identity['training_partition_sha256']=hashlib.sha256(json.dumps(identity['training_episodes'],separators=(',',':')).encode()).hexdigest()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);last=root/'last.pt'
    if last.exists() and not a.resume:raise ValueError('Existing checkpoint: specify --resume')
    saved=torch.load(last,map_location='cpu',weights_only=False) if a.resume else None
    if saved and saved['identity']!=identity:raise ValueError('Resume provenance or schedule changed')
    model=CooperativeDetector(a.assets,a.modality,pretrained=not a.resume).to(device)
    if saved:model.load_state_dict(saved['model'])
    detector=[p for n,p in model.named_parameters() if n.startswith('detector.') and p.requires_grad]
    heads=[p for n,p in model.named_parameters() if not n.startswith('detector.') and p.requires_grad]
    optimizer=torch.optim.AdamW([dict(params=detector,lr=1e-5,base_lr=1e-5),dict(params=heads,lr=1e-4,base_lr=1e-4)],weight_decay=.01)
    step=0;used=0.;best_validation=float('inf')
    if saved:optimizer.load_state_dict(saved['optimizer']);step=saved['step'];used=saved.get('gpu_hours',0.);best_validation=saved.get('best_validation',float('inf'));restore_rng(saved['rng'][rank])
    wrapper=DDP(model,device_ids=[local] if device.type=='cuda' else None,find_unused_parameters=True) if world>1 else model
    model.train();stop=[False]
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True));signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    started=time.monotonic();committed=step;allowance=a.gpu_hours
    pool=ThreadPoolExecutor(max_workers=a.prefetch_workers)
    def prepare(s):
        indices=batch_indices(dataset,a.seed,s,a.batch)[rank::world]
        return [pool.submit(dataset.__getitem__,i) for i in indices]
    pending=prepare(step) if step<a.steps else []
    while step<a.steps:
        optimizer.zero_grad(set_to_none=True);loss_total=0.;clip_count=a.batch//world
        next_pending=prepare(step+1) if step+1<a.steps else []
        for i,future in enumerate(pending):
            inputs,targets=future.result();images=move(pack_images(inputs),device);targets=move(targets,device)
            sync=contextlib.nullcontext() if world==1 or i==clip_count-1 else wrapper.no_sync()
            with sync:
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                    output=wrapper(**images,labels=[{k:t[k] for k in ('boxes','class_labels')} for t in targets])
                    losses=head_losses(output,targets);loss=losses['total']/clip_count
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite independent tracker objective')
                loss.backward();loss_total+=float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        factor=min((step+1)/1000,1)*(.1+.9*.5*(1+math.cos(math.pi*max(step-1000,0)/max(a.steps-1000,1))))
        for g in optimizer.param_groups:g['lr']=g['base_lr']*factor
        optimizer.step();step+=1;pending=next_pending
        elapsed=(time.monotonic()-started)*world/3600
        budget_stop=elapsed>=allowance
        stopped=torch.tensor(int(bool(stop[0] or budget_stop or (a.stop_file and Path(a.stop_file).exists()))),device=device)
        if world>1:dist.all_reduce(stopped,op=dist.ReduceOp.MAX)
        validation=None;improved=False
        if step%a.validate_every==0 or step==a.steps:
            validation,panel=validate(model,dataset,device,rank,world,a.development)
            improved=validation<best_validation;best_validation=min(best_validation,validation)
            if rank==0:atomic(root/'validation.json',dict(step=step,objective=validation,panel=panel,role='val'))
        # Validation consumes the same GPU allocation and belongs to the budget.
        elapsed=(time.monotonic()-started)*world/3600
        if elapsed>=allowance or stop[0] or (a.stop_file and Path(a.stop_file).exists()):stopped.fill_(1)
        if world>1:dist.all_reduce(stopped,op=dist.ReduceOp.MAX)
        should_save=(step%a.save_every==0 or step==a.steps or bool(stopped) or improved)
        if should_save:
            states=[None]*world
            if world>1:dist.all_gather_object(states,rng())
            else:states=[rng()]
            if rank==0:
                checkpoint=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),identity=identity,step=step,rng=states,
                    completed=step==a.steps,gpu_hours=used+elapsed,sampler_cursor=step*a.batch,best_validation=best_validation)
                tmp=last.with_suffix('.pt.tmp');torch.save(checkpoint,tmp);tmp.replace(last)
                if improved:
                    import shutil
                    shutil.copy2(last,root/'best.pt')
                atomic(root/'status.json',dict(step=step,total=a.steps,state='completed' if step==a.steps else 'budget_paused' if bool(stopped) else 'running',
                    gpu_hours=used+elapsed,loss=loss_total,development_only=a.development,clips_seen=step*a.batch))
            if world>1:dist.barrier()
        if rank==0:
            record=dict(step=step,loss=loss_total,gpu_hours=used+elapsed,seconds_per_update=(time.monotonic()-started)/(step-committed),validation=validation)
            atomic(root/'status.json',dict(record,total=a.steps,state='completed' if step==a.steps else 'budget_paused' if bool(stopped) else 'running',development_only=a.development,clips_seen=step*a.batch))
            with (root/'metrics.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
        if bool(stopped):break
    pool.shutdown(wait=True)
    if rank==0 and step==a.steps:
        best_path=root/'best.pt'
        if best_path.exists():
            best=torch.load(best_path,map_location='cpu',weights_only=False);best['completed']=True;best['run_completed']=True;best['training_completed_steps']=step;best['best_step']=best['step'];best['selection']='best_validation'
            temporary=best_path.with_suffix('.pt.tmp');torch.save(best,temporary);temporary.replace(best_path)
        atomic(root/'completion.json',dict(completed=True,checkpoint_sha256=sha256(last),best_checkpoint_sha256=sha256(best_path) if best_path.exists() else None,development_only=a.development,role=a.role))
    if world>1:dist.destroy_process_group()

if __name__=='__main__':main()

"""Executable observation/association pretraining on native paired factory RGB-D.

This stage trains detection evidence, NOT the complete tracking/video model.
The frozen DINO is loaded from actual supplied weights, never random features.
"""
import argparse,hashlib,json,math,os,random,time,signal,fcntl
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .data import Dataset,atomic_json,read_json,read_observation_pair
from .observation import ObservationHead,observation_loss
from .protocol import configuration
from .selection import validation_panel,validation_mean


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def frame_tensors(sample,t,device):
    x,y=sample['inputs'],sample['labels']
    def tensor(a,dtype=torch.float32):return torch.as_tensor(np.ascontiguousarray(a),dtype=dtype,device=device)
    inputs=dict(depth_mm=tensor(x['depth_mm'][t]),valid=tensor(x['depth_valid'][t],torch.bool),
                K=tensor(x['K'])[None].expand(2,-1,-1),T=tensor(x['camera_to_world'][t]),
                time_s=tensor(np.repeat(x['time_s'][t],2)),view=torch.arange(2,device=device))
    rgb=tensor(x['rgb'][t]).permute(0,3,1,2)/255
    rgb=F.interpolate(rgb,size=(336,504),mode='bilinear',align_corners=False)
    rgb=(rgb-rgb.new_tensor([.485,.456,.406])[None,:,None,None])/rgb.new_tensor([.229,.224,.225])[None,:,None,None]
    s=tensor(y['history_state'][t]);n=len(s)
    full=tensor(y['history_full_mask'][t]);visible=tensor(y['history_visible_mask'][t])
    def mask(a):return F.interpolate(a,size=(24,36),mode='area')
    target=dict(center=s[None,:,:3].expand(2,-1,-1),dimensions=s[None,:,8:11].expand(2,-1,-1),
                heading=torch.stack([s[:,6].sin(),s[:,6].cos()],-1)[None].expand(2,-1,-1),
                class_id=tensor(y['class_id'],torch.long),full_mask=mask(full),visible_mask=mask(visible),
                detectable=(tensor(y['history_visible_fraction'][t])>=.1)&(full.sum((-2,-1))>=144))
    return rgb,inputs,target


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--output',required=True)
    p.add_argument('--dino-code',required=True);p.add_argument('--dino-weights',required=True)
    p.add_argument('--seed',type=int,default=17);p.add_argument('--variant',choices=['full','rgb_only'],default='full')
    p.add_argument('--steps',type=int,default=40000);p.add_argument('--device',default='cuda');p.add_argument('--resume')
    p.add_argument('--development',action='store_true');p.add_argument('--max-seconds',type=float)
    a=p.parse_args();config=configuration(a.variant,a.seed);index=read_json(a.index)
    if index['development_only'] and not a.development:raise ValueError('Partial data requires explicit --development')
    if a.steps!=40000 and not a.development:raise ValueError('Changed stage budget requires --development')
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    torch.set_num_threads(4);device=torch.device(a.device);rng=np.random.default_rng(a.seed)
    ds=Dataset(index,role='train',primary=True,load_surfaces=True)
    val=None if a.development else Dataset(index,role='val',primary=True,load_surfaces=True)
    val_rows=[];panel=None
    if val is not None:
        val_rows,panel=validation_panel(val.rows,config['validation_panel_minimum'],config['validation_panel_seed'])
    model=ObservationHead().to(device);optim=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.999),weight_decay=.01)
    dino=torch.hub.load(a.dino_code,'dinov2_vitb14',source='local',pretrained=False).to(device)
    dino.load_state_dict(torch.load(a.dino_weights,map_location='cpu',weights_only=True));dino.eval().requires_grad_(False)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    run_lock=(output/'run.lock').open('a')
    fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=dict(stage='observation_head_pretraining',development_only=a.development,index_sha256=sha256(a.index),
                  split_sha256=index['split_sha256'],protocol_sha256=config['protocol_sha256'],
                  dino_sha256=sha256(a.dino_weights),steps=a.steps,effective_batch_pairs=16,variant=a.variant,
                  source_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
                  limitation='Observation-head/paired-association pretraining; temporal tracker not integrated.',
                  validation_panel=panel,config=config,precision='float32',
                  train_windows=len(ds),validation_windows=len(val_rows),
                  torch_version=torch.__version__,cuda_version=torch.version.cuda,
                  device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else str(device))
    step0=0;best_validation=None;best_step=None;prior_elapsed=0.
    if a.resume:
        saved=torch.load(a.resume,map_location=device,weights_only=False)
        if saved['manifest']!=manifest:raise ValueError('Resume changed data, source, weights or protocol')
        model.load_state_dict(saved['model']);optim.load_state_dict(saved['optimizer']);step0=saved['step']
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'].cpu())
        if saved['cuda_rng'] is not None:torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        best_validation=saved.get('best_validation_loss');best_step=saved.get('best_step')
        prior_elapsed=saved.get('elapsed_training_seconds',0.)
        log=output/'training.jsonl'
        if log.exists():
            retained=[line for line in log.read_text().splitlines() if json.loads(line)['step']<=step0]
            temp=log.with_suffix('.jsonl.tmp');temp.write_text('\n'.join(retained)+'\n');temp.replace(log)
    elif (output/'manifest.json').exists():raise ValueError('Existing run: use --resume or a new output directory')
    if step0>=a.steps:raise ValueError('Checkpoint already reached target updates')
    atomic_json(output/'manifest.json',manifest)
    if panel is not None:atomic_json(output/'validation-panel.json',panel)
    started=time.monotonic()
    stop_requested=False
    def request_stop(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,request_stop)
    def checkpoint(step,name='last.pt'):
        temp=output/(name+'.tmp');torch.save(dict(step=step,manifest=manifest,model=model.state_dict(),optimizer=optim.state_dict(),
          numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None,
          best_validation_loss=best_validation,best_step=best_step,
          elapsed_training_seconds=prior_elapsed+time.monotonic()-started),temp)
        temp.replace(output/name)
    def status(step,state,record=None):
        elapsed=time.monotonic()-started;count=step-step0
        atomic_json(output/'status.json',dict(status=state,pid=os.getpid(),completed_steps=step,target_steps=a.steps,
                    complete=step==a.steps,elapsed_seconds=prior_elapsed+elapsed,session_elapsed_seconds=elapsed,
                    remaining_seconds=(a.steps-step)*elapsed/count if count else None,
                    updated_unix=time.time(),best_validation_loss=best_validation,best_step=best_step,
                    development_only=a.development,last=record,
                    cuda_peak_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.))
    def validate():
        model.eval();values=[]
        with torch.no_grad():
            for row in val_rows:
                rgb,inputs,target=frame_tensors(read_observation_pair(val.root,row,5),0,device)
                out=model(dino.forward_features(rgb)['x_norm_patchtokens'],**inputs,rgb_only=a.variant=='rgb_only')
                value=float(observation_loss(model,out,target)[0])
                if not math.isfinite(value):raise RuntimeError('Nonfinite validation loss')
                values.append(value)
        model.train()
        return validation_mean(val_rows,values)
    status(step0,'running')
    if val is not None and step0==0:
        baseline,by_layout=validate()
        atomic_json(output/'initial-validation.json',dict(step=0,validation_loss=baseline,validation_by_layout=by_layout,
                    validation_windows=len(val_rows),validation_panel_sha256=panel['panel_sha256']))
        print(json.dumps(dict(step=0,initial_validation_loss=baseline)),flush=True)
    for step in range(step0,a.steps):
        factor=min((step+1)/1000,1.)*(.1+.9*.5*(1+math.cos(math.pi*max(0,step-1000)/max(1,a.steps-1000))))
        for group in optim.param_groups:group['lr']=1e-4*factor
        optim.zero_grad();total=0.;components=dict(observation=0.,association=0.)
        for _ in range(16):
            row=ds.rows[int(rng.integers(len(ds)))];t=int(rng.integers(6))
            sample=read_observation_pair(ds.root,row,t)
            rgb,inputs,target=frame_tensors(sample,0,device)
            with torch.no_grad():features=dino.forward_features(rgb)['x_norm_patchtokens']
            out=model(features,**inputs,rgb_only=a.variant=='rgb_only');loss,parts=observation_loss(model,out,target)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite observation loss')
            (loss/16).backward();total+=float(loss.detach())/16
            for key in components:components[key]+=parts[key]/16
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optim.step()
        record=dict(step=step+1,loss=total,gradient_norm=float(norm),lr=optim.param_groups[0]['lr'],elapsed_seconds=time.monotonic()-started)
        record.update(loss_components=components,elapsed_training_seconds=prior_elapsed+record['elapsed_seconds'])
        if val is not None and (step+1)%2000==0:
            record['validation_loss'],record['validation_by_layout']=validate()
            record['validation_windows']=len(val_rows)
            record['validation_panel_sha256']=panel['panel_sha256']
            if best_validation is None or record['validation_loss']<best_validation:
                best_validation=record['validation_loss'];best_step=step+1
                checkpoint(step+1,'best.pt')
        with open(output/'training.jsonl','a') as f:f.write(json.dumps(record)+'\n')
        timed_out=a.max_seconds is not None and time.monotonic()-started>=a.max_seconds
        if (step+1)%100==0 or step+1==a.steps or timed_out or stop_requested:checkpoint(step+1)
        if (step+1)%10==0 or step==step0:
            status(step+1,'running',record);print(json.dumps(record),flush=True)
        if timed_out or stop_requested:break
    status(step+1,'complete' if step+1==a.steps else 'checkpointed',record)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        import sys
        if '--output' in sys.argv:
            path=Path(sys.argv[sys.argv.index('--output')+1])/'status.json'
            if path.exists():
                state=read_json(path)
                # A second invocation must not rewrite a different live run.
                if state.get('pid')==os.getpid():
                    state.update(status='failed',error=str(exc),updated_unix=time.time())
                    atomic_json(path,state)
        raise

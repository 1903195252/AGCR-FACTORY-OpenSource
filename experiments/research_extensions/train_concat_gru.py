"""Full2048-update Concat-GRU fit on the original64 paired TRAIN families.

CPU execution uses the immutable frozen historical cache while all GPUs run
the evidence-loss replicates. Future labels are training targets only.
"""
import argparse
import json
from pathlib import Path
import signal
import time

import torch

from .region_probe import atomic_json,sha256
from .state_repair import ordinary_objective
from .train_candidate import branch_index,atomic_checkpoint
from .state_prediction_baselines import ConcatGRU

SOURCES=('experiments/research_extensions/state_prediction_baselines.py',
         'experiments/research_extensions/train_concat_gru.py',
         'experiments/research_extensions/state_repair.py',
         'experiments/paper_experiments/agcr_exp/future_prediction.py',
         'experiments/base48/model.py')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',required=True);p.add_argument('--statistics',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=17029)
    p.add_argument('--max-seconds',type=float,default=6000.)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise FileExistsError('Preserve earlier baseline attempts')
    if not 0<a.max_seconds<=6000 or a.seed!=17029:raise ValueError('Fixed baseline budget/seed required')
    torch.set_num_threads(8);torch.manual_seed(a.seed)
    cache=Path(a.cache);manifest=json.loads((cache/'manifest.json').read_text())
    data=manifest['data'];records=data['branches']
    if len(records)!=128 or len({r['family_id']for r in records})!=64 or data['test_used']is not False:
        raise ValueError('Original complete64 TRAIN-family cache required')
    if any(r['split']!='train' or r['camera']!=0 for r in records):raise ValueError('NonTRAIN cache')
    rows=[]
    for i,record in enumerate(records):
        path=cache/('%04d.pt'%i);stamp=json.loads(path.with_suffix('.json').read_text())
        if stamp!=dict(metadata=record,sha256=sha256(path),manifest_sha256=sha256(cache/'manifest.json')):
            raise ValueError('Historical cache changed')
        rows.append(torch.load(path,map_location='cpu',weights_only=False))
    groups=[[i for i,r in enumerate(records)if r['family_id']==fid]for fid in sorted({r['family_id']for r in records})]
    if any(len(g)!=2 or [records[i]['world']for i in g]!=[0,1]for g in groups):raise ValueError('Incomplete TRAIN pair')
    model=ConcatGRU(json.loads(Path(a.statistics).read_text()))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
    root=Path(__file__).resolve().parents[2]
    identity=dict(schema='agcr-concat-gru-state-baseline-1',train_families=64,train_branches=128,
        steps=2048,seed=a.seed,lr=.001,batch_worlds=2,history_frozen=True,
        cache_manifest_sha256=sha256(cache/'manifest.json'),statistics_sha256=sha256(a.statistics),
        sources={s:sha256(root/s)for s in SOURCES},
        objectives='common original joint likelihood, autoregressive motion Huber and free-prefix presence; no paired evidence or geometry loss',
        architecture='pooled aerial/ground scene concat + historical track token; two-layer GRU width256;8 shared scene modes',
        common_history='same frozen learned association/filter and observation model as isolated ablation',
        full_pipeline_retrained=False,selection_uses_test=False,future_truth_conditions=False,
        trainable_parameters=sum(v.numel()for v in model.parameters()if v.requires_grad),cpu_threads=8)
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    start=time.monotonic();step=0;model.train()
    def save():
        atomic_checkpoint(out/'last.pt',dict(identity=identity,model=model.state_dict(),
            optimizer=optimizer.state_dict(),step=step,completed=step==2048,rng=torch.get_rng_state()))
    save();atomic_json(out/'identity.json',identity)
    try:
        from experiments.base48.model import free_prefix_presence_objective
        while step<2048 and not stopped[0] and time.monotonic()-start<a.max_seconds-30:
            begin=time.monotonic();indexes=groups[branch_index(step,len(groups),a.seed)]
            optimizer.zero_grad(set_to_none=True);total=0.
            for world,index in enumerate(indexes):
                row=rows[index]
                if row['empty']:continue
                model.condition(row['history'])
                common,parts=ordinary_objective(model,row)
                rng=torch.Generator(device='cpu').manual_seed(a.seed+200003+2*step+world)
                presence,_=free_prefix_presence_objective(model,row['history'],row['reference_ids'],
                    row['future_present'],row['history_eligible'],rng)
                loss=(common+presence)/2
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite baseline loss')
                loss.backward();total+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            atomic_json(out/'status.json',dict(state='running',step=step,target_steps=2048,
                last_loss=total,gradient_norm=float(norm),step_seconds=time.monotonic()-begin,
                updated_unix=time.time(),training_device='cpu',seed=a.seed))
            if step%16==0 or step==2048:save()
        save()
        atomic_json(out/'completion.json',dict(completed=step==2048,step=step,
            checkpoint_sha256=sha256(out/'last.pt'),wall_seconds=time.monotonic()-start,identity=identity))
        atomic_json(out/'status.json',dict(state='completed'if step==2048 else'paused',step=step,
            target_steps=2048,updated_unix=time.time()))
    except BaseException as error:
        save();atomic_json(out/'status.json',dict(state='failed',step=step,error=str(error),updated_unix=time.time()));raise


if __name__=='__main__':main()

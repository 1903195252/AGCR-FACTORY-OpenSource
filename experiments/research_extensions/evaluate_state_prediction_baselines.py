"""Complete, paired160-family TEST comparison of simple state predictors.

Prediction uses six historical paired RGB-D frames and the frozen historical
frontend. References are used only after histories are constructed, to align
and score them. The original Base48 numerical state metrics are reused.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import signal
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from .region_probe import atomic_json,sha256,load_state_overlay
from .train_history_identity import load_identity_checkpoint
from .typed_evidence_repeat_test import validate_population,validate_input_boundary,budget_guard,SEEDS
from .state_prediction_baselines import ConcatGRU,constant_velocity,equal_precision_update

SCHEMA='agcr-cooperative-state-baseline-test-1'
METHODS=('shared_history_cv','equal_precision_cv','concat_gru')
ROOT=Path(__file__).resolve().parents[2]
SOURCES=('experiments/research_extensions/evaluate_state_prediction_baselines.py',
    'experiments/research_extensions/state_prediction_baselines.py',
    'experiments/research_extensions/train_concat_gru.py',
    'experiments/base48/state_eval.py','experiments/formal_evaluation/metrics.py',
    'experiments/paired_interventions/rollout.py','experiments/paired_interventions/objectives.py')


class FixedHistory:
    def __init__(self,history,dynamics):self.fixed=history;self.dynamics=dynamics
    def history(self,inputs,encoded):
        h=self.fixed
        return h['result'],h['observations'],h['shared'],h['separate']


def cv_sample(model,tokens,scene,initial,presence_logits,generator,covariance=None,**kwargs):
    return constant_velocity(dict(initial=initial,presence_logits=presence_logits,covariance=covariance),generator)


def cv_density_bank(history,slot,generator):
    from experiments.paired_interventions.rollout import _initial
    initial=history['initial'][:,slot:slot+1].expand(32,-1,-1)
    covariance=history['covariance'][:,slot:slot+1].expand(32,-1,-1,-1)
    state=_initial(initial,covariance,generator)
    paths=[]
    for step in range(1,17):
        new=state.clone();new[...,:3]=state[...,:3]+.2*step*state[...,3:6];paths.append(new)
    log_survival=torch.nn.functional.logsigmoid(history['presence_logits'][0,slot]).expand(1,1,32)
    return dict(state=torch.stack(paths,1)[None,None],mode_logits=state.new_zeros(1,1),log_survival=log_survival)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','checkpoint','output','budget-ledger','job-id'):p.add_argument('--'+name,required=True)
    p.add_argument('--test-lock',required=True)
    p.add_argument('--max-seconds',type=float,default=2400);p.add_argument('--stop-file')
    a=p.parse_args();plan=json.loads(Path(a.plan).read_text())
    budget=budget_guard(a.budget_ledger,a.job_id,plan['budget'])
    if not 30<a.max_seconds<=budget['maximum_wall_seconds']-30:raise ValueError('Preserve outer cleanup time')
    from experiments.base48 import state_eval
    from experiments.base48.train import audit_index_ancestry
    from experiments.paired_interventions.data import load_branch,to_device
    from experiments.paired_interventions.world import load_upstream
    from experiments.paired_interventions.bridge import historical_posterior
    from experiments.paired_interventions.sample import stream_seed
    from experiments.paired_interventions.objectives import candidate_log_density
    from experiments.formal_evaluation.state_likelihood import persistent_target_rollouts
    from experiments.paper_experiments.agcr_exp.stage2_system import FrozenImageEncoders
    from .state_eval_fixed import EncodedHistory
    torch.set_num_threads(4);torch.cuda.set_device('cuda:0')
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    done=json.loads(Path(a.checkpoint).with_name('completion.json').read_text())
    if not payload.get('completed') or payload.get('step')!=2048 or done.get('completed')is not True:
        raise ValueError('Concat-GRU requires a real completed2048 fit')
    identity=payload['identity']
    if (identity['steps']!=2048 or identity['train_families']!=64 or identity['seed']!=17029
        or identity['selection_uses_test']is not False or identity['future_truth_conditions']is not False
        or done['checkpoint_sha256']!=sha256(a.checkpoint)):
        raise ValueError('Baseline exposure, seed or information boundary changed')
    for name,digest in identity['sources'].items():
        if sha256(ROOT/name)!=digest:raise ValueError('Baseline source changed: '+name)
    for key,item in plan['files'].items():
        if sha256(item['path'])!=item['sha256']:raise ValueError('Frozen comparison input changed: '+key)
    files=plan['files'];paths=json.loads(Path(files['paths']['path']).read_text())
    if identity['statistics_sha256']!=sha256(paths['statistics']):raise ValueError('Training scales differ')
    manifest=Path(plan['future_cache'])/'manifest.json'
    if identity['cache_manifest_sha256']!=sha256(manifest):raise ValueError('Historical TRAIN data differ')
    test_lock=json.loads(Path(a.test_lock).read_text())
    ancestry=audit_index_ancestry(paths['families'])
    source,families=validate_population(test_lock,paths['families'],test_lock['index'],ancestry)
    model,_=load_upstream(paths['upstream'],paths['statistics'],'cuda:0')
    parent,_=load_state_overlay(files['parent']['path'])
    model.load_state_dict({k[len('upstream.'):]:v for k,v in parent['model'].items()if k.startswith('upstream.')},strict=True)
    del parent
    load_identity_checkpoint(files['history_overlay']['path'],files['parent']['path'],
                             test_lock['training_index_sha256'],upstream=model,device='cuda:0')
    model.eval().requires_grad_(False)
    gru=ConcatGRU(json.loads(Path(paths['statistics']).read_text())).cuda()
    gru.load_state_dict(payload['model'],strict=True);gru.eval().requires_grad_(False);del payload
    encoders=FrozenImageEncoders(paths['dino_code'],paths['dino_weights'],paths['wan']+'/vae','cuda:0')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    contract=dict(schema=SCHEMA+'-contract',split='test',families=[f['family_id']for f in families],
        sample_seeds=SEEDS,worlds=[0,1],camera=0,sampling_stream_training_seed=17,
        plan_sha256=sha256(a.plan),checkpoint_sha256=sha256(a.checkpoint),training=identity,
        sources={s:sha256(ROOT/s)for s in SOURCES},future_reference_scope='alignment and scoring only',
        shared_history='same immutable observation, history-refinement and association/filter weights',
        equal_precision='same associations and measurement covariances; equal scalar evidence weights; constant velocity',
        constant_velocity='same sampled initial posterior and initial existence, retained across16frames',
        concat_gru='ordinary per-view pooled feature concatenation; common history tracks; no paired evidence/geometry objective',
        density='common bandwidth0.2;32 paths per seed:8x4 for GRU,1x32 for constant velocity',
        methods=list(METHODS),selection_uses_test=False)
    if (out/'contract.json').exists() and json.loads((out/'contract.json').read_text())!=contract:
        raise ValueError('Preserve frozen baseline comparison')
    atomic_json(out/'contract.json',contract)
    report=dict(schema=SCHEMA,completed=False,split='test',contract=contract,
                methods={m:dict(rows=[])for m in METHODS},completed_families=0)
    stopped=[False];start=time.monotonic()
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def boundary():
        if stopped[0] or time.monotonic()-start>a.max_seconds-20 or(a.stop_file and Path(a.stop_file).exists()):
            raise InterruptedError('Pause at completed-family boundary')
    try:
        with torch.no_grad():
            for family in families:
                boundary();per_method={m:[]for m in METHODS};histories={m:[]for m in METHODS};references=[]
                for record in family['records']:
                    if sha256(Path(source['root'])/record['path'])!=record['sha256']:raise ValueError('Audited TEST branch changed')
                for world in (0,1):
                    boundary();record=next(r for r in family['records']if(r['world'],r['camera'])==(world,0))
                    raw=load_branch(Path(source['root'])/record['path']);validate_input_boundary(raw['inputs'])
                    branch=to_device(dict(inputs=raw['inputs'],supervision={k:raw['supervision'][k]for k in state_eval.KEEP_LABELS}),'cuda:0')
                    del raw;encoded=EncodedHistory(branch['inputs']['rgb'],encoders.history(branch['inputs']['rgb']))
                    shared=historical_posterior(model,branch['inputs'],encoded.encoded)
                    original_update=model.tracker.update
                    replacement=lambda track,evidence,initial=False:equal_precision_update(model.tracker,original_update,track,evidence,initial)
                    with patch.object(model.tracker,'update',replacement):
                        fused=historical_posterior(model,branch['inputs'],encoded.encoded)
                    for name,hist in [('shared_history_cv',shared),('equal_precision_cv',fused),('concat_gru',shared)]:
                        if name=='concat_gru':gru.condition(hist)
                        system=FixedHistory(hist,gru if name=='concat_gru'else model.dynamics)
                        context=nullcontext()if name=='concat_gru'else patch.multiple(state_eval,
                            sample_shared_future=cv_sample,innovation_nll=lambda *_args,**_kwargs:None)
                        with context:row,hist,assignment=state_eval.score_world(system,encoded,branch,family,world,17)
                        row['native_sha256']=record['sha256'];per_method[name].append(row);histories[name].append((hist,assignment))
                    references.append(branch['supervision']['state'][:,family['target_index'],:6]);del branch,encoded
                references=torch.stack(references)
                for name in METHODS:
                    for world,(history,assignment)in enumerate(histories[name]):
                        selected=torch.where(assignment['ids']==family['target_index'])[0];density=[]
                        if len(selected):
                            slot=int(selected[0])
                            if name=='concat_gru':gru.condition(history)
                            for seed in SEEDS:
                                generator=torch.Generator(device='cuda:0').manual_seed(stream_seed(17,family['family_id'],seed,'density'))
                                bank=(persistent_target_rollouts(gru,history['tokens'],history['shared'],history['initial'],
                                    history['presence_logits'],generator,slot,covariance=history['covariance'],draws=4)
                                    if name=='concat_gru'else cv_density_bank(history,slot,generator))
                                score=candidate_log_density(bank['state'][...,slot if name=='concat_gru'else 0,:6],
                                    bank['mode_logits'],references[None],gru.scale.std[:6],.2,bank['log_survival'])
                                density.append(dict(sample_seed=seed,log_density=score[0].cpu().tolist()))
                        per_method[name][world].update(density_samples=density,kernel_bandwidth=.2,
                            kernel_survival_scope='selected_focal_target_only',
                            kernel_smoothed_log_density=np.mean([s['log_density']for s in density],0).tolist()if density else None)
                    report['methods'][name]['rows'].extend(per_method[name])
                    report['methods'][name]['summary']=state_eval.aggregate(report['methods'][name]['rows'])
                report['completed_families']+=1
                atomic_json(out/'report.json',report)
                atomic_json(out/'status.json',dict(state='running',completed_families=report['completed_families'],
                    total_families=160,updated_unix=time.time()))
                del shared,fused,histories,references
        report['completed']=report['completed_families']==160
    except InterruptedError as error:report['pause_reason']=str(error)
    except BaseException as error:report['error']=type(error).__name__+': '+str(error);raise
    finally:
        report['wall_seconds']=time.monotonic()-start;atomic_json(out/'report.json',report)
        atomic_json(out/'completion.json',dict(completed=report['completed'],families=report['completed_families'],
            expected_families=160,report_sha256=sha256(out/'report.json'),contract_sha256=sha256(out/'contract.json')))
        atomic_json(out/'status.json',dict(state='completed'if report['completed']else'failed'if'error'in report else'paused',
            completed_families=report['completed_families'],total_families=160,updated_unix=time.time()))
    return 0 if report['completed']else 2


if __name__=='__main__':raise SystemExit(main())

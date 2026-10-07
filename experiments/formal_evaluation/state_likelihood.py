"""Score reference dynamics under history-conditioned mixtures (state diagnostic).

This is the exact standardized acceleration/wrapped-yaw/presence innovation
likelihood of JointDynamics, teacher-forced for density evaluation. It is not a
readout of generated pixels and not a density on unrestricted box trajectories.
Historical representatives are selected one-to-one before future scoring.
Missing references remain in the coverage denominator; duplicate and unassociated
slots remain explicit prediction errors. The KDE scores only the same selected
focal target, including its survival, and is a smoothed predictive density.
"""
import argparse,math
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import read_json,write_json,sha256,SAMPLE_SEEDS


HISTORY_SELECTION_POLICY = 'unique_reference:historical_score_desc,presence_desc,birth_asc:v1'


def unique_history_assignment(history, reference_ids, history_eligible):
    """Evaluation-only one-to-one representatives; no future-label argument.

    Keep every inferred slot in prediction. Select at most one slot per eligible
    reference using accumulated historical detection confidence, then historical
    presence and immutable birth (time, view, query). Extra slots remain explicit
    duplicates/unassociated predictions, never silently removed from the model.
    """
    raw = reference_ids.detach().cpu().numpy().astype(np.int64, copy=True)
    population = history_eligible.detach().cpu().numpy().astype(bool, copy=True)
    tracks = history['result']['tracks']
    if raw.ndim != 1 or len(raw) != len(tracks) or population.ndim != 1:
        raise ValueError('Historical assignment shape mismatch')
    if np.any(raw < -1) or np.any(raw >= len(population)):
        raise ValueError('Invalid historical reference index')
    selected = np.full(raw.shape, -1, dtype=np.int64)
    duplicate = np.zeros(raw.shape, dtype=bool)
    known = raw >= 0
    eligible = np.zeros(raw.shape, dtype=bool)
    eligible[known] = population[raw[known]]
    births = [tuple(int(x) for x in track['birth']) for track in tracks]
    if len(set(births)) != len(births):
        raise ValueError('Track births must be unique for order-independent selection')
    def priority(k):
        score = float(tracks[k]['score'])
        presence = float(tracks[k]['presence_logit'])
        if not math.isfinite(score) or not math.isfinite(presence):
            raise ValueError('Non-finite historical selection confidence')
        return (-score, -presence, births[k])
    for reference in np.unique(raw[eligible]):
        candidates = np.where(eligible & (raw == reference))[0].tolist()
        winner = min(candidates, key=priority)
        selected[winner] = reference
        duplicate[candidates] = True
        duplicate[winner] = False
    matched = int((selected >= 0).sum())
    expected = int(population.sum())
    stats = dict(
        historical_selection_policy=HISTORY_SELECTION_POLICY,
        predicted_history_slots=len(raw), matched_history_slots=int(eligible.sum()),
        matched_targets=matched, duplicate_history_slots=int(duplicate.sum()),
        unassociated_history_slots=int((raw < 0).sum()),
        ineligible_reference_slots=int((known & ~eligible).sum()),
        reference_missing_targets=expected-matched,
        historical_track_coverage=None if expected == 0 else matched/expected,
        selected_slot_by_reference={str(int(selected[k])):int(k) for k in np.where(selected >= 0)[0]},
        selected_birth_by_reference={str(int(selected[k])):list(births[k]) for k in np.where(selected >= 0)[0]},
    )
    return dict(ids=torch.as_tensor(selected,device=reference_ids.device),
                selected_mask=torch.as_tensor(selected >= 0,device=reference_ids.device),
                duplicate_mask=duplicate,unassociated_mask=raw < 0,
                unmatched_mask=selected < 0,stats=stats)


def align_unique_slots(state, present, assignment, reference_count):
    """Align fixed historical representatives; report all excess predictions."""
    state=np.asarray(state);present=np.asarray(present,dtype=bool)
    ids=assignment['ids'].detach().cpu().numpy()
    if state.ndim != 3 or present.shape != state.shape[:2] or state.shape[1] != len(ids):
        raise ValueError('Prediction shape mismatch')
    chosen=np.where(ids>=0)[0]
    if len(np.unique(ids[chosen])) != len(chosen):
        raise ValueError('Evaluation assignment is not one-to-one')
    aligned=np.full((state.shape[0],reference_count,state.shape[-1]),np.nan,np.float32)
    exists=np.zeros((state.shape[0],reference_count),bool)
    aligned[:,ids[chosen]]=state[:,chosen]
    exists[:,ids[chosen]]=present[:,chosen]
    extras={name:int(present[:,assignment[mask]].sum()) for name,mask in (
        ('duplicate_present_slot_steps','duplicate_mask'),
        ('unassociated_present_slot_steps','unassociated_mask'),
        ('unmatched_present_slot_steps','unmatched_mask'))}
    return aligned,exists,extras


def persistent_target_rollouts(model,tokens,scene,initial,initial_presence_logits,
                               generator,target_slot,covariance=None,draws=4,horizon=16,dt=.2):
    """Same full-slot free rollout; survival scores the selected focal target.

    This is a marginal target diagnostic, not the event that every duplicate or
    false-positive slot also persists. Other slots remain in the model context.
    """
    from experiments.paired_interventions.rollout import _initial,_step
    if initial.ndim!=3 or not 0<=target_slot<initial.shape[1] or not 1<=horizon<=16 or draws<1:
        raise ValueError('Invalid persistent target rollout')
    batch,count=initial.shape[:2]
    logits=model(tokens,scene,initial[:,None])['mode_logits'];modes=logits.shape[-1]
    repeat=modes*draws
    expand=lambda x:x.repeat_interleave(repeat,dim=0)
    mode_ids=torch.arange(modes,device=initial.device).repeat_interleave(draws).repeat(batch)
    state=_initial(expand(initial),None if covariance is None else expand(covariance),generator)
    states=[state]
    log_survival=F.logsigmoid(expand(initial_presence_logits))[:,target_slot]
    for _ in range(horizon):
        out=model(expand(tokens),expand(scene),torch.stack(states,1),mode_ids)
        log_survival=log_survival+F.logsigmoid(out['presence_logits'][:,0,-1,target_slot])
        states.append(_step(model,states[-1],out,generator,dt))
    paths=torch.stack(states[1:],1).reshape(batch,modes,draws,horizon,count,11)
    return dict(state=paths,mode_logits=logits,log_survival=log_survival.reshape(batch,modes,draws))


def innovation_nll(model,tokens,scene,history_reference,reference,eligible,present,score_eligible=None):
    previous=torch.cat([history_reference[None],reference[:-1]],0)
    output=model(tokens[None],scene,previous[None]);scale=model.scale
    acc=((reference[...,3:6]-previous[...,3:6])/.2-scale.acc_mean)/scale.acc_std
    logp=(-.5*((acc[None]-output['acc_mean'][0])/output['acc_scale'][0]).square()-output['acc_scale'][0].log()-.5*math.log(2*math.pi)).sum(-1)
    yaw=(reference[...,6]-previous[...,6]+math.pi)%(2*math.pi)-math.pi
    shifts=torch.arange(-4,5,device=yaw.device)*2*math.pi
    difference=(yaw[None,...,None]-output['yaw_mean'][0,...,None]+shifts)/output['yaw_scale'][0,...,None]
    logp+=torch.logsumexp(-.5*difference.square()-output['yaw_scale'][0,...,None].log()-.5*math.log(2*math.pi),-1)
    score_eligible=eligible if score_eligible is None else score_eligible
    if score_eligible.shape!=eligible.shape or bool((score_eligible & ~eligible).any()):raise ValueError('Invalid scoring subset')
    prior=torch.cat([eligible[None],present[:-1]],0)&score_eligible[None];active=prior&present
    hazard=-F.binary_cross_entropy_with_logits(output['presence_logits'][0],present[None].float().expand(8,-1,-1),reduction='none')
    scores=(logp*active[None]+hazard*prior[None]).sum((1,2))
    joint=-torch.logsumexp(output['mode_logits'][0].log_softmax(-1)+scores,0)
    return dict(joint_standardized_innovation_NLL=float(joint),per_active_step_NLL=float(joint/prior.sum().clamp_min(1)),active_steps=int(prior.sum()))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('paths','checkpoint','index','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--device',default='cuda:0');p.add_argument('--development',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--gpu-hours',type=float,default=24.)
    p.add_argument('--draws',type=int,default=8)
    a=p.parse_args()
    from .runtime import Budget
    budget=Budget(a.gpu_hours)
    if not 1<=a.draws<=8 or (not a.development and a.draws!=8):raise ValueError('Formal state diagnostics require eight fixed sample seeds')
    paths=read_json(a.paths);payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if not a.development and (payload['identity']['development_only'] or not payload['completed']):raise ValueError('Formal checkpoint required')
    if a.limit and not a.development:raise ValueError('No partial formal likelihood report')
    if not a.development:
        if sha256(a.index)!=payload['identity']['family_index_sha256']:raise ValueError('Changed family index')
        if sha256(paths['statistics'])!=payload['identity']['statistics_sha256']:raise ValueError('Changed state normalization')
    contract=dict(checkpoint_sha256=sha256(a.checkpoint),index_sha256=sha256(a.index),split=a.split,
        statistics_sha256=sha256(paths['statistics']),implementation_sha256=sha256(__file__),development_only=a.development,
        family_limit=a.limit,sample_seeds=list(SAMPLE_SEEDS[:a.draws]))
    cache=Path(a.output).with_suffix('.families');cache.mkdir(parents=True,exist_ok=True)
    if (cache/'contract.json').exists() and read_json(cache/'contract.json')!=contract:raise ValueError('State-density cache provenance changed')
    write_json(cache/'contract.json',contract)
    from experiments.paired_interventions.world import load_upstream,query_labels
    from experiments.paired_interventions.bridge import historical_posterior
    from experiments.paired_interventions.data import FamilyDataset,to_device,replay_sample
    from experiments.paired_interventions.rollout import persistent_mode_rollouts,sample_shared_future
    from experiments.paired_interventions.sample import stream_seed
    from experiments.paired_interventions.objectives import candidate_log_density
    from .metrics import aligned_state
    from experiments.paper_experiments.agcr_exp.stage2_system import FrozenImageEncoders,matched_state_supervision
    torch.set_num_threads(4)
    device=torch.device(a.device);system,_=load_upstream(paths['upstream'],paths['statistics'],device)
    state={k[len('upstream.'):]:v for k,v in payload['model'].items() if k.startswith('upstream.')}
    system.load_state_dict(state,strict=True);system.eval().requires_grad_(False)
    encoders=FrozenImageEncoders(paths['dino_code'],paths['dino_weights'],paths['wan']+'/vae',device)
    dataset=FamilyDataset(a.index,a.split);rows=[]
    for i in range(min(len(dataset),a.limit or len(dataset))):
        marker=cache/(dataset.families[i]['family_id']+'.json')
        if marker.exists():rows.extend(read_json(marker)['rows']);continue
        if budget.expired():
            write_json(a.output,dict(schema='agcr-state-density-diagnostic-2',complete=False,budget_paused=True,contract=contract,development_only=a.development,rows=rows));return
        family=to_device(dataset[i],device);target=family['metadata']['target_index'];references=torch.stack([family['branches'][w,0]['supervision']['state'][:,target,:6] for w in range(2)])
        family_rows=[]
        for w in range(2):
            branch=family['branches'][w,0];labels=replay_sample(branch)['labels'];history=historical_posterior(system,branch['inputs'],encoders.history(branch['inputs']['rgb']))
            eligible_count=int(labels['history_eligible'].sum());row=dict(family_id=family['metadata']['family_id'],world=w,reference_history_targets=eligible_count,matched_targets=0,likelihood=None,kernel_smoothed_log_density=None)
            ids=None;target_tracks=[]
            assignment=unique_history_assignment(dict(result=dict(tracks=[])),torch.empty(0,device=device,dtype=torch.long),labels['history_eligible'])
            row.update(assignment['stats'])
            row['reference_population_steps']=int((torch.cat([labels['history_eligible'][None],labels['present'][:-1]],0)&labels['history_eligible'][None]).sum())
            if not history['empty']:
                ids=query_labels(history,labels)
                known,mapped,eligible,old,reference,presence=matched_state_supervision(history['result']['initial_state'],ids,labels)
                assignment=unique_history_assignment(history,ids,labels['history_eligible'])
                row.update(assignment['stats'])
                if assignment['selected_mask'].any():
                    row['likelihood']=innovation_nll(system.dynamics,history['result']['tokens'],history['shared'],old,reference,eligible,presence,assignment['selected_mask'])
                    assert row['likelihood']['active_steps']<=row['reference_population_steps']
                target_tracks=torch.where(assignment['ids']==target)[0]
            samples=[];existence=[];density=[]
            for sample_index,sample_seed in enumerate(SAMPLE_SEEDS[:a.draws]):
                # Same sampled world for both camera paths; world number is not
                # included in the seed, matching the paired sampler's CRNs.
                generator=lambda tag:torch.Generator(device=device).manual_seed(stream_seed(payload['identity']['seed'],family['metadata']['family_id'],sample_seed,tag))
                aligned=np.full(tuple(labels['state'].shape),np.nan,np.float32);exists=np.zeros(tuple(labels['present'].shape),bool)
                mode=None;extras=dict(duplicate_present_slot_steps=0,unassociated_present_slot_steps=0,unmatched_present_slot_steps=0)
                if not history['empty']:
                    draw=sample_shared_future(system.dynamics,history['tokens'],history['shared'],history['initial'],history['presence_logits'],generator('state'),covariance=history['covariance'])
                    mode=int(draw['mode'][0]);pred=draw['state'][0].cpu().numpy();presence=draw['present'][0].cpu().numpy()
                    aligned,exists,extras=align_unique_slots(pred,presence,assignment,len(labels['history_eligible']))
                metrics=aligned_state(aligned,exists,labels['state'].cpu().numpy(),labels['present'].cpu().numpy(),labels['history_eligible'].cpu().numpy())
                samples.append(dict(sample_index=sample_index,sample_seed=sample_seed,scene_mode=mode,metrics=metrics,**extras));existence.append(exists)
                if len(target_tracks):
                    # Historical matching fixes the query; references score
                    # only free-running predictions, never select its path.
                    draws=persistent_target_rollouts(system.dynamics,history['tokens'],history['shared'],history['initial'],history['presence_logits'],generator('density'),int(target_tracks[0]),covariance=history['covariance'],draws=4)
                    log=candidate_log_density(draws['state'][...,int(target_tracks[0]),:6],draws['mode_logits'],references[None],system.dynamics.scale.std[:6],.2,draws['log_survival'])
                    density.append(dict(sample_seed=sample_seed,log_density=log[0].cpu().tolist()))
            mean_keys=('ADE_m','FDE_m','position_error_m','velocity_error_mps','heading_error_rad','size_error_m','state_missing_rate','FDE_coverage')
            row['state_samples']=samples;row['state_mean']={key:None if any(s['metrics'].get(key) is None for s in samples) else float(np.mean([s['metrics'][key] for s in samples])) for key in mean_keys}
            population=np.broadcast_to(labels['history_eligible'].cpu().numpy(),tuple(labels['present'].shape));prob=np.mean(existence,axis=0)
            row['presence_Brier']=float(np.square(prob-labels['present'].cpu().numpy())[population].mean()) if population.any() else None
            if density:
                row['kernel_smoothed_log_density']=np.mean([d['log_density'] for d in density],axis=0).tolist();row['kernel_bandwidth']=.2;row['density_samples']=density;row['kernel_survival_scope']='selected_focal_target_only; other inferred slots retained as model context'
            row['historical_track_coverage']=None if eligible_count==0 else row['matched_targets']/eligible_count;family_rows.append(row)
        write_json(marker,dict(rows=family_rows));rows.extend(family_rows)
    write_json(a.output,dict(schema='agcr-state-density-diagnostic-2',complete=True,development_only=a.development,checkpoint_sha256=sha256(a.checkpoint),index_sha256=sha256(a.index),
        sample_seeds=list(SAMPLE_SEEDS[:a.draws]),aggregation='mean_of_8' if a.draws==8 else 'development_mean',historical_selection_policy=HISTORY_SELECTION_POLICY,
        scope='internal dynamics, separate from independent video readout',likelihood_units='unique historically selected targets: standardized acceleration plus wrapped yaw/presence innovations, conditional on the original teacher-forced full-slot prefix; not full-scene likelihood',rows=rows))
if __name__=='__main__':main()

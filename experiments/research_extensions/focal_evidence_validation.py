"""Frozen, focal-target continuation diagnostics; future labels score only.

This new readout supplements, rather than replaces, the registered eight-draw
state/KDE tests. All checkpoints and the complete TEST population are locked
before inference. Missing targets remain explicit, including partner removal.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

SCHEMA = 'agcr-focal-evidence-validation-1'
ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    'experiments/research_extensions/focal_evidence_validation.py',
    'experiments/research_extensions/state_prediction_baselines.py',
    'experiments/research_extensions/train_evidence_repeats.py',
    'experiments/research_extensions/train_future_no_evidence.py',
    'experiments/research_extensions/train_history_future_extended.py',
    'experiments/research_extensions/train_history_future.py',
    'experiments/research_extensions/train_history_identity.py',
    'experiments/research_extensions/history_surface_refiner.py',
    'experiments/research_extensions/history_identity.py',
    'experiments/research_extensions/region_probe.py',
    'experiments/paired_interventions/rollout.py',
    'experiments/paired_interventions/bridge.py',
    'experiments/paired_interventions/upstream_runtime.py',
    'experiments/formal_evaluation/state_likelihood.py',
    'experiments/paper_experiments/agcr_exp/future_prediction.py',
)


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    from .region_probe import sha256
    return {s: sha256(ROOT/s) for s in SOURCES}


def frame_masks(y, target, receiver):
    """Reference-only support; onsets retain the registered three-frame rule."""
    present = np.asarray(y['present'], bool)[:, target]
    visible = np.asarray(y['visible_fraction'])[:, receiver, target] >= .1
    area = np.asarray(y['projection_area'])[:, receiver, target]
    visible &= area >= 144
    old_visible = bool(y['history_visible_fraction'][-1, receiver, target] >= .1
                       and y['history_projection_area'][-1, receiver, target] >= 144)
    old_area = y['history_projection_area'][-1, receiver, target]
    before_visible = np.r_[old_visible, visible[:-1]]
    before_area = np.r_[old_area, area[:-1]]
    before_present = np.r_[True, present[:-1]]
    starts = visible & ~before_visible & (before_area >= 144) & before_present & present
    window = starts.copy()
    window[1:] |= starts[:-1]
    window[2:] |= starts[:-2]
    return dict(full=present, hidden=present & ~visible & (area >= 144),
                reappearance=window & visible & present, onsets=starts)


def score_paths(position, existence, references, world, masks, draws):
    """No density bandwidth or future-dependent prediction/matching.

    Path error reads the deployed state (absence freezes its last position).
    Availability and accuracy on predicted-present frames are reported beside
    it. Branch success requires predicted presence throughout the scored window;
    unavailable draws count as failures, and ties count half. This is distance
    discrimination, not a calibrated probability of the true world.
    """
    refs = np.asarray(references, np.float64)
    if refs.shape != (2, 16, 3) or world not in (0, 1) or draws < 1:
        raise ValueError('Two focal references, 16 frames and positive draws required')
    matched = position is not None
    if matched:
        p, alive = np.asarray(position, np.float64), np.asarray(existence, bool)
        if p.shape != (draws, 16, 3) or alive.shape != (draws, 16) or not np.isfinite(p).all():
            raise ValueError('Complete finite position banks and presence required')
        distances = np.linalg.norm(p[:, None] - refs[None], axis=-1)
    result = dict(historical_target_matched=matched, draws=draws, windows={})
    for name in ('full', 'hidden', 'reappearance'):
        mask = np.asarray(masks[name], bool)
        if mask.shape != (16,):
            raise ValueError('A reference-only 16-frame support is required')
        count = int(mask.sum())
        row = dict(reference_frames=count, path_ADE_m=None, path_FDE_m=None,
                   predicted_present_ADE_m=None, coverage=None,
                   distance_correct_fraction=None, mean_branch_margin_m=None,
                   complete_draw_fraction=None, per_draw_ADE_m=[], branch_winners=[])
        if count:
            row['coverage'] = 0. if not matched else float(alive[:, mask].mean())
            row['complete_draw_fraction'] = 0. if not matched else float(alive[:, mask].all(1).mean())
            row['distance_correct_fraction'] = 0.
            if matched:
                errors = distances[:, world, mask]
                own = errors.mean(1)
                other = distances[:, 1-world, mask].mean(1)
                margin = other-own
                winners = (margin > 0).astype(float)+.5*(margin == 0)
                winners *= alive[:, mask].all(1)
                row.update(path_ADE_m=float(own.mean()), path_FDE_m=float(errors[:, -1].mean()),
                    predicted_present_ADE_m=float(errors[alive[:, mask]].mean()) if alive[:, mask].any() else None,
                    distance_correct_fraction=float(winners.mean()), mean_branch_margin_m=float(margin.mean()),
                    per_draw_ADE_m=own.tolist(), branch_winners=winners.tolist())
        result['windows'][name] = row
    return result


def load_selective_branch(path):
    """Decode needed members only; keep references separate from history inputs."""
    import torch
    from experiments.paired_interventions.data import CLASS_MAP
    def tensor(v):
        t = torch.from_numpy(np.ascontiguousarray(v))
        return t.float() if t.dtype == torch.float64 else t
    with np.load(path, allow_pickle=False) as z:
        K = z['K']
        if K.shape != (22, 2, 3, 3) or not np.array_equal(K, np.broadcast_to(K[0, 0], K.shape)):
            raise ValueError('Fixed shared calibration required')
        rgb = z['rgb'][:6].copy()
        depth, valid = z['depth_mm'][:6].copy(), z['valid'][:6].astype(bool)
        cameras, times = z['camera_to_world'].copy(), z['time_s'].copy()
        state, present = z['world_state'].copy(), z['present'][6:].astype(bool)
        fraction, area = z['visible_fraction'].copy(), z['projection_area'].copy()
        classes = np.asarray([CLASS_MAP[int(c)] for c in z['class_id']], np.int64)
    x = dict(rgb=tensor(rgb), depth_mm=tensor(depth), depth_valid=tensor(valid), K=tensor(K[0, 0]),
        camera_to_world=tensor(cameras[:6]), planned_camera_to_world=tensor(cameras[6:]),
        time_s=tensor(times[:6]), future_time_s=tensor(times[6:]))
    y = dict(history_state=tensor(state[:6]), state=tensor(state[6:]), present=tensor(present),
        class_id=tensor(classes), history_visible_fraction=tensor(fraction[:6]),
        history_projection_area=tensor(area[:6]), visible_fraction=tensor(fraction[6:]),
        projection_area=tensor(area[6:]))
    return dict(inputs=x, supervision=y)


def prepare(base, output):
    """CPU-only lock of existing population, weights, source and new budget."""
    import torch
    from .region_probe import atomic_json, sha256
    from .typed_evidence_repeat_test import validate_population
    from experiments.base48.train import audit_index_ancestry
    base, out = Path(base), Path(output)
    out.mkdir(parents=True, exist_ok=True)
    old = read(base/'quality_revision/evidence_validation/plan.json')
    typed = read(base/'quality_revision/typed_test_lock.json')
    paths = read(old['files']['paths']['path'])
    source, families = validate_population(typed, paths['families'], typed['index'], audit_index_ancestry(paths['families']))
    candidates = []
    for seed in (17017, 17029, 17043):
        for ev in (0, 1):
            if seed == 17029:
                cp = (base/'no_evidence_ablation/training/last.pt' if not ev else
                      Path(typed['checkpoints']['coarse_full_paired_geometry']['path']))
                module = ('train_future_no_evidence' if not ev else 'train_history_future')
            else:
                cp = base/f'quality_revision/evidence_validation/seed{seed}_ev{ev}/training/last.pt'
                module = 'train_evidence_repeats'
            from importlib import import_module
            metadata = import_module('experiments.research_extensions.'+module).load_future_checkpoint(
                cp, typed['parent'], typed['training_index_sha256'], device='cpu')
            config = metadata['identity']['configuration']
            if config['seed'] != seed or config['weights']['evidence'] != ev*.1 or metadata['step'] != 2048:
                raise ValueError('Wrong fitted future checkpoint')
            if metadata['history_checkpoint']['sha256'] != old['files']['history_overlay']['sha256']:
                raise ValueError('Every candidate must use the same frozen historical model')
            candidates.append(dict(name=f'seed{seed}_ev{ev}', seed=seed, evidence=ev*.1,
                path=str(cp), sha256=sha256(cp), module=module, metadata=metadata))
    gru_path = base/'quality_revision/evidence_validation/concat_gru_thread8_training/last.pt'
    gru_contract = read(base/'quality_revision/evidence_validation/state_baselines/contract.json')
    if sha256(gru_path) != gru_contract['checkpoint_sha256']:
        raise ValueError('Changed completed GRU comparison')
    ledger = out/'budget.json'
    plan = dict(schema=SCHEMA+'-plan', split='test', exploratory=True,
        diagnostic_scope='frozen fitted checkpoints; no training or model/test selection',
        created_unix=time.time(), family_ids=[f['family_id'] for f in families],
        informative_family_ids=[f['family_id'] for f in families if f['cohort']=='evidence'],
        index=typed['index'], index_sha256=typed['index_sha256'], population_lock=str(base/'quality_revision/typed_test_lock.json'),
        population_lock_sha256=sha256(base/'quality_revision/typed_test_lock.json'),
        files=old['files'], training_index_sha256=typed['training_index_sha256'],
        candidates=candidates, gru=dict(path=str(gru_path),sha256=sha256(gru_path),contract=gru_contract),
        sources=source_hashes(), camera=0, worlds=[0,1], horizon=16, dt_seconds=.2,
        sample_draws=64, batch_draws=16, sampling_seed=20261007,
        history_reuse='one immutable paired history per native branch for all fitted future decoders',
        partner_removal='mask partner RGB-D, encoded features, queries and scene memory through the existing availability path',
        metrics=['focal path ADE/FDE','hidden/reappearance focal path ADE','availability',
                 'distance continuation discrimination','strict two-world success','partner-removal contrast'],
        missing_policy='historical misses retained; errors undefined, availability and branch success zero',
        distinction='path error scores deployed state even after absence freezes it; predicted-present error and coverage also reported',
        branch_metric='fraction closer in mean Euclidean position distance to factual rather than counterfactual trajectory; ties half; absent draws fail',
        bootstrap=dict(repetitions=50000,seed=20261007,unit='paired scene family, with crossed resampling of three fitted future seeds'),
        budget=dict(ledger_path=str(ledger),gpu_hour_cap=2.,deadline_unix=time.time()+6*3600),
        authorization='Explicit bounded run configuration')
    if (out/'plan.json').exists():
        raise ValueError('Never overwrite a frozen focal plan')
    atomic_json(out/'plan.json', plan)
    atomic_json(ledger,dict(schema='peerdream-direction-budget-1',gpu_hour_cap=2.,authorized_gpu_hour_cap=2.,
        authorization=plan['authorization'],prior_region_probe_gpu_hours=0.,deadline_unix=plan['budget']['deadline_unix'],jobs=[]))
    return plan


def load_models(plan, device):
    import torch
    from .region_probe import load_state_overlay, sha256
    from .train_history_identity import load_identity_checkpoint
    from .state_prediction_baselines import ConcatGRU
    from experiments.paired_interventions.world import load_upstream
    from experiments.paper_experiments.agcr_exp.stage2_system import FrozenImageEncoders
    for key, item in plan['files'].items():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Changed immutable file: '+key)
    paths = read(plan['files']['paths']['path'])
    system, _ = load_upstream(paths['upstream'],paths['statistics'],device)
    parent,_ = load_state_overlay(plan['files']['parent']['path'])
    system.load_state_dict({k[len('upstream.'):]:v for k,v in parent['model'].items()
                           if k.startswith('upstream.')},strict=True)
    del parent
    load_identity_checkpoint(plan['files']['history_overlay']['path'],plan['files']['parent']['path'],
                             plan['training_index_sha256'],upstream=system,device=device)
    system.eval().requires_grad_(False)
    models = {}
    for item in plan['candidates']:
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Changed frozen future checkpoint')
        payload = torch.load(item['path'],map_location='cpu',weights_only=False)
        model = copy.deepcopy(system.dynamics)
        weights = {k[len('dynamics.'):]:v for k,v in payload['future_overlay'].items() if k.startswith('dynamics.')}
        model.load_state_dict(weights,strict=True)
        model.eval().requires_grad_(False); models[item['name']] = model
        del payload
    if sha256(plan['gru']['path']) != plan['gru']['sha256']:
        raise ValueError('Changed frozen GRU')
    cp = torch.load(plan['gru']['path'],map_location='cpu',weights_only=False)
    gru = ConcatGRU(read(paths['statistics'])).to(device)
    gru.load_state_dict(cp['model'],strict=True); gru.eval().requires_grad_(False)
    models['concat_gru'] = gru
    encoders = FrozenImageEncoders(paths['dino_code'],paths['dino_weights'],paths['wan']+'/vae',device)
    return system,models,encoders


def focal_bank(model, history, slot, family, plan, device, cv=False):
    """Batched original absorbing-presence sampler, with locked new streams."""
    import torch
    from .state_prediction_baselines import constant_velocity
    from experiments.paired_interventions.rollout import sample_shared_future
    from experiments.paired_interventions.sample import stream_seed
    positions, existence = [], []
    for offset in range(0,plan['sample_draws'],plan['batch_draws']):
        count = min(plan['batch_draws'],plan['sample_draws']-offset)
        expanded = {key:history[key].expand(count,*history[key].shape[1:])
            for key in ('tokens','shared','initial','covariance','presence_logits')}
        generator = torch.Generator(device=device).manual_seed(
            stream_seed(plan['sampling_seed'],family,offset,'focal_state'))
        if cv:
            draw = constant_velocity(expanded,generator)
        else:
            if hasattr(model,'condition'):
                model.condition(history)
            draw = sample_shared_future(model,expanded['tokens'],expanded['shared'],expanded['initial'],
                expanded['presence_logits'],generator,covariance=expanded['covariance'])
        positions.append(draw['state'][:,:,slot,:3].cpu().numpy())
        existence.append(draw['present'][:,:,slot].cpu().numpy())
    return np.concatenate(positions),np.concatenate(existence)


def run(a):
    import torch
    from .region_probe import atomic_json,sha256
    from .typed_evidence_repeat_test import budget_guard,validate_population,validate_input_boundary
    from experiments.base48.train import audit_index_ancestry
    from experiments.base48.model import receiver_only_branch
    from experiments.paired_interventions.data import to_device,historical_labels
    from experiments.paired_interventions.bridge import historical_posterior
    from experiments.paired_interventions.world import query_labels
    from experiments.formal_evaluation.state_likelihood import unique_history_assignment
    plan = read(a.plan)
    reservation = budget_guard(a.budget_ledger,a.job_id,plan['budget'])
    if a.max_seconds > reservation['maximum_wall_seconds']-30 or source_hashes()!=plan['sources']:
        raise ValueError('Budget or source contract changed')
    if plan['sample_draws'] not in (16,32,64) or plan['batch_draws'] not in (8,16,32,64):
        raise ValueError('Only explicitly locked sampling scales are admitted')
    if sha256(plan['index'])!=plan['index_sha256'] or sha256(plan['population_lock'])!=plan['population_lock_sha256']:
        raise ValueError('Changed population')
    paths = read(plan['files']['paths']['path'])
    source,families = validate_population(read(plan['population_lock']),paths['families'],
        plan['index'],audit_index_ancestry(paths['families']))
    if [f['family_id'] for f in families]!=plan['family_ids']:
        raise ValueError('Changed complete population')
    selected = [f for i,f in enumerate(families) if i%a.shards==a.shard]
    if a.benchmark:
        # Deterministic first informative family, runtime only; not selected by outcomes.
        selected = [next(f for f in selected if f['cohort']=='evidence')]
    out = Path(a.output); out.mkdir(parents=True,exist_ok=True)
    contract = dict(schema=SCHEMA+'-contract',plan_sha256=sha256(a.plan),
        shard=a.shard,shards=a.shards,families=[f['family_id'] for f in selected],benchmark=a.benchmark)
    if (out/'contract.json').exists() and read(out/'contract.json')!=contract:
        raise ValueError('Changed output contract')
    atomic_json(out/'contract.json',contract)
    start = time.monotonic(); stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):
        signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def boundary():
        if stop[0] or time.monotonic()-start>a.max_seconds-20:
            raise InterruptedError('Stop at bounded completed-family boundary')
    report = dict(schema=SCHEMA,completed=False,exploratory=True,benchmark=a.benchmark,contract=contract,
                  rows=[],family_timings=[],completed_families=0,expected_families=len(selected))
    torch.set_num_threads(4); torch.cuda.set_device('cuda:0');device='cuda:0'
    try:
        system,models,encoders = load_models(plan,device)
        report['parameters'] = {k:sum(p.numel() for p in m.parameters()) for k,m in models.items()}
        report['initialization_seconds'] = time.monotonic()-start
        with torch.no_grad():
            for family in selected:
                boundary();fid=family['family_id'];marker=out/'families'/(fid+'.json')
                if marker.exists():
                    saved=read(marker)
                    if saved.get('contract_sha256')!=sha256(out/'contract.json') or saved.get('artifact_sha256')!=sha256(marker.with_suffix('.npz')):
                        raise ValueError('Changed focal family artifact')
                    report['rows'].extend(saved['rows']);report['completed_families']+=1
                    continue
                tick=time.monotonic();bank={};rows=[];history_inputs=[]
                # Check all four audited factorial cells, even though camera0 is scored.
                for rec in family['records']:
                    if sha256(Path(source['root'])/rec['path'])!=rec['sha256']:
                        raise ValueError('Changed audited native branch')
                branches=[load_selective_branch(Path(source['root'])/next(r['path'] for r in family['records']
                    if (r['world'],r['camera'])==(world,0))) for world in (0,1)]
                references=np.stack([b['supervision']['state'][:,family['target_index'],:3].numpy() for b in branches])
                bank['references']=references
                for world,raw in enumerate(branches):
                    boundary();validate_input_boundary(raw['inputs'])
                    history_inputs.append({k:raw['inputs'][k][:,family['receiver']].clone()
                        for k in ('rgb','depth_mm','depth_valid','camera_to_world')})
                    y=raw['supervision'];target=family['target_index']
                    masks=frame_masks({k:v.numpy() for k,v in y.items()},target,family['receiver'])
                    for key,value in masks.items():bank[f'w{world}_{key}_mask']=value
                    branch=to_device(raw,device);past=historical_labels(branch)
                    eligible=past['history_detectable'].any(0).any(0)
                    encoded=encoders.history(branch['inputs']['rgb'])
                    for policy in ('paired','receiver_only'):
                        boundary()
                        if policy=='paired':x,features=branch['inputs'],encoded
                        else:
                            x=receiver_only_branch(branch,family['receiver'])['inputs']
                            keep=x['observation_available']
                            # Missing channels never contribute queries, memory or features.
                            features=tuple(torch.where(keep.reshape(6,2,*([1]*(v.ndim-2))),v,torch.zeros_like(v)) for v in encoded)
                        history=historical_posterior(system,x,features)
                        ids=query_labels(history,past)
                        assignment=unique_history_assignment(history,ids,eligible)
                        slots=torch.where(assignment['ids']==target)[0]
                        slot=int(slots[0]) if len(slots) else -1
                        names=[*models,'shared_history_cv']
                        for name in names:
                            boundary();model=models.get(name)
                            position=existence=None
                            if slot>=0:
                                position,existence=focal_bank(model,history,slot,fid,plan,device,cv=name=='shared_history_cv')
                                bank[f'w{world}_{policy}_{name}_position']=position
                                bank[f'w{world}_{policy}_{name}_present']=existence
                            score=score_paths(position,existence,references,world,masks,plan['sample_draws'])
                            rows.append(dict(family_id=fid,cohort=family['cohort'],receiver=family['receiver'],
                                world=world,input_policy=policy,method=name,target_index=target,focal_slot=slot,
                                historical_assignment=assignment['stats'],**score))
                        del history,assignment
                    del branch,encoded
                same_receiver=all(torch.equal(history_inputs[0][k],history_inputs[1][k]) for k in history_inputs[0])
                if not same_receiver:
                    raise ValueError('Receiver history is not byte-identical across the intervention pair')
                marker.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(marker.with_suffix('.npz'),**bank)
                atomic_json(marker,dict(contract_sha256=sha256(out/'contract.json'),
                    receiver_history_identical=True,artifact_sha256=sha256(marker.with_suffix('.npz')),rows=rows))
                report['rows'].extend(rows);report['completed_families']+=1
                report['family_timings'].append(dict(family_id=fid,wall_seconds=time.monotonic()-tick))
                atomic_json(out/'report.json',report)
                atomic_json(out/'status.json',dict(state='running',completed_families=report['completed_families'],
                    total_families=len(selected),updated_unix=time.time(),wall_seconds=time.monotonic()-start))
                print(json.dumps(dict(family=fid,completed=report['completed_families'],seconds=round(time.monotonic()-tick,2))),flush=True)
                del branches,bank,history_inputs
        report['completed']=report['completed_families']==len(selected)
    except InterruptedError as error:
        report['pause_reason']=str(error)
    except BaseException as error:
        report['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        report['wall_seconds']=time.monotonic()-start
        atomic_json(out/'report.json',report)
        atomic_json(out/'completion.json',dict(completed=report['completed'],report_sha256=sha256(out/'report.json'),
            contract_sha256=sha256(out/'contract.json'),families=report['completed_families'],expected_families=len(selected)))
        atomic_json(out/'status.json',dict(state='completed' if report['completed'] else 'failed' if 'error' in report else 'paused',
            completed_families=report['completed_families'],total_families=len(selected),updated_unix=time.time()))
    return 0 if report['completed'] else 2


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--base',required=True);prep.add_argument('--output',required=True)
    worker=sub.add_parser('run')
    for name in ('plan','output','budget-ledger','job-id'):worker.add_argument('--'+name,required=True)
    worker.add_argument('--shard',type=int,default=0);worker.add_argument('--shards',type=int,default=4)
    worker.add_argument('--max-seconds',type=float,required=True);worker.add_argument('--benchmark',action='store_true')
    a=p.parse_args()
    if a.command=='prepare':
        plan=prepare(a.base,a.output)
        print(json.dumps(dict(prepared=True,families=len(plan['family_ids']),draws=plan['sample_draws'],gpu_hour_cap=2.)))
        return 0
    if not 0<=a.shard<a.shards:raise ValueError('Invalid disjoint shard')
    return run(a)


if __name__=='__main__':raise SystemExit(main())

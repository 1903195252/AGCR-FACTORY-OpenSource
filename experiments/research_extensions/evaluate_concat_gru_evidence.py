"""Frozen-history TEST cache and complete ConcatGRU loss-pair evaluation.

The cache computes the historical posterior once and retains it unchanged.
Formal numerical metrics reuse Base48 score_world, including missing states.
Future labels are retained separately for historical assignment and scoring.
"""
import argparse
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from .region_probe import atomic_json, sha256, load_state_overlay
from .train_candidate import atomic_checkpoint
from .train_concat_gru_evidence import SCHEMA as TRAIN_SCHEMA, SEEDS as TRAIN_SEEDS, STEPS, to_device
from .state_prediction_baselines import ConcatGRU

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'agcr-concat-gru-evidence-test-1'
SOURCES = ('experiments/research_extensions/evaluate_concat_gru_evidence.py',
    'experiments/research_extensions/train_concat_gru_evidence.py',
    'experiments/research_extensions/state_prediction_baselines.py',
    'experiments/base48/state_eval.py',
    'experiments/formal_evaluation/metrics.py',
    'experiments/formal_evaluation/state_likelihood.py',
    'experiments/paired_interventions/rollout.py',
    'experiments/paired_interventions/objectives.py')


def load_json(path):
    return json.loads(Path(path).read_text())


def population(plan_path):
    from .typed_evidence_repeat_test import validate_population
    from experiments.base48.train import audit_index_ancestry
    plan = load_json(plan_path)
    for key, item in plan['files'].items():
        if sha256(item['path']) != item['sha256']: raise ValueError('Changed frozen input: '+key)
    if sha256(plan['index']) != plan['index_sha256'] or sha256(plan['population_lock']) != plan['population_lock_sha256']:
        raise ValueError('Changed fixed population lock')
    paths = load_json(plan['files']['paths']['path'])
    source, families = validate_population(load_json(plan['population_lock']), paths['families'],
        plan['index'], audit_index_ancestry(paths['families']))
    if [f['family_id'] for f in families] != plan['family_ids']: raise ValueError('Different TEST families')
    return plan, paths, source, families


def setup_boundary(args):
    start = time.monotonic(); stopped = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    if args.device.startswith('cuda'):
        from .typed_evidence_repeat_test import budget_guard
        budget = load_json(args.budget_ledger)
        row = budget_guard(args.budget_ledger, args.job_id, dict(ledger_path=args.budget_ledger,
            gpu_hour_cap=2., deadline_unix=budget['deadline_unix']))
        if args.max_seconds > row['maximum_wall_seconds']-30:
            raise ValueError('Preserve GPU wrapper cleanup time')
        torch.cuda.set_device(args.device)
        deadline = budget['deadline_unix']
    else:
        deadline = float('inf')
    torch.set_num_threads(args.threads)
    def boundary():
        if stopped[0] or time.monotonic()-start >= args.max_seconds-20 or time.time() >= deadline-20:
            raise InterruptedError('Pause at a completed-family boundary')
    return boundary, start


def prepare(args):
    from .train_history_identity import load_identity_checkpoint
    from .typed_evidence_repeat_test import validate_input_boundary
    from .focal_evidence_validation import load_selective_branch
    from experiments.paired_interventions.world import load_upstream
    from experiments.paired_interventions.bridge import historical_posterior
    from experiments.paper_experiments.agcr_exp.stage2_system import FrozenImageEncoders
    from experiments.base48.state_eval import KEEP_LABELS
    boundary, start = setup_boundary(args)
    plan, paths, source, families = population(args.plan)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    contract = dict(schema=SCHEMA+'-history-cache', plan_sha256=sha256(args.plan),
        families=[f['family_id'] for f in families], historical_checkpoint=plan['files']['history_overlay'],
        device=args.device, posterior_inputs='six historical paired RGB-D frames only', future_conditions=False,
        source_sha256=sha256(__file__), input_boundary_checked=True, tensor_dtype='unchanged float32')
    if (out/'contract.json').exists() and load_json(out/'contract.json') != contract:
        raise ValueError('Do not change the cache contract')
    atomic_json(out/'contract.json', contract)
    completed = []
    try:
        system, _ = load_upstream(paths['upstream'], paths['statistics'], args.device)
        parent, _ = load_state_overlay(plan['files']['parent']['path'])
        system.load_state_dict({k[len('upstream.'):]: v for k, v in parent['model'].items()
            if k.startswith('upstream.')}, strict=True); del parent
        load_identity_checkpoint(plan['files']['history_overlay']['path'], plan['files']['parent']['path'],
            plan['training_index_sha256'], upstream=system, device=args.device)
        system.eval().requires_grad_(False)
        encoders = FrozenImageEncoders(paths['dino_code'], paths['dino_weights'], paths['wan']+'/vae', args.device)
        with torch.no_grad():
            for family in families:
                boundary(); fid = family['family_id']; cp = out/(fid+'.pt'); stamp = out/(fid+'.json')
                if cp.exists() and stamp.exists():
                    old = load_json(stamp)
                    if old['sha256'] != sha256(cp) or old['contract_sha256'] != sha256(out/'contract.json'):
                        raise ValueError('Changed cached posterior')
                    completed.append(old); continue
                tick = time.monotonic()
                for r in family['records']:
                    if sha256(Path(source['root'])/r['path']) != r['sha256']:
                        raise ValueError('Changed audited TEST branch')
                cells, receiver_inputs = [], []
                for world in (0, 1):
                    record = next(r for r in family['records'] if (r['world'], r['camera']) == (world, 0))
                    raw = load_selective_branch(Path(source['root'])/record['path'])
                    validate_input_boundary(raw['inputs'])
                    receiver_inputs.append({k: raw['inputs'][k][:, family['receiver']].clone()
                        for k in ('rgb', 'depth_mm', 'depth_valid', 'camera_to_world')})
                    branch = to_device(raw, args.device)
                    encoded = encoders.history(branch['inputs']['rgb'])
                    history = historical_posterior(system, branch['inputs'], encoded)
                    cells.append(dict(world=world, native_sha256=record['sha256'],
                        history=to_device(history, 'cpu'),
                        supervision={k: to_device(branch['supervision'][k], 'cpu') for k in KEEP_LABELS}))
                    del raw, branch, encoded, history
                if not all(torch.equal(receiver_inputs[0][k], receiver_inputs[1][k]) for k in receiver_inputs[0]):
                    raise ValueError('Receiver histories differ between worlds')
                atomic_checkpoint(cp, dict(family=family, cells=cells, receiver_history_identical=True))
                receipt = dict(family_id=fid, sha256=sha256(cp), contract_sha256=sha256(out/'contract.json'),
                    native_sha256=[c['native_sha256'] for c in cells], receiver_history_identical=True)
                atomic_json(stamp, receipt); completed.append(receipt)
                atomic_json(out/'status.json', dict(state='running', completed_families=len(completed),
                    total_families=160, family_seconds=time.monotonic()-tick, updated_unix=time.time()))
                print(json.dumps(dict(family=fid, completed=len(completed), seconds=round(time.monotonic()-tick,2))), flush=True)
                del cells, receiver_inputs
    except InterruptedError:
        pass
    finally:
        result = dict(schema=SCHEMA+'-history-completion', completed=len(completed)==160,
            families=completed, expected_families=160, contract_sha256=sha256(out/'contract.json'),
            wall_seconds=time.monotonic()-start)
        atomic_json(out/'completion.json', result)
        atomic_json(out/'status.json', dict(state='completed' if result['completed'] else 'paused',
            completed_families=len(completed), total_families=160, updated_unix=time.time()))
    return 0 if len(completed)==160 else 2


def fit_identity(path, statistics_path, compact_receipt):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    done = load_json(Path(path).with_name('completion.json'))
    identity = payload['identity']
    if (payload['schema'] != TRAIN_SCHEMA or payload['completed'] is not True
            or payload['step'] != STEPS or done['completed'] is not True
            or done['checkpoint_sha256'] != sha256(path) or identity['steps'] != STEPS
            or identity['seed'] not in TRAIN_SEEDS or identity['evidence_coefficient'] not in (0., .1)
            or identity['history_frozen'] is not True or identity['future_truth_conditions'] is not False
            or identity['selection_uses_test'] is not False or identity['evidence_forward_retained'] is not True
            or identity['geometry_objective'] is not False or identity['lr'] != .001
            or identity['statistics_sha256'] != sha256(statistics_path)
            or identity['compact_sha256'] != compact_receipt['sha256']
            or identity['cache_manifest_sha256'] != compact_receipt['cache_manifest_sha256']):
        raise ValueError('Invalid completed matched GRU fit')
    for source, digest in identity['sources'].items():
        if sha256(ROOT/source) != digest: raise ValueError('Training source changed: '+source)
    return payload


def score(args):
    from experiments.base48 import state_eval
    from .evaluate_state_prediction_baselines import FixedHistory
    from .state_eval_fixed import EncodedHistory
    from .typed_evidence_repeat_test import SEEDS
    from experiments.paired_interventions.sample import stream_seed
    from experiments.paired_interventions.objectives import candidate_log_density
    from experiments.formal_evaluation.state_likelihood import persistent_target_rollouts
    boundary, start = setup_boundary(args)
    plan, paths, _, families = population(args.plan)
    cache = Path(args.history_cache); cache_contract = load_json(cache/'contract.json')
    if (cache_contract['plan_sha256'] != sha256(args.plan)
            or cache_contract['historical_checkpoint'] != plan['files']['history_overlay']):
        raise ValueError('Different frozen history')
    compact_receipt = load_json(args.compact_receipt)
    candidates, models = {}, {}
    for seed in TRAIN_SEEDS:
        for arm in (0, 1):
            name = f'seed{seed}_ev{arm}'
            path = Path(args.training)/f'seed{seed}'/name/'last.pt'
            payload = fit_identity(path, paths['statistics'], compact_receipt)
            if payload['identity']['history_checkpoint']['sha256'] != plan['files']['history_overlay']['sha256']:
                raise ValueError('Different frozen TRAIN and TEST history weights')
            model = ConcatGRU(load_json(paths['statistics'])).to(args.device)
            model.load_state_dict(payload['model'], strict=True); model.eval().requires_grad_(False)
            candidates[name] = dict(path=str(path), sha256=sha256(path), identity=payload['identity'])
            models[name] = model
    for seed in TRAIN_SEEDS:
        if candidates[f'seed{seed}_ev0']['identity']['initialization_sha256'] != candidates[f'seed{seed}_ev1']['identity']['initialization_sha256']:
            raise ValueError('Paired GRU initializations differ')
    selected = [f for i, f in enumerate(families) if i%args.shards == args.shard]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    contract = dict(schema=SCHEMA+'-contract', plan_sha256=sha256(args.plan),
        checkpoints=candidates, history_cache_contract_sha256=sha256(cache/'contract.json'),
        family_ids=[f['family_id'] for f in selected], shard=args.shard, shards=args.shards,
        full_population=[f['family_id'] for f in families], sample_seeds=SEEDS, camera=0, worlds=[0,1],
        sampling_stream_training_seed=17, density_draws_per_mode=4, kernel_bandwidth=.2,
        sources={s: sha256(ROOT/s) for s in SOURCES}, selection_uses_test=False,
        future_reference_scope='historical assignment and scoring only', sampling_device=args.device)
    if (out/'contract.json').exists() and load_json(out/'contract.json') != contract:
        raise ValueError('Changed TEST execution contract')
    atomic_json(out/'contract.json', contract)
    report = dict(schema=SCHEMA, completed=False, split='test', checkpoint_step=STEPS, sample_seeds=SEEDS,
        contract=contract, methods={n:dict(rows=[]) for n in models}, completed_families=0,
        expected_families=len(selected))
    try:
        with torch.no_grad():
            for family in selected:
                boundary(); fid=family['family_id']; marker=out/'families'/(fid+'.json')
                if marker.exists():
                    saved=load_json(marker)
                    if saved['contract_sha256'] != sha256(out/'contract.json'):
                        raise ValueError('Changed family scoring contract')
                    per_method=saved['rows']
                else:
                    cp=cache/(fid+'.pt'); stamp=load_json(cp.with_suffix('.json'))
                    if stamp['sha256'] != sha256(cp) or stamp['contract_sha256'] != sha256(cache/'contract.json'):
                        raise ValueError('Changed frozen test posterior')
                    bundle=to_device(torch.load(cp,map_location='cpu',weights_only=False), args.device)
                    if bundle['family'] != family or bundle['receiver_history_identical'] is not True:
                        raise ValueError('Wrong world pair')
                    refs=torch.stack([cell['supervision']['state'][:,family['target_index'],:6] for cell in bundle['cells']])
                    per_method={n:[] for n in models}
                    for cell in bundle['cells']:
                        world,hist=cell['world'],cell['history']
                        placeholder=torch.empty(0,device=args.device)
                        branch=dict(inputs=dict(rgb=placeholder),supervision=cell['supervision'])
                        encoded=EncodedHistory(placeholder, None)
                        for name,model in models.items():
                            model.condition(hist)
                            row,history,assignment=state_eval.score_world(FixedHistory(hist,model),encoded,branch,family,world,17)
                            row['native_sha256']=cell['native_sha256']
                            slots=torch.where(assignment['ids']==family['target_index'])[0]
                            density=[]
                            if len(slots):
                                model.condition(history);slot=int(slots[0])
                                for seed in SEEDS:
                                    rng=torch.Generator(device=args.device).manual_seed(stream_seed(17,fid,seed,'density'))
                                    bank=persistent_target_rollouts(model,history['tokens'],history['shared'],history['initial'],
                                        history['presence_logits'],rng,slot,covariance=history['covariance'],draws=4)
                                    scores=candidate_log_density(bank['state'][...,slot,:6],bank['mode_logits'],
                                        refs[None],model.scale.std[:6],.2,bank['log_survival'])
                                    density.append(dict(sample_seed=seed,log_density=scores[0].cpu().tolist()))
                            row.update(density_samples=density,kernel_bandwidth=.2,
                                kernel_smoothed_log_density=np.mean([d['log_density'] for d in density],0).tolist() if density else None,
                                kernel_survival_scope='selected_focal_target_only')
                            per_method[name].append(row)
                    marker.parent.mkdir(parents=True,exist_ok=True)
                    atomic_json(marker,dict(contract_sha256=sha256(out/'contract.json'),rows=per_method))
                    del bundle,refs
                for name in models: report['methods'][name]['rows'].extend(per_method[name])
                report['completed_families']+=1
                atomic_json(out/'status.json',dict(state='running',completed_families=report['completed_families'],
                    total_families=len(selected),updated_unix=time.time()))
                print(json.dumps(dict(family=fid,completed=report['completed_families'])),flush=True)
        report['completed']=report['completed_families']==len(selected)
    except InterruptedError:
        pass
    finally:
        for method in report['methods'].values():method['summary']=state_eval.aggregate(method['rows'])
        report['wall_seconds']=time.monotonic()-start;atomic_json(out/'report.json',report)
        atomic_json(out/'completion.json',dict(completed=report['completed'],families=report['completed_families'],
            expected_families=len(selected),report_sha256=sha256(out/'report.json'),contract_sha256=sha256(out/'contract.json')))
        atomic_json(out/'status.json',dict(state='completed' if report['completed'] else 'paused',
            completed_families=report['completed_families'],total_families=len(selected),updated_unix=time.time()))
    return 0 if report['completed'] else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','score'):
        p=sub.add_parser(command)
        p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
        p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4)
        p.add_argument('--max-seconds',type=float,required=True)
        p.add_argument('--budget-ledger');p.add_argument('--job-id')
        if command=='score':
            for name in ('training','history-cache','compact-receipt'):p.add_argument('--'+name,required=True)
            p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=4)
    args=parser.parse_args()
    return prepare(args) if args.command=='prepare' else score(args)


if __name__=='__main__':raise SystemExit(main())

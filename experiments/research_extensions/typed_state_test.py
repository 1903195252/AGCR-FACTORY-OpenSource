"""Locked full-160-family TEST scoring of completed typed history/future overlays.

No training, renderer, threshold selection or population selection occurs here.
The original VAL entrypoint and all numerical state scoring functions are reused
unchanged. A completed TEST report is not a renderer-readiness admission.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

SCHEMA = 'peerdream-typed-state-test-1'
SEEDS = [101, 211, 307, 401, 503, 601, 701, 809]
ARMS = ('coarse_full_control', 'coarse_full_paired_geometry',
        'native_full_control', 'native_full_paired_geometry')
ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    'experiments/research_extensions/typed_state_test.py',
    'experiments/research_extensions/region_probe.py',
    'experiments/research_extensions/train_history_future.py',
    'experiments/research_extensions/train_history_identity.py',
    'experiments/research_extensions/history_identity.py',
    'experiments/research_extensions/history_surface_refiner.py',
    'experiments/research_extensions/train_history_surface.py',
    'experiments/research_extensions/state_eval_fixed.py',
    'experiments/base48/state_eval.py', 'experiments/base48/train.py',
    'experiments/base48/model.py', 'experiments/base48/presence_refit.py',
    'experiments/formal_evaluation/state_likelihood.py',
    'experiments/formal_evaluation/metrics.py',
    'experiments/paired_interventions/data.py', 'experiments/paired_interventions/bridge.py',
    'experiments/paired_interventions/rollout.py', 'experiments/paired_interventions/objectives.py',
    'experiments/paired_interventions/sample.py', 'experiments/paired_interventions/world.py',
    'experiments/paper_experiments/agcr_exp/stage2_system.py',
    'experiments/paper_experiments/agcr_exp/future_prediction.py',
    'experiments/paper_experiments/agcr_exp/temporal_state.py',
    'experiments/paper_experiments/agcr_exp/observation.py',
    'experiments/paper_experiments/agcr_exp/geometry.py')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%d.tmp' % os.getpid())
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def source_contract():
    return {name: sha(ROOT / name) for name in SOURCES}


def validate_lock(lock):
    if (lock.get('schema') != SCHEMA + '-lock' or lock.get('split') != 'test'
            or lock.get('sample_seeds') != SEEDS or lock.get('sampling_stream_training_seed') != 17
            or lock.get('camera') != 0 or lock.get('worlds') != [0, 1]
            or lock.get('selection_uses_test') is not False
            or lock.get('density_draws_per_seed') != 4 or lock.get('density_bandwidth') != .2):
        raise ValueError('Changed fixed TEST protocol or information boundary')
    families = lock.get('family_ids', [])
    if len(families) != 160 or len(set(families)) != 160:
        raise ValueError('Exactly the locked 160 distinct TEST families are required')
    if any(not isinstance(f, str) or not f or Path(f).name != f for f in families):
        raise ValueError('Invalid family identity')
    if set(lock.get('checkpoints', {})) != set(ARMS):
        raise ValueError('Freeze all four predetermined overlays before TEST scoring')
    for arm, item in lock['checkpoints'].items():
        if item.get('steps') != 2048 or item.get('arm') != arm:
            raise ValueError('Only the fixed completed 2048-step candidates are supported')
        for key in ('sha256', 'history_sha256'):
            value = item.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError('Checkpoint hash must be frozen')
    return lock


def freeze(requirements_path, original_plan_path, output):
    req, plan = read(requirements_path), read(original_plan_path)
    lock = dict(schema=SCHEMA+'-lock', split='test', selection_uses_test=False,
        family_ids=req['source_test_index']['families'], sample_seeds=SEEDS,
        camera=0, worlds=[0, 1], sampling_stream_training_seed=17,
        density_draws_per_seed=4, density_bandwidth=.2,
        paths=plan['paths'], paths_sha256=plan['paths_sha256'],
        index=req['source_test_index']['path'], index_sha256=req['source_test_index']['sha256'],
        training_index_sha256=plan['index_sha256'], parent=plan['parent'], parent_sha256=plan['parent_sha256'],
        training_panel=plan['panel'], training_panel_sha256=plan['panel_sha256'],
        checkpoints={}, sources=source_contract(),
        budget=dict(ledger_path=plan['ledger'],gpu_hour_cap=plan['budget_authorization']['gpu_hour_cap'],
                    deadline_unix=plan['budget_authorization']['deadline_unix']),
        origin=dict(requirements_sha256=sha(requirements_path), completed_val_plan_sha256=sha(original_plan_path)),
        attribution='Joint evidence/motion-projection/reappearance-geometry loss contrast, not evidence loss alone',
        budget_policy='Run only as a single child of the existing budget_job ledger; no new authorization')
    for arm in ARMS:
        history = arm.rsplit('_control', 1)[0] if arm.endswith('_control') else arm.rsplit('_paired_geometry', 1)[0]
        source = req['checkpoint_locks'][arm]
        lock['checkpoints'][arm] = dict(arm=arm, path=source['path'], sha256=source['sha256'], steps=2048,
            history_path=plan['histories'][history]['path'], history_sha256=plan['histories'][history]['sha256'])
    validate_lock(lock)
    output = Path(output)
    if output.exists() and read(output) != lock:
        raise ValueError('Do not replace an existing evaluation lock')
    atomic(output, lock)
    return lock


def validate_population(lock, registered_path, source_path, ancestry):
    """Same registered-index/original-source rule as base48.state_eval."""
    if sha(registered_path) != lock['training_index_sha256'] or sha(source_path) != lock['index_sha256']:
        raise ValueError('Index differs from the frozen training/original TEST identity')
    if sha(source_path) != sha(registered_path):
        if (not ancestry or ancestry.get('source_index_sha256') != sha(source_path)
                or ancestry.get('train_val_identical') is not True):
            raise ValueError('Requested TEST index lacks unchanged TRAIN/VAL ancestry')
    source = read(source_path)
    if (source.get('schema') != 'agcr-intervention-index-1' or source.get('audit_passed') is not True
            or source.get('development_only')):
        raise ValueError('Formal TEST requires the audited native original index')
    indexed = {f['family_id']: f for f in source['families']}
    if len(indexed) != len(source['families']):
        raise ValueError('Duplicate family ID')
    tests = [f for f in source['families'] if f['split'] == 'test']
    if [f['family_id'] for f in tests] != lock['family_ids'] or len(tests) != 160:
        raise ValueError('TEST population/order differs; a 20-family video subset is not admissible')
    root = Path(source['root']).resolve()
    for family in tests:
        records = family['records']
        if len(records) != 4 or {(r['world'], r['camera']) for r in records} != {(w,c) for w in (0,1) for c in (0,1)}:
            raise ValueError('Incomplete four-cell source family')
        for row in records:
            path = (root / row['path']).resolve()
            if root not in path.parents:
                raise ValueError('Native branch escapes locked root')
    return source, tests


def validate_metadata(lock, arm, metadata, lineage):
    item = lock['checkpoints'][arm]
    config = metadata['identity']['configuration']; data = metadata['identity']['data']
    objective = 'control' if arm.endswith('_control') else 'paired_geometry'
    history = arm[:-len('_' + objective)]
    if (metadata.get('checkpoint_sha256') != item['sha256'] or metadata.get('step') != 2048
            or metadata.get('completed') is not True or metadata.get('arm') != objective
            or metadata['history_checkpoint']['sha256'] != item['history_sha256']
            or metadata['historical'].get('arm') != history
            or config.get('history_frozen') is not True or config.get('future_truth_conditions') is not False
            or data.get('split') != 'train' or data.get('test_used') is not False
            or data.get('selection_uses_validation_scores') is not False
            or data.get('index_sha256') != lock['training_index_sha256']
            or data.get('panel_sha256') != lock['training_panel_sha256']
            or lineage.get('sha256') != lock['parent_sha256']
            or lineage.get('candidate_identity') is not None
            or lineage['identity'].get('seed') != 17
            or lineage['identity'].get('family_index_sha256') != lock['training_index_sha256']):
        raise ValueError('Candidate training/parent identity or future-truth boundary differs')
    if not data.get('branches') or any(r.get('split') != 'train' or r.get('camera') != 0 for r in data['branches']):
        raise ValueError('Future overlay contains non-TRAIN exposure')


def validate_input_boundary(inputs):
    expected = {'rgb','depth_mm','depth_valid','K','camera_to_world','planned_camera_to_world','time_s','future_time_s'}
    if set(inputs) != expected or tuple(inputs['rgb'].shape[:2]) != (6,2):
        raise ValueError('Only six historical paired RGB-D observations and known plans may enter prediction')
    if (tuple(inputs['depth_mm'].shape[:2]) != (6,2) or tuple(inputs['depth_valid'].shape[:2]) != (6,2)
            or tuple(inputs['camera_to_world'].shape[:2]) != (6,2)):
        raise ValueError('Future observation accidentally entered a historical input')
    if len(inputs['planned_camera_to_world']) != 16 or len(inputs['future_time_s']) != 16:
        raise ValueError('The locked future horizon is 16 frames')
    if bool((inputs['time_s'] > 0).any()) or bool((inputs['future_time_s'] <= 0).any()):
        raise ValueError('Historical/future time boundary changed')


def rows_sha(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_family_rows(rows, family_id, target_index=None):
    if len(rows) != 2 or [r.get('world') for r in rows] != [0,1]:
        raise ValueError('A family commit requires both complete worlds')
    for row in rows:
        if row.get('family_id') != family_id:
            raise ValueError('Wrong family in state score cache')
        samples = row.get('state_samples', [])
        if ([s.get('sample_seed') for s in samples] != SEEDS
                or [s.get('sample_index') for s in samples] != list(range(8))):
            raise ValueError('All eight fixed draws must be present in order')
        for sample in samples:
            metric = sample['metrics']
            for key in ('ADE_m','FDE_m','state_missing_rate','FDE_coverage'):
                value = metric[key]
                if value is not None and not math.isfinite(value):
                    raise ValueError('Non-finite state metric')
            for key in ('reference_count','matched_count'):
                if type(metric[key]) is not int or metric[key] < 0:
                    raise ValueError('Invalid state metric denominator')
            if metric['matched_count'] > metric['reference_count']:
                raise ValueError('Matched states exceed reference denominator')
        density = row.get('density_samples')
        if density is None or (density and [s.get('sample_seed') for s in density] != SEEDS):
            raise ValueError('Incomplete original density scoring protocol')
        if target_index is not None:
            expected = str(target_index) in row['selected_slot_by_reference']
            if bool(density) != expected:
                raise ValueError('Focal density availability differs from historical assignment')
    return rows


def preflight(lock_path, arm):
    from experiments.base48.train import audit_index_ancestry
    from .region_probe import load_state_overlay
    from .train_history_future import load_future_checkpoint
    lock = validate_lock(read(lock_path))
    if lock['sources'] != source_contract():
        raise ValueError('Frozen evaluation sources changed')
    paths = read(lock['paths'])
    if sha(lock['paths']) != lock['paths_sha256'] or sha(lock['parent']) != lock['parent_sha256']:
        raise ValueError('Locked paths or parent checkpoint changed')
    item = lock['checkpoints'][arm]
    for path, expected in ((item['path'],item['sha256']), (item['history_path'],item['history_sha256']),
                           (lock['training_panel'],lock['training_panel_sha256'])):
        if sha(path) != expected:
            raise ValueError('Frozen overlay/history/panel changed: ' + path)
    ancestry = audit_index_ancestry(paths['families'])
    source, families = validate_population(lock, paths['families'], lock['index'], ancestry)
    payload, lineage = load_state_overlay(lock['parent']); del payload
    metadata = load_future_checkpoint(item['path'], lock['parent'], lock['training_index_sha256'], device='cpu')
    validate_metadata(lock, arm, metadata, lineage)
    if sha(paths['statistics']) != metadata['identity']['statistics_sha256']:
        raise ValueError('Training normalization statistics changed')
    contract = dict(schema=SCHEMA+'-contract', lock_sha256=sha(lock_path), arm=arm, split='test',
        families=lock['family_ids'], sample_seeds=SEEDS, index_sha256=sha(lock['index']), ancestry=ancestry,
        checkpoint_sha256=item['sha256'], history_checkpoint_sha256=item['history_sha256'],
        parent_sha256=lock['parent_sha256'], paths_sha256=lock['paths_sha256'],
        sources=lock['sources'], metadata=metadata, original_parent_lineage=lineage,
        sampling_stream_training_seed=17, selection_uses_test=False, future_reference_scope='scoring only',
        input_policy='paired', reference_population='same paired-history-eligible population as original Base state evaluation',
        state_scorer='experiments.base48.state_eval.score_world/aggregate unchanged',
        density_scorer='original persistent_target_rollouts and candidate_log_density, 4 paths/mode/seed, bandwidth0.2',
        training_amendment='completed typed TRAIN-only history/future overlays; no renderer readiness or video result claim')
    return lock, paths, source, families, contract


def budget_guard(ledger_path, job_id, expected):
    from .budget_job import start_ticks
    ledger = read(ledger_path)
    if (Path(ledger_path).resolve()!=Path(expected['ledger_path']).resolve()
            or ledger['gpu_hour_cap']!=expected['gpu_hour_cap']
            or ledger['deadline_unix']!=expected['deadline_unix']):
        raise ValueError('Original allocation cap, ledger and deadline must remain unchanged')
    jobs = [r for r in ledger['jobs'] if r['id'] == job_id]
    if len(jobs) != 1:
        raise ValueError('An existing budget_job reservation is required')
    row = jobs[0]
    if (row['state'] not in ('reserved','running') or row.get('gpu_count') != 1
            or row.get('wrapper_pid') != os.getppid()
            or row.get('wrapper_start_ticks') != start_ticks(os.getppid())
            or os.environ.get('CUDA_VISIBLE_DEVICES') != row.get('cuda_visible_devices')
            or time.time() >= ledger['deadline_unix']):
        raise ValueError('Run must be the live single-GPU child of its original-budget wrapper')
    return row


def run(args):
    import numpy as np
    import torch
    from experiments.base48.state_eval import KEEP_LABELS, score_world, aggregate
    from experiments.paired_interventions.data import load_branch, to_device
    from experiments.paired_interventions.world import load_upstream
    from experiments.paper_experiments.agcr_exp.stage2_system import FrozenImageEncoders
    from experiments.formal_evaluation.state_likelihood import persistent_target_rollouts
    from experiments.paired_interventions.objectives import candidate_log_density
    from experiments.paired_interventions.sample import stream_seed
    from .region_probe import load_state_overlay
    from .train_history_future import load_future_checkpoint
    frozen = validate_lock(read(args.lock))
    budget = budget_guard(args.budget_ledger, args.job_id, frozen['budget'])
    if args.max_seconds > budget['maximum_wall_seconds']-30:
        raise ValueError('Inner deadline must leave wrapper cleanup time')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    lease = (out/'.lock').open('a+'); fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start = time.monotonic(); stopped = [False]
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def boundary():
        if stopped[0] or time.monotonic()-start >= args.max_seconds-20 or (args.stop_file and Path(args.stop_file).exists()):
            raise InterruptedError('Bounded state TEST evaluation paused at family boundary')
    admitted = False
    report = dict(schema=SCHEMA, complete=False, split='test', development_only=False,
        selected_checkpoint_evaluation=True, generation_evaluated=False, ready_for_renderer=False,
        rows=[], sample_seeds=SEEDS)
    try:
        lock, paths, source, families, contract = preflight(args.lock,args.arm)
        contract_path = out/'contract.json'
        if contract_path.exists() and read(contract_path) != contract:
            raise ValueError('Existing TEST output belongs to another frozen contract')
        atomic(contract_path,contract); contract_sha=sha(contract_path); admitted=True
        report['contract']=contract
        report['checkpoint_step']=2048
        system = encoders = None
        cache=out/'families';cache.mkdir(exist_ok=True)
        torch.set_num_threads(4)
        with torch.no_grad():
            for family in families:
                boundary();fid=family['family_id'];record_hashes={}
                # Preserve original state_eval's verification of all four native cells.
                for record in family['records']:
                    boundary();path=Path(source['root'])/record['path']
                    if sha(path)!=record['sha256']:raise ValueError('Audited TEST branch changed: '+str(path))
                    record_hashes[record['path']]=record['sha256']
                marker=cache/(fid+'.json')
                if marker.exists():
                    saved=read(marker)
                    if (saved.get('contract_sha256')!=contract_sha or saved.get('source_records')!=record_hashes
                            or saved.get('rows_sha256')!=rows_sha(saved['rows'])):
                        raise ValueError('Changed TEST family cache provenance')
                    rows=validate_family_rows(saved['rows'],fid,family['target_index'])
                else:
                    if system is None:
                        boundary();torch.cuda.set_device('cuda:0')
                        system,_=load_upstream(paths['upstream'],paths['statistics'],'cuda:0')
                        payload,_=load_state_overlay(lock['parent'])
                        system.load_state_dict({k[len('upstream.'):]:v for k,v in payload['model'].items() if k.startswith('upstream.')},strict=True)
                        del payload
                        load_future_checkpoint(lock['checkpoints'][args.arm]['path'],lock['parent'],lock['training_index_sha256'],upstream=system,device='cuda:0')
                        system.eval().requires_grad_(False)
                        encoders=FrozenImageEncoders(paths['dino_code'],paths['dino_weights'],paths['wan']+'/vae','cuda:0')
                    rows=[];histories=[];references=[]
                    for world in (0,1):
                        boundary();record=next(r for r in family['records'] if (r['world'],r['camera'])==(world,0))
                        raw=load_branch(Path(source['root'])/record['path']);validate_input_boundary(raw['inputs'])
                        branch=to_device(dict(inputs=raw['inputs'],supervision={k:raw['supervision'][k] for k in KEEP_LABELS}),'cuda:0');del raw
                        row,history,assignment=score_world(system,encoders,branch,family,world,17)
                        row['native_sha256']=record['sha256'];rows.append(row);histories.append((history,assignment))
                        references.append(branch['supervision']['state'][:,family['target_index'],:6]);del branch
                    references=torch.stack(references)
                    for world,(history,assignment) in enumerate(histories):
                        boundary();selected=torch.where(assignment['ids']==family['target_index'])[0];density=[]
                        if len(selected):
                            k=int(selected[0])
                            for seed in SEEDS:
                                generator=torch.Generator(device='cuda:0').manual_seed(stream_seed(17,fid,seed,'density'))
                                draws=persistent_target_rollouts(system.dynamics,history['tokens'],history['shared'],history['initial'],history['presence_logits'],generator,k,covariance=history['covariance'],draws=4)
                                score=candidate_log_density(draws['state'][...,k,:6],draws['mode_logits'],references[None],system.dynamics.scale.std[:6],.2,draws['log_survival'])
                                density.append(dict(sample_seed=seed,log_density=score[0].cpu().tolist()))
                        rows[world].update(density_samples=density,kernel_bandwidth=.2,
                            kernel_smoothed_log_density=np.mean([s['log_density'] for s in density],0).tolist() if density else None,
                            kernel_survival_scope='selected_focal_target_only')
                    validate_family_rows(rows,fid,family['target_index'])
                    atomic(marker,dict(contract_sha256=contract_sha,source_records=record_hashes,rows_sha256=rows_sha(rows),rows=rows))
                    del histories,history,assignment,references
                report['rows'].extend(rows);report['summary']=aggregate(report['rows'])
                # Preserve the scorer; strata only partition complete family rows.
                report['strata']={cohort+'_r'+str(receiver):aggregate([r for r in report['rows'] if (r['cohort'],r['receiver'])==(cohort,receiver)])
                    for cohort in ('ambiguous','evidence') for receiver in (0,1)}
                atomic(out/'report.json',report)
                atomic(out/'status.json',dict(state='running',completed_families=len(report['rows'])//2,total_families=160,
                    arm=args.arm,split='test',updated_unix=time.time()))
            report['complete']=len(report['rows'])==320
    except InterruptedError as error:
        report['pause_reason']=str(error)
    except BaseException as error:
        report['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        if not admitted:
            atomic(out/'admission_failure.json',dict(error=report.get('error',report.get('pause_reason')),arm=args.arm,updated_unix=time.time()))
        else:
            finish_report(out,report,start,args.arm)
    return 0 if report['complete'] else 2


def finish_report(out,report,start,arm):
    report['wall_seconds']=time.monotonic()-start
    atomic(out/'report.json',report)
    atomic(out/'completion.json',dict(completed=report['complete'],split='test',families=len(report['rows'])//2,
        expected_families=160,sample_seeds=SEEDS,report_sha256=sha(out/'report.json'),
        contract_sha256=sha(out/'contract.json') if (out/'contract.json').exists() else None))
    atomic(out/'status.json',dict(state='completed' if report['complete'] else 'failed' if 'error' in report else 'paused',
        completed_families=len(report['rows'])//2,total_families=160,arm=arm,split='test',updated_unix=time.time()))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze');f.add_argument('--requirements',required=True);f.add_argument('--original-plan',required=True);f.add_argument('--output',required=True)
    for name in ('preflight','run'):
        q=sub.add_parser(name);q.add_argument('--lock',required=True);q.add_argument('--arm',choices=ARMS,required=True);q.add_argument('--output',required=True)
        if name=='run':
            q.add_argument('--budget-ledger',required=True);q.add_argument('--job-id',required=True)
            q.add_argument('--max-seconds',type=float,default=2640);q.add_argument('--stop-file')
    a=p.parse_args(argv)
    if a.action=='freeze':return freeze(a.requirements,a.original_plan,a.output)
    if a.action=='preflight':
        *_,contract=preflight(a.lock,a.arm);atomic(a.output,dict(completed=True,cpu_only=True,contract=contract));return
    if not math.isfinite(a.max_seconds) or not 0<a.max_seconds<=3600:p.error('Finite positive inner wall bound <=3600s required')
    return run(a)


if __name__=='__main__':
    code=main();raise SystemExit(code if isinstance(code,int) else 0)

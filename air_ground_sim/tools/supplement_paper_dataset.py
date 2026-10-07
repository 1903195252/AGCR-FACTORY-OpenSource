#!/usr/bin/env python3
"""Repair event groups and acquire fixed-window quotas in a versioned release.

Existing episodes are linked read-only; camera-group splits never change.
Only reference labels determine clip eligibility, never a model's outputs.
"""
import argparse
from collections import Counter, defaultdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')
from paper_dataset.common import atomic_json, digest, file_hash, pipeline_hash, ScriptValidationError
from paper_dataset.planning import plans as make_plans

REVISION = 'event-camera-v6'


def read(path):
    return json.loads(Path(path).read_text())


def group_key(p):
    return p['layout'], p['path']


def supplemental_wave(config, wave, pool, first_index):
    """One complete 20-layout × 8-event block, with paired realizations.

    Path IDs do not overlap the original 0..49 paths or the other pool.
    Class/occluder/motion combinations rotate across layouts and waves.
    """
    cfg = dict(config, paths_per_layout=8, augmentation_paths_per_train_layout=0,
               val_layouts=0, test_layouts=0)
    templates = make_plans(cfg)
    factors = make_plans(dict(cfg, train_layouts=1,
                             paths_per_layout=(wave+1)*config['train_layouts']*8))
    result = []
    for p in templates:
        j = wave * config['train_layouts'] * 8 + p['layout'] * 8 + p['path']
        # Reuse the original factor-selection policy with a rotated index.
        template = factors[2*j + p['repeat']]
        path = (1000 if pool == 'training_supplement' else 5000) + wave*8 + p['path']
        q = dict(template, index=first_index+len(result), layout=p['layout'], path=path,
                 name='l%02d_p%03d_r%d' % (p['layout'], path, p['repeat']), pool=pool,
                 split='train', geometry_revision=REVISION, supplement_wave=wave,
                 camera_seed=[config['seed'],p['layout'],path,61],
                 motion_seed=[config['seed'],p['layout'],path,p['repeat'],97])
        result.append(q)
    return result


def prepare(base, root, target):
    base=Path(base).resolve();root=Path(root).resolve()
    if root == base:raise ValueError('Supplement release must not overwrite the original')
    source=read(base/'manifest.json');original=read(base/'plans.json')
    missing={group_key(p) for p in original if not (base/'episodes'/p['name']/'complete.json').exists()}
    revision=dict(base_identity=source['identity'],base_plans_sha256=file_hash(base/'plans.json'),
                  pipeline_sha256=pipeline_hash(),runner_sha256=file_hash(__file__),
                  scheduler_sha256={name:file_hash(Path(__file__).with_name(name))
                                    for name in ('capture_task_queue.py','capture_scheduler.py')},
                  geometry_revision=REVISION,target_clips=target)
    identity=digest(revision)
    if (root/'manifest.json').exists():
        if read(root/'manifest.json')['identity'] != identity:
            raise ValueError('Source/configuration changed; use a new release, do not mix revisions')
        return
    root.mkdir(parents=True,exist_ok=True);(root/'episodes').mkdir(exist_ok=True)
    for name in ('assets','layouts','layouts.json'):
        if not (root/name).exists():(root/name).symlink_to(base/name,target_is_directory=(base/name).is_dir())
    inherited={};inherited_hashes={};updated=[]
    for p in original:
        p=dict(p)
        if group_key(p) in missing:p['geometry_revision']=REVISION
        else:
            old=base/'episodes'/p['name'];meta=read(old/'complete.json')
            inherited[p['name']]=meta['identity'];inherited_hashes[p['name']]=file_hash(old/'complete.json')
            (root/'episodes'/p['name']).symlink_to(old,target_is_directory=True)
        updated.append(p)
    manifest=dict(source,identity=identity,created_unix=time.time(),pipeline_sha256=pipeline_hash(),
                  revision=revision,base_root=str(base),inherited_episode_identities=inherited,
                  inherited_marker_sha256=inherited_hashes,base_episodes=len(original),
                  repair_groups=[list(k) for k in sorted(missing)],
                  repair_episodes=[p['name'] for p in updated if group_key(p) in missing],
                  quota_target=target,supplement_policy='complete_balanced_blocks_until_both_pools_reach_target')
    atomic_json(root/'config.json',source['config']);atomic_json(root/'plans.json',updated)
    atomic_json(root/'manifest.json',manifest)


def quick_quota(root, assignments):
    """The exact existing reference-only history coverage rule at four starts."""
    import h5py
    import numpy as np
    from agcr_exp.data import plan_role
    from agcr_exp.protocol import DEFAULT
    counts=Counter();events=defaultdict(Counter);layouts=defaultdict(Counter)
    for marker in sorted((root/'episodes').glob('*/complete.json')):
        meta=read(marker);role=plan_role(meta,assignments)
        if role not in ('train','augmentation_train'):continue
        with h5py.File(marker.parent/'frames.h5','r') as h:
            b=(h['visible_fraction'][:]>=DEFAULT['v_min']) & (h['full_area'][:]>=DEFAULT['area_min'])
            visible=h['visible_area'][:]>0
            for start in DEFAULT['evaluation_starts']:
                eligible=b[start:start+6].any(axis=(0,1))
                future=visible[start+6:start+22].any(axis=(0,1))
                if not (future & ~eligible).any():
                    counts[role]+=1;events[role][meta['event']]+=1;layouts[role][meta['layout']]+=1
    return dict(eligible_clips={r:counts[r] for r in ('train','augmentation_train')},
                by_event={r:dict(v) for r,v in events.items()},
                by_layout={r:dict(v) for r,v in layouts.items()})


def paired_specs(scene, plans):
    """Select one physical placement accepted by both motion realizations."""
    failures=[]
    for attempt in range(3):
        try:
            specs=[scene.spec(p,attempts=[attempt]) for p in plans]
            if any(s['physical_attempt'] != attempt for s in specs):
                raise ValueError('Preflight returned a different fixture attempt')
            if any(s['camera'] != specs[0]['camera'] or s['fixtures'] != specs[0]['fixtures'] for s in specs):
                raise ValueError('Paired camera/fixture mismatch')
            return specs
        except ScriptValidationError as exc:
            failures.append(str(exc))
    raise ScriptValidationError('No common accepted physical placement: '+json.dumps(failures))


def capture_group(scene, plans, config, root, manifest):
    """Use the same preflight/capture path, without retaining a bound-method cycle."""
    from paper_dataset.capture import collect_episode
    specs=paired_specs(scene,plans)
    had_override='spec' in scene.__dict__
    override=scene.__dict__.get('spec')
    try:
        for p,s in zip(plans,specs):
            def cached_spec(plan,s=s):
                scene.configure_fixtures(s)
                return s
            scene.spec=cached_spec
            result=collect_episode(scene,p,config,root,manifest['identity'])
            print(json.dumps(dict(status='complete',name=p['name'],seconds=result['seconds'])),flush=True)
    finally:
        if had_override:scene.spec=override
        else:del scene.spec


def worker(root, taskfile, worker_id, workers):
    from paper_dataset.capture import SceneCapture, collect_episode
    root=Path(root);manifest=read(root/'manifest.json');config=manifest['config']
    tasks=read(taskfile);groups=defaultdict(list)
    for p in tasks:groups[group_key(p)].append(p)
    chosen=[(key,ps) for i,(key,ps) in enumerate(sorted(groups.items())) if i%workers==worker_id]
    scene=None;layout=None;failures=[]
    try:
        for key,ps in chosen:
            if (root/'STOP').exists():raise RuntimeError('STOP requested')
            if all((root/'episodes'/p['name']/'complete.json').exists() for p in ps):continue
            if layout!=key[0]:
                if scene:scene.close()
                scene=SceneCapture(root/'layouts'/('%02d.xml'%key[0]),config);layout=key[0]
            try:
                # Preflight both realizations, then require a common fixture
                # attempt. Their historical geometry must be identical.
                specs=paired_specs(scene,ps)
                original_spec=scene.spec
                try:
                    for p,s in zip(ps,specs):
                        # Configure the exact preflighted geometry before capture.
                        def cached_spec(plan, s=s):
                            scene.configure_fixtures(s)
                            return s
                        scene.spec=cached_spec
                        r=collect_episode(scene,p,config,root,manifest['identity'])
                        print(json.dumps(dict(status='complete',name=p['name'],seconds=r['seconds'])),flush=True)
                finally:scene.spec=original_spec
            except Exception as exc:
                record=dict(group=list(key),plans=ps,error=str(exc),time=time.time())
                atomic_json(root/'rejected'/('l%02d_p%03d.json'%key),record)
                failures.append(record);print(json.dumps(dict(status='rejected',**record)),flush=True)
    finally:
        if scene:scene.close()
    if failures:raise SystemExit(2)


def run_tasks(root, tasks, label, workers):
    taskfile=root/'queues'/(label+'.json');atomic_json(taskfile,tasks)
    (root/'logs').mkdir(exist_ok=True);running=[]
    for i in range(min(workers,len({group_key(p) for p in tasks}))):
        log=(root/'logs'/('%s_%02d.log'%(label,i))).open('a')
        proc=subprocess.Popen([sys.executable,__file__,'--out',str(root),'--taskfile',str(taskfile),
                               '--worker',str(i),'--workers',str(min(workers,len({group_key(p) for p in tasks})))],stdout=log,stderr=subprocess.STDOUT)
        running.append((proc,log))
    codes=[]
    for proc,log in running:codes.append(proc.wait());log.close()
    missing=[p['name'] for p in tasks if not (root/'episodes'/p['name']/'complete.json').exists()]
    if any(codes) or missing:raise RuntimeError('%s incomplete: %s'%(label,missing))


def orchestrate_static(args):
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'supplement.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(args.base,root,args.target)
        from agcr_exp.data import assignments,build_index
        from agcr_exp.selection import quota_report
        fixed_roles=assignments(args.base)
        if assignments(root)!=fixed_roles:raise ValueError('Existing reader/ID-test split changed')
        manifest=read(root/'manifest.json');plans=read(root/'plans.json')
        progress=dict(status='repairing',pid=os.getpid(),started_unix=time.time(),target=args.target,
                      repair_episodes=len(manifest['repair_episodes']),dataset_root=str(root))
        atomic_json(root/'supplement_status.json',progress)
        try:
            repair=[p for p in plans if p['name'] in manifest['repair_episodes']]
            run_tasks(root,repair,'repair',args.workers)
            progress.update(status='supplementing',repair_complete=True)
            # Resume any already planned wave before considering additional plans.
            for wave in range(args.max_waves):
                current=[p for p in plans if p.get('supplement_wave')==wave]
                if current:run_tasks(root,current,'wave_%02d'%wave,args.workers)
                quota=quick_quota(root,fixed_roles)
                progress.update(quota,completed_episodes=len(list((root/'episodes').glob('*/complete.json'))),
                                planned_episodes=len(plans),updated_unix=time.time())
                atomic_json(root/'supplement_status.json',progress)
                if all(n>=args.target for n in quota['eligible_clips'].values()):break
                if current:continue
                additions=[]
                for role,pool in [('train','training_supplement'),('augmentation_train','augmentation')]:
                    if quota['eligible_clips'][role]<args.target:
                        additions.extend(supplemental_wave(manifest['config'],wave,pool,len(plans)+len(additions)))
                if shutil.disk_usage(root).free < (manifest['config']['disk_reserve_gib']+len(additions)*.06)*1024**3:
                    raise RuntimeError('Insufficient disk reserve for the next full balanced block')
                plans.extend(additions);atomic_json(root/'plans.json',plans)
                manifest['planned_episodes']=len(plans);manifest['plans_sha256']=file_hash(root/'plans.json')
                atomic_json(root/'manifest.json',manifest)
                progress.update(planned_episodes=len(plans),wave=wave,updated_unix=time.time())
                atomic_json(root/'supplement_status.json',progress)
                run_tasks(root,additions,'wave_%02d'%wave,args.workers)
            quota=quick_quota(root,fixed_roles)
            if not all(n>=args.target for n in quota['eligible_clips'].values()):
                raise RuntimeError('Quota not reached within the declared maximum waves')
            progress.update(quota,status='validating',updated_unix=time.time())
            atomic_json(root/'supplement_status.json',progress)
            with (root/'logs'/'validation.log').open('w') as log:
                subprocess.run([sys.executable,str(Path(__file__).with_name('validate_paper_dataset.py')),
                                '--root',str(root),'--export','--require-complete'],stdout=log,stderr=subprocess.STDOUT,check=True)
            index=build_index(root,root/'experiment_index.json')
            report=quota_report(index,args.target);atomic_json(root/'downstream_quota.json',report)
            if not report['reference_quota_ready']:raise RuntimeError('Formal index quota differs from capture quota')
            if assignments(root)!=fixed_roles:raise ValueError('Original split changed')
            for name,sha in manifest['inherited_marker_sha256'].items():
                if file_hash(root/'episodes'/name/'complete.json')!=sha:raise ValueError('Inherited marker changed: '+name)
            progress.update(status='complete',completed_episodes=len(plans),planned_episodes=len(plans),
                            split_sha256=index['split_sha256'],completed_unix=time.time())
            atomic_json(root/'supplement_status.json',progress)
        except BaseException as exc:
            progress.update(status='failed',error=str(exc),updated_unix=time.time())
            atomic_json(root/'supplement_status.json',progress)
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base');p.add_argument('--out',required=True)
    p.add_argument('--target',type=int,default=4000);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--max-waves',type=int,default=12)
    p.add_argument('--taskfile');p.add_argument('--worker',type=int)
    p.add_argument('--queue-file');p.add_argument('--run-id');p.add_argument('--controller-pid',type=int)
    a=p.parse_args()
    if a.workers<1 or a.target<1:raise ValueError('Invalid workers/target')
    if a.queue_file:
        from capture_scheduler import dynamic_worker
        dynamic_worker(a)
    elif a.taskfile:worker(a.out,a.taskfile,a.worker,a.workers)
    elif not a.base:p.error('--base is required for the orchestrator')
    else:
        from capture_scheduler import orchestrate_dynamic
        orchestrate_dynamic(a)


if __name__=='__main__':main()

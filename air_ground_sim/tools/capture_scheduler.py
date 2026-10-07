"""Eight persistent workers pull complete camera groups from a shared queue."""
import fcntl
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from capture_task_queue import TaskQueue, live_worker_pids
from paper_dataset.common import atomic_json, file_hash, CaptureQualityError
from supplement_paper_dataset import read, prepare, group_key, supplemental_wave, quick_quota, capture_group


def worker_state(root, slot, run_id, status, **fields):
    atomic_json(root/'scheduler'/'workers'/('%02d.json'%slot),
                dict(worker=slot,pid=os.getpid(),run_id=run_id,status=status,updated_unix=time.time(),**fields))


def dynamic_worker(args):
    from paper_dataset.capture import SceneCapture
    root=Path(args.out);manifest=read(root/'manifest.json');config=manifest['config']
    queue=TaskQueue(root,args.queue_file,args.run_id)
    scene=None;layout=None;groups_done=0
    try:
        while os.getppid()==args.controller_pid:
            if (root/'STOP').exists() or queue.document()['sealed']:break
            claim=queue.take(layout)
            if claim is None:
                worker_state(root,args.worker,args.run_id,'waiting',groups_done=groups_done)
                time.sleep(1)
                continue
            with claim:
                worker_state(root,args.worker,args.run_id,'capturing',group=list(claim.key),
                             names=[p['name'] for p in claim.plans],groups_done=groups_done)
                print(json.dumps(dict(status='claimed',group=list(claim.key),worker=args.worker,
                                      names=[p['name'] for p in claim.plans],time=time.time())),flush=True)
                try:
                    if layout!=claim.key[0]:
                        if scene:scene.close()
                        scene=None
                        gc.collect()
                        scene=SceneCapture(root/'layouts'/('%02d.xml'%claim.key[0]),config)
                        layout=claim.key[0]
                    capture_group(scene,claim.plans,config,root,manifest)
                    if not queue.finished(claim.plans):raise RuntimeError('Pair missing completion markers')
                    groups_done+=1
                    if groups_done%8==0:gc.collect()
                except Exception as exc:
                    scope='scene' if isinstance(exc,CaptureQualityError) else 'global'
                    claim.fail(exc,scope=scope)
                    record=dict(group=list(claim.key),plans=claim.plans,error=str(exc),scope=scope,time=time.time())
                    atomic_json(root/'rejected'/('l%02d_p%03d.json'%claim.key),record)
                    print(json.dumps(dict(status='rejected',**record)),flush=True)
                    worker_state(root,args.worker,args.run_id,'quarantined' if scope=='scene' else 'failed',
                                 error=str(exc),groups_done=groups_done)
                    if scope=='global':return
                    # Release the claim and take another group. Reconstruct the
                    # renderer after a rejection to discard partial scene state.
                    if scene:scene.close()
                    scene=None;layout=None;gc.collect()
    finally:
        if scene:scene.close()
        scene=None
        gc.collect()
        worker_state(root,args.worker,args.run_id,'exited',groups_done=groups_done)


def roles_needing_more(tasks, completed, roles, quota, target):
    """Do not add a block if in-flight fixed windows could already meet quota."""
    from agcr_exp.data import plan_role
    from agcr_exp.protocol import DEFAULT
    capacity={r:0 for r in ('train','augmentation_train')}
    for p in tasks:
        role=plan_role(p,roles)
        if p['name'] not in completed and role in capacity:
            capacity[role]+=len(DEFAULT['evaluation_starts'])
    return [r for r in capacity if quota[r]+capacity[r]<target]


def pending_groups(plans, completed, failures):
    """Quarantine affects scheduling, never the final completeness requirement."""
    fatal=[f for f in failures if f.get('scope','scene')!='scene']
    if fatal:raise RuntimeError('Capture infrastructure failure: '+json.dumps(fatal))
    blocked={tuple(f['group']) for f in failures}
    remaining=[p for p in plans if p['name'] not in completed]
    runnable=[p for p in remaining if group_key(p) not in blocked]
    quarantined=sorted({group_key(p) for p in remaining if group_key(p) in blocked})
    return remaining,runnable,quarantined


def orchestrate_dynamic(args):
    from agcr_exp.data import assignments,build_index
    from agcr_exp.selection import quota_report
    root=Path(args.out).resolve();root.mkdir(parents=True,exist_ok=True)
    runner=Path(__file__).with_name('supplement_paper_dataset.py')
    with (root/'supplement.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(args.base,root,args.target)
        roles=assignments(args.base)
        if assignments(root)!=roles:raise ValueError('Existing split changed')
        manifest=read(root/'manifest.json');plans=read(root/'plans.json')
        queue_path=root/'queues'/'live.json';run_id=uuid.uuid4().hex
        # Old workers finish their current pairs after losing their controller.
        # Wait for them before launching replacements, so concurrency never doubles.
        deadline=time.monotonic()+300
        while live_worker_pids(root,queue_path):
            if time.monotonic()>deadline:raise RuntimeError('Previous workers still active; no duplicate pool started')
            time.sleep(2)
        doc=dict(run_id=run_id,controller_pid=os.getpid(),workers=args.workers,sealed=False,tasks=plans)
        atomic_json(queue_path,doc)
        queue=TaskQueue(root,queue_path,run_id)
        progress=dict(status='supplementing',scheduler='shared-paired-group-queue',run_id=run_id,
                      pid=os.getpid(),configured_workers=args.workers,started_unix=time.time(),
                      target=args.target,dataset_root=str(root),repair_episodes=len(manifest['repair_episodes']))
        atomic_json(root/'supplement_status.json',progress)
        (root/'logs').mkdir(exist_ok=True)
        processes={};restarts={i:0 for i in range(args.workers)}

        def launch(slot):
            log=(root/'logs'/('dynamic_%02d.log'%slot)).open('a')
            try:
                proc=subprocess.Popen([sys.executable,str(runner),'--out',str(root),'--queue-file',str(queue_path),
                     '--run-id',run_id,'--controller-pid',str(os.getpid()),'--worker',str(slot),'--workers',str(args.workers)],
                     stdout=log,stderr=subprocess.STDOUT)
            finally:log.close()
            processes[slot]=proc

        def close_pool():
            doc['sealed']=True;atomic_json(queue_path,doc)
            for proc in processes.values():
                try:proc.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    try:proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait()

        try:
            for slot in range(args.workers):launch(slot)
            last_quota=0;quota=None
            while True:
                if (root/'STOP').exists():raise RuntimeError('STOP requested; draining paired groups')
                failed=[read(p) for p in queue.failures.glob('*.json')]
                # A rejected physical event cannot stop unrelated acquisitions.
                # Unknown failures (disk, renderer, I/O, etc.) remain fatal.
                if any(f.get('scope','scene')!='scene' for f in failed):
                    raise RuntimeError('Capture infrastructure failure: '+json.dumps(failed))
                for slot,proc in list(processes.items()):
                    code=proc.poll()
                    if code is not None:
                        restarts[slot]+=1
                        if restarts[slot]>3:raise RuntimeError('Repeated worker exit: slot %d, code %s'%(slot,code))
                        print(json.dumps(dict(status='worker_restart',worker=slot,exit_code=code,time=time.time())),flush=True)
                        launch(slot)
                completed={p.parent.name for p in (root/'episodes').glob('*/complete.json')}
                remaining,runnable,quarantined=pending_groups(plans,completed,failed)
                groups_left=len({group_key(p) for p in runnable})
                if quota is None or time.monotonic()-last_quota>=20 or not runnable:
                    quota=quick_quota(root,roles);last_quota=time.monotonic()
                # Refill before the tail: workers need not wait for a batch barrier.
                if groups_left<=args.workers*2:
                    quota=quick_quota(root,roles);last_quota=time.monotonic()
                    needed=roles_needing_more(runnable,completed,roles,quota['eligible_clips'],args.target)
                    if needed:
                        wave=max((p.get('supplement_wave',-1) for p in plans),default=-1)+1
                        if wave>=args.max_waves:raise RuntimeError('Quota not reached within maximum waves')
                        additions=[]
                        for role,pool in [('train','training_supplement'),('augmentation_train','augmentation')]:
                            if role in needed:additions.extend(supplemental_wave(manifest['config'],wave,pool,len(plans)+len(additions)))
                        if shutil.disk_usage(root).free<(manifest['config']['disk_reserve_gib']+len(additions)*.06)*1024**3:
                            raise RuntimeError('Insufficient disk reserve for next balanced block')
                        plans.extend(additions);atomic_json(root/'plans.json',plans)
                        manifest.update(planned_episodes=len(plans),plans_sha256=file_hash(root/'plans.json'))
                        atomic_json(root/'manifest.json',manifest)
                        doc['tasks']=plans;atomic_json(queue_path,doc)
                        progress['wave']=wave
                        remaining,runnable,quarantined=pending_groups(plans,completed,failed)
                        groups_left=len({group_key(p) for p in runnable})
                progress.update(quota,completed_episodes=len(completed),planned_episodes=len(plans),
                                remaining_groups=groups_left,repair_complete=all(n in completed for n in manifest['repair_episodes']),
                                quarantined_groups=[list(k) for k in quarantined],
                                quarantine_count=len(quarantined),
                                worker_restarts=restarts,updated_unix=time.time())
                atomic_json(root/'supplement_status.json',progress)
                if not runnable and quarantined:
                    close_pool()
                    progress.update(status='needs_repair',error='All runnable groups finished; quarantined groups require repair before final validation',
                                    updated_unix=time.time())
                    atomic_json(root/'supplement_status.json',progress)
                    return
                if not remaining:
                    if not all(v>=args.target for v in quota['eligible_clips'].values()):
                        raise RuntimeError('No remaining tasks but quota is not met')
                    break
                time.sleep(2)
            close_pool()
            progress.update(status='validating',updated_unix=time.time());atomic_json(root/'supplement_status.json',progress)
            with (root/'logs'/'validation.log').open('w') as log:
                subprocess.run([sys.executable,str(runner.with_name('validate_paper_dataset.py')),
                    '--root',str(root),'--export','--require-complete'],stdout=log,stderr=subprocess.STDOUT,check=True)
            index=build_index(root,root/'experiment_index.json')
            report=quota_report(index,args.target);atomic_json(root/'downstream_quota.json',report)
            if not report['reference_quota_ready']:raise RuntimeError('Formal quota check failed')
            if assignments(root)!=roles:raise ValueError('Original split changed')
            for name,sha in manifest['inherited_marker_sha256'].items():
                if file_hash(root/'episodes'/name/'complete.json')!=sha:raise ValueError('Inherited marker changed: '+name)
            progress.update(status='complete',completed_unix=time.time(),split_sha256=index['split_sha256'])
            atomic_json(root/'supplement_status.json',progress)
        except BaseException as exc:
            close_pool()
            progress.update(status='failed',error=str(exc),updated_unix=time.time())
            atomic_json(root/'supplement_status.json',progress)
            raise

"""Dependency scheduler for real training, sampling and independent evaluation.

A process exit code is not scientific completion. Each job must satisfy its
artifact predicate, and missing external methods stay explicitly blocked.
"""
import argparse,fcntl,json,os,signal,subprocess,time,uuid
from pathlib import Path
from experiments.paired_interventions.train import atomic_json
from .storage import capacity_reason


def satisfied(marker):
    path=Path(marker['path'])
    if not path.is_file():return False
    try:value=json.loads(path.read_text())
    except (ValueError,OSError):return False
    for dotted,expected in marker.get('predicate',{'completed':True}).items():
        current=value
        for key in dotted.split('.'):
            if not isinstance(current,dict) or key not in current:return False
            current=current[key]
        if current!=expected:return False
    return True


def readiness(job,complete):
    missing=[name for name in job.get('dependencies',[]) if name not in complete]
    missing+=['path:'+path for path in job.get('requires',[]) if not Path(path).exists()]
    if job.get('external_completion') and not satisfied(job['external_completion']):missing.append('external_completion')
    if job.get('blocked_reason'):missing.append(job['blocked_reason'])
    if not missing and job.get('storage_capacity'):
        reason=capacity_reason(job['storage_capacity'])
        if reason:missing.append(reason)
    return missing


def bounded_command(job,remaining):
    command=list(job['command']);gpus=job.get('gpu_count',0)
    if '--gpu-hours' in command:
        position=command.index('--gpu-hours')+1;command[position]=str(min(float(command[position]),remaining))
    if job.get('resume_flag') and Path(job.get('checkpoint','__missing__')).is_file() and job['resume_flag'] not in command:
        command += [job['resume_flag']]
        if job.get('resume_takes_path',True):command.append(job['checkpoint'])
    return command


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--gpu-hours',type=float,required=True);p.add_argument('--run',action='store_true');a=p.parse_args()
    if a.gpu_hours<=0:raise ValueError('Positive explicit GPU budget required')
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);plan=json.loads(Path(a.plan).read_text());jobs=plan['jobs']
    names=[j['id'] for j in jobs]
    if len(names)!=len(set(names)):raise ValueError('Duplicate experiment job IDs')
    if any(set(j.get('dependencies',[]))-set(names) for j in jobs):raise ValueError('Unresolved dependency names')
    if not a.run:print(json.dumps(dict(jobs=len(jobs),started=False)));return
    lock=(root/'.runner.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=json.loads((root/'budget.json').read_text()) if (root/'budget.json').exists() else {}
    used=float(prior.get('used_gpu_hours',0.));stopping=[False];current=[None]
    def stop(*_):stopping[0]=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    completed={j['id'] for j in jobs if satisfied(j['completion'])};failed={};attempted=set()
    def publish(state,active=None):
        blocked={j['id']:readiness(j,completed) for j in jobs if j['id'] not in completed}
        atomic_json(root/'status.json',dict(state=state,pid=os.getpid(),updated_unix=time.time(),active_job=active,
            completed=sorted(completed),total=len(jobs),failures=failed,blocked=blocked,used_gpu_hours=used,budget_gpu_hours=a.gpu_hours,
            actual_results='read each completed artifact; none inferred from exit codes'))
        atomic_json(root/'budget.json',dict(used_gpu_hours=used,limit_gpu_hours=a.gpu_hours))
    while len(completed)<len(jobs):
        if stopping[0]:publish('paused');return
        ready=[j for j in jobs if j['id'] not in completed|attempted and not readiness(j,completed)]
        if not ready:publish('blocked');return
        job=ready[0];name=job['id'];gpus=job.get('gpu_count',0);remaining=a.gpu_hours-used
        if gpus and remaining<=0:publish('budget_reached');return
        command=bounded_command(job,remaining);logdir=root/'logs';logdir.mkdir(exist_ok=True)
        stop_file=None
        if job.get('safe_stop_file'):
            # A fresh per-invocation path avoids treating an earlier graceful
            # stop as a new stop request on resume.
            stop_file=root/'stop_requests'/(uuid.uuid4().hex+'.json')
            stop_file.parent.mkdir(parents=True,exist_ok=True)
            if '--stop-file' in command:command[command.index('--stop-file')+1]=str(stop_file.resolve())
            else:command+=['--stop-file',str(stop_file.resolve())]
        atomic_json(logdir/(name+'.launch.json'),dict(command=command,started_unix=time.time(),gpu_count=gpus))
        previous=used;begin=time.monotonic();sent=False
        with (logdir/(name+'.log')).open('a') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);current[0]=child
            while child.poll() is None:
                used=previous+(time.monotonic()-begin)*gpus/3600
                if (stopping[0] or (gpus and used>=a.gpu_hours)) and not sent:
                    # Training has its own bound and saves at an update boundary.
                    # Readers and samplers handle SIGTERM at a clip boundary.
                    if stop_file is not None:
                        atomic_json(stop_file,dict(requested=True,reason='runner_pause' if stopping[0] else 'budget'))
                    elif '--gpu-hours' not in command or stopping[0]:os.killpg(child.pid,signal.SIGTERM)
                    sent=True
                publish('running',name);time.sleep(5)
        used=previous+(time.monotonic()-begin)*gpus/3600;attempted.add(name)
        if child.returncode==0 and satisfied(job['completion']):completed.add(name)
        elif used>=a.gpu_hours or stopping[0]:publish('checkpoint_saved',name);return
        else:failed[name]=dict(exit_code=child.returncode,reason='artifact predicate not satisfied' if child.returncode==0 else 'process failed')
    publish('complete');atomic_json(root/'completion.json',dict(completed=True,jobs=sorted(completed)))
if __name__=='__main__':main()

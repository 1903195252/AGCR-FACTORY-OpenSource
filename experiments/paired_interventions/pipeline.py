"""Sequential revised training with a persistent shared GPU-hour budget.

The original observation checkpoint remains read-only. This revision warm-starts
its new dynamics/reader objective, then trains depth and joint video. Phase four
is dispatched only after a completed formal generator and explicit job gates.
"""
import argparse,json,os,signal,subprocess,sys,time,uuid,fcntl
from pathlib import Path
from .train import atomic_json


def stage_four_accounting(limit,total_used,stage_four_used,base=None):
    """Runner budgets are cumulative; parent totals already include its ledger."""
    if min(limit,total_used,stage_four_used)<0:raise ValueError('Negative GPU accounting')
    if base is None:base=max(0.,total_used-stage_four_used)
    if base<0:raise ValueError('Negative pre-stage-four usage')
    return float(base),max(0.,limit-base),max(total_used,base+stage_four_used)


def stage_four_readiness(plan_path):
    """A present DAG can contain explicitly unimplemented external methods."""
    if not plan_path or not Path(plan_path).is_file():return dict(integrated=False,all_ready=False,blocked=[])
    try:plan=json.loads(Path(plan_path).read_text())
    except (OSError,ValueError):return dict(integrated=False,all_ready=False,blocked=[dict(id='stage_four_plan',reason='invalid_plan')])
    blocked={j['id']:j['blocked_reason'] for j in plan.get('jobs',[]) if j.get('blocked_reason')}
    for name in plan.get('missing_implementations',[]):blocked.setdefault(name,'missing_implementation')
    return dict(integrated=True,all_ready=not blocked,blocked=[dict(id=name,reason=reason) for name,reason in sorted(blocked.items())])


def stages(paths,output,variant,seed,world_size,python=sys.executable,development=False,execution_policy='standard'):
    prefix=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(world_size),
            '--module','experiments.paired_interventions.train']
    if world_size==1:prefix=[python,'-m','experiments.paired_interventions.train']
    result=[];previous=None
    for phase in ('dynamics','depth','video'):
        run=Path(output)/(phase+'_seed'+str(seed))
        command=prefix+['--paths',str(Path(paths).resolve()),'--output',str(run.resolve()),'--phase',phase,
            '--variant',variant,'--seed',str(seed),'--execution-policy',execution_policy]
        if previous:command+=['--init',str(previous/('last.pt' if development else 'best.pt'))]
        if development:command+=['--development','--families-only','--steps','2','--batch',str(world_size)]
        result.append(dict(phase=phase,run=str(run.resolve()),command=command));previous=run
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('paths','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--variant',default='full');p.add_argument('--seed',type=int,default=17)
    p.add_argument('--world-size',type=int,default=2);p.add_argument('--gpu-hours',type=float,required=True)
    p.add_argument('--previous-gpu-hours',type=float,default=0.,help='Already consumed before this revised pipeline')
    p.add_argument('--stage-four-plan',help='Machine-readable DAG consumed after completed video training')
    p.add_argument('--development',action='store_true');p.add_argument('--run',action='store_true')
    p.add_argument('--execution-policy',choices=['standard','optimized_v5'],default='standard')
    a=p.parse_args();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    paths=json.loads(Path(a.paths).read_text());plan=stages(a.paths,root,a.variant,a.seed,a.world_size,development=a.development,execution_policy=a.execution_policy)
    atomic_json(root/'plan.json',dict(stages=plan,budget_gpu_hours=a.gpu_hours,previous_gpu_hours=a.previous_gpu_hours,development_only=a.development,execution_policy=a.execution_policy))
    if not a.run:print(json.dumps(dict(plan=str(root/'plan.json'),started=False)));return
    if a.gpu_hours<=a.previous_gpu_hours:raise ValueError('No remaining GPU budget')
    lock=(root/'.dispatcher.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=json.loads((root/'budget.json').read_text()) if (root/'budget.json').exists() else {}
    used=max(float(prior.get('used_gpu_hours',a.previous_gpu_hours)),a.previous_gpu_hours)
    stage_four_base=prior.get('stage_four_base_gpu_hours')
    if prior and prior.get('limit_gpu_hours')!=a.gpu_hours:raise ValueError('Explicit unchanged budget required on automatic resume')
    stop=[False];stop_file=root/('stop-request-'+uuid.uuid4().hex+'.json')
    def signal_stop(*_):
        stop[0]=True;atomic_json(stop_file,dict(requested=True,reason='pipeline_signal'))
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    titles={'dynamics':'成对数据动力学与独立训练读出头','depth':'深度解码器预热','video':'可见性感知联合 RGB-D 视频生成'}
    phase_state={row['phase']:'pending' for row in plan}
    def publish(state,row=None,**extra):
        readiness=stage_four_readiness(a.stage_four_plan)
        phases=[dict(phase=1,title='观测与关联',state='complete',tasks=[])]
        for number,keys,title in [(2,['dynamics'],'时序状态与几何'),(3,['depth','video'],'联合 RGB-D 生成')]:
            values=[phase_state[key] for key in keys]
            ps='complete' if all(x=='complete' for x in values) else next((x for x in values if x!='complete' and x!='pending'),'pending')
            phases.append(dict(phase=number,title=title,state=ps,tasks=[dict(id=k,title=titles[k],state=phase_state[k]) for k in keys]))
        downstream=extra.get('stage_four')
        fourth_state=('complete' if downstream.get('state')=='complete' else downstream.get('state','pending')) if downstream is not None else ('pending' if a.stage_four_plan else 'integration_required')
        current=downstream.get('active_job') if downstream is not None else None
        tasks=[] if current is None else [dict(id=current,title=current,state=fourth_state)]
        phases.append(dict(phase=4,title='外部对比、独立评测与下游训练',state=fourth_state,tasks=tasks,
            integrated=readiness['integrated'],external_blockers=readiness['blocked']))
        atomic_json(root/'status.json',dict(schema='agcr-pipeline-1',pid=os.getpid(),updated_unix=time.time(),state=state,
            execution_policy=a.execution_policy,
            phase='stage_four' if downstream is not None else row['phase'] if row else None,
            current_task=current if downstream is not None else row['phase'] if row else None,
            active_run=str(root/'stage_four') if downstream is not None else row['run'] if row else None,
            phases=phases,all_stages_ready=readiness['all_ready'],stage_four_integrated=readiness['integrated'],
            stage_four_external_blockers=readiness['blocked'],
            complete=state=='experiments_complete',development_only=a.development,
            user_paused=bool(stop[0] or (not a.development and Path(paths['pause_marker']).exists())),
            subsequent_gpu_hours=a.gpu_hours,subsequent_gpu_hours_used=used,budget_exhausted=used>=a.gpu_hours,**extra))
        atomic_json(root/'budget.json',dict(limit_gpu_hours=a.gpu_hours,used_gpu_hours=used,stage_four_base_gpu_hours=stage_four_base,updated_unix=time.time(),
            accounting='allocated training GPUs multiplied by elapsed wall seconds; prior original run included'))
    for row in plan:
        run=Path(row['run']);complete=run/'completion.json'
        if complete.exists():
            info=json.loads(complete.read_text())
            if info['completed'] and info['development_only']==a.development:phase_state[row['phase']]='complete';continue
        if stop[0] or (not a.development and Path(paths['pause_marker']).exists()):
            phase_state[row['phase']]='paused_user';publish('paused',row);return
        remaining=a.gpu_hours-used
        if remaining<=0:phase_state[row['phase']]='budget_reached';publish('budget_reached',row);return
        run.mkdir(parents=True,exist_ok=True)
        command=row['command']+['--gpu-hours',str(remaining),'--stop-file',str(stop_file)]
        if (run/'last.pt').exists():command+=['--resume',str(run/'last.pt')]
        phase_state[row['phase']]='running';publish('running',row)
        atomic_json(run/'pipeline-launch.json',dict(started_unix=time.time(),command=command,world_size=a.world_size))
        before=time.monotonic();charged=used
        with (run/'console.log').open('a') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            while child.poll() is None:
                used=charged+(time.monotonic()-before)*a.world_size/3600
                publish('running',row);time.sleep(5)
        used=charged+(time.monotonic()-before)*a.world_size/3600
        if child.returncode:
            phase_state[row['phase']]='failed';publish('failed',row,exit_code=child.returncode);return
        info=json.loads(complete.read_text())
        if not info['completed']:
            phase_state[row['phase']]='budget_reached' if used>=a.gpu_hours else 'checkpointed'
            publish('checkpoint_saved',row,step=info['step']);return
        phase_state[row['phase']]='complete'
    publish('training_complete',plan[-1])
    if a.stage_four_plan and (stop[0] or a.gpu_hours<=used):
        state='paused' if stop[0] else 'budget_reached'
        publish(state,stage_four=dict(state=state,active_job=None));return
    if a.stage_four_plan and a.gpu_hours>used and not stop[0]:
        if not a.development and Path(paths['pause_marker']).exists():publish('paused',stage_four=dict(state='paused',active_job=None));return
        ledger=root/'stage_four/budget.json'
        prior_four=json.loads(ledger.read_text()).get('used_gpu_hours',0.) if ledger.exists() else 0.
        stage_four_base,runner_cap,used=stage_four_accounting(a.gpu_hours,used,float(prior_four),stage_four_base)
        if runner_cap<=prior_four:publish('budget_reached',stage_four=dict(state='budget_reached',active_job=None,used_gpu_hours=prior_four));return
        publish('stage_four_starting',stage_four=dict(state='starting',active_job=None,used_gpu_hours=prior_four))
        command=[sys.executable,'-m','experiments.experiment_runner.run','--plan',a.stage_four_plan,
            '--output',str(root/'stage_four'),'--gpu-hours',str(runner_cap),'--run']
        child=subprocess.Popen(command);sent=False
        while child.poll() is None:
            requested=stop[0] or (not a.development and Path(paths['pause_marker']).exists())
            if requested and not sent:child.send_signal(signal.SIGTERM);sent=True
            status_file=root/'stage_four/status.json'
            if status_file.exists():
                downstream=json.loads(status_file.read_text());used=stage_four_base+downstream.get('used_gpu_hours',prior_four)
                publish('stage_four_stopping' if sent else 'stage_four_running',stage_four=downstream)
            time.sleep(5)
        status_file=root/'stage_four/status.json'
        downstream=json.loads(status_file.read_text()) if status_file.exists() else dict(state='failed',active_job=None,used_gpu_hours=prior_four)
        used=stage_four_base+downstream.get('used_gpu_hours',prior_four)
        publish('paused' if sent else 'experiments_complete' if downstream['state']=='complete' else 'stage_four_returned',
            stage_four_exit=child.returncode,stage_four=downstream)
    lock.close()
if __name__=='__main__':main()

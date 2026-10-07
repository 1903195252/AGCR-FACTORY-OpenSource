"""Executable stage-four DAG; every completion marker has an explicit predicate."""
import argparse,json,sys
from pathlib import Path
from .train import atomic


def build_plan(index,assets,output,generated=None,gpu_hours_per_job=24.,gpus=2,python=None,zones=None):
    """generated=[{method,seed,modality,manifest,completion}] from real samplers.

    The caller schedules ready jobs, supplies its remaining global GPU budget,
    and appends --resume only when the training output/last.pt exists. An exit
    code of zero WITHOUT the declared completion predicate means incomplete.
    """
    if gpu_hours_per_job<=0 or gpus<1:raise ValueError('Explicit positive GPU budget required')
    root=Path(output);python=python or sys.executable;jobs=[]
    def add(name,module,args,deps=(),requires=(),gpu=0,marker=None,predicate=None):
        args=list(args)
        if gpu and module in {'experiments.downstream_tracking.calibrate','experiments.downstream_tracking.annotations','experiments.downstream_tracking.rgb_acceptance'}:args+=['--gpu-hours',gpu_hours_per_job]
        command=[python,'-m',module]+list(map(str,args))
        jobs.append(dict(id=name,command=command,dependencies=list(deps),requires=list(map(str,requires)),gpu_count=gpu,
            completion=dict(path=str(marker),predicate=predicate or {'completed':True}),output_directory=str(root/name),status='not_run',
            budget_gpu_hours=gpu_hours_per_job if gpu else 0))
    def train(name,role,modality,seed,cohort=None,deps=()):
        args=['--index',index,'--assets',assets,'--output',root/name,'--role',role,'--modality',modality,'--seed',seed,'--gpu-hours',gpu_hours_per_job]
        stop_file=root/name/'STOP';args+=['--stop-file',stop_file]
        if cohort:args+=['--cohort',cohort]
        add(name,'experiments.downstream_tracking.train',args,deps,[index,assets]+([cohort] if cohort else []),gpus,root/name/'completion.json')
        jobs[-1]['command']=[python,'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(gpus),'-m']+jobs[-1]['command'][2:]
        jobs[-1]['safe_stop_file']=str(stop_file);jobs[-1]['resume_flag']='--resume';jobs[-1]['resume_takes_path']=False; jobs[-1]['checkpoint']=str(root/name/'last.pt')
    def association(name,checkpoint,dependency):
        path=root/name/'association.json'
        add(name,'experiments.downstream_tracking.calibrate',['--mode','association','--index',index,'--checkpoint',checkpoint,'--assets',assets,'--output',path],
            [dependency],[checkpoint,index],1,path,{'fit_role':'val'})
        return path
    readers={}
    for role in ('annotation','video_evaluation'):
        for modality in ('rgb','rgbd'):
            name=role+'_'+modality;train(name,role,modality,17029)
            checkpoint=root/name/'best.pt';assoc=association(name+'_association',checkpoint,name)
            readers[role,modality]=(name,checkpoint,assoc)
    thresholds={}
    for modality in ('rgb','rgbd'):
        name,checkpoint,assoc=readers['annotation',modality];job=name+'_quality';path=root/job/'thresholds.json'
        module='experiments.downstream_tracking.rgb_acceptance' if modality=='rgb' else 'experiments.downstream_tracking.calibrate'
        args=(['--mode','calibrate','--input',index] if modality=='rgb' else ['--mode','annotation','--index',index,'--modality','rgbd'])
        add(job,module,args+['--checkpoint',checkpoint,'--assets',assets,'--association',assoc,'--output',path],
            [name+'_association'],[checkpoint,assoc],1,path,{'fit_role':'val'})
        thresholds[modality]=(job,path)
    def evaluation(name,modality,seed,training_dependency):
        checkpoint=root/name/'best.pt'
        # Association weights stay fixed across data regimes: use corresponding R-only validation calibration.
        reference='R_%s_s%d'%(modality,seed);association_job=reference+'_association';assoc=root/association_job/'association.json'
        if name==reference:association(reference+'_association',checkpoint,name)
        for role in ('id_test','test'):
            job=name+'_evaluate_'+role;result=root/job/'results.json'
            args=['--index',index,'--checkpoint',checkpoint,'--assets',assets,'--association',assoc,'--role',role,'--output',root/job,'--gpu-hours',gpu_hours_per_job]
            if zones:args+=['--zones',zones]
            add(job,'experiments.formal_evaluation.downstream',args,[training_dependency,association_job],[checkpoint,assoc],1,root/job/'completion.json')
            jobs[-1]['shared_association_reference']=reference
    def cohort_job(name,regime,modality,seed,accepted=None,deps=()):
        cohort=root/'cohorts'/(name+'.json');args=['--index',index,'--output',cohort,'--regime',regime,'--seed',seed,'--annotation-modality',modality]
        if accepted:args+=['--accepted-generated',accepted]
        add('cohort_'+name,'experiments.downstream_tracking.cohort',args,deps,[index]+([accepted] if accepted else []),0,cohort,{'clip_count':8000 if regime in ('R_repeat','R_plus_G','R_plus_new_render') else 4000})
        train(name,'downstream',modality,seed,cohort,['cohort_'+name]);evaluation(name,modality,seed,name)
    for seed in (17,29,43):
        for modality in ('rgb','rgbd'):
            for regime in ('R','R_repeat','R_plus_new_render'):cohort_job('%s_%s_s%d'%(regime,modality,seed),regime,modality,seed)
    for item in generated or []:
        modality=item['modality'];seed=item['seed'];method=item['method']
        if seed not in (17,29,43) or modality not in ('rgb','rgbd'):raise ValueError('Unregistered replicate/modality')
        name='annotate_%s_%s_s%d'%(method,modality,seed);reader,checkpoint,assoc=readers['annotation',modality]
        proposal=root/name/'proposals.json';accepted=root/name/'accepted.json';threshold_job,threshold=thresholds[modality]
        module='experiments.downstream_tracking.rgb_acceptance' if modality=='rgb' else 'experiments.downstream_tracking.annotations'
        args=['--mode','annotate','--input',item['manifest'],'--checkpoint',checkpoint,'--assets',assets,'--association',assoc,'--output',proposal]
        if modality=='rgbd':args+=['--modality','rgbd']
        add(name,module,args,[reader+'_association'],[item['manifest'],item['completion'],checkpoint,assoc],1,proposal)
        jobs[-1]['external_completion']={'path':item['completion'],'predicate':{'completed':True}}
        accept_name=name+'_accept'
        add(accept_name,module,['--mode','accept','--input',proposal,'--thresholds',threshold,'--output',accepted],
            [name,threshold_job],[proposal,threshold],0,accepted,{'completed':True})
        for regime in ('G','R_G','R_plus_G'):
            base_name='%s_%s_%s_s%d'%(method,regime,modality,seed)
            cohort_job(base_name,regime,modality,seed,accepted,[accept_name])
            if modality=='rgbd':
                # Depth-isolation table: exact SAME accepted RGB-D labels/cohort, only learner input changes.
                rgb_name=base_name+'_rgb_learner';same_cohort=root/'cohorts'/(base_name+'.json')
                train(rgb_name,'downstream','rgb',seed,same_cohort,['cohort_'+base_name])
                evaluation(rgb_name,'rgb',seed,rgb_name)
    # Each method/regime is compared with the same rendered-only learner at
    # all three training seeds; episode-paired tests use fixed R calibration.
    groups={}
    for job in list(jobs):
        name=job['id']
        if '_evaluate_' not in name:continue
        training,role=name.rsplit('_evaluate_',1)
        if '_s' not in training:continue
        stem,seedpart=training.rsplit('_s',1)
        seed=int(seedpart.split('_')[0])
        modality='rgb' if seedpart.endswith('_rgb_learner') else stem.rsplit('_',1)[-1]
        if stem.startswith('R_') and stem==('R_'+modality):continue
        group=(stem+('_rgb_learner' if seedpart.endswith('_rgb_learner') else ''),role,modality)
        groups.setdefault(group,{})[seed]=name
    for (stem,role,modality),members in groups.items():
        if set(members)!={17,29,43}:continue
        left=[root/members[seed]/'report.json' for seed in (17,29,43)]
        reference=['R_%s_s%d_evaluate_%s'%(modality,seed,role) for seed in (17,29,43)]
        right=[root/name/'report.json' for name in reference]
        name=stem+'_compare_R_'+role;out=root/name/'comparison.json'
        add(name,'experiments.formal_evaluation.compare_downstream',
            ['--left']+left+['--right']+right+['--metrics','R3','IDF1','HOTA','ID_switches','entry_F1','--output',out],
            list(members.values())+reference,left+right,0,out,{'schema':'agcr-downstream-paired-tests-1'})
    return dict(schema='agcr-downstream-jobs-2',jobs=jobs,formal_training_started=False,
        scheduler_contract='Completion predicates are mandatory; budget-paused training is incomplete even on exit0. Preserve job checkpoints. Never synthesize missing cohorts.',
        output_metrics='formal_evaluation.downstream results; no scores until execution',actual_results=None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--assets',required=True);p.add_argument('--output',required=True)
    p.add_argument('--generated',help='JSON list method/seed/modality/manifest/completion');p.add_argument('--gpu-hours-per-job',type=float,required=True)
    p.add_argument('--gpus',type=int,default=2);p.add_argument('--zones');a=p.parse_args()
    plan=build_plan(a.index,a.assets,a.output,json.loads(Path(a.generated).read_text()) if a.generated else [],a.gpu_hours_per_job,a.gpus,zones=a.zones)
    atomic(Path(a.output)/'jobs.json',plan);print(json.dumps(dict(jobs=len(plan['jobs']),manifest=str(Path(a.output)/'jobs.json'))))

if __name__=='__main__':main()

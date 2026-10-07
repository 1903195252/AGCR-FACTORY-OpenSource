"""Audit manifests and expand the causal experiment matrix without launching jobs."""
import importlib.util
from pathlib import Path
from .selection import quota_report
from .protocol import DEFAULT,VARIANTS,EXPERIMENTS,EXTERNAL,RETRAIN,configuration,digest

# Operational capabilities; never advertise all configured variants as implemented.
IMPLEMENTED={
 'index':True,'rgbd_reader':True,'observation_corruptions':True,'mechanism_kernels':True,
 'depth_state_metrics':True,'paired_statistics':True,'downstream_cohort_builder':True,
 'full_observation_tracker':False,'joint_geometry_training':False,
 'integrated_wan_rgbd_training':False,'independent_annotation_reader':False,
 'independent_evaluation_reader':False,'downstream_tracker_training':False,
 'external_factory_adaptations':False,
}


def audit(index,assets=None):
    issues=[]
    quotas=quota_report(index)
    for role,pool in quotas['pools'].items():
        if pool['deficit_clips']:
            issues.append('Downstream fixed-window quota deficit '+role+': '+str(pool['deficit_clips']))
    if index['development_only']:issues.append('Data index is development-only; formal results disabled')
    counts={r['role']:r['windows'] for r in index['counts'] if r['cohort']=='history_covered'}
    for role in ['train','val','id_test','test','annotation_train','evaluation_reader_train']:
        if counts.get(role,0)==0:issues.append('No primary windows in '+role)
    # Each camera trajectory must belong to exactly one independent role.
    groups={}
    for row in index['windows']:
        group=row['camera_group'];old=groups.setdefault(group,row['role'])
        if old!=row['role']:issues.append('Camera trajectory leaks across roles: '+group)
    for name,ok in IMPLEMENTED.items():
        if not ok:issues.append('Integration required: '+name)
    for name,path in (assets or {}).items():
        if not Path(path).exists():issues.append('Missing asset '+name+': '+str(path))
    return dict(formal_ready=not issues,issues=sorted(set(issues)),capabilities=IMPLEMENTED,
                split_sha256=index['split_sha256'],counts=counts,downstream_quota=quotas,
                dependencies={name:importlib.util.find_spec(name) is not None for name in ['torch','h5py','scipy','diffusers','peft','lpips','trackeval']})


def matrix(index,experiments=None):
    jobs=[]
    quotas=quota_report(index) if 'windows' in index else None
    for eid in experiments or list(EXPERIMENTS):
        exp=EXPERIMENTS[eid]
        for variant in exp['variants']:
            for seed in DEFAULT['training_seeds']:
                base=dict(experiment=eid,variant=variant,training_seed=seed,split_sha256=index['split_sha256'],
                          dataset_identity=index['dataset_identity'],development_only=index['development_only'])
                if eid=='E9':
                    conditions=[]
                    for size in DEFAULT['downstream']['sizes']:
                        conditions.extend([dict(regime='R',total_clips=size,generated_fraction=0.),dict(regime='G',total_clips=size,generated_fraction=1.)])
                        conditions.extend(dict(regime='R_G',total_clips=size,generated_fraction=f) for f in [.25,.5,.75])
                    conditions.extend(dict(regime=r,total_clips=4000) for r in ['R_repeat','R_plus_G','R_plus_new_render'])
                elif eid=='E10':conditions=[dict(cohort=c) for c in exp['cohorts']]
                else:conditions=[dict(intervention=i) for i in exp['interventions']]
                for condition in conditions:
                    row=dict(base,**condition)
                    row['kind']='external_baseline' if variant in EXTERNAL else 'internal_control_or_method'
                    row['config_sha256']=digest(EXTERNAL[variant]) if variant in EXTERNAL else configuration(variant,seed)['protocol_sha256']
                    row['sample_count']=8;row['status']='prepared_not_executed'
                    if eid=='E9' and quotas is not None:
                        row['reference_quota_ready']=quotas['reference_quota_ready']
                        needed=['train','augmentation_train'] if condition['regime']=='R_plus_new_render' else ['train']
                        missing={r:max(0,condition['total_clips']-quotas['pools'][r]['eligible_clips']) for r in needed}
                        if any(missing.values()):
                            row['status']='insufficient_fixed_window_reference_data';row['deficit_clips']=missing
                    if eid=='E10' and condition['cohort'] in ['ood_class','ood_layout_class'] and index.get('heldout_class') is None:
                        row['status']='requires_class_excluded_training_index'
                    if eid=='E10' and condition['cohort']=='physical_qualitative':row['quantitative']=False
                    row['job_id']=digest(row)[:20]
                    jobs.append(row)
    training={};stage_names=list(DEFAULT['stages'])
    for job in jobs:
        variant=job['variant'];seed=job['training_seed']
        if variant in EXTERNAL:
            job['training_dependencies']=['external_adaptation:'+variant+':'+str(seed)];continue
        owner='full' if RETRAIN[variant]=='inference_intervention' else variant
        first=stage_names.index(RETRAIN[owner]);deps=[]
        for j,stage in enumerate(stage_names):
            stage_owner='full' if j<first else owner
            spec=dict(stage=stage,variant=stage_owner,seed=seed,split_sha256=index['split_sha256'],
                      updates=DEFAULT['stages'][stage],config_sha256=configuration(stage_owner,seed)['protocol_sha256'])
            key=digest(spec)[:20];spec['dependencies']=deps[-1:];spec['training_id']=key
            spec['status']='ready_entrypoint' if stage=='observation' and stage_owner in ['full','rgb_only'] else 'integration_required'
            training[key]=spec;deps.append(key)
        job['training_dependencies']=deps
    return dict(schema='agcr-run-matrix-1',jobs=jobs,job_count=len(jobs),training_jobs=list(training.values()),
                unique_internal_training_stages=len(training),downstream_quota=quotas,
                note='Configuration expansion is not execution. Dependent model integrations are audited separately.')

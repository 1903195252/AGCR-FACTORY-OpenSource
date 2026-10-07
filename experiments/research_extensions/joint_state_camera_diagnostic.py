"""CPU-only state/geometry diagnosis of the complete matched joint8 DEV16.

This reads committed samples, never reconstructs or reloads their learned
history. Internal state accuracy is not generated-image accuracy. Identical
camera-path state draws are an architectural property, not a learned effect.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from .region_probe import sha256

SCHEMA='agcr-joint8-state-camera-dev-diagnostic-1'
ARMS=('joint_feedback','detached_feedback')
FAMILIES=('l20_evidence_r1_worker_ground_structure_005',
          'l23_evidence_r0_forklift_ground_structure_004')
SEEDS=(101,211)
ROOT=Path(__file__).resolve().parents[2]
HELPER=Path(__file__).with_name('camera_metrics.py')
KEYS=('internal_state','state_present','historical_reference_ids','historical_selected_slot_mask',
      'history_eligible_reference','predicted_full_layers','predicted_visible_layers',
      'initial_state','class_id','time_s','K','camera_to_world')
REF_KEYS=('world_state','present','instance_id','class_id','visible_fraction',
          'projection_area','camera_to_world','K','time_s')


def read(path):return json.loads(Path(path).read_text())


def option(command,flag):
    if command.count(flag)!=1:raise ValueError('Exactly one '+flag+' required')
    return command[command.index(flag)+1]


def helper():
    spec=importlib.util.spec_from_file_location('_frozen_camera_diagnostic',HELPER)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def require_mechanism(payload,arm):
    from .joint_feedback_candidate import REQUIRED_HOOKS
    coverage=payload.get('feedback_coverage',{});hooks=coverage.get('hook_nonzero_steps',{})
    if (payload.get('completed')is not True or payload.get('step')!=8 or payload.get('target_steps')!=8
            or payload.get('feedback_mechanism_admitted')is not True
            or type(coverage.get('active_steps'))is not int or not 0<coverage['active_steps']<=8
            or any(type(hooks.get(k))is not int or not 0<=hooks[k]<=8 for k in REQUIRED_HOOKS)
            or (arm=='joint_feedback'and not all(hooks[k]>0 for k in REQUIRED_HOOKS))
            or (arm=='detached_feedback'and any(hooks[k] for k in REQUIRED_HOOKS))):
        raise ValueError('Both actual eight-update feedback mechanisms must be admitted')


def training_contract(plan_path):
    import torch
    from .joint_feedback_candidate import validate_joint_payload,load_joint_step_gate
    from .train_joint_feedback import execution_recipe,validate_resume
    plan=read(plan_path);rows=plan['configurations']
    if len(rows)!=2 or {r['arm']for r in rows}!=set(ARMS)or any(r['steps']!=8 for r in rows):
        raise ValueError('Exactly the two matched8 configurations are required')
    result={};lineage={str(plan_path):sha256(plan_path)}
    roots={Path(option(r['train_argv'],'--output')).parent.parent for r in rows}
    if len(roots)!=1:raise ValueError('Matched training chain root differs')
    root=roots.pop();receipt=read(root/'receipt.json');done=read(root/'completion.json')
    if (done.get('completed')is not True or done.get('steps_per_arm')!=8 or done.get('seed')!=17033
            or sorted(done.get('arms',[]))!=sorted(ARMS)
            or done.get('receipt_sha256')!=sha256(root/'receipt.json')
            or receipt.get('plan_sha256')!=sha256(plan_path)):
        raise ValueError('The complete matched training chain is required')
    for path in(root/'receipt.json',root/'completion.json'):lineage[str(path)]=sha256(path)
    for row in rows:
        arm=row['arm'];c=row['train_argv'];folder=Path(option(c,'--output'))
        if option(c,'--arm')!=arm or option(c,'--steps')!='8'or option(c,'--seed')!='17033':
            raise ValueError('Matched update/seed/arm changed')
        weight_paths={name:Path(option(c,flag))for name,flag in(
            ('renderer_checkpoint','--checkpoint'),('state_checkpoint','--state-checkpoint'),
            ('future_checkpoint','--future-checkpoint'))}
        weights={k:sha256(v)for k,v in weight_paths.items()}
        paths=read(option(c,'--paths'));index_sha=sha256(paths['families'])
        native=Path(option(c,'--native-probe-report'))
        gate=load_joint_step_gate(native,weights);recipe=execution_recipe(gate)
        if not recipe['three_device']:raise ValueError('Measured three-device native Adam required')
        payload=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        identity=read(folder/'identity.json');validate_resume(payload,identity)
        actual=validate_joint_payload(payload,weight_paths['future_checkpoint'],weight_paths['state_checkpoint'],
            index_sha,renderer_sha=weights['renderer_checkpoint'])
        require_mechanism(payload,arm)
        if (actual!=identity or identity['arm']!=arm or identity['native_gate_sha256']!=sha256(native)
                or identity.get('execution_activation_offload')!=recipe['offload']):
            raise ValueError('Joint history/renderer checkpoint provenance changed')
        status=read(folder/'status.json');admission=read(root/(arm+'_checkpoint.json'))
        hashes=dict(checkpoint_sha256=sha256(folder/'last.pt'),identity_sha256=sha256(folder/'identity.json'))
        if (status.get('state')!='completed'or status.get('completed')is not True
                or admission.get('completed')is not True or admission.get('step')!=8
                or any(admission.get(k)!=v for k,v in hashes.items())):
            raise ValueError('Atomic arm completion identity does not match weights')
        result[arm]=dict(identity=identity,**hashes,native_gate_sha256=sha256(native),
            predictions=Path(option(row['sample_argv'],'--output')),weights=weights)
        for path in(folder/'last.pt',folder/'identity.json',folder/'status.json',root/(arm+'_checkpoint.json'),
                     native,native.with_name('completion.json')):lineage[str(path)]=sha256(path)
        lineage.update({str(p):weights[k]for k,p in weight_paths.items()})
        del payload
    if result[ARMS[0]]['weights']!=result[ARMS[1]]['weights']or result[ARMS[0]]['native_gate_sha256']!=result[ARMS[1]]['native_gate_sha256']:
        raise ValueError('Matched arms do not share initializer and native admission')
    return result,lineage


def panel_contract(root,arm,training):
    manifest=read(root/'manifest.json');done=read(root/'completion.json');index=read(root/'evaluation_index.json')
    expected=dict(method='peerdream_joint_'+arm,split='val',development_only=True,
        formal_experiment_result=False,whole_corpus_completed=False,sampling_steps=50,
        future_reference_conditioning=False,same_world_state_draw_across_camera_paths=True,
        candidate_completed=True,candidate_step=8,training_seed=17,
        selected_family_ids=list(FAMILIES),worlds=[0,1],cameras=[0,1],sample_indices=[0,1],sample_seeds=list(SEEDS),
        candidate_identity=training['identity'],checkpoint_sha256=training['checkpoint_sha256'],
        native_index_sha256=training['identity']['index_sha256'],
        renderer_checkpoint_sha256=training['weights']['renderer_checkpoint'],
        state_checkpoint_sha256=training['weights']['state_checkpoint'])
    if any(type(manifest.get(k))is not type(v)or manifest.get(k)!=v for k,v in expected.items()):
        raise ValueError('The exact joint8 DEV16 population or candidate changed')
    surface=manifest.get('history_surface_conditioning',{})
    if surface.get('joint_upstream_preserved')is not True or surface.get('post_import_initializer_reload')is not False:
        raise ValueError('Jointly learned history was not preserved by the sampler')
    if (done.get('completed')is not True or done.get('generated_cells')!=16 or done.get('selected_cells')!=16
            or done.get('manifest_sha256')!=sha256(root/'manifest.json')
            or manifest.get('index_sha256')!=sha256(root/'evaluation_index.json')):
        raise ValueError('All sixteen committed cells must be complete')
    families=index.get('families',[])
    if (len(families)!=2 or {r['family_id']for r in families}!=set(FAMILIES)
            or any(r.get('split')!='val'or len(r['records'])!=4
                or {(x['world'],x['camera'])for x in r['records']}!={(0,0),(0,1),(1,0),(1,1)} for r in families)):
        raise ValueError('Reference DEV16 index changed')
    sources=manifest.get('provenance',{}).get('sources',{})
    if not {'sample_candidate.py','sample_joint_feedback.py'}<={Path(name).name for name in sources}:
        raise ValueError('Typed joint sampling source provenance missing')
    for name,digest in sources.items():
        if sha256(name)!=digest:raise ValueError('Sampling source changed: '+name)
    return manifest,index


def pair_metrics(first,second,ref_first,ref_second,effective_first,effective_second,focal):
    keys=('internal_state','state_present','historical_reference_ids','historical_selected_slot_mask',
          'history_eligible_reference','initial_state','class_id','time_s','K')
    for key in keys:
        if first[key].shape!=second[key].shape:raise ValueError('Paired camera slot identity changed')
    for key in('world_state','present','K','time_s','class_id','instance_id'):
        if not np.array_equal(ref_first[key],ref_second[key]):raise ValueError('Passive camera reference changed its world')
    if not np.array_equal(ref_first['camera_to_world'][:6],ref_second['camera_to_world'][:6]):
        raise ValueError('Camera intervention changed historical poses')
    delta=np.linalg.norm(first['internal_state'][...,:3].astype(float)-second['internal_state'][...,:3],axis=-1)
    p0,r0=effective_first;p1,r1=effective_second
    full=np.broadcast_to(ref_first['present'][6:,None]&first['history_eligible_reference'][None,None],r0.shape)
    responses={}
    for name,mask in(('all_history_eligible',full),('focal_history_eligible',full&(np.arange(full.shape[-1])==focal)[None,None])):
        changed=(r0!=r1)&mask;correct=(p0==r0)&(p1==r1)&changed
        responses[name]=dict(reference_visibility_changed_steps=int(changed.sum()),
            predicted_geometry_exact_pair_response_steps=int(correct.sum()),
            predicted_geometry_changed_steps=int(((p0!=p1)&changed).sum()),
            predicted_geometry_correct_pair_fraction=None if not changed.any()else float(correct.sum()/changed.sum()))
    return dict(exact_invariance={key:bool(np.array_equal(first[key],second[key]))for key in keys},
        world_position_difference_mean_m=None if not delta.size else float(delta.mean()),
        world_position_difference_max_m=None if not delta.size else float(delta.max()),
        camera_translation_difference_max_m=float(np.linalg.norm(first['camera_to_world'][...,:3,3]-second['camera_to_world'][...,:3,3],axis=-1).max()),
        reference_camera_response=responses)


def score_panel(root,arm,training,helper_module):
    from .sample_candidate import validate_cell_marker
    manifest,index=panel_contract(root,arm,training);manifest_sha=sha256(root/'manifest.json')
    records=[];pairs=[];lineage={str(root/name):sha256(root/name)for name in('manifest.json','completion.json','evaluation_index.json')}
    for family in index['families']:
        name=family['family_id'];refs={}
        for branch in family['records']:
            path=Path(index['root'])/branch['path']
            if sha256(path)!=branch['sha256']:raise ValueError('Missing or changed reference branch')
            with np.load(path,allow_pickle=False)as archive:refs[branch['world'],branch['camera']]={k:archive[k]for k in REF_KEYS}
            lineage[str(path)]=branch['sha256']
        for world in range(2):
            for draw,seed in enumerate(SEEDS):
                arrays=[];effective=[]
                for camera in range(2):
                    marked=validate_cell_marker(root,name,world,camera,draw,manifest_sha)
                    if marked is None:raise ValueError('Uncommitted selected cell')
                    path=root/name/('world%d_camera%d_sample%02d.npz'%(world,camera,draw))
                    with np.load(path,allow_pickle=False)as archive:a={k:archive[k]for k in KEYS}
                    ref=refs[world,camera]
                    eligible=((ref['visible_fraction'][:6]>=.1)&(ref['projection_area'][:6]>=144)).any((0,1))
                    if not np.array_equal(a['history_eligible_reference'],eligible):raise ValueError('History population changed')
                    if not np.array_equal(a['historical_selected_slot_mask'],a['historical_reference_ids']>=0):raise ValueError('Historical representative mask changed')
                    if (not np.allclose(a['time_s'],ref['time_s'][6:],rtol=0,atol=1e-6)
                            or not np.allclose(a['K'],ref['K'],rtol=0,atol=1e-6)
                            or not np.isfinite(a['internal_state']).all()):raise ValueError('State, time or calibration invalid')
                    for key in('predicted_full_layers','predicted_visible_layers'):
                        if a[key].shape!=(16,2,len(a['historical_reference_ids'])+1,60,92)or not np.isfinite(a[key]).all():
                            raise ValueError('Saved target geometry layer layout changed')
                    metrics,p,r=helper_module.reference_metrics(a,ref,family['target_index'])
                    records.append(dict(family_id=name,world=world,camera=camera,draw=draw,seed=seed,
                        reference_available=True,metrics=metrics,source_sha256=marked['sha256']))
                    arrays.append(a);effective.append((p,r));lineage[str(path)]=marked['sha256']
                    lineage[str(path.with_suffix('.complete.json'))]=sha256(path.with_suffix('.complete.json'))
                pairs.append(dict(family_id=name,world=world,draw=draw,seed=seed,
                    **pair_metrics(*arrays,refs[world,0],refs[world,1],*effective,family['target_index'])))
    populations=('all_history_eligible','focal_history_eligible')
    by_camera={str(c):{pop:helper_module.aggregate_reference([r for r in records if r['camera']==c],pop)
        for pop in populations}for c in range(2)}
    unique={pop:helper_module.aggregate_reference([r for r in records if r['camera']==0],pop)for pop in populations}
    by_family={family:{str(c):{pop:helper_module.aggregate_reference(
        [r for r in records if r['family_id']==family and r['camera']==c],pop)
        for pop in populations}for c in range(2)}for family in FAMILIES}
    for data in [*by_camera.values(),unique,*(data for family in by_family.values()for data in family.values())]:
        for value in data.values():value['state_missing_rate']=None if value['state_coverage']is None else 1-value['state_coverage']
    return dict(clips=records,camera_pairs=pairs,by_camera=by_camera,
        by_family_camera=by_family,state_summary_unique_world_draw=unique,
        all_camera_states_exactly_equal=all(all(p['exact_invariance'].values())for p in pairs)),lineage


def run(plan_path,output):
    output=Path(output)
    if output.exists():raise FileExistsError('Preserve previous diagnostics; use a fresh output')
    training,lineage=training_contract(plan_path);code=helper();arms={};indices=[]
    for arm in ARMS:
        arms[arm],sources=score_panel(training[arm]['predictions'],arm,training[arm],code)
        lineage.update(sources);indices.append(read(training[arm]['predictions']/'evaluation_index.json'))
    if indices[0]!=indices[1]:raise ValueError('Matched methods use different reference population')
    for source in(__file__,HELPER,ROOT/'experiments/formal_evaluation/metrics.py',
                  ROOT/'experiments/paper_experiments/agcr_exp/metrics.py'):lineage[str(source)]=sha256(source)
    report=dict(schema=SCHEMA,completed=True,split='val',development_only=True,formal_experiment_result=False,
        protocol=dict(families=list(FAMILIES),sample_seeds=list(SEEDS),clips_per_arm=16,
            optimizer_seed=17033,sampling_stream_training_seed=17,history_frames=6,future_frames=16,
            state_accuracy_scope='Internal history-inferred shared state; not generated-image accuracy',
            state_population='All reference-present history-eligible targets; focal target separately; missing retained',
            state_camera_difference_scope='Architectural reuse of one state/noise draw across passive camera paths',
            geometry_visibility_scope='Saved conditioning M/P; not independent RGB image readout',
            geometry_rule='M/P>=0.1 and cropped P mass*64>=144 pixels and predicted presence; layer=slot+1,crop=1:91',
            statistical_scope='Two-family DEV diagnosis only; no formal TEST or superiority inference',
            ablation_scope='Upstream feedback detach versus full feedback; camera objective enabled in both'),
        arms=arms,source_sha256=lineage)
    output.mkdir(parents=True)
    from .region_probe import atomic_json
    atomic_json(output/'report.json',report)
    atomic_json(output/'completion.json',dict(schema=SCHEMA,completed=True,report_sha256=sha256(output/'report.json')))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();report=run(a.plan,a.output)
    print(json.dumps({arm:result['by_camera']for arm,result in report['arms'].items()},allow_nan=False))


if __name__=='__main__':main()

"""Fixed-size cohorts and per-target, per-frame generated-data acceptance."""
import copy
from collections import Counter
import numpy as np
from .protocol import digest, DEFAULT
from .selection import stratified_order

RULES = {
    'rgbd': {'state_error_m':'max', 'surface_error_mm':'max', 'mask_iou':'min',
             'identity_similarity':'min', 'annotation_confidence':'min',
             'paired_center_error_m':'max'},
    'rgb': {'ground_contact_error_px':'max', 'identity_similarity':'min',
            'annotation_confidence':'min', 'paired_center_error_m':'max'},
}


def fit_thresholds(validation_records, split_sha256, modality='rgbd'):
    if modality not in RULES:
        raise ValueError('Unknown annotation modality')
    if not validation_records or any(r.get('role') != 'val' for r in validation_records):
        raise ValueError('Threshold fitting requires validation records ONLY')
    if any(r.get('split_sha256') != split_sha256 or r.get('correctly_associated') is not True or r.get('source')!='rendered'
           for r in validation_records):
        raise ValueError('Calibration requires correctly associated records from the frozen split')
    readers = {r.get('annotation_reader_sha256') for r in validation_records}
    if len(readers) != 1 or None in readers:
        raise ValueError('Calibration reader provenance missing or inconsistent')
    if len({r['uid'] for r in validation_records}) != len(validation_records):
        raise ValueError('Duplicate calibration observations')
    thresholds = {}
    for key, direction in RULES[modality].items():
        # Pair statistics are applicable only to jointly visible observations.
        applicable = [r for r in validation_records if key != 'paired_center_error_m'
                      or r.get('jointly_visible') is True]
        if not applicable:
            raise ValueError('No applicable calibration observations for ' + key)
        values = np.array([r.get(key, np.nan) for r in applicable], float)
        if not np.isfinite(values).all():
            raise ValueError('Missing/nonfinite calibration statistic: ' + key)
        if np.any(values < 0) or (direction == 'min' and np.any(values > 1)):
            raise ValueError('Calibration statistic outside its declared domain: ' + key)
        thresholds[key] = dict(direction=direction,
                              value=float(np.quantile(values, .95 if direction=='max' else .05)),
                              observations=len(values))
    result = dict(schema='agcr-acceptance-2', fit_role='val', modality=modality,
                  split_sha256=split_sha256, annotation_reader_sha256=next(iter(readers)),
                  thresholds=thresholds, validation_uids=sorted(r['uid'] for r in validation_records))
    result['threshold_sha256'] = digest(result)
    return result


def _threshold_failure(value, rule):
    if not isinstance(value, (int,float,np.number)) or not np.isfinite(value):
        return True
    if value < 0 or (rule['direction']=='min' and value > 1):
        return True
    return value > rule['value'] if rule['direction']=='max' else value < rule['value']


def _clip_failures(row, thresholds):
    fail = []
    if row.get('role') != 'train': fail.append('not_generator_train_origin')
    if row.get('error_reference') != 'sampled_state': fail.append('not_sampled_state_consistency')
    if row.get('visibility_source') != 'conditioned_geometry': fail.append('visibility_source_missing')
    if row.get('frames') != 16 or row.get('views') != ['air','ground']: fail.append('not_paired_16_future_frames')
    if row.get('includes_history', True): fail.append('history_not_generated')
    if row.get('annotation_modality') != thresholds['modality']: fail.append('annotation_modality_mismatch')
    raw_ids=row.get('conditioned_target_ids', [])
    if any(k is None or isinstance(k,bool) or not isinstance(k,(str,int)) for k in raw_ids):
        fail.append('invalid_conditioned_identity')
    ids = [str(k) for k in raw_ids]
    if not ids: fail.append('no_conditioned_targets')
    if len(set(ids)) != len(ids): fail.append('duplicate_conditioned_identity')
    expected_frames = {(t,v) for t in range(16) for v in ['air','ground']}
    coverage = row.get('reader_frame_coverage', [])
    seen_frames = {(r['frame'],r['view']) for r in coverage}
    if seen_frames != expected_frames or len(coverage) != 32:
        fail.append('reader_did_not_process_all_frames')
    grid = row.get('intended_visibility', [])
    expected = {(t,v,k) for t,v in expected_frames for k in ids}
    intended = {(r['frame'],r['view'],str(r['target_id'])):r.get('visible') for r in grid}
    if len(grid)!=len(intended) or set(intended)!=expected or any(type(v) is not bool for v in intended.values()):
        fail.append('incomplete_intended_visibility')
    raw_map=row.get('history_identity_map', {})
    if any(v is None or isinstance(v,bool) or not isinstance(v,(str,int)) for v in raw_map.values()):
        fail.append('invalid_history_identity_map')
    identity_map = {str(k):str(v) for k,v in raw_map.items()}
    if set(identity_map)!=set(ids) or len(set(identity_map.values()))!=len(ids):
        fail.append('invalid_history_identity_map')
    observations = {}
    for obs in row.get('observations', []):
        key = (obs['frame'],obs['view'],str(obs.get('target_id')))
        confidence = obs.get('class_confidence')
        if not isinstance(confidence,(int,float)) or not np.isfinite(confidence) or not 0<=confidence<=1:
            fail.append('invalid_detection_confidence'); continue
        if key[:2] not in expected_frames:
            fail.append('detection_outside_clip'); continue
        if key not in intended or not intended[key]:
            if confidence > .5: fail.append('extra_or_hidden_target_detection')
            continue
        if key in observations: fail.append('duplicate_target_detection')
        observations[key] = obs
    for key, visible in intended.items():
        if not visible: continue
        obs = observations.get(key)
        if obs is None:
            fail.append('missing_visible_target'); continue
        if str(obs.get('reader_identity')) != identity_map.get(key[2]):
            fail.append('identity_inconsistency')
        for name, rule in thresholds['thresholds'].items():
            if name=='paired_center_error_m': continue
            if _threshold_failure(obs.get(name),rule): fail.append(name)
    # Compute cross-view disagreement from reader centers, not a clip-average claim.
    for t in range(16):
        for k in ids:
            a,g = (t,'air',k),(t,'ground',k)
            if not (intended.get(a) and intended.get(g)): continue
            if a not in observations or g not in observations: continue
            pa=np.asarray(observations[a].get('center_world_m',[]),float)
            pg=np.asarray(observations[g].get('center_world_m',[]),float)
            if pa.shape!=(3,) or pg.shape!=(3,) or not np.isfinite(pa).all() or not np.isfinite(pg).all():
                fail.append('missing_paired_center'); continue
            if _threshold_failure(float(np.linalg.norm(pa-pg)),thresholds['thresholds']['paired_center_error_m']):
                fail.append('paired_center_error_m')
    return sorted(set(fail))


def accept_generated(proposals, thresholds, annotation_reader_sha256):
    if thresholds.get('schema')!='agcr-acceptance-2' or thresholds.get('fit_role')!='val':
        raise ValueError('Legacy/incomplete acceptance thresholds; recalibrate on validation')
    if set(thresholds['thresholds'])!=set(RULES[thresholds['modality']]):
        raise ValueError('Missing mandatory acceptance rules')
    signature=thresholds['threshold_sha256']
    if digest({k:v for k,v in thresholds.items() if k!='threshold_sha256'})!=signature:
        raise ValueError('Threshold artifact changed after calibration')
    if thresholds['annotation_reader_sha256']!=annotation_reader_sha256:
        raise ValueError('Reader differs from the validation-calibrated reader')
    kept=[];reasons=Counter();rejected=[];uids=set()
    for row in proposals:
        if row['uid'] in uids: raise ValueError('Duplicate proposal uid')
        uids.add(row['uid'])
        if row['split_sha256']!=thresholds['split_sha256']: raise ValueError('Acceptance/split mismatch')
        if row['annotation_reader_sha256']!=annotation_reader_sha256: raise ValueError('Annotation reader mismatch')
        fail=_clip_failures(row,thresholds)
        if fail:
            reasons.update(fail);rejected.append(dict(uid=row['uid'],reasons=fail))
        else:
            kept.append(dict(copy.deepcopy(row),acceptance_schema='agcr-acceptance-2',
                             acceptance_threshold_sha256=signature))
    count=len(uids)
    return kept,dict(proposal_count=count,accepted_count=len(kept),
                     acceptance_rate=None if count==0 else len(kept)/count,
                     rejection_reasons=dict(reasons),rejected=rejected,
                     threshold_sha256=signature)


def make_cohort(real,generated,regime,total=4000,fraction=.5,seed=17,new_render=None,annotation_modality='rgbd'):
    """The sampling unit is a synchronized 16-frame clip, never an individual view."""
    def choose(rows,n,kind,allowed_role):
        if any(r['role']!=allowed_role for r in rows):raise ValueError('Wrong source role for '+kind)
        if len({r['uid'] for r in rows})!=len(rows):raise ValueError('Duplicate '+kind+' clip uid')
        if n==0:return []
        if len(rows)<n:raise ValueError('Insufficient '+kind+' clips: %d/%d; run quota audit / supplementary capture'%(len(rows),n))
        for r in rows:
            if kind=='generated':
                if r.get('acceptance_schema')!='agcr-acceptance-2' or not r.get('acceptance_threshold_sha256'):
                    raise ValueError('Generated clip did not pass complete acceptance checks')
            else:
                if r.get('start') not in DEFAULT['evaluation_starts'] or not r.get('history_covered'):
                    raise ValueError('Rendered cohort requires the four fixed history-covered windows')
            if r.get('frames')!=16 or r.get('views')!=['air','ground'] or r.get('includes_history') is not False:
                raise ValueError('Invalid paired future clip')
        ranked=stratified_order(rows,seed,limit=n)
        return [dict(r,source=kind) for r in ranked]
    # Same ranked real/generated pools yield nested fractions without cherry picking.
    if total<=0 or not 0<=fraction<=1:raise ValueError('Invalid sample budget')
    if annotation_modality not in RULES:raise ValueError('Unknown annotation modality')
    if regime=='R':result=choose(real,total,'rendered','train')
    elif regime=='G':result=choose(generated,total,'generated','train')
    elif regime=='R_G':
        ng=round(total*fraction)
        if not 0<=ng<=total:raise ValueError('Invalid generated fraction')
        result=choose(real,total-ng,'rendered','train')+choose(generated,ng,'generated','train')
    elif regime=='R_repeat':
        base=choose(real,total,'rendered','train');result=base+[dict(r,repeat=1) for r in base]
    elif regime=='R_plus_G':result=choose(real,total,'rendered','train')+choose(generated,total,'generated','train')
    elif regime=='R_plus_new_render':result=choose(real,total,'rendered','train')+choose(new_render or [],total,'additional_rendered','augmentation_train')
    else:raise ValueError(regime)
    if regime=='R_plus_new_render':
        original={r['camera_group'] for r in result if r['source']=='rendered'}
        additional={r['camera_group'] for r in result if r['source']=='additional_rendered'}
        if original & additional:raise ValueError('Additional rendering must use new camera-path groups')
    grows=[r for r in result if r['source']=='generated']
    if any(r['annotation_modality']!=annotation_modality for r in grows):raise ValueError('Annotation pipeline differs from comparison modality')
    if len({r['acceptance_threshold_sha256'] for r in grows})>1:raise ValueError('Mixed acceptance thresholds')
    signatures={r['split_sha256'] for r in result}
    if len(signatures)!=1:raise ValueError('Mixed splits in downstream training')
    return dict(schema='agcr-downstream-1',regime=regime,clips=result,clip_count=len(result),seed=seed,
                split_sha256=next(iter(signatures)),updates=40000,batch_size=16,
                modalities=['RGB','RGB-D'] if annotation_modality=='rgbd' else ['RGB'],
                same_retained_clips_for_both_modalities=annotation_modality=='rgbd',
                annotation_modality=annotation_modality,
                generated_selected=sum(r['source']=='generated' for r in result),
                retention_rate=None if not generated else sum(r['source']=='generated' for r in result)/len(generated),
                unique_episodes=len({r['episode'] for r in result}),
                unique_camera_groups=len({r['camera_group'] for r in result}),
                counts_by_source=dict(Counter(r['source'] for r in result)),
                selection='reference-metadata-stratified-nested-v1',
                cohort_sha256=digest(result))

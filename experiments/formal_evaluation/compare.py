"""Paired training-seed comparisons with scene-parent bootstrap and Holm tests."""
import argparse
import numpy as np
from .common import read_json,write_json
from .statistics import paired_cluster_bootstrap,paired_cluster_wilcoxon,holm,mean_reducer
from .tracking import combine_tracking,restore_tracking_arrays

RATIOS={'R3':('R3_success','R3_events'),
    'false_appearance_pixel_fraction':('false_appearance_pixels','fully_occluded_projection_pixels'),
    'false_appearance_frame_rate':('false_appearance_target_frames','fully_occluded_target_frames'),
    'visible_mask_IoU':('mask_intersection','mask_union'),
    'joint_detection_coverage':('jointly_detected_count','jointly_visible_count'),
    'cross_view_ID_agreement':('identity_agreement_count','jointly_detected_count'),
    'early_false_visible_fraction':('early_false_frames','early_reference_hidden_frames'),
    'late_false_visible_fraction':('late_false_frames','late_reference_hidden_frames')}


def reducer_for(metric):
    if metric in ('HOTA','IDF1','ID_switches'):
        def reduce(rows):
            values=[]
            for draw in range(8):
                stats=[restore_tracking_arrays(r['samples'][draw]['tracking_statistics']) for r in rows]
                values.append(combine_tracking(stats)[metric])
            return float(np.mean(values))
        return reduce
    if metric in RATIOS:
        n,d=RATIOS[metric]
        def reduce(rows):
            values=[]
            for draw in range(8):
                numerator=sum(r['samples'][draw]['counts'][n] for r in rows);denominator=sum(r['samples'][draw]['counts'][d] for r in rows)
                if denominator==0:return None
                values.append(numerator/denominator)
            return float(np.mean(values))
        return reduce
    if metric=='visibility_F1':
        def reduce(rows):
            out=[]
            for draw in range(8):
                c={k:sum(r['samples'][draw]['counts']['visibility_'+k] for r in rows) for k in ('tp','fp','fn')}
                den=2*c['tp']+c['fp']+c['fn']
                if not den:return None
                out.append(2*c['tp']/den)
            return float(np.mean(out))
        return reduce
    if metric in ('FVD','mAP_50_95','AP50'):raise ValueError('Corpus metric requires stored feature/detection recomputation, never average per-clip values')
    return mean_reducer(metric)


def load_reports(paths,development=False,allow_diagnostics=False):
    rows=[];seeds=[];provenance=[]
    for path in paths:
        r=read_json(path)
        if not r.get('complete'):raise ValueError('Incomplete evaluation report')
        diagnostic=bool(r.get('diagnostic_only') or r.get('generator_manifest',{}).get('diagnostic_only'))
        if diagnostic and not allow_diagnostics:raise ValueError('Diagnostic/oracle evaluation is excluded from primary comparisons')
        if not development and r.get('development_only'):raise ValueError('Development scores are not formal evidence')
        seed=r['generator_manifest']['training_seed'];seeds.append(seed);rows.extend(r['rows']);provenance.append(dict(r['reader_manifest'],diagnostic_only=diagnostic))
    if len(set(seeds))!=len(seeds):raise ValueError('Duplicate training seed')
    if not development and sorted(seeds)!=[17,29,43]:raise ValueError('Formal comparison requires all three training seeds')
    return rows,provenance


def common_center_population(left,right,metric):
    """Common matched reference keys, alongside each method's full coverage."""
    import copy
    from .statistics import record_key
    a={record_key(r):copy.deepcopy(r) for r in left};b={record_key(r):copy.deepcopy(r) for r in right}
    if set(a)!=set(b):raise ValueError('Unpaired conditional-error populations')
    view='air' if metric.startswith('air_') else 'ground' if metric.startswith('ground_') else None
    final=metric.endswith('FDE_m');counts=dict(reference=0,left_matched=0,right_matched=0,common_matched=0)
    for key in a:
        av=[];bv=[]
        for draw in range(8):
            def entries(row):
                items=row['samples'][draw]['per_reference_centers']
                return {(x['frame'],x['view'],x['reference_id']):x for x in items if (view is None or x['view']==view) and (not final or x['frame']==21)}
            aa,bb=entries(a[key]),entries(b[key])
            if set(aa)!=set(bb):raise ValueError('Methods changed reference target/frame set')
            shared=[k for k in aa if aa[k]['matched'] and bb[k]['matched']]
            counts['reference']+=len(aa);counts['left_matched']+=sum(x['matched'] for x in aa.values());counts['right_matched']+=sum(x['matched'] for x in bb.values());counts['common_matched']+=len(shared)
            av.append(None if not shared else float(np.mean([aa[k]['center_error_m'] for k in shared])))
            bv.append(None if not shared else float(np.mean([bb[k]['center_error_m'] for k in shared])))
        a[key]['metrics'][metric]=None if any(x is None for x in av) else float(np.mean(av))
        b[key]['metrics'][metric]=None if any(x is None for x in bv) else float(np.mean(bv))
    return list(a.values()),list(b.values()),counts


def validate_shared_reader(left,right):
    fields=('checkpoint_sha256','association_sha256','reader_postprocessing','reader_source_sha256')
    if any(left.get(key)!=right.get(key) for key in fields):
        raise ValueError('Methods must share frozen reader weights, thresholds and inference implementation')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--left',nargs='+',required=True);p.add_argument('--right',nargs='+',required=True)
    p.add_argument('--metrics',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--development',action='store_true');p.add_argument('--replicates',type=int,default=2000)
    p.add_argument('--allow-oracle-diagnostics',action='store_true',help='Explicit diagnostic comparison, kept outside primary hypothesis tables')
    a=p.parse_args()
    if not a.development and a.replicates!=2000:raise ValueError('Formal bootstrap uses 2000 replicates')
    left,lp=load_reports(a.left,a.development,a.allow_oracle_diagnostics);right,rp=load_reports(a.right,a.development,a.allow_oracle_diagnostics)
    diagnostic=any(p.get('diagnostic_only') for p in lp+rp)
    for l,r in zip(lp,rp):
        validate_shared_reader(l,r)
    results=[];pvalues=[]
    for metric in a.metrics:
        reducer=reducer_for(metric);aa,bb=left,right;coverage=None
        if metric.endswith('_center_ADE_m') or metric.endswith('_center_FDE_m'):
            aa,bb,coverage=common_center_population(left,right,metric)
        try:
            boot=paired_cluster_bootstrap(aa,bb,reducer,a.replicates)
            test=paired_cluster_wilcoxon(aa,bb,reducer);pvalues.append(test['pvalue'])
            results.append(dict(metric=metric,status='ok',**boot,wilcoxon=test,conditional_common_population=coverage))
        except ValueError as exc:
            results.append(dict(metric=metric,status='not_estimable',reason=str(exc),ci95=None));pvalues.append(None)
    # Planned family size remains the number of requested primary hypotheses,
    # including non-estimable ones conservatively assigned p=1 for correction.
    adjusted=holm([1. if p is None else p for p in pvalues])
    for row,p in zip(results,adjusted):row['holm_adjusted_p']=None if row['status']!='ok' else p
    write_json(a.output,dict(schema='agcr-diagnostic-paired-tests-1' if diagnostic else 'agcr-paired-primary-tests-1',
        diagnostic_only=diagnostic,development_only=a.development,multiple_testing='Holm',hypothesis_family=a.metrics,comparisons=results))
if __name__=='__main__':main()

"""Paired data-regime comparisons, recomputing tracking/event statistics."""
import argparse
import numpy as np
from .common import read_json,write_json
from .statistics import paired_cluster_bootstrap,paired_cluster_wilcoxon,holm
from .tracking import combine_tracking,restore_tracking_arrays


def reducer(metric):
    if metric in ('HOTA','IDF1','ID_switches'):
        return lambda rows:combine_tracking([restore_tracking_arrays(r['tracking_statistics']) for r in rows])[metric]
    if metric=='R3':
        def f(rows):
            n=sum(r['events']['R3_success'] for r in rows);d=sum(r['events']['R3_events'] for r in rows)
            return None if not d else n/d
        return f
    if metric=='entry_F1':
        def f(rows):
            if any(r['entry'] is None for r in rows):return None
            c={k:sum(r['entry'][k] for r in rows) for k in ('tp','fp','fn')};den=2*c['tp']+c['fp']+c['fn']
            return None if not den else 2*c['tp']/den
        return f
    raise ValueError('Select HOTA,IDF1,ID_switches,R3 or entry_F1; mAP requires detection-level COCO resampling')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--left',nargs='+',required=True);p.add_argument('--right',nargs='+',required=True)
    p.add_argument('--metrics',nargs='+',default=['R3','IDF1']);p.add_argument('--output',required=True);p.add_argument('--development',action='store_true');p.add_argument('--replicates',type=int,default=2000);a=p.parse_args()
    def load(paths):
        rows=[];seeds=[];calibration={}
        for path in paths:
            report=read_json(path)
            if not report.get('complete') or (report.get('development_only') and not a.development):raise ValueError('Complete formal report required')
            seeds.append(report['training_seed']);rows.extend(report['rows']);calibration[report['training_seed']]=report['association_sha256']
        if len(set(seeds))!=len(seeds) or (not a.development and sorted(seeds)!=[17,29,43]):raise ValueError('Three unique matched training seeds required')
        return rows,calibration
    left,lcal=load(a.left);right,rcal=load(a.right)
    if lcal!=rcal:raise ValueError('Regimes must share R-only validation association thresholds')
    if not a.development and a.replicates!=2000:raise ValueError('Formal bootstrap replicates=2000')
    output=[];pvalues=[]
    for metric in a.metrics:
        reduce=reducer(metric)
        try:
            estimate=paired_cluster_bootstrap(left,right,reduce,a.replicates);test=paired_cluster_wilcoxon(left,right,reduce)
            output.append(dict(metric=metric,status='ok',**estimate,wilcoxon=test));pvalues.append(test['pvalue'])
        except ValueError as exc:output.append(dict(metric=metric,status='not_estimable',reason=str(exc)));pvalues.append(1.)
    for r,pvalue in zip(output,holm(pvalues)):r['holm_adjusted_p']=pvalue if r['status']=='ok' else None
    write_json(a.output,dict(schema='agcr-downstream-paired-tests-1',development_only=a.development,hypothesis_family=a.metrics,multiple_testing='Holm',comparisons=output))
if __name__=='__main__':main()

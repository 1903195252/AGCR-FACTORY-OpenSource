"""Scene-parent cluster inference; all descendants and eight draws stay together."""
from collections import defaultdict
import numpy as np

def holm(pvalues):
    p=np.asarray(pvalues,float)
    if p.ndim!=1 or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid p-values')
    order=np.argsort(p);q=np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))).clip(0,1)
    out=np.empty_like(p);out[order]=q;return out.tolist()

def record_key(r):
    return (str(r['layout']),str(r['scene_parent_id']),str(r['unit_id']),int(r['training_seed']))

def paired_cluster_bootstrap(left,right,reducer,replicates=2000,seed=17029):
    """reducer computes a metric from all records of one layout and seed.

    Tracking reducers must recombine official sufficient statistics. Each seed
    is evaluated separately then seeds and layouts receive equal weight. Missing
    predictions with a positive reference population are never discarded.
    """
    a={record_key(r):r for r in left};b={record_key(r):r for r in right}
    if not a or len(a)!=len(left) or len(b)!=len(right) or set(a)!=set(b):raise ValueError('Paired full populations required')
    groups=defaultdict(lambda:defaultdict(list))
    for key in sorted(a):groups[key[0]][key[1]].append(key)
    seeds=sorted({k[3] for k in a})
    for layout,parents in groups.items():
        for parent,keys in parents.items():
            units=defaultdict(set)
            for k in keys:units[k[2]].add(k[3])
            if any(s!=set(seeds) for s in units.values()):raise ValueError('Every unit must retain all paired training seeds')
    def value(collection,draw):
        values=[]
        for keys in draw:
            per_seed=[]
            for training_seed in seeds:
                selected=[collection[k] for k in keys if k[3]==training_seed]
                v=reducer(selected)
                if v is None or not np.isfinite(v):raise ValueError('Undefined positive-population metric; report coverage/failure instead of dropping it')
                per_seed.append(v)
            values.append(np.mean(per_seed))
        return float(np.mean(values))
    whole=[[k for values in parents.values() for k in values] for parents in groups.values()]
    av,bv=value(a,whole),value(b,whole);rng=np.random.default_rng(seed);d=[]
    for _ in range(replicates):
        draw=[]
        for parents in groups.values():
            keys=sorted(parents);take=rng.integers(0,len(keys),len(keys))
            draw.append([k for index in take for k in parents[keys[index]]])
        d.append(value(a,draw)-value(b,draw))
    return dict(left=av,right=bv,difference=av-bv,ci95=np.quantile(d,[.025,.975]).tolist(),
        replicates=replicates,bootstrap_seed=seed,cluster='scene_parent_within_layout',layout_weighting='equal',
        training_seeds=seeds,parents={k:len(v) for k,v in groups.items()})

def paired_cluster_wilcoxon(left,right,reducer):
    from scipy.stats import wilcoxon
    a={record_key(r):r for r in left};b={record_key(r):r for r in right}
    if not a or len(a)!=len(left) or len(b)!=len(right) or set(a)!=set(b):raise ValueError('Unpaired populations')
    groups=defaultdict(list)
    for k in a:groups[k[:2]].append(k)
    differences=[]
    for keys in groups.values():
        seeds=sorted({k[3] for k in keys});ds=[]
        for s in seeds:
            selected=[k for k in keys if k[3]==s]
            x,y=reducer([a[k] for k in selected]),reducer([b[k] for k in selected])
            if x is None or y is None:raise ValueError('Undefined cluster value')
            ds.append(x-y)
        differences.append(np.mean(ds))
    d=np.asarray(differences)
    if not np.isfinite(d).all():raise ValueError('Nonfinite cluster difference')
    if np.all(d==0):return dict(statistic=0.,pvalue=1.,parents=len(d))
    r=wilcoxon(d,zero_method='wilcox',alternative='two-sided')
    return dict(statistic=float(r.statistic),pvalue=float(r.pvalue),parents=len(d))

def mean_reducer(metric):
    def reduce(rows):
        v=[r['metrics'][metric] for r in rows]
        return None if any(x is None for x in v) else float(np.mean(v))
    return reduce

def ratio_reducer(numerator,denominator):
    def reduce(rows):
        n=sum(r['counts'][numerator] for r in rows);d=sum(r['counts'][denominator] for r in rows)
        return None if d==0 else n/d
    return reduce

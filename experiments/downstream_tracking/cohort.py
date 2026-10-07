"""Build manuscript-matched immutable data regimes using existing acceptance checks."""
import argparse,json
from pathlib import Path
from experiments.paper_experiments.agcr_exp.downstream import make_cohort
from experiments.paper_experiments.agcr_exp.selection import rendered_candidates
from .data import sha256
from experiments.paper_experiments.agcr_exp.selection import stratified_order
from experiments.paper_experiments.agcr_exp.protocol import digest
from collections import Counter


def make_rgb_cohort(real,generated,regime,total,seed,new_render=None):
    def choose(rows,n,source,role):
        if len(rows)<n:raise ValueError('Insufficient '+source+' clips; no duplicate padding permitted')
        if len({r['uid'] for r in rows})!=len(rows):raise ValueError('Duplicate candidate uid')
        if any(r['role']!=role for r in rows):raise ValueError('Training split violation')
        if source=='generated' and any(r.get('acceptance_schema')!='agcr-acceptance-3-rgb' for r in rows):
            raise ValueError('External comparisons require the common trajectory-free RGB annotation protocol for every method')
        return [dict(r,source=source) for r in stratified_order(rows,seed,limit=n)]
    if regime=='R':clips=choose(real,total,'rendered','train')
    elif regime=='G':clips=choose(generated,total,'generated','train')
    elif regime=='R_G':clips=choose(real,total-total//2,'rendered','train')+choose(generated,total//2,'generated','train')
    elif regime=='R_repeat':
        base=choose(real,total,'rendered','train');clips=base+[dict(r,repeat=1) for r in base]
    elif regime=='R_plus_G':clips=choose(real,total,'rendered','train')+choose(generated,total,'generated','train')
    elif regime=='R_plus_new_render':
        clips=choose(real,total,'rendered','train')+choose(new_render or [],total,'additional_rendered','augmentation_train')
        if {r['camera_group'] for r in clips if r['source']=='rendered'}&{r['camera_group'] for r in clips if r['source']=='additional_rendered'}:
            raise ValueError('Additional rendering must use independent camera groups')
    else:raise ValueError('Unknown regime')
    signatures={r['split_sha256'] for r in clips}
    if len(signatures)!=1:raise ValueError('Inconsistent frozen split')
    return dict(schema='agcr-downstream-2-rgb',regime=regime,clips=clips,clip_count=len(clips),seed=seed,
        split_sha256=next(iter(signatures)),updates=40000,batch_size=16,modalities=['RGB'],annotation_modality='rgb',
        annotation_protocol='independent_rgb_trajectory_free_v1',counts_by_source=dict(Counter(r['source'] for r in clips)),
        cohort_sha256=digest(clips),trajectory_filter=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--accepted-generated')
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=17);p.add_argument('--total',type=int,default=4000)
    p.add_argument('--regime',choices=['R','G','R_G','R_repeat','R_plus_G','R_plus_new_render'],required=True)
    p.add_argument('--annotation-modality',choices=['rgb','rgbd'],default='rgbd');a=p.parse_args()
    index=json.loads(Path(a.index).read_text());generated=[]
    if a.accepted_generated:
        document=json.loads(Path(a.accepted_generated).read_text());generated=document['accepted']
        for row in generated:
            if row.get('development_only') and not index['development_only']:raise ValueError('Development generation cannot enter a formal cohort')
            for k in ('rgbd_path','annotation_path','annotation_sha256'):
                if k not in row:raise ValueError('Accepted proposal lacks '+k)
            if sha256(row['annotation_path'])!=row['annotation_sha256']:raise ValueError('Annotation changed')
    if a.annotation_modality=='rgb':
        cohort=make_rgb_cohort(rendered_candidates(index),generated,a.regime,a.total,a.seed,rendered_candidates(index,'augmentation_train'))
    else:
        cohort=make_cohort(rendered_candidates(index),generated,a.regime,a.total,.5,a.seed,
                           rendered_candidates(index,'augmentation_train'),a.annotation_modality)
    cohort.update(index_sha256=sha256(a.index),development_only=index['development_only'])
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(cohort,indent=2)+'\n')
    print(json.dumps({k:v for k,v in cohort.items() if k!='clips'}))

if __name__=='__main__':main()

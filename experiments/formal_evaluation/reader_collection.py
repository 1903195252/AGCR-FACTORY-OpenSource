"""Dedicated reader-domain repair; separate data, candidate weights and budget.

This never changes a generator, an existing checkpoint, its training identity,
or test thresholds. Completed candidates still require independent acceptance.
"""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import random
from pathlib import Path
import threading
import time

import numpy as np

from .common import read_json,write_json,sha256
from .runtime import Budget
from .clean_reference import select_families

SCHEMA='agcr-independent-reader-domain-repair-1'


def immutable(path,value):
    path=Path(path)
    if path.exists() and read_json(path)!=value:raise ValueError('Changed registered reader-repair contract')
    write_json(path,value)


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def plan_reader_families(ordinary,excluded,output):
    """Sixteen new complete factorial families, balanced over class/direction."""
    occupied={r['scene_parent_id'] for r in excluded['families']}
    layouts=sorted({int(r['layout']) for r in ordinary['windows'] if r['role']=='evaluation_reader_train'})
    if len(layouts)<8:raise ValueError('Need at least eight native TRAIN layouts')
    selected=[layouts[k*(len(layouts)-1)//7] for k in range(8)]
    jobs=[];classes=('worker','forklift','agv','uav')
    for k,layout in enumerate(selected):
        for receiver in (0,1):
            serial=1000+2*k+receiver
            parent='native_layout_%02d_intervention_%03d'%(layout,serial)
            if parent in occupied:raise ValueError('Dedicated reader ancestry already used by generator')
            jobs.append(dict(dataset_root=ordinary['dataset_root'],output=str(Path(output).resolve()),
                layout=layout,split='train',cohort=('evidence','ambiguous')[k//4],receiver=receiver,
                serial=serial,target_class=classes[k%4],occluder=('rack','ground_structure')[(k+receiver)%2],
                margin=.3,gpu=0,reader_role='evaluation_reader_adaptation_train',scene_parent_id=parent))
    return jobs


def verify_exclusion(index,excluded,new_audit,old_audit):
    if not index.get('audit_passed') or not new_audit.get('passed') or not old_audit.get('passed'):
        raise ValueError('Full original pixel/control audit required')
    if len(index['families'])!=16:raise ValueError('Dedicated reader corpus requires exactly sixteen complete families')
    if any(r['split']!='train' for r in index['families']):raise ValueError('Reader adaptation cannot consume validation or test families')
    old_ids={r['family_id'] for r in excluded['families']};old_parents={r['scene_parent_id'] for r in excluded['families']}
    if any(r['family_id'] in old_ids or r['scene_parent_id'] in old_parents for r in index['families']):
        raise ValueError('Reader/generator family ancestry overlap')
    audited={r['family_id']:r for r in old_audit['families']}
    if old_ids-set(audited):raise ValueError('Original history audit does not cover excluded generator corpus')
    signatures={tuple(sorted(h.items())) for r in old_audit['families'] for h in r['history_sha256'].values()}
    new_rows={r['family_id']:r for r in new_audit['families']}
    if set(new_rows)!={r['family_id'] for r in index['families']}:raise ValueError('Incomplete new-family audit')
    for row in new_rows.values():
        if not row.get('passed') or len(row['history_sha256'])!=4:raise ValueError('Missing four-cell history audit')
        if any(tuple(sorted(h.items())) in signatures for h in row['history_sha256'].values()):
            raise ValueError('Dedicated reader history duplicates existing generator history')
    return dict(passed=True,independent_scene_parents=16,complete_cells=64,
        no_generator_family_overlap=True,no_generator_history_duplicate=True,
        new_family_checks='original audit_family/audit_manifest including pixel, visibility, camera and dynamics controls',
        original_history_source='previous complete source audit, frozen by SHA in collection manifest')


def collect(a):
    from experiments.paired_interventions.collect import bounded_capture
    from experiments.paired_interventions.data import build_family_index
    ordinary=read_json(a.ordinary_index);excluded=read_json(a.exclude_index)
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    if root==Path(excluded['root']).resolve() or root==Path(ordinary['dataset_root']).resolve():raise ValueError('Separate reader dataset output required')
    jobs=plan_reader_families(ordinary,excluded,root)
    contract=dict(schema=SCHEMA,scope='dedicated_reader_train_collection',ordinary_index_sha256=sha256(a.ordinary_index),
        excluded_index_sha256=sha256(a.exclude_index),excluded_audit_sha256=sha256(a.exclude_audit),jobs=jobs,
        source_sha256={str(p):sha256(p) for p in [Path(__file__),Path(__file__).parent.parent/'paired_interventions/collect.py',
            Path(__file__).parent.parent/'paired_interventions/plan_families.py',Path(__file__).parent.parent/'paired_interventions/render_family.py',
            Path(__file__).parent.parent/'paired_interventions/audit.py']})
    immutable(root/'reader_collection_manifest.json',contract)
    budget=Budget(a.gpu_hours);stopping=[False];finished=threading.Event()
    def watch():
        while not finished.wait(.25):
            # Do not schedule another ~80s family too near the external cap.
            if budget.expired() or budget.elapsed_hours()*3600>=a.gpu_hours*3600-110 or Path(a.stop_file).exists():stopping[0]=True
    threading.Thread(target=watch,daemon=True).start();results=[]
    try:
        for row in bounded_capture(jobs,a.workers,1,stopping):
            results.append(row);write_json(root/'status.json',dict(state='collecting',finished=len(results),families=16,results=results))
            print(json.dumps({k:v for k,v in row.items() if k!='traceback'}),flush=True)
    finally:finished.set()
    if len(results)!=16 or any(not r['passed'] for r in results):
        write_json(root/'completion.json',dict(completed=False,state='paused_or_rejected',finished=len(results)));return
    # Retain the original full audit; these dedicated few-shot families do not
    # claim to satisfy the unrelated 512-family generator collection quota.
    index=build_family_index(root,root/'index.building.json',development=True)
    audit=read_json(root/'global_audit.json');report=verify_exclusion(index,excluded,audit,read_json(a.exclude_audit))
    index.update(reader_role='evaluation_reader_adaptation_train',dedicated_reader_corpus=True,
                 exclusion=report,collection_contract_sha256=sha256(root/'reader_collection_manifest.json'))
    write_json(root/'index.json',index);write_json(root/'exclusion_audit.json',report)
    write_json(root/'completion.json',dict(completed=True,families=16,cells=64,index_sha256=sha256(root/'index.json'),diagnostic_only=True))

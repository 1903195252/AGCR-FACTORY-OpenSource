#!/usr/bin/env python3
"""Report actual accepted counts separately from prescribed quotas."""
import argparse,json,time
from pathlib import Path
from collections import Counter
from paper_dataset.common import atomic_json
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();r=Path(a.root)
manifest=json.loads((r/'manifest.json').read_text());plans=json.loads((r/'plans.json').read_text())
planned=Counter((j['split'],j['pool'],j['event']) for j in plans);accepted=Counter();total=0;bytes_=0;seconds=0
for file in (r/'episodes').glob('*/complete.json'):
    m=json.loads(file.read_text());accepted[m['split'],m['pool'],m['event']]+=1
    total+=1;bytes_+=m['data_bytes'];seconds+=m['seconds']
report=dict(checked_unix=time.time(),purpose=manifest['purpose'],accepted_episodes=total,
    planned_episodes=len(plans),synchronous_timestamps=total*manifest['config']['frames_per_episode'],
    rgb_images=total*manifest['config']['frames_per_episode']*2,data_gib=bytes_/1024**3,
    mean_episode_worker_seconds=seconds/max(1,total),rejected_records=len(list((r/'rejected').glob('*.json'))),
    quotas=[dict(split=k[0],pool=k[1],event=k[2],planned=v,accepted=accepted[k],missing=v-accepted[k]) for k,v in planned.items()])
atomic_json(r/'progress.json',report);print(json.dumps(report,indent=2))

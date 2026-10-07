"""Immutable provenance and protocol contracts, independent of generator modules."""
import hashlib
import json
from pathlib import Path
import numpy as np

SAMPLE_SEEDS=(101,211,307,401,503,601,701,809)
TRAINING_SEEDS=(17,29,43)
VIEWS=('air','ground')
NATIVE_CLASSES={3:1,4:2,13:3,14:4}

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def jsonable(x):
    if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,np.ndarray):return jsonable(x.tolist())
    if isinstance(x,np.generic):return jsonable(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(jsonable(value),indent=2,allow_nan=False)+'\n');q.replace(p)

def read_json(path):return json.loads(Path(path).read_text())
def read_jsonl(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def write_jsonl(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(''.join(json.dumps(jsonable(r),allow_nan=False)+'\n' for r in rows));q.replace(p)

ORACLE_CONDITIONS={'oracle_state','oracle_state_and_geometry'}


def require_completed_corpus(root):
    """Accept the native and external atomic marker names, never file presence alone."""
    markers=[Path(root)/name for name in ('completion.json','completed.json')]
    found=[read_json(p) for p in markers if p.is_file()]
    if not found or any(m.get('completed') is not True for m in found):
        raise ValueError('Incomplete corpus: completed=true marker required')
    return True


def prediction_manifest(root,development=False,allow_oracle_diagnostics=False):
    m=read_json(Path(root)/'manifest.json')
    seeds=m.get('sample_seeds',[])
    if not development and (seeds!=list(SAMPLE_SEEDS) or m.get('development_only')):
        raise ValueError('Formal evaluation requires eight fixed seeds and formal predictions')
    if not seeds or len(set(seeds))!=len(seeds) or any(s not in SAMPLE_SEEDS for s in seeds):
        raise ValueError('Missing/unknown sample seeds')
    if not m.get('checkpoint_sha256'):raise ValueError('Generator checkpoint provenance required')
    if m.get('conditions') in ORACLE_CONDITIONS:
        if not allow_oracle_diagnostics:raise ValueError('Oracle conditioning requires explicit --allow-oracle-diagnostics; excluded from primary predictions')
        if (m.get('diagnostic_only') is not True or m.get('downstream_augmentation') is not False
            or m.get('eligible_population')!='all_history_eligible' or not m.get('condition_source_sha256')
            or m.get('future_rgb_input') is not False):
            raise ValueError('Oracle diagnostic provenance/population restrictions missing')
    elif m.get('conditions') not in ('history_inferred_state_only','history_only','history_predicted_scene'):
        raise ValueError('Formal generation must exclude future reference conditioning')
    return m

def frame_key(r):return (str(r['episode']),int(r['frame']),str(r['view']))

def frame_index(rows):
    out={frame_key(r):r for r in rows}
    if len(out)!=len(rows):raise ValueError('Duplicate episode/frame/view')
    for r in rows:
        if r['view'] not in VIEWS:raise ValueError('Unknown view')
        ids=[str(d['track_id']) for d in r['detections']]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate track in a camera frame')
        for d in r['detections']:
            b=np.asarray(d['box_xyxy'],float)
            if b.shape!=(4,) or not np.isfinite(b).all() or (b[2:]<b[:2]).any():raise ValueError('Invalid pixel box')
    return out

def require_reader_provenance(manifest,role='video_evaluation',development=False):
    if manifest.get('role')!=role:raise ValueError('Reader role mismatch')
    if not manifest.get('checkpoint_sha256') or not manifest.get('training_partition_sha256'):
        raise ValueError('Independent reader checkpoint and partition provenance required')
    if manifest.get('generator_auxiliary_reader',False):raise ValueError('Generator auxiliary reader cannot score image quality')
    if manifest.get('future_reference_input',False):raise ValueError('Future reference leakage in visual inference')
    if not development and (manifest.get('development_only') or not manifest.get('completed_training')):
        raise ValueError('Formal reader must have completed its independent training')

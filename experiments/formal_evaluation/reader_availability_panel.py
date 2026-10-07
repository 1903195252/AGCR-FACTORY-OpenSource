"""Freeze all available complete draw-zero families by provenance, not quality."""
import argparse,time
from pathlib import Path
from .common import read_json,write_json,sha256

SCHEMA='agcr-diagnostic-available-family-panel-1'
CELL_FILES=tuple('world%d_camera%d_sample00.npz'%(w,c) for w in range(2) for c in range(2))


def inspect_family(root,family_id):
    folder=Path(root)/family_id;marker=folder/'draw_00.json'
    if not marker.exists():raise ValueError('draw_zero_marker_missing')
    record=read_json(marker)
    if set(record.get('files',{}))!=set(CELL_FILES):raise ValueError('draw_zero_requires_exactly_four_cells')
    for name in CELL_FILES:
        path=folder/name
        if not path.is_file():raise ValueError('cell_missing:'+name)
        if sha256(path)!=record['files'][name]:raise ValueError('cell_checksum_mismatch:'+name)
    return dict(family_id=family_id,marker_sha256=sha256(marker),files={n:record['files'][n] for n in CELL_FILES})


def create_panel(index_path,predictions,output):
    if Path(output).exists():raise ValueError('Availability snapshots are immutable; use a new file')
    index=read_json(index_path);manifest=read_json(Path(predictions)/'manifest.json')
    if manifest['split']!='test' or manifest['index_sha256']!=sha256(index_path):raise ValueError('Use registered test corpus and original index')
    ids=manifest['selected_family_ids'];allowed={f['family_id'] for f in index['families'] if f['split']=='test'}
    if not ids or len(ids)!=len(set(ids)) or not set(ids)<=allowed:raise ValueError('Invalid registered family list')
    accepted=[];unavailable=[]
    for family_id in ids:
        try:accepted.append(inspect_family(predictions,family_id))
        except (ValueError,FileNotFoundError) as exc:unavailable.append(dict(family_id=family_id,reason=str(exc)))
    if not accepted:raise ValueError('No complete draw-zero family is available')
    panel=dict(schema=SCHEMA,diagnostic_only=True,partial_corpus=True,created_unix=time.time(),
        selection='all complete draw-zero four-cell groups available at snapshot; no scores/images used',
        generator_manifest_sha256=sha256(Path(predictions)/'manifest.json'),index_sha256=sha256(index_path),
        sample_index=0,sample_seed=manifest['sample_seeds'][0],registered_family_ids=ids,
        selected_family_ids=[r['family_id'] for r in accepted],families=accepted,unavailable=unavailable,
        expected_cells=4*len(accepted),registered_total_cells=4*len(ids),formal_corpus_completed=False,
        source_sha256=sha256(__file__))
    write_json(output,panel);return panel


def validate_panel(path,predictions,index_path,split,rerun):
    if split!='test' or not rerun:raise ValueError('Availability panel is only for explicit diagnostic test-pixel reruns')
    panel=read_json(path);manifest=read_json(Path(predictions)/'manifest.json')
    if panel.get('schema')!=SCHEMA or panel.get('diagnostic_only') is not True or panel.get('formal_corpus_completed') is not False:
        raise ValueError('Not a diagnostic availability snapshot')
    if panel['generator_manifest_sha256']!=sha256(Path(predictions)/'manifest.json') or panel['index_sha256']!=sha256(index_path):
        raise ValueError('Availability panel corpus or index changed')
    if panel['registered_family_ids']!=manifest['selected_family_ids'] or panel['sample_seed']!=manifest['sample_seeds'][0] or panel['sample_index']!=0:
        raise ValueError('Availability panel changed original registration')
    ids=panel['selected_family_ids']
    if not ids or len(ids)!=len(set(ids)) or ids!=[r['family_id'] for r in panel['families']] or not set(ids)<=set(panel['registered_family_ids']):
        raise ValueError('Invalid selected whole families')
    if panel['expected_cells']!=4*len(ids) or panel['registered_total_cells']!=4*len(panel['registered_family_ids']):
        raise ValueError('Incorrect partial/full population denominator')
    for record in panel['families']:
        if inspect_family(predictions,record['family_id'])!=record:raise ValueError('Availability snapshot pixels or marker changed')
    return panel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('index','predictions','output'):p.add_argument('--'+name,required=True)
    a=p.parse_args();panel=create_panel(a.index,a.predictions,a.output)
    print('diagnostic panel: %d cells of %d registered; draw zero only'%(panel['expected_cells'],panel['registered_total_cells']))


if __name__=='__main__':main()

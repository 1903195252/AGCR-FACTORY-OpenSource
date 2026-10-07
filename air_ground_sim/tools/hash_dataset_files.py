"""Create per-scene SHA-256 lists; optionally follow a running scene exporter."""
import argparse
import hashlib
import json
from pathlib import Path
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--dataroot',type=Path,required=True)
p.add_argument('--follow',action='store_true')
a=p.parse_args();root=a.dataroot
out=root/'checksums';out.mkdir(exist_ok=True)
seen=set();files_total=0;bytes_total=0
while True:
    for chunk in sorted((root/'chunks').glob('*.json')):
        if chunk.stem in seen:continue
        records=json.loads(chunk.read_text());scene=records['tables']['scene'][0]['name']
        ext=json.loads((root/'extensions'/f'{scene}.json').read_text())
        paths=[r['filename'] for r in records['tables']['sample_data']]
        paths.extend(r[k] for r in ext for k in ('semantic_filename','instance_filename'))
        assert len(paths)==len(set(paths))
        lines=[]
        for relative in sorted(paths):
            file=root/relative;h=hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
            lines.append(h.hexdigest()+'  '+relative+'\n');bytes_total+=file.stat().st_size
        tmp=out/(scene+'.sha256.tmp');tmp.write_text(''.join(lines));tmp.replace(out/(scene+'.sha256'))
        seen.add(chunk.stem);files_total+=len(paths)
        print(f'Hashed {len(seen)} scenes, {files_total} image/label files',flush=True)
    manifest=root/'manifest.json'
    complete=manifest.exists() and json.loads(manifest.read_text()).get('status')=='complete'
    if complete or not a.follow:break
    time.sleep(5)
report=dict(status='complete' if complete else 'partial',hashed_scenes=len(seen),files=files_total,bytes=bytes_total,algorithm='SHA-256',relative_path_base='dataset root')
if complete:
    m=json.loads(manifest.read_text());assert files_total==m['images']*3
(root/'integrity_report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

"""Fetch the official RT-DETR-R18 safetensors, preserving the exact Hub revision."""
import argparse,hashlib,json,urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--endpoint',default='https://huggingface.co');a=p.parse_args()
    base=a.endpoint.rstrip('/');repo='PekingU/rtdetr_r18vd';root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(base+'/api/models/'+repo,timeout=45) as f:metadata=json.load(f)
    revision=metadata['sha'];files={}
    for name in ('config.json','preprocessor_config.json','model.safetensors'):
        path=root/name
        if not path.exists():
            tmp=path.with_suffix(path.suffix+'.part')
            with urllib.request.urlopen(base+'/'+repo+'/resolve/'+revision+'/'+name,timeout=120) as response,tmp.open('wb') as out:
                while True:
                    chunk=response.read(8*1024*1024)
                    if not chunk:break
                    out.write(chunk)
            tmp.replace(path)
        files[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    provenance=dict(repository=repo,revision=revision,files=files,source='https://huggingface.co/'+repo,
        official_architecture='RT-DETR-R18',weights_scope='COCO pretrained detector; tracking heads initialized independently')
    (root/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(provenance))

if __name__=='__main__':main()

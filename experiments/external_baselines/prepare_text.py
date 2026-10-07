"""Cache the official CogVideoX empty-string text condition, with source hashes."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import urllib.parse
import urllib.request


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def download(root, workers):
    repo='ZhipuAI/CogVideoX-5b-I2V'
    base='https://modelscope.cn/api/v1/models/'+repo
    with urllib.request.urlopen(base+'/repo/files?Revision=master&Recursive=true',timeout=30) as response:
        entries=json.load(response)['Data']['Files']
    entries=[x for x in entries if x['Type']=='blob' and x['Path'].startswith(('text_encoder/','tokenizer/'))]
    if not entries or any(not x.get('Sha256') for x in entries):
        raise ValueError('Official text assets require published content hashes')
    root.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for item in entries:
        path=root/item['Path'];path.parent.mkdir(parents=True,exist_ok=True)
        if path.is_file() and path.stat().st_size==item['Size'] and sha256(path)==item['Sha256']:
            continue
        pieces=path.with_name(path.name+'.parts');pieces.mkdir(exist_ok=True)
        for start in range(0,item['Size'],64*1024*1024):
            end=min(start+64*1024*1024,item['Size'])-1
            jobs.append((item,start,end,pieces/('%d-%d'%(start,end))))
    def get(job):
        item,start,end,path=job
        if path.exists() and path.stat().st_size==end-start+1:
            return
        url=base+'/repo?'+urllib.parse.urlencode(dict(Revision='master',FilePath=item['Path']))
        for attempt in range(4):
            try:
                req=urllib.request.Request(url,headers={'Range':'bytes=%d-%d'%(start,end)})
                with urllib.request.urlopen(req,timeout=45) as response:
                    if response.status!=206 or response.headers.get('Content-Range')!='bytes %d-%d/%d'%(start,end,item['Size']):
                        if not (start==0 and end==item['Size']-1 and response.status==200):
                            raise ValueError('Asset server returned the wrong byte range')
                    temp=path.with_suffix('.tmp')
                    with temp.open('wb') as output:
                        shutil.copyfileobj(response,output,4*1024*1024)
                if temp.stat().st_size!=end-start+1:
                    raise ValueError('Incomplete asset range')
                temp.replace(path)
                return
            except Exception:
                if attempt==3:
                    raise
                time.sleep(2**attempt)
    with ThreadPoolExecutor(workers) as pool:
        for count,future in enumerate(as_completed([pool.submit(get,job) for job in jobs]),1):
            future.result()
            if count%16==0:
                print(json.dumps(dict(ranges=count,total=len(jobs))),flush=True)
    for item in entries:
        path=root/item['Path'];pieces=path.with_name(path.name+'.parts')
        if pieces.exists():
            temp=path.with_suffix('.assembling')
            with temp.open('wb') as output:
                for start in range(0,item['Size'],64*1024*1024):
                    end=min(start+64*1024*1024,item['Size'])-1
                    with (pieces/('%d-%d'%(start,end))).open('rb') as stream:
                        shutil.copyfileobj(stream,output,8*1024*1024)
            if sha256(temp)!=item['Sha256']:
                raise ValueError('Official asset checksum failed: '+item['Path'])
            temp.replace(path);shutil.rmtree(pieces)
    return dict(repository=repo,provider='ModelScope official ZhipuAI repository',files=entries)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
    p.add_argument('--device',default='cuda:1')
    a=p.parse_args()
    provenance=download(a.root,a.workers)
    import torch
    from transformers import AutoTokenizer, T5EncoderModel
    tokenizer=AutoTokenizer.from_pretrained(a.root/'tokenizer',local_files_only=True)
    model=T5EncoderModel.from_pretrained(a.root/'text_encoder',local_files_only=True,
                                         torch_dtype=torch.bfloat16).to(a.device).eval().requires_grad_(False)
    tokens=tokenizer([''],padding='max_length',max_length=226,truncation=True,
                     add_special_tokens=True,return_attention_mask=True,return_tensors='pt')
    with torch.no_grad():
        value=model(tokens.input_ids.to(a.device),tokens.attention_mask.to(a.device)).last_hidden_state
        length=int(tokens.attention_mask.sum())
        value=value.cpu()
    if value.shape!=(1,226,4096) or not torch.isfinite(value).all() or not value[:,:length].abs().any():
        raise ValueError('Invalid official empty prompt condition')
    torch.save(value,a.root/'empty_text.pt')
    provenance.update(prompt='',sequence_length=226,valid_tokens=length,shape=list(value.shape),
                      output_sha256=sha256(a.root/'empty_text.pt'))
    (a.root/'empty_text.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(dict(ready=True,path=str(a.root/'empty_text.pt'),valid_tokens=length)),flush=True)


if __name__=='__main__':
    main()

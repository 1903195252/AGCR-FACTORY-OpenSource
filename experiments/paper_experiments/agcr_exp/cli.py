import argparse,json
from pathlib import Path
from .data import atomic_json,build_index,read_json,Dataset,assert_no_future_leak
from .protocol import write_protocol
from .runs import audit,matrix
from .selection import quota_report,validation_panel


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('protocol');a.add_argument('--output',required=True)
    a=sub.add_parser('index');a.add_argument('--root',required=True);a.add_argument('--output',required=True)
    a.add_argument('--allow-partial',action='store_true');a.add_argument('--heldout-class',type=int,choices=[3,4,13,14])
    for name in ['audit','matrix','quota','validation-panel']:
        a=sub.add_parser(name);a.add_argument('--index',required=True);a.add_argument('--output',required=True)
        if name=='matrix':a.add_argument('--experiments',nargs='+')
    a=sub.add_parser('inspect-window');a.add_argument('--index',required=True);a.add_argument('--role',default='train');a.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='protocol':write_protocol(args.output);return
    if args.command=='index':
        r=build_index(args.root,args.output,args.allow_partial,args.heldout_class)
        print(json.dumps(dict(windows=len(r['windows']),counts=r['counts'],split_sha256=r['split_sha256']),indent=2));return
    index=read_json(args.index)
    if args.command=='audit':result=audit(index)
    elif args.command=='matrix':result=matrix(index,args.experiments)
    elif args.command=='quota':result=quota_report(index)
    elif args.command=='validation-panel':
        _,result=validation_panel(Dataset(index,'val').rows)
        result['split_sha256']=index['split_sha256']
    else:
        ds=Dataset(index,args.role,load_surfaces=True);sample=ds[0];assert_no_future_leak(sample['inputs'])
        result=dict(metadata=sample['metadata'],inputs={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in sample['inputs'].items()},
                    labels={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in sample['labels'].items()})
    atomic_json(args.output,result)
    print(json.dumps(result if args.command!='matrix' else dict(job_count=result['job_count']),indent=2))

if __name__=='__main__':main()

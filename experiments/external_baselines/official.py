"""Load pinned upstream implementations, retaining original model computation.

Only documented fixed-grid constants are parameterized. No proprietary model is
substituted and no upstream source tree is modified.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

LOCK=json.loads(Path(__file__).with_name('official_sources.json').read_text())


def verify_source(root,name):
    root=Path(root)
    if (root/'.git').exists():
        commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    else:
        commit=(root/'UPSTREAM_COMMIT').read_text().strip()
    if commit!=LOCK[name]['commit']: raise RuntimeError('Unreviewed upstream revision: '+commit)
    return root


def package(name,path):
    if name not in sys.modules:
        value=types.ModuleType(name);value.__path__=[str(path)];sys.modules[name]=value


def module(name,path,transform=None):
    path=Path(path);source=path.read_text()
    if transform: source=transform(source)
    value=types.ModuleType(name);value.__file__=str(path);value.__package__=name.rsplit('.',1)[0]
    sys.modules[name]=value
    exec(compile(source,str(path),'exec'),value.__dict__)
    return value


def syncam_module(root):
    root=verify_source(root,'syncammaster')/'diffsynth'/'models'
    package('_agcr_syncam',root);module('_agcr_syncam.utils',root/'utils.py')
    def patch(s):
        old='h, w = 30, 52  # h, w  hard code'
        if s.count(old)!=1: raise RuntimeError('Unexpected upstream SynCamMaster grid')
        return s.replace(old,'h, w = self.factory_grid')
    return module('_agcr_syncam.wan_video_dit',root/'wan_video_dit.py',patch)


def shareverse_module(root,frames,height,width,dim=3072,layers=42):
    root=verify_source(root,'shareverse')/'stage2'/'shareverse'/'models'
    prefix='_agcr_shareverse_%s_%s_%s_%s'%(frames,height,width,dim)
    package(prefix,root);package(prefix+'.transformers',root/'transformers')
    def shape_patch(s):
        return s.replace('from .embeddings import apply_rotary_emb','from diffusers.models.embeddings import apply_rotary_emb').replace('f, h, w, c = 9, 30, 45, 3072',f'f, h, w, c = {frames}, {height}, {width}, {dim}').replace('T=9, H=30, W=45',f'T={frames}, H={height}, W={width}')
    module(prefix+'.attention_processor',root/'attention_processor.py',shape_patch)
    module(prefix+'.normalization',root/'normalization.py',shape_patch)
    def transformer_patch(s):
        s=shape_patch(s).replace('view(batch_size, 9,','view(batch_size, %d,'%frames)
        s=s.replace('nn.Conv2d(24, 3072,','nn.Conv2d(24, %d,'%dim)
        s=s.replace('for _ in range(21)','for _ in range(%d)'%((layers+1)//2))
        return s
    return module(prefix+'.transformers.cogvideox_transformer_3d',root/'transformers'/'cogvideox_transformer_3d.py',transformer_patch)

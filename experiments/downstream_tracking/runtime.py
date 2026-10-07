"""Atomic per-episode/clip caches and bounded execution for reader jobs."""
import hashlib,json,signal,time
from pathlib import Path
from .data import sha256
from .train import atomic


class ReaderRuntime:
    def __init__(self,output,gpu_hours,inputs,options=None):
        if gpu_hours<=0:raise ValueError('Reader job requires a positive GPU-hour budget')
        self.started=time.monotonic();self.seconds=gpu_hours*3600;self.stopped=False
        self.root=Path(str(output)+'.cache');self.root.mkdir(parents=True,exist_ok=True)
        identity=dict(inputs={str(Path(p).resolve()):sha256(p) for p in inputs if p},options=options,
            source={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
        path=self.root/'identity.json'
        if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('Reader resume input/config/source changed')
        atomic(path,identity)
        signal.signal(signal.SIGTERM,lambda *_:setattr(self,'stopped',True));signal.signal(signal.SIGINT,lambda *_:setattr(self,'stopped',True))
    def path(self,key):return self.root/(hashlib.sha256(str(key).encode()).hexdigest()+'.json')
    def get(self,key):
        path=self.path(key)
        return json.loads(path.read_text())['value'] if path.exists() else None
    def put(self,key,value):atomic(self.path(key),dict(key=str(key),value=value))
    def pause(self):
        if self.stopped or time.monotonic()-self.started>=self.seconds:
            atomic(self.root/'status.json',dict(completed=False,status='paused',wall_seconds=time.monotonic()-self.started));return True
        return False

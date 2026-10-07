"""Extend a completed 256-update future run in a new, explicitly audited root.

The native objective, typed checkpoint loader, optimizer, RNG and data stream are
unchanged. Only the declared training horizon grows. Original checkpoints and
their completed protocols remain immutable. This is a horizon extension, not an
identity-preserving restart of the earlier 256-update protocol.
"""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from . import train_history_future as native
from .region_probe import atomic_json, sha256
from .train_candidate import atomic_checkpoint, supervise

SCHEMA = 'peerdream-history-future-extension-1'


def extend_payload(payload, target_steps):
    config = payload['identity']['configuration']
    if (payload.get('completed') is not True or payload.get('step') != 256
            or payload.get('target_steps') != 256 or config.get('steps') != 256
            or type(target_steps) is not int or not 256 < target_steps <= 4096
            or not payload.get('optimizer') or set(payload.get('rng', {})) != {'cpu', 'cuda'}):
        raise ValueError('Require a completed typed 256-step optimizer/RNG checkpoint')
    result = dict(payload, target_steps=target_steps, completed=False)
    result['identity'] = copy.deepcopy(payload['identity'])
    result['identity']['configuration']['steps'] = target_steps
    return result


def prepare(args):
    out = Path(args.output).resolve(); old = Path(args.initial_checkpoint).resolve()
    if old.parent == out or old.parent in out.parents or out in old.parents:
        raise ValueError('The original checkpoint namespace must remain separate')
    if sha256(old) != args.initial_checkpoint_sha256:
        raise ValueError('Initial completed checkpoint hash changed')
    payload = torch.load(old, map_location='cpu', weights_only=False)
    paths = json.loads(Path(args.paths).read_text()); index_sha = sha256(paths['families'])
    native.validate_checkpoint(payload, args.parent, index_sha, require_complete=True)
    config = payload['identity']['configuration']
    if (config['arm'] != args.arm or config['seed'] != args.seed or config['lr'] != args.lr
            or payload['history_checkpoint'] != dict(path=str(Path(args.history_overlay).resolve()), sha256=sha256(args.history_overlay))
            or payload['identity']['data']['panel_sha256'] != sha256(args.panel)):
        raise ValueError('Extension must use the same arm, history, seed, learning rate and TRAIN panel')
    root = Path(__file__).resolve().parents[2]
    if any(payload['identity']['sources'].get(name) != sha256(root/name) for name in native.SOURCES):
        raise ValueError('Native objective or deployment source changed')
    derived = extend_payload(payload, args.steps)
    native.validate_checkpoint(derived, args.parent, index_sha)
    out.mkdir(parents=True, exist_ok=True)
    if any(x.name != '.extension.lock' for x in out.iterdir()):
        raise FileExistsError('Fresh extension destination required; attempts are never reset')
    atomic_checkpoint(out/'last.pt', derived)
    receipt = dict(schema=SCHEMA, initial_checkpoint=str(old), initial_checkpoint_sha256=sha256(old),
        initial_step=256, target_steps=args.steps, additional_updates=args.steps-256,
        derived_initial_checkpoint_sha256=sha256(out/'last.pt'),
        history_checkpoint=derived['history_checkpoint'], identity=derived['identity'],
        extension_source_sha256=sha256(__file__), optimizer_preserved=True,
        cpu_rng_preserved=True, cuda_rng_preserved=True,
        changed_fields=['target_steps', 'completed', 'identity.configuration.steps'],
        unchanged_native_sources=True, parent_checkpoint_unmodified=True,
        execution='original_train_history_future.run_with_extended_horizon',
        diagnostic_only=True, formal_experiment_result=False, prepared_unix=time.time())
    atomic_json(out/'extension.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('paths','parent','history-overlay','history-cache','panel','cache','output',
                'initial-checkpoint','initial-checkpoint-sha256'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--arm',choices=('control','paired_geometry'),required=True)
    parser.add_argument('--steps',type=int,default=2048)
    parser.add_argument('--seed',type=int,default=17029); parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--device',default='cuda:0'); parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--save-every',type=int,default=16); parser.add_argument('--max-seconds',type=float,default=10710.)
    parser.add_argument('--stop-file'); parser.add_argument('--worker-report',help=argparse.SUPPRESS)
    parser.add_argument('--watchdog-stop',help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if (not 256 < args.steps <= 4096 or not 0 < args.max_seconds <= 10800
            or not math.isfinite(args.lr) or not 0 < args.lr <= .001
            or not 1 <= args.threads <= 16 or args.save_every < 1):
        parser.error('Require bounded extended training and finite matching optimizer settings')
    args.resume = True; args.cache_only = False
    if args.worker_report:
        receipt = json.loads((Path(args.output)/'extension.json').read_text())
        if (receipt['extension_source_sha256'] != sha256(__file__)
                or receipt['initial_checkpoint_sha256'] != args.initial_checkpoint_sha256
                or receipt['target_steps'] != args.steps
                or sha256(Path(args.output)/'last.pt') != receipt['derived_initial_checkpoint_sha256']):
            raise ValueError('Initial extension receipt or derived checkpoint changed before worker')
        return native.run(args)
    out = Path(args.output).resolve()
    protected = [Path(x).resolve() for x in (args.cache,args.history_cache,Path(args.parent).parent,
                                            Path(args.history_overlay).parent,Path(args.initial_checkpoint).parent)]
    if any(out == path or path in out.parents or out in path.parents for path in protected):
        raise ValueError('Output overlaps original assets or caches')
    out.mkdir(parents=True,exist_ok=True)
    with (out/'.extension.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        prepare(args)
        attempt=out/'attempts'/('%d_%d'%(time.time_ns(),os.getpid())); attempt.mkdir(parents=True)
        command=[sys.executable,'-m','experiments.research_extensions.train_history_future_extended',
                 *(sys.argv[1:] if argv is None else argv),'--worker-report',str(attempt/'report.json'),
                 '--watchdog-stop',str(attempt/'STOP')]
        code=supervise(command,out,attempt,args.max_seconds,60.,args.stop_file)
        payload=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        receipt=json.loads((out/'extension.json').read_text())
        if sha256(args.initial_checkpoint)!=args.initial_checkpoint_sha256:
            raise ValueError('Original checkpoint changed during extension')
        atomic_json(out/'extension_completion.json',dict(schema=SCHEMA,completed=payload.get('completed') is True,
            step=payload.get('step'),target_steps=args.steps,initial_step=256,
            additional_completed_updates=payload.get('step',256)-256,final_checkpoint_sha256=sha256(out/'last.pt'),
            extension_receipt_sha256=sha256(out/'extension.json'),original_checkpoint_sha256=args.initial_checkpoint_sha256,
            native_sources_unchanged=payload.get('identity',{}).get('sources')==receipt['identity']['sources'],
            original_checkpoint_unmodified=True,returncode=code))
        return code


if __name__=='__main__': raise SystemExit(main())

"""Fifty-step VAL sampling of the independently adapted condition renderer."""
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

from . import sample_candidate as sampler
from .region_probe import sha256
from .train_condition_renderer import fixed_future, validate_candidate, validate_adapter_tensors, forward_scopes, SOURCES


def main(argv=None):
    cli = list(sys.argv[1:] if argv is None else argv)
    if cli.count('--future-checkpoint') != 1 or '--history-surface-checkpoint' in cli:
        raise ValueError('Exactly one --future-checkpoint required')
    translated = ['--history-surface-checkpoint' if x=='--future-checkpoint' else x for x in cli]
    args = sampler.arguments(translated)
    if not args.candidate_checkpoint:
        raise ValueError('An adapted renderer candidate must be explicit')
    root = Path(args.output).resolve()
    for path in (args.paths,args.panel,args.renderer_checkpoint,args.state_checkpoint,args.candidate_checkpoint,args.history_surface_checkpoint):
        source = Path(path).resolve()
        if root == source or root in source.parents:
            raise ValueError('Output overlaps immutable input')
    root.mkdir(parents=True,exist_ok=True)
    if not args.worker:
        from experiments.base48.sample import exclusive
        with exclusive(root/'.sampling.lock'):
            return sampler.supervise([sys.executable,'-m','experiments.research_extensions.sample_condition_renderer',*cli,'--worker'],root,args.max_seconds,args.stop_file)
    import torch
    checkpoint = torch.load(args.candidate_checkpoint,map_location='cpu',weights_only=False)
    config = checkpoint['identity']['configuration']
    geometry, renderer = config['neutralize_geometry_omega'], config['neutralize_renderer_omega']
    del checkpoint
    old_install, old_provenance = sampler.install_state_overlay, sampler.provenance
    def install(model,path,expected_index_sha256=None):
        value = old_install(model,path,expected_index_sha256)
        future = fixed_future(args.history_surface_checkpoint,path,expected_index_sha256,model.upstream,args.device)
        return dict(value,future_conditioning=future,mode='frozen_history_future2048_renderer_adaptation')
    def candidate(payload,renderer_sha,state_sha,index_sha,allow_partial=False):
        return validate_candidate(payload,renderer_sha,state_sha,index_sha,args.history_surface_checkpoint,args.state_checkpoint,allow_partial)
    def surface(model,path,parent,index_sha,device):
        value = fixed_future(path,parent,index_sha,model.upstream,device)
        return dict(value,sampler_extension_source_sha256=sha256(__file__))
    def method(state,renderer_candidate=None,history_surface=None):
        if renderer_candidate is None or history_surface is None:
            raise ValueError('Both verified state and renderer candidates required')
        return 'peerdream_condition_renderer_'+renderer_candidate['configuration']['arm']
    def provenance(*a,**kw):
        value = old_provenance(*a,**kw)
        for name in SOURCES:
            path = Path(__file__).with_name(name)
            value['sources'][str(path.resolve())] = sha256(path)
        return value
    with ExitStack() as stack:
        for key,value in (('install_state_overlay',install),('validate_candidate',candidate),
                ('validate_candidate_tensors',validate_adapter_tensors),
                ('install_surface_candidate',surface),('method_identity',method),('provenance',provenance)):
            stack.enter_context(patch.object(sampler,key,value))
        stack.enter_context(forward_scopes(geometry,renderer))
        return sampler.main(translated)


if __name__ == '__main__':
    raise SystemExit(main())

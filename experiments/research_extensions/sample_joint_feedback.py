"""Fixed native50 VAL sampling of a completed typed joint-feedback candidate.

No TEST selection is available here. The learned upstream is preserved after
import: historical/future initializer weights are never reloaded over it.
"""
from contextlib import ExitStack
from pathlib import Path
import sys
from unittest.mock import patch

from . import sample_candidate as sampler
from .region_probe import sha256
from .train_condition_renderer import fixed_future,forward_scopes
from .joint_feedback_candidate import validate_joint_payload,validate_joint_tensors


def main(argv=None):
    cli=list(sys.argv[1:]if argv is None else argv)
    if cli.count('--future-checkpoint')!=1 or '--history-surface-checkpoint'in cli:
        raise ValueError('Exactly one registered future initializer required')
    translated=['--history-surface-checkpoint'if x=='--future-checkpoint'else x for x in cli]
    a=sampler.arguments(translated)
    if not a.candidate_checkpoint or a.allow_partial_candidate:
        raise ValueError('Only a completed joint checkpoint may be sampled in this comparison')
    root=Path(a.output).resolve()
    for p in(a.paths,a.panel,a.renderer_checkpoint,a.state_checkpoint,a.candidate_checkpoint,a.history_surface_checkpoint):
        source=Path(p).resolve()
        if root==source or root in source.parents:raise ValueError('Output overlaps immutable input')
    root.mkdir(parents=True,exist_ok=True)
    if not a.worker:
        from experiments.base48.sample import exclusive
        with exclusive(root/'.sampling.lock'):
            return sampler.supervise([sys.executable,'-m','experiments.research_extensions.sample_joint_feedback',*cli,'--worker'],root,a.max_seconds,a.stop_file)
    original_install,original_provenance=sampler.install_state_overlay,sampler.provenance
    installed={}
    def install(model,path,expected_index_sha256=None):
        state=original_install(model,path,expected_index_sha256)
        future=fixed_future(a.history_surface_checkpoint,path,expected_index_sha256,model.upstream,a.device)
        installed.update(future=future,model=model)
        return dict(state,joint_initializer=future,mode='joint_history_state_video')
    def candidate(payload,renderer_sha,state_sha,index_sha,allow_partial=False):
        identity=validate_joint_payload(payload,a.history_surface_checkpoint,a.state_checkpoint,index_sha,renderer_sha=renderer_sha)
        if identity['ancestry'].get('renderer_checkpoint_sha256')!=renderer_sha or identity['ancestry'].get('state_checkpoint_sha256')!=state_sha:
            raise ValueError('Joint renderer/state parents differ')
        for name,digest in identity['source_sha256'].items():
            if sha256(Path(__file__).with_name(name))!=digest:raise ValueError('Joint training source changed: '+name)
        return identity
    def surface(model,path,parent,index_sha,device):
        # This callback follows candidate import in the shared sampler. It
        # records provenance only, never copying the old initializer tensors.
        if installed.get('model')is not model or path!=a.history_surface_checkpoint:
            raise ValueError('Joint structure was not installed before learned weights')
        return dict(installed['future'],post_import_initializer_reload=False,
                    joint_upstream_preserved=True,sampler_extension_source_sha256=sha256(__file__))
    def method(state,renderer_candidate=None,history_surface=None):
        if renderer_candidate is None or not history_surface.get('joint_upstream_preserved'):
            raise ValueError('Joint candidate provenance missing')
        return 'peerdream_joint_'+renderer_candidate['arm']
    def provenance(*args,**kwargs):
        value=original_provenance(*args,**kwargs)
        for name in('sample_joint_feedback.py','joint_feedback_candidate.py','train_joint_feedback.py'):
            p=Path(__file__).with_name(name);value['sources'][str(p.resolve())]=sha256(p)
        return value
    with ExitStack()as stack:
        for name,callback in(('install_state_overlay',install),('validate_candidate',candidate),
                ('validate_candidate_tensors',validate_joint_tensors),('install_surface_candidate',surface),
                ('method_identity',method),('provenance',provenance)):
            stack.enter_context(patch.object(sampler,name,callback))
        stack.enter_context(forward_scopes(True,True))
        return sampler.main(translated)


if __name__=='__main__':raise SystemExit(main())

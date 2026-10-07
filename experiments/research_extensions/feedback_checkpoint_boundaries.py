"""Execute the two composite RGB-D feedback boundaries without checkpointing.

The Wan blocks, native causal residuals, depth blocks, DINO and reader keep
their existing checkpoints. Only the redundant enclosing ``generated_readout``
and per-view ``one`` boundaries execute directly. This avoids checkpoint
re-entry across invocation-local VAE caches and nested saved-tensor hooks.
All operations, random draws, loss terms and input gradients are retained.
"""
from contextlib import contextmanager
from unittest.mock import patch


BOUNDARIES = frozenset((
    'JointSystem.feedback.<locals>.generated_readout',
    'JointSystem.decode.<locals>.one',
))


@contextmanager
def flat_feedback_boundaries(*, _module=None):
    if _module is None:
        from experiments.paired_interventions import system as _module
    original = _module.checkpoint
    record = dict(policy='leaf_checkpoints_without_composite_feedback_boundaries',
                  direct_calls={name: 0 for name in sorted(BOUNDARIES)},
                  other_module_calls_preserved=0,
                  leaf_checkpoints_preserved=True, random_draws_changed=False)

    def dispatch(function, *args, **kwargs):
        name = getattr(function, '__qualname__', '')
        if name not in BOUNDARIES:
            record['other_module_calls_preserved'] += 1
            return original(function, *args, **kwargs)
        if kwargs != dict(use_reentrant=False):
            raise ValueError('Composite checkpoint contract changed: '+name)
        record['direct_calls'][name] += 1
        return function(*args)

    with patch.object(_module, 'checkpoint', dispatch):
        yield record

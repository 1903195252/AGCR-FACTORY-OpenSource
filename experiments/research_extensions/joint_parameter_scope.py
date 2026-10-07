"""Shared exact parameter domain for native joint admission and training."""
from contextlib import contextmanager

@contextmanager
def parameter_scope(model):
    """Train upstream plus condition/adapters/gate; keep the renderer prior fixed.

    The existing learned observation/filter/dynamics/geometry modules can
    receive actual decoded feedback. The independent auxiliary generated-image
    reader, original Wan including LoRA, depth decoder and image encoders remain
    frozen. The latter retain input derivatives. No module.train() call changes
    historical dropout or the inference mechanism.
    """
    original = [(p,p.requires_grad) for p in model.parameters()]
    try:
        model.requires_grad_(False)
        model.upstream.requires_grad_(True)
        model.condition.requires_grad_(True)
        model.wan.adapters.requires_grad_(True)
        model.wan.gate.requires_grad_(True)
        model.reader.requires_grad_(False)
        groups = model.trainable_groups()
        allowed = ('upstream.', 'condition.', 'wan.adapters.', 'wan.gate.', 'wan.encoder.')
        names = {id(p):name for name,p in model.named_parameters()}
        actual = [id(p) for g in groups for p in g['params']]
        expected = {id(p) for p in model.parameters() if p.requires_grad}
        if not actual or len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError('Joint optimizer parameters are empty, duplicated or incomplete')
        if {g['name'] for g in groups} != {'upstream','new'}:
            raise ValueError('Joint candidate requires upstream and new groups only')
        if any(not names[x].startswith(allowed) for x in actual):
            raise ValueError('A frozen reader/backbone/depth parameter entered joint training')
        yield groups
    finally:
        for p,flag in original:
            p.requires_grad_(flag)

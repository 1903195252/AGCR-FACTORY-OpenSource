"""Optional saved-tensor parameter exclusion across explicit residency moves.

No production entrypoint imports this scope yet. It changes only which known
parameter/buffer storage aliases bypass activation copying, including the plain
FrozenImageEncoders/NativeVAE wrappers. It never registers arbitrary moved
activations as parameters. The store retains its final, refreshed registry;
all temporary move hooks are restored on normal or exceptional exit.
"""
from contextlib import ExitStack,contextmanager
from unittest.mock import patch

import torch

from .decode_activation_offload import storage_key


def parameter_and_buffer_tensors(model):
    encoders=model.encoders
    modules=list(dict.fromkeys((model,encoders.vae,encoders.dino)))
    values=[value for module in modules for value in list(module.parameters())+list(module.buffers())]
    values += [encoders.mean,encoders.std,model.vae.mean,model.vae.std]
    if any(not isinstance(value,torch.Tensor)for value in values):
        raise TypeError('Fixed storage registry contains a non-tensor normalization value')
    return list(dict.fromkeys(values))


@contextmanager
def track_fixed_parameter_storages(model,store):
    """Refresh after every known _move_data call, including failed partial moves.

    Enter inside the activation-store scope and outside depth/VAE placements
    and joint_backward. This also covers staged_video_backward's internal
    backbone offload/restore. No model weights, precision, RNG or gradients are
    changed by this registry.
    """
    from . import execution,depth_device_residency,vae_decoder_device_residency
    namespaces=(execution,depth_device_residency,vae_decoder_device_residency)
    record=dict(policy='registered_and_plain_encoder_storages_refreshed_after_moves',
        refreshes=0,move_calls=0,move_errors=0,tracked_tensors=0,unique_storages=0,
        activation_tensors_registered=False,gradients_detached=False,
        weights_modified=False,move_hooks_restored=False)
    def refresh():
        values=parameter_and_buffer_tensors(model)
        store.fixed={storage_key(value)for value in values}
        record.update(refreshes=record['refreshes']+1,tracked_tensors=len(values),unique_storages=len(store.fixed))
    refresh()
    originals={module:module._move_data for module in namespaces}
    try:
        with ExitStack()as stack:
            for module,original in originals.items():
                def moved(tensor,device,_original=original):
                    record['move_calls']+=1
                    try:return _original(tensor,device)
                    except BaseException:
                        record['move_errors']+=1
                        raise
                    finally:refresh()
                stack.enter_context(patch.object(module,'_move_data',moved))
            yield record
    finally:
        refresh()
        record['move_hooks_restored']=all(module._move_data is original for module,original in originals.items())
        if not record['move_hooks_restored']:
            raise RuntimeError('A parameter residency move hook was not restored')

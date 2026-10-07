"""Three-device native Adam admission preserving exact saved-tensor layouts.

The device threshold is explicit on this independent entrypoint. The original
entrypoint retains its64GiB default. Native frames, diffusion steps, losses,
precision and gradient routes remain unchanged; all three cards share one budget.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch


def main(argv=None):
    cli=list(sys.argv[1:]if argv is None else argv)
    extra=argparse.ArgumentParser(add_help=False)
    extra.add_argument('--cpu-activation-gib',type=float,default=128.)
    extra.add_argument('--depth-device',required=True)
    extra.add_argument('--vae-device',required=True)
    option,remaining=extra.parse_known_args(cli)
    if not 0<option.cpu_activation_gib<=128:raise ValueError('CPU activation bound must be <=128GiB')
    from . import probe_joint_adam_native as original
    a=original.arguments(remaining);root=Path(a.output).resolve()
    devices=(a.device,option.depth_device,option.vae_device)
    if len(set(devices))!=3 or any(not d.startswith('cuda:') or not d[5:].isdigit() for d in devices):
        raise ValueError('Three distinct explicitly reserved CUDA devices are required')
    if not a.worker:
        for name in('paths','renderer_checkpoint','state_checkpoint','future_checkpoint'):
            source=Path(getattr(a,name)).resolve()
            if root==source or root in source.parents:raise ValueError('Output overlaps immutable input')
        if root.exists()and any(root.iterdir()):raise FileExistsError('New independent offload admission output required')
        root.mkdir(parents=True,exist_ok=True)
        from .sample_candidate import supervise
        return supervise([sys.executable,'-m','experiments.research_extensions.probe_joint_adam_three_device_layout',*cli,'--worker'],root,a.max_seconds,None)
    # Match the original worker's allocator before importing torch-heavy code.
    import os
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    from . import joint_feedback_candidate as candidate
    from .joint_graph_offload import joint_graph_saved_tensors_on_cpu
    from . import feedback_recompute_offload as offload_module
    from .layout_saved_activation import LayoutActivationStore
    from .residency_fixed_storage import track_fixed_parameter_storages
    from .vae_decode_local_cache import reentrant_native_features
    from .depth_device_residency import frozen_depth_on_device
    from .vae_decoder_device_residency import frozen_vae_decoder_on_device
    import torch
    properties=torch.cuda.get_device_properties(option.depth_device)
    if properties.total_memory<30*2**30:raise ValueError('Secondary native depth device must have >=30GiB')
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction,option.depth_device)
    torch.cuda.reset_peak_memory_stats(option.depth_device)
    vae_properties=torch.cuda.get_device_properties(option.vae_device)
    if vae_properties.total_memory<30*2**30:raise ValueError('Native VAE device must have >=30GiB')
    torch.cuda.set_per_process_memory_fraction(a.memory_fraction,option.vae_device)
    torch.cuda.reset_peak_memory_stats(option.vae_device)
    from .region_probe import atomic_json,sha256
    backward=candidate.joint_backward;stores=[];feature_stores=[];device_records=[];cache_records=[];boundary_records=[];vae_records=[];fixed_records=[]
    def execute(model,*args,**kwargs):
        with joint_graph_saved_tensors_on_cpu(model,option.cpu_activation_gib)as(store,record):
            boundary_records.append(record)
            stores.append(store)
            with track_fixed_parameter_storages(model,store) as fixed_record:
                fixed_records.append(fixed_record)
                with reentrant_native_features(model,store)as(feature_store,cache_record):
                    feature_stores.append(feature_store)
                    cache_records.append(cache_record)
                    with frozen_depth_on_device(model,option.depth_device)as placed:
                        device_records.append(placed)
                        # Depth Parameter objects retain identity while storage moves;
                        # refresh alias exclusions on their actual execution devices.
                        with frozen_vae_decoder_on_device(model,option.vae_device) as vae_placed:
                            vae_records.append(vae_placed)
                            return backward(model,*args,**kwargs)
    try:
        with patch.object(offload_module,'ActivationStore',LayoutActivationStore),\
                patch.object(candidate,'joint_backward',execute):
            code=original.worker(a,minimum_device_gib=30)
    finally:
        for name in('report.json','failure.json'):
            p=root/name
            if not p.exists():continue
            report=json.loads(p.read_text())
            report['execution_activation_offload']=dict(enabled=True,max_gib=option.cpu_activation_gib,
                scope='joint_forward_decode_readout_and_per_view_backward',dtype_preserved=True,model_parameters_retained_on_gpu=True,
                native_feature_residency=True,native_feature_depth_transfer='per_level',
                cpu_limit_includes_native_features=True,invocation_local_vae_cache=True,
                invocation_scoped_feature_hooks=True,cpu_capacity_policy='shared_128GiB_including_features_with16GiB_reserve',
                recomputation_hook_installed_inside_per_view=True,joint_forward_scope=True,
                saved_tensor_layout='exact_storage_span_shape_stride_offset_v1',
                saved_parameter_registry='registered_and_plain_encoder_storages_refreshed_after_moves')
            report['execution_device_admission']=dict(minimum_device_gib=30,
                original_entrypoint_default_minimum_device_gib=64,physical_properties_unmodified=True,
                expected_physical_gpu_count=3)
            report['native_cache_records']=cache_records
            report['fixed_storage_records']=fixed_records
            report['execution_feedback_checkpoints']=dict(policy='joint_forward_and_reentrant_view_saved_tensor_offload',leaf_checkpoints_preserved=True)
            report['feedback_checkpoint_records']=boundary_records
            report['execution_depth_residency']=dict(enabled=True,device=option.depth_device,
                gpu_count=3,scope='frozen_depth_only',dtype_preserved=True,input_gradients_preserved=True)
            report['depth_residency_records']=device_records
            report['execution_vae_residency']=dict(enabled=True,device=option.vae_device,gpu_count=3,
                scope='frozen_decoder_and_post_quant_only',dtype_preserved=True,
                input_gradients_preserved=True,encoder_device_unchanged=True,
                scaling_device_unchanged=True,output_device='original_latent_device')
            report['vae_residency_records']=vae_records
            report['vae_gpu_peak']=dict(device=option.vae_device,name=vae_properties.name,
                peak_allocated_gib=torch.cuda.max_memory_allocated(option.vae_device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(option.vae_device)/2**30)
            report['secondary_gpu_peak']=dict(device=option.depth_device,name=properties.name,
                peak_allocated_gib=torch.cuda.max_memory_allocated(option.depth_device)/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved(option.depth_device)/2**30)
            report['cpu_activation_records']=[dict(store.report(),container_before=store.container_before,
                container_after=store.container_after)for store in stores]
            report['native_feature_records']=[store.report()for store in feature_stores]
            report.setdefault('source_sha256',{}).update({n:sha256(Path(__file__).with_name(n))for n in
                ('probe_joint_adam_three_device_layout.py','layout_saved_activation.py','residency_fixed_storage.py','vae_decoder_device_residency.py','joint_graph_offload.py','feedback_recompute_offload.py','feedback_checkpoint_boundaries.py','feedback_reentrant_boundaries.py','vae_decode_local_cache.py','depth_device_residency.py','decode_activation_offload.py','vae_feature_residency.py')})
            atomic_json(p,report)
            if name=='report.json':
                receipt=json.loads((root/'completion.json').read_text());receipt['report_sha256']=sha256(p)
                atomic_json(root/'completion.json',receipt)
    return code


if __name__=='__main__':raise SystemExit(main())

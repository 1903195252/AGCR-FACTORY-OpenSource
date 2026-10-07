"""Verify that native samplers use the implementation that produced weights."""
from pathlib import Path

from experiments.paired_interventions.data import file_sha256


def implementation_hashes(method, source, adapter_root=None):
    source = Path(source)
    result = dict(
        native_source_sha256={str(p.relative_to(source)): file_sha256(p)
                              for p in source.rglob('*.py') if '.git' not in p.parts},
        source_hashes={p.name: file_sha256(p) for p in Path(__file__).parent.glob('*.py')},
    )
    if method == 'magicdrive_v2':
        root = Path(adapter_root) if adapter_root else Path(__file__).parent.parent / 'external_magicdrive'
        files = {p.name: file_sha256(p) for p in root.glob('*.py')}
        files['vendor/factory_patch_provenance.json'] = file_sha256(source / 'factory_patch_provenance.json')
        result['native_factory_adapter_sha256'] = files
    return result


def verify_sampling_implementation(identity, method, source, text, modality,
                                   forecaster_checkpoint=None, adapter_root=None):
    if identity['method'] != method or identity['modality'] != modality or identity['phase'] != 'video':
        raise ValueError('Sampling requires the matching native video-stage checkpoint')
    expected = implementation_hashes(method, source, adapter_root)
    expected['text_sha256'] = file_sha256(text)
    expected['forecaster_checkpoint_sha256'] = file_sha256(forecaster_checkpoint) if forecaster_checkpoint else None
    for name, current in expected.items():
        if identity.get(name) != current:
            raise ValueError('Sampling implementation differs from training checkpoint: ' + name)

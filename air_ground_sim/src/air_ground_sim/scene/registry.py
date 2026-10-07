"""Model identity is checked before native loading. No implicit lightweight fallback."""
import ctypes
import hashlib
import json
from pathlib import Path
import mujoco
from ..paths import BUNDLE, CONFIG


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''): h.update(chunk)
    return h.hexdigest()


def profiles():
    return json.loads((CONFIG/'models.json').read_text())


def load_model(profile='factory_industrial', verify=True):
    registry = profiles()
    if profile not in registry: raise ValueError('Unknown model profile: '+profile)
    spec = registry[profile]
    library = Path(mujoco.__file__).parent/'libmujoco.so.3.12.1'
    identity=json.loads((BUNDLE/'runtime/identity.json').read_text())
    if identity.get('source_library_sha256',identity['library_sha256'])!=spec['native_io']['library_sha256']:
        raise RuntimeError('Packaged native engine source differs from this model profile.')
    if sha256(library) != identity['library_sha256']:
        raise RuntimeError('Native engine identity differs from this profile.')
    if not hasattr(ctypes.CDLL(str(library)), 'mj_loadModelBuffer64'):
        raise RuntimeError('The complete model requires the bundled >2 GiB native I/O extension.')
    path = BUNDLE/spec['model']
    if verify and sha256(path) != spec['model_sha256']:
        raise ValueError('Model checksum mismatch: '+str(path))
    return mujoco.MjModel.from_binary_path(str(path)), spec


def verify_authoring(profile):
    spec = profiles()[profile]; xml = BUNDLE/spec['authoring']
    if sha256(xml) != spec['authoring_sha256']: raise ValueError('Authoring XML mismatch')
    records = json.loads((xml.parent/'assets.json').read_text())['files']
    for row in records:
        if sha256(xml.parent/row['file']) != row['sha256']:
            raise ValueError('Asset mismatch: '+row['file'])
    return {'profile': profile, 'verified_assets': len(records), 'xml_sha256': sha256(xml)}

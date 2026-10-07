"""Deterministic identity, atomic records, calibration and configuration."""
import hashlib
import json
import os
import uuid
from pathlib import Path
import numpy as np

class CaptureQualityError(ValueError):
    """A scene failed a data-quality check; other scene groups may continue."""

class ScriptValidationError(CaptureQualityError):
    """No accepted physical realization of the requested event was found."""

NAMESPACE = uuid.UUID('7eb81139-d76e-49ef-b4c6-645d702d9ef8')
CATEGORIES = [
    ('ignore', 'Unassigned'), ('flat.floor', 'Factory floor'),
    ('static.structure', 'Walls, columns, ceiling and lights'),
    ('human.pedestrian.worker', 'Factory worker'),
    ('vehicle.forklift', 'Forklift including mast and forks'),
    ('factory.production_line', 'Production line'),
    ('factory.workbench', 'Assembly bench and tools'),
    ('factory.control_cabinet', 'Control cabinet'),
    ('factory.wip_cart', 'Work in progress cart'),
    ('factory.storage_rack', 'Storage rack and cargo'),
    ('factory.pallet_load', 'Pallet and load'),
    ('static.safety_equipment', 'Barriers and fire equipment'),
    ('flat.marking', 'Floor markings'),
    ('vehicle.agv', 'Automated guided vehicle'),
    ('vehicle.uav', 'Flying unmanned aerial vehicle target'),
    ('occluder.robot_arm', 'Industrial manipulator and held workpiece'),
    ('occluder.suspended_load', 'Suspended panel or load'),
    ('occluder.ground_structure', 'Ground-supported structure'),
]

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def token(*args):
    return uuid.uuid5(NAMESPACE, '/'.join(map(str, args))).hex

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def episode_identity(manifest, name):
    """An explicit per-episode lineage for unchanged data in a revised release."""
    return manifest.get('inherited_episode_identities', {}).get(name, manifest['identity'])

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)

def load_config(path):
    c = json.loads(Path(path).read_text())
    if c['width'] != 720 or c['height'] != 480 or c['fps'] != 5:
        raise ValueError('Paper capture requires 720x480 at 5 Hz')
    if c['history_frames'] != 6 or c['future_frames'] != 16:
        raise ValueError('Paper temporal contract is six history and sixteen future frames')
    if c['max_targets'] > 6 or c['max_targets'] < 2:
        raise ValueError('Target capacity must be 2..6')
    if c['motion_repeats'] != 2:
        raise ValueError('Each camera path must have two motion realizations')
    if max(c['evaluation_window_starts']) + 22 > c['frames_per_episode']:
        raise ValueError('Evaluation window crosses episode boundary')
    return c

def rotation_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])

def quaternion(R):
    import mujoco
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=np.float64).ravel())
    if q[0] < 0:
        q = -q
    return q.tolist()

def intrinsics(c):
    f = c['width'] / (2 * np.tan(np.deg2rad(c['horizontal_fov_deg'] / 2)))
    return np.array([[f, 0, (c['width']-1)/2], [0, f, (c['height']-1)/2], [0, 0, 1.]])

def camera_basis(yaw, pitch):
    forward = np.array([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)])
    right = np.array([np.sin(yaw), -np.cos(yaw), 0.])
    return np.column_stack([right, np.cross(forward, right), forward])

def bbox(mask):
    yy, xx = np.nonzero(mask)
    if not len(xx):
        return [0, 0, 0, 0]
    return [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]

def pipeline_hash():
    here = Path(__file__).parent
    return digest({p.name: file_hash(p) for p in sorted(here.glob('*.py'))})

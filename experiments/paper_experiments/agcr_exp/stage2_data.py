"""Compact stage-two supervision; reference data never enters history inputs."""
import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from .data import CLASS_MAP, Dataset, atomic_json, read_json, transform_states, world_to_anchor
from .protocol import digest
from .train_observation import sha256


def area8(value):
    """Exact pixel-area means on the padded 60 x 92 geometry grid."""
    value = np.asarray(value, dtype=np.float32)
    shape = value.shape[:-2]
    small = value.reshape(*shape, 60, 8, 90, 8).mean(axis=(-3, -1))
    return np.pad(small, [(0, 0)]*len(shape)+[(0, 0), (1, 1)])


def patch20(value):
    value = np.asarray(value, dtype=np.float32)
    return value.reshape(*value.shape[:-2], 24, 20, 36, 20).mean(axis=(-3, -1))


def depth8(value):
    """Area-pooled front depth, weighted by reference support, in millimeters."""
    valid = np.isfinite(value) & (value >= 100) & (value <= 60000)
    support = area8(valid)
    return area8(np.where(valid, value, 0))/np.maximum(support, 1e-8), support


def isolated(h, t, view, count):
    full = np.zeros((count, 480, 720), np.float32)
    for k in range(count):
        x0, y0, x1, y1 = h['isolated/boxes'][t, view, k]
        offset, size = h['isolated/offsets'][t, view, k]
        if size:
            full[k, y0:y1, x0:x1] = h['isolated/depth_mm'][offset:offset+size].reshape(y1-y0, x1-x0)
    return full


def load_window(job):
    """Only 12 historical RGB-D images and one auxiliary supervision pair native-sized.

    Future surfaces are reduced on CPU. This does not change native depth inputs,
    optical-axis units, episode partitions, or target counts.
    """
    root, row, auxiliary_offset = job
    directory = Path(root)/'episodes'/row['episode']
    start, anchor = row['start'], row['anchor']
    frames = list(range(start, start+22))
    with h5py.File(directory/'frames.h5', 'r') as h:
        transform, yaw = world_to_anchor(h['ego_to_world'][anchor, 1])
        state = transform_states(h['state'][start:start+22], transform, yaw).astype('f4')
        count = state.shape[1]
        classes = np.array([CLASS_MAP[int(x)] for x in h['class_id'][:]], np.int64)
        history_full, history_visible = [], []
        front, support, visible = [], [], []
        auxiliary_full = None
        for offset, t in enumerate(frames):
            objects = np.stack([isolated(h, t, v, count) for v in range(2)])
            instance = h['instance'][t]
            masks = instance[:, None] == np.arange(1, count+1)[None, :, None, None]
            if offset < 6:
                history_full.append(patch20(objects > 0))
                history_visible.append(patch20(masks))
            else:
                z, p = depth8(objects)
                front.append(z); support.append(p); visible.append(area8(masks))
                if offset == 6+auxiliary_offset:
                    auxiliary_full = patch20(objects > 0)
                    auxiliary_visible = patch20(masks)
        def rgb(t):
            return np.stack([np.asarray(Image.open(directory/'rgb'/('%s_%04d.jpg' % (name, t))))
                             for name in ('CAM_AIR', 'CAM_GROUND')])
        aux_t = start+6+auxiliary_offset
        camera = (transform@h['camera_to_world'][start:start+22]).astype('f4')
        time = (h['time_s'][start:start+22]-h['time_s'][anchor]).astype('f4')
        inputs = dict(rgb=np.stack([rgb(t) for t in frames[:6]]),
                      depth_mm=h['depth_mm'][start:start+6], depth_valid=h['valid'][start:start+6].astype(bool),
                      K=h['K'][:].astype('f4'), camera_to_world=camera[:6],
                      planned_camera_to_world=camera[6:], time_s=time[:6], future_time_s=time[6:])
        fraction = h['visible_fraction'][start:start+22]
        full_area = h['full_area'][start:start+22]
        static_z, static_p = depth8(h['static_depth_mm'][start+6:start+22])
        labels = dict(state=state[6:], history_state=state[:6], class_id=classes,
                      history_eligible=np.asarray(row['history_eligible']),
                      present=np.ones((16, count), bool), history_present=np.ones((6, count), bool),
                      history_full_mask=np.asarray(history_full), history_visible_mask=np.asarray(history_visible),
                      history_detectable=(fraction[:6] >= .1)&(full_area[:6] >= 144),
                      isolated_depth_mm=np.asarray(front), full_mask=np.asarray(support),
                      visible_mask=np.asarray(visible), static_depth_mm=static_z, static_support=static_p)
        auxiliary = dict(rgb=rgb(aux_t), depth_mm=h['depth_mm'][aux_t], valid=h['valid'][aux_t].astype(bool),
                         K=inputs['K'], T=camera[6+auxiliary_offset], state=state[6+auxiliary_offset],
                         class_id=classes, full_mask=auxiliary_full, visible_mask=auxiliary_visible,
                         detectable=(fraction[6+auxiliary_offset] >= .1)&(full_area[6+auxiliary_offset] >= 144))
    return dict(inputs=inputs, labels=labels, auxiliary=auxiliary, uid=row['uid'], auxiliary_offset=auxiliary_offset)


class Moments:
    def __init__(self, width):
        self.count = 0
        self.sum = np.zeros(width, np.float64)
        self.cross = np.zeros((width, width), np.float64)

    def add(self, values):
        values = np.asarray(values, np.float64).reshape(-1, self.sum.size)
        self.count += len(values); self.sum += values.sum(0); self.cross += values.T@values

    def result(self, floor):
        if not self.count:
            raise ValueError('No training observations for normalization')
        mean = self.sum/self.count
        cov = self.cross/self.count-np.outer(mean, mean)
        cov = (cov+cov.T)*.5
        std = np.sqrt(np.maximum(np.diag(cov), np.asarray(floor)**2))
        cov += np.diag(np.maximum(0, np.asarray(floor)**2-np.diag(cov)))
        return dict(count=self.count, mean=mean.tolist(), std=std.tolist(), covariance=cov.tolist())


def training_statistics(index_path, output):
    """Fit with eligible TRAIN windows only; never read val/test states."""
    index = read_json(index_path)
    ds = Dataset(index, 'train', primary=True)
    by_episode = defaultdict(list)
    for row in ds.rows:
        by_episode[row['episode']].append(row)
    state_stats, acceleration = Moments(11), Moments(3)
    velocities = {k: Moments(3) for k in range(1, 5)}
    for number, (episode, rows) in enumerate(sorted(by_episode.items())):
        with h5py.File(ds.root/'episodes'/episode/'frames.h5', 'r') as h:
            raw, ego = h['state'][:], h['ego_to_world'][:, 1]
            classes = np.asarray([CLASS_MAP[int(c)] for c in h['class_id'][:]])
        for row in rows:
            transform, yaw = world_to_anchor(ego[row['anchor']])
            states = transform_states(raw[row['start']:row['start']+22], transform, yaw)
            selected = np.asarray(row['history_eligible'], bool)
            state_stats.add(states[:, selected])
            acceleration.add(np.diff(states[:, selected, 3:6], axis=0)/.2)
            for k in range(1, 5):
                mask = (classes == k)&selected
                if mask.any():
                    velocities[k].add(states[:, mask, 3:6])
        if number % 100 == 0:
            print('train statistics: %d/%d episodes' % (number+1, len(by_episode)), flush=True)
    result = dict(schema='agcr-train-state-normalization-1', role='train', index_sha256=sha256(index_path),
                  split_sha256=index['split_sha256'], windows=len(ds), episodes=len(by_episode),
                  selected_uid_sha256=digest([row['uid'] for row in ds.rows]),
                  weighting='uniform_eligible_training_window_and_present_target_time',
                  state=state_stats.result([.1]*3+[.05]*3+[.1,.05]+[.05]*3),
                  acceleration=acceleration.result([.05]*3),
                  class_velocity={str(k): v.result([.05]*3) for k, v in velocities.items()})
    atomic_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    training_statistics(args.index, args.output)

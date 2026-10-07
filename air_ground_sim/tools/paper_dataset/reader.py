"""Forecast API: history and planned cameras are separated from future supervision."""
from pathlib import Path
import h5py
import numpy as np
from PIL import Image

def read_window(root,episode,start,history=6,future=16):
    directory=Path(root)/'episodes'/episode
    stop=start+history+future;anchor=start+history
    with h5py.File(directory/'frames.h5','r') as h:
        if start<0 or stop>len(h['time_s']):raise IndexError('Window outside episode')
        rgb=np.stack([np.stack([np.asarray(Image.open(directory/'rgb'/('%s_%04d.jpg'%(ch,t))))
              for ch in ['CAM_AIR','CAM_GROUND']]) for t in range(start,stop)])
        inputs=dict(rgb=rgb[:history],depth_mm=h['depth_mm'][start:anchor],
            depth_valid=h['valid'][start:anchor],K=h['K'][:],
            camera_to_world=h['camera_to_world'][start:anchor],
            ego_to_world=h['ego_to_world'][start:anchor],
            planned_camera_to_world=h['camera_to_world'][anchor:stop],
            time_s=h['time_s'][start:stop]-h['time_s'][anchor-1])
        supervision=dict(rgb=rgb[history:],depth_mm=h['depth_mm'][anchor:stop],
            depth_valid=h['valid'][anchor:stop],state=h['state'][anchor:stop],
            instance=h['instance'][anchor:stop],semantic=h['semantic'][anchor:stop],
            visible_fraction=h['visible_fraction'][anchor:stop],full_area=h['full_area'][anchor:stop],
            static_depth_mm=h['static_depth_mm'][anchor:stop])
        # Ground-truth membership is an evaluation target, never an input feature.
        seen=(h['visible_fraction'][start:anchor]>=.1)&(h['full_area'][start:anchor]>=144)
        supervision['historically_visible_target']=seen.any(axis=(0,1))
        return dict(inputs=inputs,supervision=supervision)

def isolated_depth(h,frame,view,instance):
    box=h['isolated/boxes'][frame,view,instance];offset,count=h['isolated/offsets'][frame,view,instance]
    patch=h['isolated/depth_mm'][offset:offset+count]
    return box,patch.reshape(int(box[3]-box[1]),int(box[2]-box[0]))

def balanced_event_windows(root,seed=17,pool='main',split='train'):
    """Infinite index iterator: event -> camera group -> episode/window, uniformly."""
    import json
    rows=json.loads((Path(root)/'windows.json').read_text());groups={}
    for row in rows:
        if row['split']!=split or row['pool']!=pool or not row['event_window']:continue
        groups.setdefault(row['event'],{}).setdefault(row['matched_camera_path'],[]).append(row)
    if len(groups)!=8:raise ValueError('All eight event groups must be present before balanced training')
    rng=np.random.default_rng(seed);events=sorted(groups)
    while True:
        event=events[int(rng.integers(len(events)))];paths=sorted(groups[event])
        path=paths[int(rng.integers(len(paths)))];candidates=groups[event][path]
        yield candidates[int(rng.integers(len(candidates)))]

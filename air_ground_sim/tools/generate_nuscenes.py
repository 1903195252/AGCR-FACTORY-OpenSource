"""Generate/resume synchronized UAV3D-style factory data in nuScenes tables."""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from air_ground_sim.dataset.exporter import export_dataset

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--config',type=Path,default=ROOT/'configs/dataset_uav3d.json')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--resume',action='store_true')
p.add_argument('--scenes',type=int)
p.add_argument('--samples-per-scene',type=int)
a=p.parse_args();config=json.loads(a.config.read_text())
if a.scenes is not None:config['scenes']=a.scenes
if a.samples_per_scene is not None:config['samples_per_scene']=a.samples_per_scene
export_dataset(config,a.output,a.resume)

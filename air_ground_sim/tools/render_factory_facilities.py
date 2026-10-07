"""Inspect native facility geometry; --review excludes only the two robots.

The final default path always loads the complete registered factory profile.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MUJOCO_GL','egl')
sys.path.insert(0,str(ROOT/'src'))
from air_ground_sim import Simulator
from air_ground_sim.scene.appearance import IndustrialAppearance
from air_ground_sim.scene.registry import sha256
import mujoco

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--review',action='store_true')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--width',type=int,default=1600)
p.add_argument('--height',type=int,default=1000)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sim=None
if a.review:
    source=ROOT/'assets/factory_industrial/authoring/model.xml'
    tree=ET.parse(source);root=tree.getroot();world=root.find('worldbody')
    for body in list(world.findall('body')):
        if body.get('name') in ('dog_mobile_base','uav_fuav_root'):world.remove(body)
    for tag in ('actuator','equality','contact','sensor','tendon','keyframe'):
        for section in root.findall(tag):root.remove(section)
    asset=root.find('asset');used={g.get('mesh') for g in world.iter('geom')}
    for node in list(asset):
        if node.tag=='mesh' and node.get('name') not in used:asset.remove(node)
        elif node.get('file'):node.set('file',str((source.parent/node.get('file')).resolve()))
    file=a.output/'factory_review.xml'
    tree.write(file,encoding='utf-8',xml_declaration=True)
    print('Compiling visual facility review (robots excluded only from this review)...',flush=True)
    m=mujoco.MjModel.from_xml_path(str(file))
    d=mujoco.MjData(m)
    appearance=IndustrialAppearance(m)
    identity=sha256(source)
    mujoco.mj_saveModel(m,str(a.output/'factory_review.mjb'))
else:
    sim=Simulator('factory_industrial')
    m,d=sim.model,sim.data
    appearance=sim.appearance
    identity=sim.model_spec['model_sha256']
mujoco.mj_forward(m,d)
def position(body,height):return d.xpos[m.body(body).id]+[0,0,height]
views=[
    ('assembly',position('industrial_assembly_03',1.02),4.2,130,-18),
    ('control_cabinet',position('industrial_control_03',1.15),3.5,125,-12),
    ('wip_cart',position('industrial_wip_03',.75),3.0,135,-16),
    ('production',[-113,-39,2.],62,220,-14),
    ('warehouse',[-110,-112,2.],65,218,-17),
    ('factory_overview',[0,0,0],620,130,-53),
]
m.vis.global_.offwidth=a.width;m.vis.global_.offheight=a.height;m.vis.map.zfar=4.
option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
records=[]
with mujoco.Renderer(m,width=a.width,height=a.height,max_geom=m.ngeom+2048) as renderer:
    for name,target,distance,azimuth,elevation in views:
        camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
        camera.lookat[:]=target;camera.distance=distance;camera.azimuth=azimuth;camera.elevation=elevation
        appearance.prepare(m,target,distance,renderer);mujoco.mj_forward(m,d)
        renderer.update_scene(d,camera=camera,scene_option=option)
        Image.fromarray(renderer.render()).save(a.output/(name+'.png'))
        records.append(dict(file=name+'.png',lookat=np.asarray(target).tolist(),distance=distance,azimuth=azimuth,elevation=elevation))
        print('Rendered '+name,flush=True)
report=dict(renderer='MuJoCo native Classic',profile='factory_industrial',review_without_robots=a.review,
    source_identity=identity,resolution=[a.width,a.height],geoms=m.ngeom,meshes=m.nmesh,textures=m.ntex,
    views=records,physics_steps=0)
(a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
thumb_w=700;thumb_h=round(a.height/a.width*thumb_w)
sheet=Image.new('RGB',(thumb_w*2,(thumb_h+35)*3),(25,31,38));draw=ImageDraw.Draw(sheet)
for i,row in enumerate(records):
    x=i%2*thumb_w;y=i//2*(thumb_h+35)
    sheet.paste(Image.open(a.output/row['file']).resize((thumb_w,thumb_h),Image.Resampling.LANCZOS),(x,y+35))
    draw.text((x+14,y+12),'MuJoCo | '+row['file'].split('.')[0],fill='white')
sheet.save(a.output/'preview.jpg',quality=93)
if sim:sim.close()

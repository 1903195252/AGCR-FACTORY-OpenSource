"""Fast geometric audit using the exact native primitive collision shapes.

The small model is an audit artifact, never a replacement for the full simulator.
"""
import argparse
import hashlib
from copy import deepcopy
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import air_ground_sim
import mujoco

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--xml',type=Path,default=ROOT/'assets/factory_industrial/authoring/model.xml')
p.add_argument('--output',type=Path,default=ROOT/'outputs/factory_industrial_audit')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
source=ET.parse(a.xml).getroot()
root=ET.Element('mujoco',model='collision_layout_audit')
ET.SubElement(root,'compiler',angle='radian')
world=ET.SubElement(root,'worldbody')


def copy_static(src,dst):
    for g in src.findall('geom'):
        if int(g.get('contype','0'))==0 or g.get('type') in ('mesh','plane'):continue
        g=deepcopy(g)
        for key in ('material','mesh','class','mass','density'):g.attrib.pop(key,None)
        dst.append(g)
    for b in src.findall('body'):
        if b.get('mocap')=='true' or b.find('freejoint') is not None or b.findall('joint'):continue
        child=ET.SubElement(dst,'body',**{k:b.get(k) for k in ('name','pos','quat') if b.get(k) is not None})
        copy_static(b,child)


copy_static(source.find('worldbody'),world)
ET.ElementTree(root).write(a.output/'collision_audit.xml',encoding='utf-8',xml_declaration=True)
print('Compiling collision-only audit model...',flush=True)
m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
d=mujoco.MjData(m);mujoco.mj_forward(m,d)
names=[m.geom(i).name for i in range(m.ngeom)]
new=np.array([n.startswith('industrial_') for n in names])
assembly=np.arange(m.nbody)
for b in range(1,m.nbody):
    parent=int(m.body_parentid[b])
    if parent:assembly[b]=assembly[parent]
geom_assembly=assembly[m.geom_bodyid]
relocated=np.array([m.body(int(b)).name.startswith('layout_line_') for b in geom_assembly])
centers=d.geom_xpos.copy();halves=np.zeros_like(centers)
for i in range(m.ngeom):
    r=d.geom_xmat[i].reshape(3,3);s=m.geom_size[i];typ=m.geom_type[i]
    if typ==mujoco.mjtGeom.mjGEOM_BOX:halves[i]=abs(r)@s
    elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:halves[i]=np.sqrt(r[:,0]**2+r[:,1]**2)*s[0]+abs(r[:,2])*s[1]
    elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:halves[i]=s[0]+abs(r[:,2])*s[1]
    elif typ==mujoco.mjtGeom.mjGEOM_SPHERE:halves[i]=s[0]
    else:halves[i]=m.geom_rbound[i]
lo=centers-halves;hi=centers+halves
issues=[]
for i in np.flatnonzero(new|relocated):
    # Parts within one manufactured fixture deliberately touch/overlap.
    candidate=np.flatnonzero(np.all(np.minimum(hi,hi[i])-np.maximum(lo,lo[i])>.001,axis=1)
                             & (geom_assembly!=geom_assembly[i]))
    for j in candidate:
        if (new[j] or relocated[j]) and j<i:continue
        distance=mujoco.mj_geomDistance(m,d,int(i),int(j),.1,None)
        if distance<-.005:
            issues.append(dict(a=names[i],b=names[j],penetration_m=float(-distance)))

# Geometric road corridors are swept at operating height, excluding road markings
# and overhead services. This is a clearance audit, not autonomous path planning.
routes=[('south_main',[-184,-82],[184,-82],.8),
        ('central_south_lane',[-184,-4],[184,-4],.8),
        ('central_north_lane',[-184,4],[184,4],.8),
        ('north_south',[0,-138],[0,85],.8),
        ('west_walk_south',[-182,-76.6],[-18,-76.6],.28),
        ('east_walk_south',[18,-76.6],[182,-76.6],.28),
        ('west_walk_north',[-182,84.5],[-18,84.5],.28),
        ('east_walk_north',[18,84.5],[182,84.5],.28)]
routes += [('logistics_south',[-188,-138],[185,-138],.8),
           ('logistics_west',[-188,-138],[-188,-84],.8),
           ('logistics_east',[185,-138],[185,-84],.8),
           ('logistics_north',[-188,-84],[185,-84],.8),
           ('south_cross_lower',[-185,-86],[185,-86],.8),
           ('south_cross_upper',[-185,-78],[185,-78],.8),
           ('central_left_outer',[-4.2,-138],[-4.2,88],.8),
           ('central_left_inner',[-1.2,-138],[-1.2,88],.8),
           ('central_right_inner',[1.2,-138],[1.2,88],.8),
           ('central_right_outer',[4.2,-138],[4.2,88],.8),
           ('utility_lower',[-185,86.3],[185,86.3],.8),
           ('utility_upper',[-185,93],[185,93],.8),
           ('west_walk_outer',[-182,-76.6],[-182,84.5],.28),
           ('west_walk_inner',[-18,-76.6],[-18,84.5],.28),
           ('east_walk_inner',[11,-76.6],[11,84.5],.28),
           ('east_walk_outer',[182,-76.6],[182,84.5],.28)]
routes += [('utility_walk_south',[-182.5,90],[182.5,90],.415),
           ('utility_walk_north',[-182.5,138],[182.5,138],.415),
           ('utility_walk_west',[-182.5,90],[-182.5,138],.415),
           ('utility_walk_east',[182.5,90],[182.5,138],.415),
           ('warehouse_walk_south',[-180,-142],[-20,-142],.415),
           ('warehouse_walk_north',[-180,-88],[-20,-88],.415),
           ('warehouse_walk_west',[-180,-142],[-180,-88],.415),
           ('warehouse_walk_east',[-20,-142],[-20,-88],.415),
           ('central_walk_west',[-6.8,-135],[-6.8,85],.415),
           ('central_walk_east',[6.8,-135],[6.8,85],.415),
           ('central_walk_south',[-6.8,-135],[6.8,-135],.415),
           ('central_walk_north',[-6.8,85],[6.8,85],.415)]
route_report=[]
for name,start,end,radius in routes:
    # Measured from native actor contact shapes: truck half-width 0.910002 m,
    # worker half-width 0.364792 m; include 10 cm / 5 cm lateral margin.
    walking='walk' in name
    radius=.415 if walking else 1.011
    if walking:
        start=[v if v!=-76.6 else -76.3 for v in start]
        end=[v if v!=-76.6 else -76.3 for v in end]
        if name.startswith('east_walk'):
            if start[0]==18:start[0]=11
            if end[0]==18:end[0]=11
    start,end=np.array(start),np.array(end)
    low=np.minimum(start,end)-radius;high=np.maximum(start,end)+radius
    hit=(hi[:,2]>.005)&(lo[:,2]<(2. if walking else 3.05))&np.all(hi[:,:2]>low,axis=1)&np.all(lo[:,:2]<high,axis=1)
    blocked=[names[i] for i in np.flatnonzero(hit)]
    route_report.append(dict(name=name,start=start.tolist(),end=end.tolist(),swept_halfwidth_m=radius,blocking_geoms=blocked))
report=dict(audit='Native primitive collision geometry; no proxy size changes',geoms=m.ngeom,
    source_xml_sha256=hashlib.sha256(a.xml.read_bytes()).hexdigest(),
    new_contact_geoms=int(new.sum()),inter_fixture_penetrations=issues,roads=route_report)
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(new_contact_geoms=int(new.sum()),penetrations=len(issues),examples=issues[:12],
    route_blockers={r['name']:r['blocking_geoms'][:8] for r in route_report}),indent=2),flush=True)

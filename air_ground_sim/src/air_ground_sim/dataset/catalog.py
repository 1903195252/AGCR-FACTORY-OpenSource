"""Auditable geom-to-class and object ownership, never inferred from RGB."""
import json
import re
from collections import defaultdict,Counter
import numpy as np
import mujoco
from ..paths import CONFIG
from .geometry import geom_local_corners,quaternion

# Custom factory ontology, not the nuScenes driving benchmark's ten classes.
CATEGORIES=[
 ('ignore','Unresolved/mixed source ownership; excluded from semantic evaluation',(0,0,0)),
 ('flat.floor','Factory floor and aisle coating',(95,100,110)),
 ('static.structure','Walls, columns, roof, beams and lighting',(150,165,180)),
 ('human.pedestrian.worker','Whole worker, including workwear',(235,80,80)),
 ('vehicle.forklift','Forklift including forks and mast',(245,170,35)),
 ('robot.ground','Wheeled quadruped',(40,200,200)),
 ('robot.uav','Original factory UAV',(160,80,230)),
 ('factory.production_line','Whole original production cell, conveyor and machines',(75,120,240)),
 ('factory.workbench','Assembly bench with attached tools and bins',(220,140,90)),
 ('factory.control_cabinet','Electrical/control cabinet',(100,205,145)),
 ('factory.wip_cart','Line-side material cart and its bins',(225,110,195)),
 ('factory.storage_rack','Storage rack including stored cargo',(170,155,70)),
 ('factory.pallet_load','Pallet staging assembly and cargo',(195,125,45)),
 ('static.safety_barrier','Guardrails and rack protection',(245,220,45)),
 ('static.utility','Service pipes, trays and tanks',(80,160,150)),
 ('static.safety_equipment','Fire equipment and emergency signs',(220,60,120)),
 ('flat.marking','Paint and flush floor labels',(245,245,220)),
 ('static.terrain','Authored rough terrain testing surfaces',(110,85,65)),
]
CLASS={n:i for i,(n,_,_) in enumerate(CATEGORIES)}
PALETTE=np.array([c for _,_,c in CATEGORIES],dtype=np.uint8)


def source_class(s):
    if s.startswith(('rack_',)):return 11
    if s.startswith(('warehouse_pallet_',)):return 12
    if s.startswith(('electrical_cabinet','electrical_panel')):return 9
    if s.startswith(('forklift_','mobile_robot_')):return 4
    if s.startswith(('pedestrian_','worker_')):return 3
    if s.startswith(('fire_extinguisher','extinguisher_','emergency_exit','emergency_valve')):return 15
    if s.startswith(('process_tank','utility_header','tank_drop')):return 14
    if s.startswith(('hazard','dock_bumper','rough_zone_guard')):return 13
    if s.startswith('rough_zone_'):return 17
    if s.startswith(('exterior_wall','wall_trim','clerestory','facade_rib','steel_column','column_foot',
                     'roof_','loading_door','factory_foundation')):return 2
    if s.startswith('new::service_trolley'):return 10
    return 0


def body_class(name):
    for prefix,cls,instance in [
        (('worker_','pedestrian_'),3,True),(('forklift_','mobile_robot_'),4,True),
        (('dog_mobile_base',),5,True),(('uav_fuav_root',),6,True),
        (('layout_line_',),7,True),(('industrial_assembly_',),8,True),
        (('industrial_control_',),9,True),(('industrial_wip_',),10,True),
        (('industrial_staging_',),12,True),(('industrial_service_',),14,False),
        (('industrial_rack_',),13,False),(('industrial_fire_',),15,False),
        (('industrial_training_', 'industrial_median_'),16,False)]:
        if name.startswith(prefix):return cls,instance
    return 0,False


class ObjectCatalog:
    def __init__(self,model,data):
        self.model=model
        provenance=json.loads((CONFIG/'factory_semantic_sources.json').read_text())['batches']
        self.geom_class=np.zeros(model.ngeom,np.uint16)
        self.geom_instance=np.zeros(model.ngeom,np.uint16)
        roots=np.zeros(model.nbody,int)
        for b in range(1,model.nbody):
            p=int(model.body_parentid[b]);roots[b]=roots[p] if p else b
        groups=defaultdict(list);classes={};owners={};ignored=[]
        for g in range(model.ngeom):
            if model.geom_group[g] not in (0,1,2,4) or model.geom_rgba[g,3]<=0:continue
            name=model.geom(g).name;body=int(roots[model.geom_bodyid[g]])
            bname=model.body(body).name if body else ''
            cls,thing=body_class(bname);key=bname if thing else None
            # Floor paint belongs to the floor, not the standing object's box.
            if name.endswith(('_zone_front','_zone_back','_zone_left','_zone_right')) or '_floor_' in name:
                cls,key=16,None
            if name=='factory_floor_400x300' or 'aisle' in name:cls,key=1,None
            elif name.startswith('sensor_roof_panel'):cls,key=2,None
            elif not cls:
                record=provenance.get(name)
                if record:
                    kinds={source_class(s) for s in record['sources']}
                    cls=next(iter(kinds)) if len(kinds)==1 else 0
                if not cls:ignored.append(name)
            # Explicit material marking rule also catches fixture zone lines/text.
            mat=int(model.geom_matid[g]);material=model.mat(mat).name if mat>=0 else ''
            if material.startswith('industrial_marking') and data.geom_xpos[g,2]<.01:cls,key=16,None
            self.geom_class[g]=cls
            if key:
                groups[key].append(g);classes[key]=cls;owners[key]=body
        self.objects=[]
        for idx,(name,geoms) in enumerate(sorted(groups.items()),1):
            body=owners[name];root_r=data.xmat[body].reshape(3,3)
            points=[]
            for g in geoms:
                world=geom_local_corners(model,g)@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
                points.append((world-data.xpos[body])@root_r)
            points=np.concatenate(points);lo,hi=points.min(0),points.max(0)
            self.objects.append(dict(id=idx,name=name,body=body,category_id=classes[name],geoms=geoms,
                local_center=(lo+hi)/2,extent_xyz=hi-lo,dynamic=bool(model.body_mocapid[body]>=0),
                articulated=classes[name] in (5,6)))
            self.geom_instance[geoms]=idx
        self.by_id={o['id']:o for o in self.objects}
        self.ignored_geoms=ignored

    def boxes(self,data,offset):
        out={}
        for obj in self.objects:
            b=obj['body'];r=data.xmat[b].reshape(3,3)
            extent=obj['extent_xyz'];center=obj['local_center']
            if obj['articulated']:
                # Bounding box follows current joint pose, not just its root.
                points=[]
                for g in obj['geoms']:
                    world=geom_local_corners(self.model,g)@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
                    points.append((world-data.xpos[b])@r)
                points=np.concatenate(points);lo,hi=points.min(0),points.max(0)
                center=(lo+hi)/2;extent=hi-lo
            out[obj['id']]=dict(translation=(data.xpos[b]+r@center+offset).tolist(),
                size=extent[[1,0,2]].tolist(),rotation=quaternion(r))
        return out

    def masks(self,seg):
        valid=(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(seg[:,:,0]>=0)
        ids=seg[:,:,0].clip(0)
        return np.where(valid,self.geom_class[ids],0).astype(np.uint16),np.where(valid,self.geom_instance[ids],0).astype(np.uint16)

    def manifest(self):
        return dict(categories=[dict(id=i,name=n,description=d,color=list(c)) for i,(n,d,c) in enumerate(CATEGORIES)],
            objects=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in o.items()} for o in self.objects],
            ignored_geom_names=self.ignored_geoms,
            geometric_class_counts=dict(Counter(int(x) for x in self.geom_class)))

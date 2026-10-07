"""Independent inspection layouts built from the project's native factory assets."""
import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from .common import atomic_json, digest

LANES = np.array([3.0, 8.0, 13.0, 18.0, 23.0, 28.0])

def vec(x): return ' '.join(format(float(v), '.9g') for v in x)

def make_layout(index, config):
    rng = np.random.default_rng(np.random.SeedSequence([config['seed'], index, 1041]))
    racks = []
    for row, y in enumerate(LANES[2:]):
        for slot, x in enumerate([-8., 0., 8.]):
            if rng.random() < .22 and slot != 1: continue
            racks.append(dict(x=float(x+rng.uniform(-1.0,1.0)), y=float(y-2.5),
                              half_length=float(rng.uniform(1.6,3.1)), height=float(rng.uniform(3.2,5.1)),
                              fill=float(rng.uniform(.65,1.0)), row=row))
    canopies = []
    for k, y in enumerate(LANES[2:]):
        if rng.random() < .55:
            canopies.append(dict(x=float(rng.choice([-6.,0.,6.])+rng.uniform(-1,1)),
                                 y=float(y), half_length=float(rng.uniform(2.0,3.5)),
                                 z=float(rng.uniform(4.1,5.4)), row=k))
    if not canopies: canopies.append(dict(x=0.,y=18.,half_length=2.7,z=4.5,row=3))
    result = dict(index=index, racks=racks, canopies=canopies,
                  production_x=[float(-26+rng.uniform(-2,2)),float(26+rng.uniform(-2,2))],
                  fixtures=[dict(x=float(x+rng.uniform(-.7,.7)),y=float(y+rng.uniform(-1.0,1.0)))
                            for x in [-17.,17.] for y in [-1.,10.,21.,32.]],
                  light=float(rng.uniform(.55,.85)), floor=float(rng.uniform(.30,.43)),
                  zones=[dict(id='restricted_aisle_0',polygon=[[-1,1.6],[3,1.6],[3,4.4],[-1,4.4]])])
    # Geometry-only fingerprint: lighting does not establish a distinct layout.
    result['geometry_sha256'] = digest({k:v for k,v in result.items() if k not in ('index','light','floor')})
    return result


def build_scene(library_dir, layout, destination, config):
    library_dir = Path(library_dir)
    library = ET.parse(library_dir/'library.xml').getroot()
    root = ET.Element('mujoco', model='agcr_paper_layout_%02d'%layout['index'])
    ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='true',fusestatic='false')
    ET.SubElement(root,'option',gravity='0 0 -9.81',timestep='.002')
    for default in library.findall('default'):root.append(copy.deepcopy(default))
    size=ET.SubElement(root,'size',nuser_geom='1')
    vis=ET.SubElement(root,'visual')
    ET.SubElement(vis,'global',offwidth=str(config['width']),offheight=str(config['height']))
    ET.SubElement(vis,'quality',shadowsize=str(config['shadowsize']),offsamples='0')
    ET.SubElement(vis,'headlight',ambient='.28 .28 .28',diffuse='.35 .35 .35',specular='.15 .15 .15')
    asset=copy.deepcopy(library.find('asset'));root.append(asset)
    for n in asset:
        if n.get('file'): n.set('file',str((library_dir/n.get('file')).resolve()))
    f=layout['floor']
    ET.SubElement(asset,'texture',name='paper_floor_texture',type='2d',builtin='checker',
                  rgb1=vec([f,f+.025,f+.035]),rgb2=vec([f+.025,f+.05,f+.06]),width='512',height='512')
    ET.SubElement(asset,'material',name='paper_floor',texture='paper_floor_texture',texrepeat='24 20',
                  specular='.18',shininess='.25',reflectance='.03')
    world=ET.SubElement(root,'worldbody')
    def box(name,pos,size,color,cls=2,parent=world):
        return ET.SubElement(parent,'geom',name=name,type='box',pos=vec(pos),size=vec(size),rgba=vec(color),
                             group='0',contype='0',conaffinity='0',user=str(cls))
    ET.SubElement(world,'geom',name='floor',type='plane',pos='0 10 0',size='42 31 .1',
                  material='paper_floor',group='0',contype='0',conaffinity='0',user='1')
    for x in [-40,40]: box('wall_x'+str(x),[x,10,8],[.15,30,8],[.67,.71,.73,1])
    for y in [-20,40]: box('wall_y'+str(y),[0,y,8],[40,.15,8],[.67,.71,.73,1])
    box('roof',[0,10,16],[40,30,.12],[.53,.57,.6,1])
    for x in [-36,-18,0,18,36]:
        for y in [-18,38]: box('column_%s_%s'%(x,y),[x,y,8],[.2,.2,8],[.25,.32,.37,1])
        box('roof_beam'+str(x),[x,10,15.6],[.14,29.9,.2],[.27,.32,.36,1])
    for n,(x,y) in enumerate([(-15,-8),(15,-8),(-15,22),(15,22)]):
        light=layout['light']
        ET.SubElement(world,'light',name='light%d'%n,pos=vec([x,y,15.4]),dir='0 0 -1',
                      directional='false',diffuse=vec([light]*3),specular='.12 .12 .12',castshadow='true')
        box('luminaire%d'%n,[x,y,15.6],[2,.2,.06],[.94,.94,.9,1])
    def clone(kind,name,pos,yaw=0,cls=6,mocap=False):
        body=copy.deepcopy(library.find('./worldbody/body[@name="'+kind+'"]'))
        body.set('name',name);body.set('pos',vec(pos));body.set('quat',vec([math.cos(yaw/2),0,0,math.sin(yaw/2)]))
        if mocap: body.set('mocap','true')
        for j,n in enumerate(body.iter()):
            if n is not body and n.get('name'):n.set('name',name+'_'+str(j))
            if n.tag=='geom':
                n.set('user',str(cls));n.set('group','1' if mocap else '0')
        world.append(body)
    for i,x in enumerate(layout['production_x']):clone('production','production_%d'%i,[x,10,0],cls=5)
    for i,pose in enumerate(layout['fixtures']):
        x,y=pose['x'],pose['y'];yaw=0 if x<0 else math.pi
        clone('assembly','assembly_%d'%i,[x,y,0],yaw,6)
        clone('control','control_%d'%i,[x,y+2.1,0],yaw,7)
        clone('wip','wip_%d'%i,[x,y-1.8,0],yaw,8)
    for n,r in enumerate(layout['racks']):
        x,y,L,H=r['x'],r['y'],r['half_length'],r['height']
        for side in [-1,1]:
            for dy in [-.62,.62]:
                box('rack%d_post%s_%s'%(n,side,dy),[x+side*L,y+dy,H/2],[.065,.065,H/2],[.1,.27,.39,1],9)
        for level,z in enumerate(np.linspace(.22,H-.25,4)):
            box('rack%d_shelf%d'%(n,level),[x,y,z],[L+.08,.72,.065],[.82,.33,.06,1],9)
            if level<3:
                crate_h=(H-.47)/6-.08
                for j,dx in enumerate(np.linspace(-L+.48,L-.48, max(2,int(L*2)))):
                    box('rack%d_crate%d_%d'%(n,level,j),[x+dx,y,z+crate_h+.08],
                        [.39,.57,crate_h*r['fill']],[.55+.035*j,.40+.025*j,.25+.02*j,1],9)
                    box('rack%d_strap%d_%d'%(n,level,j),[x+dx,y-.575,z+crate_h+.08],
                        [.025,.008,crate_h*r['fill']],[.75,.66,.45,1],9)
    for n,c in enumerate(layout['canopies']):
        x,y,L,z=c['x'],c['y'],c['half_length'],c['z']
        box('canopy%d_roof'%n,[x,y,z],[L,1.7,.13],[.29,.38,.43,1],2)
        for dx in [-L+.1,L-.1]:
            for dy in [-1.65,1.65]:
                box('canopy%d_post%s_%s'%(n,dx,dy),[x+dx,y+dy,z/2],[.07,.07,z/2],[.27,.35,.40,1],2)
    for y in LANES:
        for dy in [-1.4,1.4]:box('lane_%s_%s'%(y,dy),[0,y+dy,.003],[14,.025,.002],[.91,.67,.11,1],12)
    for i,z in enumerate(layout['zones']):
        (x0,y0),(x1,_),(_,y1),_=z['polygon']
        for x in [x0,x1]:box('zone%d_x%s'%(i,x),[x,(y0+y1)/2,.009],[.035,(y1-y0)/2,.002],[.91,.27,.07,1],12)
        for y in [y0,y1]:box('zone%d_y%s'%(i,y),[(x0+x1)/2,y,.009],[(x1-x0)/2,.035,.002],[.91,.27,.07,1],12)
    # Two candidates per lane allow the class to vary without recompiling a layout.
    for k in range(config['max_targets']):
        clone('forklift','forklift_%d'%k,[0,0,-100],cls=4,mocap=True)
        clone('worker','worker_%d'%k,[0,0,-100],cls=3,mocap=True)
        clone('forklift','loaded_forklift_%d'%k,[0,0,-100],cls=4,mocap=True)
        loaded=world.find('./body[@name="loaded_forklift_%d"]'%k)
        cargo=box('forklift%d_cargo'%k,[1.5,0,1.6],[.95,.70,1.0],[.58,.43,.27,1],4,loaded)
        cargo.set('group','1')
        agv=ET.SubElement(world,'body',name='agv_%d'%k,mocap='true',pos='0 0 -100')
        box('agv%d_chassis'%k,[0,0,.31],[.72,.49,.20],[.18,.24,.29,1],13,agv)
        box('agv%d_deck'%k,[0,0,.54],[.68,.46,.04],[.80,.62,.13,1],13,agv)
        for x in [-.45,.45]:
            for y in [-.49,.49]:
                ET.SubElement(agv,'geom',name='agv%d_wheel%s_%s'%(k,x,y),type='cylinder',
                    pos=vec([x,y,.18]),size='.18 .065',quat='.707106781  .707106781 0 0',
                    rgba='.055 .06 .065 1',user='13',group='1',contype='0',conaffinity='0')
        for x in [-.73,.73]:
            box('agv%d_bumper%s'%(k,x),[x,0,.27],[.055,.51,.075],[.055,.065,.07,1],13,agv)
            box('agv%d_indicator%s'%(k,x),[x*1.075,0,.38],[.006,.15,.025],[.2,.85,.45,1],13,agv)
        ET.SubElement(agv,'geom',name='agv%d_lidar'%k,type='cylinder',pos='.51 0 .65',size='.07 .06',
                      rgba='.12 .15 .17 1',user='13',group='1',contype='0',conaffinity='0')
        box('agv%d_payload'%k,[-.13,0,.77],[.35,.32,.18],[.61,.47,.31,1],13,agv)
        for node in agv.findall('geom'):node.set('group','1')
        clone('uav','uav_%d'%k,[0,0,-100],cls=14,mocap=True)
    # Scripted fixture bodies are fixed throughout each episode. Their positions
    # implement occlusion events; they are not toggled in the rendered images.
    for view in ['air','ground']:
        for kind,cls in [('rack',9),('robot_arm',15),('suspended_load',16),('ground_structure',17)]:
            body=ET.SubElement(world,'body',name='occluder_'+view+'_'+kind,mocap='true',pos='0 0 -100')
            if kind=='rack':
                # Dense stored goods provide an opaque silhouette; frame/supports
                # remain explicit geometry and follow the same transforms.
                box('occ_'+view+'_'+kind+'_cargo',[0,0,0],[1,.28,1],[.61,.46,.29,1],cls,body)
                for sx in [-1,1]:
                    box('occ_'+view+'_'+kind+'_post'+str(sx),[sx*1.03,0,0],[.04,.36,1.08],[.12,.28,.39,1],cls,body)
                for zz in [-1,0,1]:
                    box('occ_'+view+'_'+kind+'_shelf'+str(zz),[0,0,zz],[1.09,.35,.035],[.86,.38,.065,1],cls,body)
            elif kind=='robot_arm':
                box('occ_'+view+'_'+kind+'_workpiece',[0,0,0],[1,.23,1],[.63,.66,.68,1],cls,body)
                for j,(x,z) in enumerate([(1.25,-.5),(1.65,-1.1),(1.65,-1.8)]):
                    box('occ_'+view+'_'+kind+'_link'+str(j),[x,0,z],[.22,.25,.55],[.86,.34,.07,1],cls,body)
            elif kind=='suspended_load':
                box('occ_'+view+'_'+kind+'_load',[0,0,0],[1,.20,1],[.36,.43,.48,1],cls,body)
                for sx in [-.65,.65]:
                    box('occ_'+view+'_'+kind+'_cable'+str(sx),[sx,0,2],[.012,.012,1.], [.15,.16,.17,1],cls,body)
            else:
                box('occ_'+view+'_'+kind+'_panel',[0,0,0],[1,.3,1],[.50,.55,.57,1],cls,body)
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(root).write(destination,encoding='utf-8',xml_declaration=True)
    return destination

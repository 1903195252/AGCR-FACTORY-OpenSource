"""Blender 5: build an editable, articulated wheeled quadruped and MJCF meshes.

blender --background --factory-startup --python tools/build_dog_blender.py
All measurements are meters; X forward, Y left, Z up. Meshes and collision
frames derive from the same link definitions. No source CAD is overwritten.
"""
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / "authoring/dog_rebuilt"
ASSET = ROOT / "assets/dog_rebuilt"
MESH = ASSET / "meshes"
for folder in (AUTHOR, MESH):
    folder.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.context.scene.unit_settings.system = "METRIC"
bpy.context.scene.unit_settings.scale_length = 1

COLORS = {
    "shell": (.055, .095, .125, 1),
    "alloy": (.50, .57, .61, 1),
    "edge": (.19, .25, .28, 1),
    "rubber": (.025, .029, .034, 1),
    "orange": (.95, .25, .045, 1),
    "glass": (.012, .04, .06, 1),
    "led": (.14, .8, .82, 1),
}
materials = {}
for name, color in COLORS.items():
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = .7 if name == "rubber" else .34
    bsdf.inputs["Metallic"].default_value = .75 if name in ("alloy", "edge") else .15
    materials[name] = mat

links = {}
parts = []
colliders = []


def link(name, position, parent=None, joint=None, mass=0, inertia=None):
    obj = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(obj)
    obj.empty_display_type = "ARROWS"
    obj.empty_display_size = .022
    obj.location = position
    obj["mujoco_link"] = name
    if parent:
        obj.parent = links[parent]["object"]
        obj.location = Vector(position) - Vector(links[parent]["position"])
    links[name] = dict(object=obj, position=list(position), parent=parent,
                       joint=joint, mass=mass, inertia=inertia)
    return name


def finish(obj, name, owner, material, bevel=0):
    obj.name = name
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        mod = obj.modifiers.new("Machined edge radius", "BEVEL")
        mod.width = bevel
        mod.segments = 3
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_apply(modifier=mod.name)
    obj.data.materials.append(materials[material])
    world = obj.matrix_world.copy()
    obj.parent = links[owner]["object"]
    obj.matrix_world = world
    obj["mujoco_body"] = owner
    obj["material_name"] = material
    parts.append(obj)
    return obj


def box(name, owner, pos, dimensions, material, bevel=.001):
    bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
    obj = bpy.context.object
    obj.dimensions = dimensions
    return finish(obj, name, owner, material, bevel)


def cylinder(name, owner, pos, radius, depth, material, axis=(0, 1, 0), bevel=.0005):
    bpy.ops.mesh.primitive_cylinder_add(vertices=96, radius=radius, depth=depth, location=pos)
    obj = bpy.context.object
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = Vector(axis).to_track_quat("Z", "Y")
    return finish(obj, name, owner, material, bevel)


def beam(name, owner, start, end, width, depth, material):
    a, b = Vector(start), Vector(end)
    bpy.ops.mesh.primitive_cube_add(size=1, location=(a+b)/2)
    obj = bpy.context.object
    obj.dimensions = (width, depth, (b-a).length)
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = (b-a).to_track_quat("Z", "Y")
    return finish(obj, name, owner, material, .002)


def collision(name, owner, kind, pos, size, **extra):
    local = Vector(pos) - Vector(links[owner]["position"])
    colliders.append(dict(name=name, owner=owner, type=kind, pos=list(local), size=size, **extra))


base = link("dog_mobile_base", (0, 0, 0), mass=1.48, inertia=(.0041, .0075, .0090))
box("sealed_lower_chassis", base, (0, 0, .002), (.216, .134, .055), "shell", .010)
box("upper_service_cover", base, (0, 0, .028), (.204, .127, .018), "edge", .007)
box("underside_skid", base, (0, 0, -.025), (.178, .106, .008), "alloy", .003)
collision("dog_belly_collision", base, "box", (0, 0, .002), [.108, .067, .0275])
collision("dog_skid_collision", base, "box", (0, 0, -.025), [.089, .053, .004])
# Four substantial brackets terminate inside both the chassis and the deck.
for x in (-.080, .080):
    for y in (-.056, .056):
        box(f"deck_mount_{x}_{y}", base, (x, y, .037), (.025, .021, .031), "alloy", .002)
        collision(f"deck_mount_contact_{x}_{y}", base, "box", (x,y,.037), [.0125,.0105,.0155])
for y in (-.063, .063):
    box(f"deck_support_rail_{y}", base, (0, y, .047), (.202, .012, .012), "edge", .002)
    collision(f"deck_rail_contact_{y}", base, "box", (0,y,.047), [.101,.006,.006])
box("load_bearing_deck_base", base, (0, 0, .054), (.260, .180, .006), "edge", .001)
for x in (-.120,.120):
    box(f"deck_end_rim_{x}",base,(x,0,.059),(.020,.180,.004),"edge",.0005)
for y in (-.0815,.0815):
    box(f"deck_side_rim_{y}",base,(0,y,.059),(.220,.017,.004),"edge",.0005)
box("inset_docking_surface", base, (0, 0, .059), (.220, .146, .004), "rubber", .0005)
collision("dog_chassis_collision", base, "box", (0,0,.056), [.130,.090,.005])
for x in (-.117, .117):
    for y in (-.077, .077):
        cylinder(f"deck_fastener_{x}_{y}", base, (x,y,.0604), .003, .0012, "alloy", axis=(0,0,1), bevel=.0002)
for y in (-.081, .081):
    box(f"deck_safety_strip_{y}", base, (0, y, .06055), (.148,.004,.001), "orange", .0003)
    box(f"chassis_side_rail_{y}", base, (0,y*.85,.009), (.15,.006,.015), "edge", .002)
    for i in range(7):
        box(f"vent_{y}_{i}", base, (-.030+i*.010, y*.842, .024), (.004,.002,.013), "rubber", .0006)
# Flush front stereo sensor and bumpers, with physical envelopes.
box("front_sensor_housing", base, (.109,0,.014), (.016,.083,.026), "edge", .004)
collision("front_sensor_collision", base, "box", (.109,0,.014), [.008,.0415,.013])
for y in (-.023,.023):
    cylinder(f"front_camera_{y}", base, (.1175,y,.016), .006, .002, "glass", axis=(1,0,0))
box("front_status_light", base, (.1175,0,.008), (.002,.016,.003), "led", .0003)
for x in (-.112, .112):
    box(f"bumper_{x}", base, (x,0,-.014), (.014,.123,.016), "rubber", .003)
    collision(f"bumper_contact_{x}", base, "box", (x,0,-.014), [.007,.0615,.008])

wheel_radius = .035
wheel_track = .232
leg_specs = [("左前", "lf", .075, 1), ("右前", "rf", .075, -1),
             ("左后", "lr", -.075, 1), ("右后", "rr", -.075, -1)]
for chinese, tag, x, side in leg_specs:
    hip_pos = Vector((x, side*.080, .004))
    knee_pos = Vector((x-.055, side*.104, -.045))
    wheel_pos = Vector((x-.015, side*.116, -.100))
    axis = (0, -side, 0)
    hip = link(f"dog_{tag}_upper", hip_pos, base,
               dict(name=f"dog_{chinese}跨关节", axis=axis, range=(-.26,.26)),
               mass=.12, inertia=(.00012,.00012,.000055))
    knee = link(f"dog_{tag}_lower", knee_pos, hip,
                dict(name=f"dog_{chinese}膝关节", axis=axis, range=(-.44,.44)),
                mass=.09, inertia=(.00008,.00008,.000035))
    wheel = link(f"dog_{tag}_wheel", wheel_pos, knee,
                 dict(name=f"dog_{chinese}轮", axis=axis),
                 mass=.085, inertia=(.000029,.000052,.000029))
    cylinder(f"{tag}_hip_motor", hip, hip_pos, .021, .034, "edge")
    cylinder(f"{tag}_hip_endcap", hip, hip_pos+Vector((0,side*.018,0)), .016, .003, "orange")
    collision(f"{tag}_hip_contact", hip, "cylinder", hip_pos, [.021,.017], quat=[.7071067812,.7071067812,0,0])
    beam(f"{tag}_upper_link", hip, hip_pos, knee_pos, .022,.021,"alloy")
    a=hip_pos-Vector(links[hip]["position"]);b=knee_pos-Vector(links[hip]["position"])
    colliders.append(dict(name=f"{tag}_upper_contact",owner=hip,type="capsule",size=[.011],fromto=list(a)+list(b)))
    cylinder(f"{tag}_knee_housing", knee, knee_pos, .016, .025, "edge")
    cylinder(f"{tag}_knee_bolt", knee, knee_pos+Vector((0,side*.014,0)), .007,.003,"orange")
    beam(f"{tag}_lower_link", knee, knee_pos, wheel_pos, .018,.016,"alloy")
    a=knee_pos-Vector(links[knee]["position"]);b=wheel_pos-Vector(links[knee]["position"])
    colliders.append(dict(name=f"{tag}_lower_contact",owner=knee,type="capsule",size=[.009],fromto=list(a)+list(b)))
    cylinder(f"{tag}_tire", wheel, wheel_pos, wheel_radius, .024, "rubber", bevel=.0015)
    # The cylinder contacts use the same axle, rolling radius and tread width.
    collision(f"dog_{tag}_tire_collision",wheel,"cylinder",wheel_pos,[wheel_radius,.0105],
              quat=[.7071067812,.7071067812,0,0],friction="1 .005 .0001",condim="6")
    for face in (-1,1):
        y=face*.012
        cylinder(f"{tag}_rim_{face}",wheel,wheel_pos+Vector((0,y,0)),.023,.0018,"alloy")
        cylinder(f"{tag}_hub_{face}",wheel,wheel_pos+Vector((0,y+face*.0012,0)),.010,.003,"edge")
        cylinder(f"{tag}_axle_cap_{face}",wheel,wheel_pos+Vector((0,y+face*.003,0)),.004,.002,"orange")
        for i in range(6):
            angle=i*math.tau/6
            p=wheel_pos+Vector((.016*math.cos(angle),y+face*.0012,.016*math.sin(angle)))
            cylinder(f"{tag}_rim_recess_{face}_{i}",wheel,p,.0038,.001,"rubber",bevel=.0001)
    for i in range(32):
        angle=i*math.tau/32
        p=wheel_pos+Vector((.031*math.cos(angle),side*.0122,.031*math.sin(angle)))
        obj=box(f"{tag}_sidewall_grip_{i}",wheel,p,(.005,.0008,.0014),"edge",.0002)
        obj.rotation_euler[1]=-angle

bpy.context.view_layer.update()

# Export evaluated, triangulated meshes in owning link coordinates. Baking this
# transform is essential: Blender's global coordinates must not leak into MJCF.
asset=ET.Element("asset")
for name,color in COLORS.items():
    ET.SubElement(asset,"material",name="rebuilt_"+name,rgba=" ".join(map(str,color)),
                  specular=".3",shininess=".25")
mesh_records=[]
for obj in parts:
    owner=obj["mujoco_body"]
    depsgraph=bpy.context.evaluated_depsgraph_get()
    evaluated=obj.evaluated_get(depsgraph)
    mesh=evaluated.to_mesh()
    mesh.calc_loop_triangles()
    transform=links[owner]["object"].matrix_world.inverted() @ obj.matrix_world
    vertices=[transform @ v.co for v in mesh.vertices]
    file=MESH/(obj.name+".obj")
    with file.open("w") as stream:
        for v in vertices:stream.write("v %.9g %.9g %.9g\n"%tuple(v))
        for tri in mesh.loop_triangles:stream.write("f %d %d %d\n"%tuple(i+1 for i in tri.vertices))
    evaluated.to_mesh_clear()
    ET.SubElement(asset,"mesh",name="rebuilt_"+obj.name,file="meshes/"+file.name)
    mesh_records.append(dict(name=obj.name,body=owner,material=obj["material_name"]))


def fmt(value):
    return " ".join(f"{v:.10g}" for v in value) if isinstance(value,(list,tuple,Vector)) else str(value)


world=ET.Element("worldbody")
bodies={}
for name,spec in links.items():
    parent=spec["parent"]
    local=Vector(spec["position"])-(Vector(links[parent]["position"]) if parent else Vector((0,0,0)))
    if not parent:local.z=.136
    body=ET.SubElement(bodies[parent] if parent else world,"body",name=name,pos=fmt(local))
    bodies[name]=body
    if not parent:ET.SubElement(body,"freejoint",name="dog_base_free")
    joint=spec["joint"]
    if joint:
        attributes=dict(name=joint["name"],axis=fmt(joint["axis"]),type="hinge",
                        damping=".001" if "range" not in joint else ".2",
                        armature=".00002" if "range" not in joint else ".001")
        if "range" in joint:attributes.update(range=fmt(joint["range"]),limited="true")
        ET.SubElement(body,"joint",**attributes)
    # Nominal development inertias; explicitly positive and physically realizable.
    com=(0,0,.012) if not parent else ((-.027,.012*(1 if spec["position"][1]>0 else -1),-.024)
        if name.endswith("upper") else ((.020,0,-.0275) if name.endswith("lower") else (0,0,0)))
    ET.SubElement(body,"inertial",mass=str(spec["mass"]),pos=fmt(com),diaginertia=fmt(spec["inertia"]))
for record in mesh_records:
    ET.SubElement(bodies[record["body"]],"geom",name="rebuilt_"+record["name"],
                  type="mesh",mesh="rebuilt_"+record["name"],material="rebuilt_"+record["material"],
                  contype="0",conaffinity="0",group="1",mass="0")
for record in colliders:
    attr={k:fmt(v) for k,v in record.items() if k!="owner"}
    attr.update(contype="1",conaffinity="1",group="3",rgba=".8 .2 .1 .2",solref=".006 1",solimp=".95 .99 .001")
    ET.SubElement(bodies[record["owner"]],"geom",**attr)
deck_geom = bodies[base].find("geom[@name='dog_chassis_collision']")
deck_top = float(deck_geom.get("pos").split()[2]) + float(deck_geom.get("size").split()[2])
ET.SubElement(bodies[base],"site",name="gate_dog_attachment",pos=f"0 0 {deck_top:.15g}",size=".003",rgba="0 0 0 0")
ET.SubElement(bodies[base],"camera",name="ground_rgbd",pos=".120 0 .016",quat="-.4743855 -.5243647 .5243647 .4743855",fovy="58")
actuator=ET.Element("actuator")
for chinese,_,_,_ in leg_specs:
    for joint_kind in ("跨关节","膝关节"):
        name="dog_"+chinese+joint_kind
        ET.SubElement(actuator,"position",name=name+"_position",joint=name,kp="35",kv=".6",
                      ctrllimited="true",ctrlrange="-.26 .26" if joint_kind=="跨关节" else "-.44 .44",
                      forcelimited="true",forcerange="-2.5 2.5")
    name="dog_"+chinese+"轮"
    ET.SubElement(actuator,"motor",name=name+"_motor",joint=name,ctrllimited="true",ctrlrange="-.46 .46")
fragment=ET.Element("mujoco",model="Blender_rebuilt_wheeled_quadruped")
ET.SubElement(fragment,"compiler",angle="radian",inertiafromgeom="false",autolimits="true")
ET.SubElement(fragment,"option",timestep=".0005",gravity="0 0 -9.81",integrator="implicitfast")
visual=ET.SubElement(fragment,"visual")
ET.SubElement(visual,"global",offwidth="1920",offheight="1200")
ET.SubElement(visual,"headlight",ambient=".3 .3 .3",diffuse=".7 .7 .7")
ET.SubElement(fragment,"statistic",center="0 0 .12",extent=".6")
fragment.append(asset)
ET.SubElement(world,"geom",name="factory_floor_400x300",type="plane",size="2 2 .1",rgba=".32 .36 .40 1",contype="2",conaffinity="1")
ET.SubElement(world,"light",pos="-1 -2 3",dir=".2 .3 -1",directional="true",diffuse=".8 .8 .8")
fragment.append(world);fragment.append(actuator)
# Explicit parent filters avoid applying contact forces inside shared joint housings.
contact=ET.SubElement(fragment,"contact")
for name,spec in links.items():
    if spec["parent"]:
        ET.SubElement(contact,"exclude",body1=name,body2=spec["parent"])
ET.indent(fragment)
ET.ElementTree(fragment).write(ASSET/"model.xml",encoding="utf-8",xml_declaration=True)
specification=dict(wheel_radius_m=wheel_radius,wheel_track_m=wheel_track,deck_top_root_m=.061,
                   root_initial_height_m=.136,mass_kg=sum(x["mass"] for x in links.values()),
                   mesh_count=len(parts),collision_geoms=len(colliders),
                   nominal_inertias=True,blender_version=bpy.app.version_string,
                   visual_and_collision_frames="shared link local frames",source_script="tools/build_dog_blender.py")
(ASSET/"design.json").write_text(json.dumps(specification,indent=2)+"\n")

# Save a native editable Blender assembly with studio objects in a separate collection.
studio=bpy.data.collections.new("STUDIO - not exported to MuJoCo")
bpy.context.scene.collection.children.link(studio)
def studio_object(obj):
    for collection in list(obj.users_collection):collection.objects.unlink(obj)
    studio.objects.link(obj)

# Shift the assembly root only for the Blender studio; exported mesh coordinates stay local.
links[base]["object"].location.z=.135
bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,0))
floor=bpy.context.object;floor.name="Studio ground";studio_object(floor)
floor_mat=bpy.data.materials.new("Studio warm gray");floor_mat.diffuse_color=(.16,.19,.22,1)
floor.data.materials.append(floor_mat)
for name,pos,power,size in [("Key",(.2,-.4,.75),3,.6),("Fill",(-.4,.3,.45),2,.5),("Rim",(.3,.5,.7),4,.4)]:
    bpy.ops.object.light_add(type="AREA",location=pos)
    light=bpy.context.object;light.name=name;light.data.energy=power;light.data.shape="DISK";light.data.size=size
    light.rotation_euler=(Vector((0,0,.12))-light.location).to_track_quat("-Z","Y").to_euler();studio_object(light)
bpy.ops.object.camera_add(location=(.43,-.49,.32))
camera=bpy.context.object;camera.name="Presentation camera";studio_object(camera)
camera.rotation_euler=(Vector((0,0,.105))-camera.location).to_track_quat("-Z","Y").to_euler()
camera.data.type="ORTHO";camera.data.ortho_scale=.48
scene=bpy.context.scene;scene.camera=camera
scene.render.engine="CYCLES";scene.cycles.samples=40;scene.cycles.use_denoising=True
scene.render.resolution_x=1600;scene.render.resolution_y=1200;scene.render.resolution_percentage=100
scene.world.color=(.18,.18,.18)
scene.view_settings.view_transform="AgX"
bpy.ops.object.select_all(action="DESELECT")
links[base]["object"].select_set(True);bpy.context.view_layer.objects.active=links[base]["object"]
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type=="VIEW_3D":
            area.spaces.active.region_3d.view_distance=.65
            area.spaces.active.region_3d.view_location=(0,0,.11)
            area.spaces.active.clip_start=.001
bpy.ops.wm.save_as_mainfile(filepath=str(AUTHOR/"dog_rebuilt.blend"))
scene.render.filepath=str(AUTHOR/"blender_preview.png")
bpy.ops.render.render(write_still=True)
print("BUILT",json.dumps(specification),flush=True)

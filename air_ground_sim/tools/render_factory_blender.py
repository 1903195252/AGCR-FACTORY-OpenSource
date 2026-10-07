"""Blender CLI: exact MuJoCo snapshot with procedural PBR materials and Cycles.

blender -b --python tools/render_factory_blender.py -- --snapshot outputs/industrial_snapshot
No geometry is moved or enlarged for the image. Shaders use bump, not displacement.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector


ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--snapshot', type=Path, required=True)
parser.add_argument('--output', type=Path, default=ROOT / 'outputs/industrial_cycles')
parser.add_argument('--blend', type=Path, default=ROOT / 'authoring/factory_industrial/factory_industrial.blend')
parser.add_argument('--samples', type=int, default=64)
parser.add_argument('--width', type=int, default=1600)
parser.add_argument('--height', type=int, default=1000)
parser.add_argument('--views', nargs='+', default=['dog', 'team', 'production', 'warehouse'])
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
args.output.mkdir(parents=True, exist_ok=True)
args.blend.parent.mkdir(parents=True, exist_ok=True)
snapshot = json.loads((args.snapshot / 'scene.json').read_text())
appearance_file = args.snapshot / 'appearance.json'
if not appearance_file.exists():
    appearance_file = ROOT / 'configs/appearance_industrial.json'
appearance_bytes = appearance_file.read_bytes()
config = json.loads(appearance_bytes)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.render.engine = 'CYCLES'
scene.cycles.samples = args.samples
scene.cycles.use_denoising = True
scene.cycles.adaptive_threshold = .025
scene.cycles.max_bounces = 7
scene.cycles.diffuse_bounces = 3
scene.cycles.glossy_bounces = 4
scene.cycles.transmission_bounces = 4
scene.cycles.transparent_max_bounces = 6
prefs = bpy.context.preferences.addons['cycles'].preferences
try:
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    gpu = [d for d in prefs.devices if d.type == 'OPTIX']
    for device in prefs.devices:
        device.use = device in gpu
    if gpu:
        scene.cycles.device = 'GPU'
except (TypeError, RuntimeError):
    pass
scene.render.resolution_x = args.width
scene.render.resolution_y = args.height
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'AgX - Medium High Contrast'
scene.view_settings.exposure = -.4
world = bpy.data.worlds.new('Soft industrial daylight')
world.use_nodes = True
world.node_tree.nodes['Background'].inputs['Color'].default_value = (.64, .73, .86, 1)
world.node_tree.nodes['Background'].inputs['Strength'].default_value = .25
scene.world = world

materials = {}


def material(record):
    color = record['rgba'][:3]
    kind = record['material_class']
    key = (record['material'], *[round(c, 5) for c in color], round(record['rgba'][3], 4))
    if key in materials:
        return materials[key]
    mat = bpy.data.materials.new(record['material'] + '_PBR')
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    values = config['material_classes'][kind]
    bsdf.inputs['Base Color'].default_value = (*color, 1)
    bsdf.inputs['Roughness'].default_value = values['roughness']
    bsdf.inputs['Metallic'].default_value = values['metallic']
    if kind == 'paint':
        bsdf.inputs['Coat Weight'].default_value = .2
        bsdf.inputs['Coat Roughness'].default_value = .32
    if kind == 'glass':
        bsdf.inputs['Transmission Weight'].default_value = max(0., 1-record['rgba'][3])
        bsdf.inputs['IOR'].default_value = 1.46
    if kind == 'emissive':
        bsdf.inputs['Emission Color'].default_value = (*color, 1)
        bsdf.inputs['Emission Strength'].default_value = 2.5
    # Object coordinates retain metre scale on mesh imports. Noise is procedural,
    # self contained in the blend file, and never offsets the physical surface.
    coord = nodes.new('ShaderNodeTexCoord')
    noise = nodes.new('ShaderNodeTexNoise')
    links.new(coord.outputs['Object'], noise.inputs['Vector'])
    noise.inputs['Scale'].default_value = {'floor': 95, 'concrete': 65, 'rubber': 700, 'metal': 450, 'wood': 24}.get(kind, 280)
    noise.inputs['Detail'].default_value = 3
    noise.inputs['Roughness'].default_value = .72
    ramp = nodes.new('ShaderNodeValToRGB')
    variation = .24 if kind in ('floor','concrete','wood') else .07
    ramp.color_ramp.elements[0].position = .18
    ramp.color_ramp.elements[1].position = .82
    ramp.color_ramp.elements[0].color = (*[c*(1-variation) for c in color], 1)
    ramp.color_ramp.elements[1].color = (*[min(1,c*(1+variation)) for c in color], 1)
    links.new(noise.outputs['Fac'], ramp.inputs[0])
    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    rough = nodes.new('ShaderNodeMapRange')
    rough.inputs['To Min'].default_value = max(.06, values['roughness']-.1)
    rough.inputs['To Max'].default_value = min(.96, values['roughness']+.1)
    links.new(noise.outputs['Fac'], rough.inputs['Value'])
    links.new(rough.outputs[0], bsdf.inputs['Roughness'])
    bump = nodes.new('ShaderNodeBump')
    bump.inputs['Strength'].default_value = .22
    bump.inputs['Distance'].default_value = .00035 if kind in ('floor','concrete') else .000025
    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    if kind in ('floor', 'concrete'):
        # Restrained broad mottling in the coating, independent of the micrograin.
        broad = nodes.new('ShaderNodeTexNoise')
        broad.inputs['Scale'].default_value = .7
        broad.inputs['Detail'].default_value = 2
        links.new(coord.outputs['Object'], broad.inputs['Vector'])
        mix = nodes.new('ShaderNodeMixRGB')
        mix.blend_type = 'MULTIPLY'
        mix.inputs[0].default_value = .16
        links.new(ramp.outputs['Color'], mix.inputs[1])
        links.new(broad.outputs['Fac'], mix.inputs[2])
        links.new(mix.outputs[0], bsdf.inputs['Base Color'])
    materials[key] = mat
    return mat


factory = bpy.data.collections.new('Factory - MuJoCo visual meshes')
robots = bpy.data.collections.new('Robots - simulated pose')
scene.collection.children.link(factory)
scene.collection.children.link(robots)
meshes = {}
with np.load(args.snapshot / 'geometry.npz') as archive:
    for n, i in enumerate(snapshot['mesh_ids']):
        vertices, faces = archive[f'v{i}'], archive[f'f{i}']
        mesh = bpy.data.meshes.new(f'mujoco_mesh_{i}')
        mesh.vertices.add(len(vertices))
        mesh.vertices.foreach_set('co', vertices.ravel())
        mesh.loops.add(faces.size)
        mesh.loops.foreach_set('vertex_index', faces.ravel())
        mesh.polygons.add(len(faces))
        mesh.polygons.foreach_set('loop_start', np.arange(len(faces), dtype=np.int32)*3)
        mesh.polygons.foreach_set('loop_total', np.full(len(faces), 3, dtype=np.int32))
        mesh.polygons.foreach_set('use_smooth', np.ones(len(faces), dtype=bool))
        mesh.update()
        mesh.set_sharp_from_angle(angle=math.radians(40))
        meshes[i] = mesh
        if n % 700 == 0:
            print(f'Imported {n}/{len(snapshot["mesh_ids"])} meshes', flush=True)

primitive_meshes = {}
for record in snapshot['geoms']:
    typ = record['type']
    if typ == 7:
        mesh = meshes[record['mesh']]
        obj = bpy.data.objects.new(record['name'], mesh)
        (robots if record['body'].startswith(('dog_', 'uav_')) else factory).objects.link(obj)
    else:
        size = record['size']
        key = (typ, tuple(size))
        cached = primitive_meshes.get(key)
        if cached is not None:
            obj = bpy.data.objects.new(record['name'], cached)
        elif typ in (0, 6):
            bpy.ops.mesh.primitive_cube_add(size=2)
            obj = bpy.context.object
            obj.scale = size if typ == 6 else [max(200.,size[0]), max(150.,size[1]), .001]
        elif typ in (2, 4):
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=1)
            obj = bpy.context.object
            obj.scale = [size[0]]*3 if typ == 2 else size
        elif typ == 5:
            bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=size[0], depth=2*size[1])
            obj = bpy.context.object
        else:
            raise ValueError(f'Unsupported visible geom type {typ}: {record["name"]}')
        # Apply scale so procedural texture coordinates are measured in metres.
        if cached is None:
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
            primitive_meshes[key] = obj.data
        obj.name = record['name']
        for collection in list(obj.users_collection):
            collection.objects.unlink(obj)
        factory.objects.link(obj)
    rotation = Matrix(record['rotation']).to_4x4()
    rotation.translation = Vector(record['position'])
    if typ == 0:
        rotation.translation.z -= .001  # box stands in for the plane, top remains Z=0.
    obj.matrix_world = rotation
    mat = material(record)
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    obj.material_slots[0].link = 'OBJECT'
    obj.material_slots[0].material = mat
    obj['mujoco_geom_id'] = record['id']
    obj['mujoco_body'] = record['body']

lights = bpy.data.collections.new('Industrial soft lighting')
scene.collection.children.link(lights)
for x in range(-175, 176, 50):
    for y in range(-125, 126, 50):
        data = bpy.data.lights.new(f'Ceiling softbox {x} {y}', 'AREA')
        data.energy = 3500
        data.shape = 'DISK'
        data.size = 16
        data.color = (1., .91, .79)
        obj = bpy.data.objects.new(data.name, data)
        lights.objects.link(obj)
        obj.location = (x,y,11.5)
sun = bpy.data.lights.new('Daylight through open factory roof', 'SUN')
sun.energy = 1.0
sun.angle = math.radians(18)
sun.color = (.83, .91, 1.)
obj = bpy.data.objects.new(sun.name, sun)
lights.objects.link(obj)
obj.rotation_euler = Vector((.35,.25,-1)).to_track_quat('-Z','Y').to_euler()

dog = np.array(snapshot['ground_position'])
air = np.array(snapshot['air_position'])
views = {
    'dog': (dog+[0,0,-.025], .65, 140, -23),
    'team': ((air+dog)/2, 1.8, 135, -18),
    'production': ([-113,-39,2], 62, 220, -14),
    'warehouse': ([-110,-112,2], 65, 218, -17),
    'aisle': ([-113,-39,1.4], 42, 220, -2),
    'overview': ([0,0,0], 620, 130, -53),
}
if snapshot['profile']=='factory_industrial':
    layout=json.loads((ROOT/'configs/factory_industrial_layout.json').read_text())
    positions={item['name']:np.array(item['position']) for item in layout['modules']}
    bench=positions['industrial_assembly_03']
    views['workcell']=(bench+[.7,-1.,.85],6.,135,-16)
    views['assembly']=(bench+[0,0,1.02],4.2,130,-18)
    views['cabinet']=(positions['industrial_control_03']+[0,0,1.15],3.5,125,-12)
cameras = {}
for name, (target,distance,azimuth,elevation) in views.items():
    cam = bpy.data.cameras.new(name)
    cam.type = 'PERSP'
    cam.sensor_fit = 'VERTICAL'
    cam.sensor_height = 24
    cam.lens = 12 / math.tan(math.radians(45)/2)
    cam.clip_start, cam.clip_end = .02, 1200
    obj = bpy.data.objects.new(name, cam)
    scene.collection.objects.link(obj)
    a, e = math.radians(azimuth), math.radians(elevation)
    # MuJoCo azimuth is the viewing direction; eye is behind lookat.
    offset = np.array([-math.cos(a)*math.cos(e),-math.sin(a)*math.cos(e),-math.sin(e)]) * distance
    obj.location = np.array(target)+offset
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat('-Z','Y').to_euler()
    cameras[name] = obj
scene.camera = cameras[args.views[0]]
scene['source_profile'] = snapshot['profile']
scene['physics_time_s'] = snapshot['time_s']
scene['model_sha256'] = snapshot['model_sha256']
scene['geometry_contract'] = 'MuJoCo visible geometry and simulated transforms; shading-only PBR bump'
bpy.ops.wm.save_as_mainfile(filepath=str(args.blend.resolve()), compress=True)
report = dict(engine='Blender Cycles', blender_version=bpy.app.version_string,
    appearance_sha256=hashlib.sha256(appearance_bytes).hexdigest(),
    samples=args.samples, resolution=[args.width,args.height], model_sha256=snapshot['model_sha256'],
    physics_time_s=snapshot['time_s'], source_snapshot=str(args.snapshot.resolve()),
    visible_geoms=len(snapshot['geoms']), meshes=len(meshes), materials=len(materials), images=[])
for name in args.views:
    scene.camera = cameras[name]
    scene.render.filepath = str((args.output / (name+'.png')).resolve())
    bpy.ops.render.render(write_still=True)
    report['images'].append(dict(file=name+'.png',camera=views[name]))
    print(f'Rendered {name}', flush=True)
(args.output / 'manifest.json').write_text(json.dumps(report,indent=2,default=lambda x: x.tolist())+'\n')

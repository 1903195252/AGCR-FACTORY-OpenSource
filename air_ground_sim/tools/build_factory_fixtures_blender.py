"""Generate bevelled, reusable MJCF fixture meshes in Blender."""
import json
from pathlib import Path
import sys
import bpy

root=Path(sys.argv[sys.argv.index('--')+1]).resolve()
shapes=json.loads((root/'shapes.json').read_text())
bpy.ops.wm.read_factory_settings(use_empty=True)
(root/'meshes').mkdir(exist_ok=True)
for i,(name,spec) in enumerate(shapes.items()):
    if spec['type']=='box':
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj=bpy.context.object
        obj.dimensions=spec['dimensions']
        bpy.ops.object.transform_apply(location=False,rotation=False,scale=True)
        bevel=obj.modifiers.new('Manufactured edge radius','BEVEL')
        bevel.width=min(.012,min(spec['dimensions'])*.12)
        bevel.segments=3
        bpy.ops.object.modifier_apply(modifier=bevel.name)
    else:
        curve=bpy.data.curves.new(name,'FONT')
        curve.body=spec['text']
        curve.size=spec['size']
        curve.extrude=.0001
        curve.resolution_u=2
        obj=bpy.data.objects.new(name,curve)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active=obj
        obj.select_set(True)
        bpy.ops.object.convert(target='MESH')
    obj.name=name
    mesh=obj.data
    mesh.calc_loop_triangles()
    with (root/'meshes'/(name+'.obj')).open('w') as f:
        f.write('# Blender fixture mesh; metres, local coordinates\n')
        for v in mesh.vertices:
            f.write('v %.9g %.9g %.9g\n'%tuple(v.co))
        for t in mesh.loop_triangles:
            f.write('f %d %d %d\n'%tuple(v+1 for v in t.vertices))
    obj.location=(i%12*3, i//12*3, 0)
    obj.select_set(False)
bpy.context.scene.unit_settings.system='METRIC'
bpy.ops.wm.save_as_mainfile(filepath=str(root/'fixture_library.blend'),compress=True)
print(f'Exported {len(shapes)} reusable fixture meshes',flush=True)

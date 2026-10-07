"""Export self-contained editable MJCF, optionally keeping one scene component."""
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
from ..paths import BUNDLE
from .registry import profiles


def export_scene(profile,component,output):
    if component not in ('all','factory','uav','dog'):raise ValueError(component)
    output=Path(output)
    if output.exists():raise FileExistsError('Use a new directory to preserve prior exports')
    source=BUNDLE/profiles()[profile]['authoring']
    tree=ET.parse(source);root=tree.getroot();world=root.find('worldbody')
    roots={'uav':'uav_fuav_root','dog':'dog_mobile_base'}
    parents={child:node for node in root.iter() for child in node}
    if component!='all':
        keep=set(roots.values()) if component=='factory' else {roots[component]}
        if component=='factory':
            for node in list(world.findall('body')):
                if node.get('name') in keep:world.remove(node)
        else:
            chosen=root.find(".//body[@name='%s']"%roots[component])
            if chosen is None:raise ValueError('Missing component root')
            for node in list(world):
                if node is chosen or node.tag=='light' or node.get('name')=='factory_floor_400x300':continue
                world.remove(node)
        # Remove references to bodies/joints/sites/geoms no longer in the tree.
        names={kind:{node.get('name') for node in world.iter(kind) if node.get('name')}
               for kind in ('body','joint','freejoint','site','geom','camera')}
        names['body'].add('world');names['joint']|=names['freejoint']
        reference_keys={'body':'body','body1':'body','body2':'body','joint':'joint',
                        'joint1':'joint','joint2':'joint','site':'site','site1':'site','site2':'site',
                        'geom':'geom','geom1':'geom','geom2':'geom','target':'body'}
        for section in ('actuator','equality','contact','sensor','tendon'):
            parent=root.find(section)
            if parent is None:continue
            for node in list(parent):
                if any(key in reference_keys and value not in names[reference_keys[key]]
                       for desc in node.iter() for key,value in desc.attrib.items()):parent.remove(node)
        for node in list(root.findall('keyframe')):root.remove(node)
    # Keep visual and contact meshes referenced by the retained world, without changing vertices.
    asset=root.find('asset')
    asset_nodes=set(asset)
    used={key:{node.get(key) for node in root.iter() if node not in asset_nodes and node.get(key)}
          for key in ('mesh','material','texture','hfield','skin')}
    for material in asset.findall('material'):
        if material.get('name') in used['material'] and material.get('texture'):used['texture'].add(material.get('texture'))
    for node in list(asset):
        if node.tag in used and node.get('name') not in used[node.tag]:asset.remove(node)
    output.mkdir(parents=True)
    files=[]
    for node in asset:
        if not node.get('file'):continue
        original=source.parent/node.get('file');target=output/'assets'/original.name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():shutil.copy2(original,target)
        node.set('file','assets/'+original.name)
        files.append('assets/'+original.name)
    tree.write(output/'model.xml',encoding='utf-8',xml_declaration=True)
    (output/'export.json').write_text(json.dumps(dict(profile=profile,component=component,
        source_model_sha256=profiles()[profile]['model_sha256'],files=sorted(set(files)),
        physics_parameters_changed=False,compiled=False,
        note='Isolated components retain authored initial poses and the factory floor; custom edits require recompilation.'),indent=2)+'\n')
    return output/'model.xml'

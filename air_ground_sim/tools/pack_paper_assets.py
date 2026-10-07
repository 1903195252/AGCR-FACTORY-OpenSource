"""Extract reusable original factory modules with source hashes (no figure scene)."""
import argparse
import copy
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from paper_dataset.common import file_hash

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    source = ET.parse(a.source).getroot()
    assets = {n.get('name'): n for n in source.find('asset')}
    selected = {
        'forklift': 'forklift_dynamic_00', 'worker': 'worker_dynamic_00',
        'assembly': 'industrial_assembly_01', 'control': 'industrial_control_01',
        'wip': 'industrial_wip_01', 'staging': 'industrial_staging_01',
        'production': 'layout_line_r00_c00',
        'uav': 'uav_fuav_root',
    }
    root = ET.Element('mujoco', model='original_factory_asset_library')
    for default in source.findall('default'):root.append(copy.deepcopy(default))
    asset = ET.SubElement(root, 'asset')
    world = ET.SubElement(root, 'worldbody')
    counts = {}
    for kind, name in selected.items():
        item = source.find('.//body[@name="' + name + '"]')
        if item is None:
            raise ValueError('Missing native component ' + name)
        body = copy.deepcopy(item)
        body.set('name', kind)
        body.set('pos', '0 0 0')
        body.set('quat', '1 0 0 0')
        for parent in body.iter():
            for node in list(parent):
                if node.tag in ('joint', 'freejoint', 'inertial', 'site', 'camera'):
                    parent.remove(node)
                elif node.tag == 'geom':
                    rgba = node.get('rgba', '1 1 1 1').split()
                    if node.get('group') == '3' or float(rgba[-1]) == 0:
                        parent.remove(node)
                    else:
                        node.set('contype', '0'); node.set('conaffinity', '0')
        body.attrib.pop('mocap', None)
        world.append(body)
        counts[kind] = len(body.findall('.//geom'))
    used = set()
    for node in world.iter():
        for key in ('mesh', 'material'):
            if node.get(key): used.add(node.get(key))
    while True:
        before = set(used)
        for name in list(used):
            for key in ('texture',):
                if assets[name].get(key): used.add(assets[name].get(key))
        if used == before: break
    mapping = {}
    for name in sorted(used):
        node = copy.deepcopy(assets[name])
        if node.get('file'):
            src = (a.source.parent / node.get('file')).resolve()
            if not src.is_file(): raise FileNotFoundError(src)
            if src not in mapping: mapping[src] = 'assets/' + file_hash(src) + src.suffix
            node.set('file', mapping[src])
        asset.append(node)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.output, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.writestr('library.xml', ET.tostring(root, encoding='utf-8'))
        done = set()
        for src, dest in mapping.items():
            if dest not in done: z.write(src, dest); done.add(dest)
        z.writestr('provenance.json', json.dumps({
            'source': 'air_ground_sim/assets/factory_industrial/authoring/model.xml',
            'source_xml_sha256': file_hash(a.source), 'components': selected,
            'visual_geom_counts': counts,
            'scope': 'Original factory modules rearranged into independently specified inspection layouts; not the Fig2 scene or the unchanged 400x300m factory.',
            'assets': {dest: file_hash(src) for src, dest in mapping.items()},
        }, indent=2))
    print(json.dumps({'file': str(a.output), 'bytes': a.output.stat().st_size, 'geoms': counts}))

if __name__ == '__main__': main()

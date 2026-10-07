#!/usr/bin/env python3
"""Export the actual visible MuJoCo geometry/state for Blender, without recompile."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from air_ground_sim import Simulator
from air_ground_sim.io import restore_physics_state, save_physics_state
from air_ground_sim.scene.appearance import material_class
from air_ground_sim.paths import CONFIG
import mujoco


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', default='factory_rebuilt')
    p.add_argument('--replay', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    appearance_bytes = (CONFIG / 'appearance_industrial.json').read_bytes()
    (args.output / 'appearance.json').write_bytes(appearance_bytes)
    with Simulator(args.profile) as sim:
        m, d = sim.model, sim.data
        if args.replay:
            restore_physics_state(m, d, args.replay, sim.model_spec['model_sha256'])
        mujoco.mj_forward(m, d)
        visible = np.flatnonzero((m.geom_group < 3) & (m.geom_rgba[:, 3] > 0))
        used = sorted(set(int(m.geom_dataid[g]) for g in visible if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH))
        arrays = {}
        for i in used:
            v, nv = int(m.mesh_vertadr[i]), int(m.mesh_vertnum[i])
            f, nf = int(m.mesh_faceadr[i]), int(m.mesh_facenum[i])
            arrays[f'v{i}'] = m.mesh_vert[v:v+nv]
            arrays[f'f{i}'] = m.mesh_face[f:f+nf]
        print(f'Exporting {len(visible)} geoms / {len(used)} visible meshes', flush=True)
        np.savez_compressed(args.output / 'geometry.npz', **arrays)
        records = []
        for g in visible:
            mat = int(m.geom_matid[g])
            rgba = m.geom_rgba[g].copy()
            # Same rule as mjv_updateScene: default geom gray inherits its material.
            if mat >= 0 and np.array_equal(rgba, np.array([.5, .5, .5, 1], np.float32)):
                rgba = m.mat_rgba[mat].copy()
            name = m.mat(mat).name if mat >= 0 else 'unassigned'
            records.append(dict(id=int(g), name=m.geom(g).name, body=m.body(int(m.geom_bodyid[g])).name,
                type=int(m.geom_type[g]), mesh=int(m.geom_dataid[g]), size=m.geom_size[g].tolist(),
                position=d.geom_xpos[g].tolist(), rotation=d.geom_xmat[g].reshape(3,3).tolist(),
                material=name, material_class=material_class(name), rgba=rgba.tolist(),
                emission=float(m.mat_emission[mat]) if mat >= 0 else 0.))
        state = sim.state()
        result = dict(profile=args.profile, time_s=sim.time, model_sha256=sim.model_spec['model_sha256'],
            appearance_sha256=hashlib.sha256(appearance_bytes).hexdigest(),
            appearance=sim.appearance_name, geometry_source='Exact compiled MuJoCo visible meshes and world transforms',
            air_position=state['air']['position_m'].tolist(), ground_position=state['ground']['position_m'].tolist(),
            mesh_ids=used, geoms=records)
        (args.output / 'scene.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        save_physics_state(sim, args.output / 'physics_state.npz')
        print(args.output / 'scene.json', flush=True)


if __name__ == '__main__':
    main()

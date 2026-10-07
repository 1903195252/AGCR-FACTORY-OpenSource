"""Portable, documented exports, not learning checkpoints."""
import json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image


def jsonable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:jsonable(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [jsonable(v) for v in value]
    return value


def export_observations(observations, directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    for robot,obs in observations.items():
        rgbd,cloud=obs['rgbd'],obs['lidar']
        Image.fromarray(rgbd['rgb']).save(directory/(robot+'_rgb.png'))
        depth_mm=rgbd['depth_mm'] if 'depth_mm' in rgbd else np.asarray(rgbd['depth_m'])*1000
        np.savez_compressed(directory/(robot+'_sensors.npz'),depth_mm=np.asarray(depth_mm,dtype=np.float32),depth_valid=rgbd['valid'],
            K=rgbd['K'],camera_position=rgbd['camera_position'],camera_rotation=rgbd['camera_rotation'],
            timestamp_s=rgbd['timestamp_s'],lidar_xyz_world=cloud['xyz_world'],
            lidar_ranges_m=cloud['ranges_m'],lidar_valid=cloud['valid'],lidar_origin=cloud['origin'])
        write_ply(directory/(robot+'_lidar.ply'),cloud['xyz_world'])
    (directory/'schema.json').write_text(json.dumps(dict(schema='air_ground_sensors_v2',
        units=dict(depth_mm='mm',camera_position='m',lidar_xyz_world='m',lidar_ranges_m='m',lidar_origin='m',timestamp_s='s'),
        depth_definition='optical-axis Z; float32; convert mm to m before world-frame back-projection',
        camera_axes='CV: X right, Y down, Z forward; camera_rotation maps to world',
        invalid_depth=0,point_frame='world',metadata_labels_included=False),indent=2)+'\n')


def write_ply(path, xyz):
    xyz=np.asarray(xyz,dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not np.isfinite(xyz).all():raise ValueError('Finite Nx3 point array required')
    with Path(path).open('x') as stream:
        stream.write(f'ply\nformat ascii 1.0\nelement vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\nend_header\n')
        np.savetxt(stream,xyz,fmt='%.7g')


def save_physics_state(sim,path):
    """Exact native integration state; NOT controller/observer or policy state."""
    state_type=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(sim.model,state_type))
    mujoco.mj_getState(sim.model,sim.data,state,state_type)
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream,integration_state=state,state_type=int(state_type),
            model_sha256=sim.model_spec['model_sha256'],time_s=sim.time,
            controller_state_included=False)


def restore_physics_state(model,data,path,model_sha256):
    """For render/replay or a separately initialized custom controller, not automatic resume."""
    with np.load(path,allow_pickle=False) as saved:
        if str(saved['model_sha256'])!=model_sha256:raise ValueError('Snapshot model mismatch')
        state_type=int(saved['state_type']);state=saved['integration_state']
        if state.shape!=(mujoco.mj_stateSize(model,state_type),):raise ValueError('State size mismatch')
        mujoco.mj_setState(model,data,state,state_type)
    mujoco.mj_forward(model,data)


def export_interface(sim,path):
    m=sim.model
    payload=dict(profile=sim.profile,model_sha256=sim.model_spec['model_sha256'],timestep_s=sim.timestep,
        appearance=sim.appearance_name,
        counts={key:int(getattr(m,key)) for key in ('nq','nv','nu','nbody','ngeom','nmesh','ncam','neq')},
        joints=[dict(name=m.joint(i).name,type=int(m.jnt_type[i]),qpos=int(m.jnt_qposadr[i]),
                     dof=int(m.jnt_dofadr[i])) for i in range(m.njnt)],
        actuators=[dict(name=m.actuator(i).name,control_range=m.actuator_ctrlrange[i],
                        limited=bool(m.actuator_ctrllimited[i])) for i in range(m.nu)],
        cameras=[dict(name=m.camera(i).name,body=m.body(int(m.cam_bodyid[i])).name,fovy_deg=float(m.cam_fovy[i])) for i in range(m.ncam)],
        attachment=sim.attachment.specification(),drive_geometry=sim.model_spec['drive_geometry'],
        state_contract='qpos free-joint quaternion wxyz; MuJoCo free-joint angular qvel is body-local',
        action_contract='air x/y/z/yaw; ground x/y/yaw; world meters/radians; fold arm angle degrees')
    Path(path).write_text(json.dumps(jsonable(payload),indent=2)+'\n')

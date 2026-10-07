"""Full contact physics with replaceable references and explicit actuator access."""
import math
from pathlib import Path
import numpy as np
import mujoco
from .scene.registry import load_model
from .scene.actors import SmoothActors
from .control.air import AirController
from .control.ground import GroundController, attached_ground_hold, wheel_floor_contacts
from .control.references import References
from .dynamics.attachment import MagneticAttachment


class Simulator:
    def __init__(self, profile='factory_industrial', *, dynamic_actors=True, verify=True, appearance='industrial'):
        if appearance not in ('industrial', 'legacy'):
            raise ValueError('appearance must be industrial or legacy')
        self.profile=profile
        self.model,self.model_spec=load_model(profile,verify)
        self.appearance_name=appearance
        from .scene.appearance import IndustrialAppearance
        self.appearance=IndustrialAppearance(self.model) if appearance=='industrial' else None
        self.dynamic_actors=bool(dynamic_actors)
        self.data=mujoco.MjData(self.model)
        self._sensors=None;self._viewer=None
        self.reset()

    def reset(self, *, air_xyz=(-180.,-82.,.074), ground_xy=(-180.,-81.5)):
        """Explicit episode boundary: initializes poses and all controller/observer state."""
        air_xyz=np.asarray(air_xyz,float);ground_xy=np.asarray(ground_xy,float)
        if air_xyz.shape!=(3,) or ground_xy.shape!=(2,) or not np.isfinite(np.r_[air_xyz,ground_xy]).all():
            raise ValueError('Finite air XYZ and ground XY required')
        mujoco.mj_resetData(self.model,self.data)
        self.ground=GroundController(self.model,self.data,
            route=np.stack((ground_xy,ground_xy+[1.,0.])),loop_route=False,
            yaw_rate_feedback=True,**self.model_spec['drive_geometry'])
        self.ground.place_initial()
        self.air=AirController(self.model,self.data,0.)
        self.air.initial_target=air_xyz.copy();self.air.place_initial();self.air.align_body_origin_to_target()
        self.actors=SmoothActors(self.model,self.data,self.model_spec.get('actor_route_variant','baseline')) if self.dynamic_actors else None
        mujoco.mj_forward(self.model,self.data)
        self.references=References(np.r_[self.data.xpos[self.air.root],0.],np.r_[ground_xy,0.])
        self.air.executor=self.ground.executor=self.references
        self.attachment=MagneticAttachment(self.model,self.data,surface_spec=self.model_spec['surface_spec'])
        self.air.bind_payload(self.ground,self.attachment)
        self._wheel_bodies={int(self.model.jnt_bodyid[self.model.joint('dog_'+leg+'轮').id])
                            for leg in ('左前','右前','左后','右后')}
        self._mode=None;self._payload_fraction=0.;self.last_control={}
        self.steps=0
        self.prepare_visuals(self.data.xpos[self.ground.body], 10.)
        return self.state()

    @property
    def time(self): return float(self.data.time)

    @property
    def timestep(self): return float(self.model.opt.timestep)

    def command(self, *, air=None, ground=None, duration=4., fold_deg=None):
        """World-frame air [x,y,z,yaw], ground [x,y,yaw]; SI units, yaw radians.

        Smooth references are not collision-free route planning. Caller supplies
        safe targets and durations. Fold is the mechanical arm angle in degrees.
        """
        # Validate everything first, so an invalid second robot cannot partly commit.
        for value,n in ((air,4),(ground,3)):
            if value is not None and (np.asarray(value).shape!=(n,) or not np.isfinite(value).all()):
                raise ValueError('Invalid target')
        if not math.isfinite(duration) or duration<=0:raise ValueError('Positive duration required')
        if fold_deg is not None:self.air.fold_servo_target(fold_deg)
        if air is not None:self.references.goto('air',air,self.time,duration)
        if ground is not None:self.references.goto('ground',ground,self.time,duration)
        if fold_deg is not None:self.air.command_fold(fold_deg,self.time)

    def ground_supported(self):
        return any(force>1. for _,_,_,force in wheel_floor_contacts(self.model,self.data,self._wheel_bodies))

    def try_attach(self):
        """Requires real mating contact, geometric alignment, low speed and dwell."""
        changed=self.attachment.try_lock(self.time)
        if changed:mujoco.mj_forward(self.model,self.data)
        return changed

    def release(self):
        changed=self.attachment.release(self.ground_supported())
        if changed:mujoco.mj_forward(self.model,self.data)
        return changed

    def step(self, n=1, *, actuator_callback=None):
        """Integrate n physics ticks, never teleport either robot.

        A callback(model,data,sim) replaces BOTH reference controllers for the
        episode and must provide all motor commands/aerodynamic forces itself.
        Switching control modes mid-episode is rejected to avoid stale observers.
        """
        if type(n) is not int or n<1:raise ValueError('n must be a positive integer')
        mode='actuator' if actuator_callback is not None else 'reference'
        if self._mode not in (None,mode):raise ValueError('Reset before switching control modes')
        self._mode=mode
        for _ in range(n):
            self.data.xfrc_applied[:]=0.;self.data.qfrc_applied[:]=0.
            if self.actors:self.actors.update(self.data)
            mujoco.mj_step1(self.model,self.data)
            if actuator_callback is not None:
                self.data.ctrl[:]=0.
                actuator_callback(self.model,self.data,self)
            else:
                target=1. if self.attachment.active else 0.
                self._payload_fraction+=float(np.clip(target-self._payload_fraction,-self.timestep,self.timestep))
                self.air.controller.mass=self.air.air_mass+self._payload_fraction*self.air.dog_mass
                gr=(attached_ground_hold(self.ground,self.ground_supported()) if self.attachment.active
                    else self.ground.apply(self.time))
                self.last_control=dict(air=self.air.apply(self.time),ground=gr)
            mujoco.mj_step2(self.model,self.data)
            self.steps+=1
            wrench=self.attachment.solved_wrench()
            if wrench['overload']:raise RuntimeError('Finite-load attachment broke: '+str(wrench))
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all() or any(w.number for w in self.data.warning):
                raise RuntimeError('Nonfinite state or native MuJoCo warning at '+str(self.time))
        # Keep returned pose/camera timestamps aligned with the integrated qpos.
        mujoco.mj_kinematics(self.model,self.data)
        mujoco.mj_comPos(self.model,self.data)
        mujoco.mj_camlight(self.model,self.data)
        if self._viewer is not None:
            with self._viewer.lock():
                # Viewer GL context is fixed at creation; keep its 12 m shadow radius.
                self.prepare_visuals(self._viewer.cam.lookat, 12. / 1.15)
                mujoco.mj_camlight(self.model,self.data)
        return self.state()

    def state(self):
        def pose(body):
            return dict(position_m=self.data.xpos[body].copy(),quaternion_wxyz=self.data.xquat[body].copy())
        return dict(time_s=self.time,air=pose(self.air.root),ground=pose(self.ground.body),
                    qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),ctrl=self.data.ctrl.copy(),
                    attached=self.attachment.active,physics_steps=self.steps)

    def observe(self, *, width=320,height=240,rings=16,azimuth_samples=180):
        """Real RGB-D and geometric LiDAR; no labels, task metadata or map oracle."""
        from .sensors.bridge import SensorBridge
        if self._sensors and (self._sensors.width,self._sensors.height)!=(width,height):
            self._sensors.close();self._sensors=None
        if self._sensors is None:self._sensors=SensorBridge(self.model,width,height,appearance=self.appearance)
        mujoco.mj_forward(self.model,self.data)
        return {r:dict(rgbd=self._sensors.rgbd(self.data,r),
                       lidar=self._sensors.lidar(self.data,r,rings,azimuth_samples)) for r in ('air','ground')}

    def render(self, *, width=960,height=640,lookat=None,distance=5.,azimuth=135.,elevation=-25.):
        camera=mujoco.MjvCamera();mujoco.mjv_defaultCamera(camera)
        camera.lookat[:]=self.data.xpos[self.ground.body] if lookat is None else lookat
        camera.distance=distance;camera.azimuth=azimuth;camera.elevation=elevation
        option=mujoco.MjvOption();option.geomgroup[:]=[1,1,1,0,0,0]
        self.prepare_visuals(camera.lookat, distance)
        self.model.vis.global_.offwidth=max(width,self.model.vis.global_.offwidth)
        self.model.vis.global_.offheight=max(height,self.model.vis.global_.offheight)
        mujoco.mj_forward(self.model,self.data)
        with mujoco.Renderer(self.model,width=width,height=height,max_geom=int(self.model.ngeom)+2048) as renderer:
            renderer.update_scene(self.data,camera=camera,scene_option=option)
            return renderer.render().copy()

    def prepare_visuals(self, lookat, distance, renderer=None):
        """Call before mj_forward/update_scene when reusing an external renderer."""
        if self.appearance:
            self.appearance.prepare(self.model, lookat, distance, renderer)

    def launch_viewer(self):
        from mujoco import viewer
        if self._viewer is None:
            self.prepare_visuals(self.data.xpos[self.ground.body], 12. / 1.15)
            mujoco.mj_camlight(self.model,self.data)
            self._viewer=viewer.launch_passive(self.model,self.data)
            self._viewer.cam.lookat[:]=self.data.xpos[self.ground.body]
            self._viewer.cam.distance=6.;self._viewer.cam.elevation=-25.
            self._viewer.opt.geomgroup[:]=[1,1,1,0,0,0]
        return self._viewer

    def close(self):
        if self._sensors:self._sensors.close();self._sensors=None
        if self._viewer:self._viewer.close();self._viewer=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

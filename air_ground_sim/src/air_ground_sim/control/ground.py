from __future__ import annotations
import math
import numpy as np
import mujoco
from .ground_rate import YawRateFeedback

def body_angular_velocity(model, data, body_id):
    velocity = np.empty(6, dtype=np.float64)
    # BODY/local uses principal-inertia axes; XBODY/local uses the actual
    # body frame, also when a full inertia tensor rotates its principal axes.
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body_id, velocity, 1)
    return velocity[:3].copy()

def wheel_floor_contacts(model,data,wheel_body_ids):
    """Identify load-bearing wheel contacts by owning bodies, not mesh names."""
    floor=model.geom('factory_floor_400x300').id
    contacts=[]
    for i,c in enumerate(data.contact):
        pair=(int(c.geom1),int(c.geom2))
        if floor not in pair: continue
        other=pair[1] if pair[0]==floor else pair[0]
        if int(model.geom_bodyid[other]) not in wheel_body_ids: continue
        force=np.zeros(6);mujoco.mj_contactForce(model,data,i,force)
        contacts.append((i,other,float(c.dist),max(0.,float(force[0]))))
    return contacts

def timed_dog_class():
    from .ground_base import PhysicalDog as base

    class TimedDog(base):
        executor = None

        def __init__(self,*args,wheel_radius_m=.0225,wheel_track_m=.198,yaw_rate_feedback=False,**kwargs):
            # Explicit platform geometry: legacy callers retain their original
            # values, while a registered hardware candidate can supply its CAD
            # rolling radius and lateral contact-centre separation.
            if not np.isfinite([wheel_radius_m,wheel_track_m]).all() or min(wheel_radius_m,wheel_track_m)<=0:
                raise ValueError('Wheel radius and track must be finite and positive')
            self.wheel_radius_m=float(wheel_radius_m)
            self.wheel_track_m=float(wheel_track_m)
            super().__init__(*args,**kwargs)
            # Fixed added hardware need not share a name prefix. Its real body
            # ownership, including fused bodies, determines the nominal load.
            actual_mass=float(self.model.body_subtreemass[self.body])
            if not np.isfinite([actual_mass,self.mass]).all() or min(actual_mass,self.mass)<=0:
                raise ValueError('Positive physical dog subtree mass required')
            self.scale*=actual_mass/self.mass
            self.mass=actual_mass
            if type(yaw_rate_feedback) is not bool:
                raise ValueError('Yaw-rate feedback must be explicitly enabled or disabled')
            self.yaw_servo = YawRateFeedback() if yaw_rate_feedback else None

        def place_initial(self):
            if float(self.data.time)!=0.:
                raise ValueError('Dog placement is initialization only')
            height=float(self.model.qpos0[self.qpos+2])
            if not math.isfinite(height):
                raise ValueError('Finite authored dog initial height required')
            super().place_initial()
            self.data.qpos[self.qpos+2]=height

        def apply(self,t):
            position,velocity,_ = self.executor.reference('ground',t)
            current=self.data.xpos[self.body,:2]
            yaw=math.atan2(self.data.xmat[self.body,3],self.data.xmat[self.body,0])
            c,s=math.cos(yaw),math.sin(yaw)
            dx,dy=position[:2]-current
            longitudinal,lateral=c*dx+s*dy,-s*dx+c*dy
            yaw_error=math.atan2(math.sin(position[2]-yaw),math.cos(position[2]-yaw))
            # Conventional reference tracking, declared separately from the
            # learned task/action policy. No pose writes or external body wrench.
            desired_forward=float(np.clip(c*velocity[0]+s*velocity[1]+longitudinal,-.6,.6))
            desired_yaw=float(np.clip(velocity[2]+.5*yaw_error+.5*lateral,-.25,.25))
            dt=float(self.model.opt.timestep)
            self.forward_command += float(np.clip(desired_forward-self.forward_command,-1.2*dt,.8*dt))
            self.yaw_rate_command += float(np.clip(desired_yaw-self.yaw_rate_command,-.6*dt,.6*dt))
            forward,yaw_rate=self.forward_command,self.yaw_rate_command
            measured_yaw_rate = None
            if self.yaw_servo is not None:
                omega = body_angular_velocity(self.model, self.data, self.body)
                rotation = self.data.xmat[self.body].reshape(3, 3)
                denominator = rotation[2, 1]**2 + rotation[2, 2]**2
                if denominator < .25:
                    raise ValueError('Ground chassis tilt exceeds planar rate-control support')
                measured_yaw_rate = float((rotation[2, 1]*omega[1] + rotation[2, 2]*omega[2]) / denominator)
                yaw_rate = self.yaw_servo.update(yaw_rate, measured_yaw_rate, dt)
            left,right=forward-yaw_rate*self.wheel_track_m*.5,forward+yaw_rate*self.wheel_track_m*.5
            wheel_target=np.array([-left,right,-left,right])/self.wheel_radius_m
            wheel_control=[]
            for target,actuator,qvel in zip(wheel_target,self.wheel_actuators,self.wheel_qvel):
                value=float(np.clip(.014*self.scale*(target-self.data.qvel[qvel]),-.46,.46))
                self.data.ctrl[actuator]=value
                wheel_control.append(value)
            phase=2.*math.pi*1.6*t
            gain=float(np.clip(abs(forward)/3.,0.,1.)*np.clip(1.-abs(yaw_error)/.35,0.,1.))
            hip=gain*.045*np.sin(phase+np.array([math.pi,math.pi,0.,0.]))
            knee=.060+.050*np.sin(phase+np.array([0.,math.pi,math.pi,0.])+math.pi/3.)
            for value,actuator in zip(hip,self.hip_actuators): self.data.ctrl[actuator]=float(value)
            for value,actuator in zip(knee,self.knee_actuators): self.data.ctrl[actuator]=float(value)
            return dict(desired_position=position.tolist(),desired_velocity=velocity.tolist(),
                position_error_m=float(np.linalg.norm(position[:2]-current)),yaw_error_rad=yaw_error,
                forward_command_mps=forward,yaw_rate_command_rps=yaw_rate,wheel_torque=wheel_control,
                requested_yaw_rate_rps=self.yaw_rate_command, measured_yaw_rate_rps=measured_yaw_rate,
                chassis_rate_feedback=self.yaw_servo is not None,
                external_body_wrench=False,terrain_mode='flat_floor_wheeled_suspension')

    return TimedDog

def attached_ground_hold(dog, ground_supported=False):
    """Neutral airborne suspension; existing wheel-speed servo brakes on support.

    No wheel drive in free flight. On touchdown, zero motor torque would allow
    rolling; a zero-speed target uses the same finite torque/gain as driving.
    """
    torque=np.zeros(len(dog.wheel_actuators))
    if ground_supported:
        torque=np.clip(-.014*dog.scale*dog.data.qvel[dog.wheel_qvel],-.46,.46)
    dog.data.ctrl[dog.wheel_actuators]=torque
    dog.data.ctrl[dog.hip_actuators]=0.
    dog.data.ctrl[dog.knee_actuators]=.06
    dog.forward_command=dog.yaw_rate_command=0.
    if getattr(dog, 'yaw_servo', None) is not None:
        dog.yaw_servo.reset()
    return dict(mode='landed_wheel_brake' if ground_supported else 'carried_actuator_hold',
                wheel_torque=torque.tolist(),external_body_wrench=False,
                wheel_braking_ground_supported=bool(ground_supported))

GroundController = timed_dog_class()

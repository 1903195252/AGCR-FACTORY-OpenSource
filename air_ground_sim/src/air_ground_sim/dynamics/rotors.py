from __future__ import annotations
import math
import numpy as np
import mujoco
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class RotorParameters:
    kf: float = 1.5e-5
    kq: float = 2.4e-7
    tau: float = .020
    speed_limit: float = 2500.
    torque_limit: float = 1.5
    linear_drag: float = .0245
    angular_drag: float = .0025
    # Registered indices go (-x,+y), (-x,-y), (+x,-y), (+x,+y).
    # Native spin is opposite the old steady body-yaw sign at each location.
    spin_signs: tuple = (-1, 1, -1, 1)

    def validate(self):
        for name in ['kf', 'kq', 'linear_drag', 'angular_drag']:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError('Finite nonnegative aerodynamic coefficients required')
        for name in ['tau', 'speed_limit', 'torque_limit']:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError('Finite positive drive parameters required')
        if not self.spin_signs or any(s not in [-1, 1] for s in self.spin_signs):
            raise ValueError('Explicit +/-1 physical spin signs required')

class NativeRotorDynamics:
    def __init__(self, model, data, parameters=RotorParameters()):
        parameters.validate()
        if model.opt.density != 0 or model.opt.viscosity != 0:
            raise ValueError('Automatic fluid forces would duplicate this aerodynamic model')
        self.model, self.parameters = model, parameters
        self.signs = np.asarray(parameters.spin_signs, dtype=float)
        self.root = int(model.body('uav_fuav_root').id)
        self.joints, self.bodies, self.sites, self.actuators, self.dofs = [], [], [], [], []
        for i in range(len(self.signs)):
            joint = model.joint('rotor_spin_%d' % i)
            body = model.body('uav_motor_rotor_%d' % i)
            site = model.site('uav_rotor_axis_%d' % i)
            actuator = model.actuator('rotor_drive_%d' % i)
            if int(joint.type[0]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise ValueError('A physical hinge rotor is required')
            if int(joint.bodyid[0]) != body.id or int(site.bodyid[0]) != body.id:
                raise ValueError('Rotor site, body and native joint do not agree')
            if (int(actuator.trntype[0]) != int(mujoco.mjtTrn.mjTRN_JOINT)
                    or int(actuator.trnid[0]) != joint.id
                    or not np.array_equal(actuator.gear, [1, 0, 0, 0, 0, 0])
                    or int(actuator.dyntype[0]) != int(mujoco.mjtDyn.mjDYN_NONE)
                    or int(actuator.gaintype[0]) != int(mujoco.mjtGain.mjGAIN_FIXED)
                    or int(actuator.biastype[0]) != int(mujoco.mjtBias.mjBIAS_NONE)
                    or actuator.gainprm[0] != 1):
                raise ValueError('Expected an unscaled direct torque port')
            self.joints.append(joint.id)
            self.bodies.append(body.id)
            self.sites.append(site.id)
            self.actuators.append(actuator.id)
            self.dofs.append(int(joint.dofadr[0]))
        # Positions/velocities must be current when creating the binding.
        self.axis_inertia = np.array([self._axis_inertia(data, j, b)
                                     for j, b in zip(self.joints, self.bodies)])
        if np.any(self.axis_inertia <= 0) or not np.isfinite(self.axis_inertia).all():
            raise ValueError('Positive finite native rotor-subtree inertia required')

    def _axis_inertia(self, data, joint, rotor):
        axis, anchor = data.xaxis[joint], data.xanchor[joint]
        total = 0.
        for body in range(rotor, self.model.nbody):
            ancestor = body
            while ancestor > 0 and ancestor != rotor:
                ancestor = int(self.model.body_parentid[ancestor])
            if ancestor != rotor:
                continue
            if body != rotor and self.model.body_jntnum[body]:
                raise ValueError('Moving rotor descendants need a separate inertia model')
            rotation = data.ximat[body].reshape(3, 3)
            tensor = rotation@np.diag(self.model.body_inertia[body])@rotation.T
            offset = data.xipos[body]-anchor
            total += float(axis@tensor@axis+self.model.body_mass[body]*np.linalg.norm(np.cross(axis, offset))**2)
        return total

    def relative_speeds(self, data):
        return data.qvel[self.dofs].copy()

    def observe(self, data):
        """Read native spin and aerodynamic loads without changing any state."""
        relative = self.relative_speeds(data)
        axes = data.xaxis[self.joints].copy()
        velocity = np.empty(6)
        air = []
        for axis, body in zip(axes, self.bodies):
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
            air.append(float(axis@velocity[:3]))
        air = np.asarray(air)
        return dict(relative_speed_rad_s=relative, air_axis_speed_rad_s=air, axes_world=axes,
                    aerodynamic_torque_axis_Nm=-self.parameters.kq*air*abs(air),
                    thrust_N=self.parameters.kf*self.signs*air*abs(air))

    def frame_control_wrench(self, data, motor_torque_Nm):
        """Known thrust and motor reaction for the retained frame controller.

        Gyroscopic/shape coupling stays in the native multibody dynamics. This
        observer input is not an additional force applied to the simulation.
        Motor torque is the last native actuator effort, not steady air drag.
        """
        torque = np.asarray(motor_torque_Nm, dtype=float)
        if torque.shape != self.signs.shape or not np.isfinite(torque).all():
            raise ValueError('One finite native motor effort per rotor required')
        measured = self.observe(data)
        force, moment = np.zeros(3), np.zeros(3)
        center = data.subtree_com[self.root]
        for thrust, effort, axis, site in zip(measured['thrust_N'], torque, measured['axes_world'], self.sites):
            value = thrust*axis
            force += value
            moment += np.cross(data.site_xpos[site]-center, value)-effort*axis
        rotation = data.xmat[self.root].reshape(3, 3)
        return dict(force_world_N=force, torque_body_Nm=rotation.T@moment)

    def apply(self, data, command_rad_s, wind_world=(0., 0., 0.)):
        """Call after mj_step1, before mj_step2; caller clears external force once.

        This method adds its force contribution and sets its own actuator ports.
        It never edits qpos/qvel, unrelated controls, or other force sources.
        Free-air coefficients are used; surface/ground-effect integration remains
        an explicit environment-layer requirement rather than an implicit plane.
        """
        parameters = self.parameters
        command = np.asarray(command_rad_s, dtype=float)
        wind = np.asarray(wind_world, dtype=float)
        if (command.shape != self.signs.shape or not np.isfinite(command).all()
                or np.any(command < 0) or np.any(command > parameters.speed_limit)
                or wind.shape != (3,) or not np.isfinite(wind).all()):
            raise ValueError('Finite bounded rotor commands and a world-frame wind vector required')
        measured = self.observe(data)
        relative, axes = measured['relative_speed_rad_s'], measured['axes_world']
        velocity = np.empty(6)
        air = measured['air_axis_speed_rad_s']
        drag_axis, thrust = measured['aerodynamic_torque_axis_Nm'], measured['thrust_N']
        requested = self.axis_inertia/parameters.tau*(self.signs*command-relative)-drag_axis
        torque = self.signs*np.clip(self.signs*requested, 0., parameters.torque_limit)
        torque[command == 0] = 0.
        data.ctrl[self.actuators] = torque
        force, moment = np.zeros(3), np.zeros(3)
        center = data.subtree_com[self.root]
        for value, drag, axis, site, body in zip(thrust, drag_axis, axes, self.sites, self.bodies):
            point = data.site_xpos[site]
            local_force, local_torque = value*axis, drag*axis
            mujoco.mj_applyFT(self.model, data, local_force, local_torque, point, body, data.qfrc_applied)
            force += local_force
            moment += np.cross(point-center, local_force)+local_torque
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.root, velocity, 0)
        relative_wind = velocity[3:]-wind
        body_force = -parameters.linear_drag*np.linalg.norm(relative_wind)*relative_wind
        body_torque = -parameters.angular_drag*velocity[:3]
        point = data.xipos[self.root]
        mujoco.mj_applyFT(self.model, data, body_force, body_torque, point, self.root, data.qfrc_applied)
        return dict(relative_speed_rad_s=relative, air_axis_speed_rad_s=air,
                    motor_torque_Nm=torque, thrust_N=thrust, aerodynamic_torque_axis_Nm=drag_axis,
                    aerodynamic_spin_power_W=drag_axis*air,
                    external_force_world_N=force+body_force,
                    external_torque_about_com_world_Nm=moment+np.cross(point-center, body_force)+body_torque)

    def allocation_body(self, data):
        """Steady nominal 6D wrench per positive rotor thrust, in root axes.

        This does not stand in for transient native motor reaction or momentum.
        """
        if self.parameters.kf <= 0:
            raise ValueError('Positive KF required for force-normalized allocation')
        rotation = data.xmat[self.root].reshape(3, 3)
        center = data.subtree_com[self.root]
        columns = []
        for sign, joint, site in zip(self.signs, self.joints, self.sites):
            axis = data.xaxis[joint]
            lever = data.site_xpos[site]-center
            torque = np.cross(lever, axis)-sign*self.parameters.kq/self.parameters.kf*axis
            columns.append(np.r_[rotation.T@axis, rotation.T@torque])
        return np.asarray(columns).T

    def manifest(self):
        return dict(parameters=asdict(self.parameters), axis_inertia_kg_m2=self.axis_inertia.tolist(),
                    spin_state='Only native qpos/qvel; no external Omega integration',
                    thrust_torque_application='Rotor subtree aerodynamic wrench; motor joint internal effort',
                    motor_torque_limit_is_measured=False, propeller_reverse_model_calibrated=False,
                    zero_command_mode='coast; no motor torque or phase lock',
                    free_air_only=True, formal_model_accepted=False)

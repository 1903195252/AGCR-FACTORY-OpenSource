from __future__ import annotations
import math
import numpy as np
import mujoco
from .native_air import NativeUAVFlightController
from ..dynamics.payload import PayloadInertia, bind_inertia_schedule, rigid_assembly_inertia, subtree_ids
from .angles import yaw_near_reference

class AirController(NativeUAVFlightController):
    executor = None

    def __init__(self, model, data, duration):
        # Native rotor inertia construction needs current axes/positions.
        mujoco.mj_forward(model, data)
        super().__init__(model, data)
        joint = int(model.body_jntadr[self.root])
        self.qpos = int(model.jnt_qposadr[joint])
        self.qvel = int(model.jnt_dofadr[joint])
        self.initial_target = np.array([0., 0., .074])
        self.connection = self.dog = None
        self._fold_segment = (0., 45., 45.)
        self.air_mass = float(model.body_subtreemass[self.root])
        self.air_bodies = subtree_ids(model, self.root)

    def place_initial(self):
        if self.data.time != 0:
            raise ValueError('Placement is allowed only before physical integration')
        self.data.qpos[self.qpos:self.qpos+3] = self.initial_target
        self.data.qpos[self.qpos+3:self.qpos+7] = [1., 0., 0., 0.]
        self.data.qvel[self.qvel:self.qvel+6] = 0.

    def align_body_origin_to_target(self):
        if self.data.time != 0:
            raise ValueError('No runtime body-origin reset')
        mujoco.mj_forward(self.model, self.data)
        self.data.qpos[self.qpos:self.qpos+3] += self.initial_target-self.data.xpos[self.root]
        mujoco.mj_forward(self.model, self.data)

    def reference(self, t):
        p, v, a = self.executor.reference('air', t)
        return p[:3], np.array([0., 0., p[3]]), v[:3], np.array([0., 0., v[3]]), a[:3], np.array([0., 0., a[3]])

    def state17(self):
        state = super().state17()
        if self.executor is not None:
            target = self.executor.reference('air', float(self.data.time))[0][3]
            attitude = state[self.legacy.IDX_ETA].copy()
            attitude[2] = yaw_near_reference(float(attitude[2]), float(target))
            state[self.legacy.IDX_ETA] = attitude
        return state

    def fold_at(self, t):
        start, a, b = self._fold_segment
        u = float(np.clip((t-start)/.6, 0., 1.))
        return a+(b-a)*(10*u**3-15*u**4+6*u**5)

    def command_fold(self, degrees, t):
        # Validate with the same mechanical inverse used by the actual servo.
        self.fold_servo_target(degrees)
        self._fold_segment = (float(t), self.fold_at(t), float(degrees))

    def fold_settled(self, tolerance_deg=1.):
        angles = np.degrees(self.geometry.alpha_deploy+self.data.qpos[self.arms])
        return bool(np.max(np.abs(angles-self._fold_segment[2])) <= tolerance_deg)

    def bind_payload(self, dog, connection):
        self.dog, self.connection = dog, connection
        self.dog_mass = float(self.model.body_subtreemass[dog.body])
        def unloaded(alpha):
            return rigid_assembly_inertia(self.model, self.data, self.air_bodies, self.root)
        inertia = PayloadInertia(self.model, self.data, self, dog, connection, unloaded)
        bind_inertia_schedule(self.controller, inertia)
        self.inverse_inertia = lambda alpha: np.linalg.inv(inertia(alpha))

    def center_offset(self):
        if self.connection is None or not self.connection.active:
            return np.zeros(3)
        payload = float(np.clip(self.controller.mass-self.air_mass, 0., self.dog_mass))
        center_air = self.data.subtree_com[self.root]
        center = (self.air_mass*center_air+payload*self.data.subtree_com[self.dog.body])/(self.air_mass+payload)
        return center_air-center

    def allocation(self):
        matrix = super().allocation()
        offset = self.data.xmat[self.root].reshape(3, 3).T@self.center_offset()
        matrix[1, :] += offset[1]
        matrix[2, :] -= offset[0]
        return matrix

    def known_control_wrench(self, effort):
        known = super().known_control_wrench(effort)
        rotation = self.data.xmat[self.root].reshape(3, 3)
        known['torque_body_Nm'] += rotation.T@np.cross(self.center_offset(), known['force_world_N'])
        return known

    def apply(self, t):
        info = super().apply(self.reference(t), self.fold_at(t))
        return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in info.items()}

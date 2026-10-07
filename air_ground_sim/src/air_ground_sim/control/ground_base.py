from __future__ import annotations
import math
import numpy as np
import mujoco
DOG_MASS_REFERENCE_KG = 0.890333548
DOG_ROUTE = np.array([[-180.,-81.5],[-179.,-81.5]])

class PhysicalDog:

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, route: np.ndarray | None=None, loop_route: bool=True) -> None:
        self.model = model
        self.data = data
        self.route = np.asarray(DOG_ROUTE if route is None else route, dtype=float)
        if self.route.ndim != 2 or self.route.shape[1] != 2:
            raise ValueError('dog route must have shape [N,2]')
        if len(self.route) < 2:
            raise ValueError('dog route requires at least two waypoints')
        self.loop_route = bool(loop_route)
        self.completed = False
        self.joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'dog_base_free')
        self.body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'dog_mobile_base')
        self.qpos = int(model.jnt_qposadr[self.joint])
        self.qvel = int(model.jnt_dofadr[self.joint])
        wheel_names = ('dog_左前轮', 'dog_右前轮', 'dog_左后轮', 'dog_右后轮')
        self.wheel_qvel = []
        self.wheel_qpos = []
        self.wheel_actuators = []
        for name in wheel_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + '_motor')
            self.wheel_qvel.append(int(model.jnt_dofadr[jid]))
            self.wheel_qpos.append(int(model.jnt_qposadr[jid]))
            self.wheel_actuators.append(aid)
        leg_names = ('左前', '右前', '左后', '右后')
        self.hip_qpos = []
        self.knee_qpos = []
        self.hip_actuators = []
        self.knee_actuators = []
        for leg in leg_names:
            hip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'dog_{leg}跨关节')
            knee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f'dog_{leg}膝关节')
            self.hip_qpos.append(int(model.jnt_qposadr[hip]))
            self.knee_qpos.append(int(model.jnt_qposadr[knee]))
            self.hip_actuators.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'dog_{leg}跨关节_position'))
            self.knee_actuators.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f'dog_{leg}膝关节_position'))
        self.mass = float(sum((model.body_mass[i] for i in range(model.nbody) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or '').startswith('dog_'))))
        self.scale = self.mass / DOG_MASS_REFERENCE_KG
        self.waypoint = 1
        self.forward_command = 0.0
        self.yaw_rate_command = 0.0

    def place_initial(self) -> None:
        self.data.qpos[self.qpos:self.qpos + 7] = (self.route[0, 0], self.route[0, 1], 0.101, 1.0, 0.0, 0.0, 0.0)

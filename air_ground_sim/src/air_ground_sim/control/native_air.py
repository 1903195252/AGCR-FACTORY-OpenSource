from __future__ import annotations
import math
import numpy as np
import mujoco
import json
from argparse import Namespace
from ..paths import CONFIG
from ..kinematics.folding import FourBarGeometry
from ..dynamics.rotors import NativeRotorDynamics

def legacy_args():
    return Namespace(method='M1', fg=2.5, feo_gain_scale=1., feo_leak_scale=1.,
                     dhat_max_scale=1., feo_pos_comp_scale=1., feo_att_comp_scale=1.,
                     pos_gain_scale=1., att_gain_scale=1.)

class PhysicsTickClock:
    """Schedule on integer physics ticks, without accumulating a second clock.

    MuJoCo accumulates floating-point time at each physics step. Independently
    adding the controller period drifts relative to that clock over long runs.
    The alignment tolerance below is 1/1000 of a step, not permission to skip
    a control tick; actual missed updates and backwards time remain errors.
    """
    def __init__(self, timestep, period, origin):
        if not all(math.isfinite(v) for v in (timestep, period, origin)) or timestep <= 0 or period <= 0:
            raise ValueError('Finite positive physics step and control period required')
        self.stride = round(period/timestep)
        if self.stride < 1 or not math.isclose(self.stride*timestep, period, rel_tol=0., abs_tol=1e-12):
            raise ValueError('Physics step must divide the retained flight-control period')
        self.timestep, self.origin = float(timestep), float(origin)
        self.next_tick, self.last_tick = 0, 0

    def take_update(self, stamp):
        if not math.isfinite(stamp):
            raise ValueError('Finite simulation time required')
        elapsed_ticks = (stamp-self.origin)/self.timestep
        tick = round(elapsed_ticks)
        if abs(elapsed_ticks-tick) > .001:
            raise ValueError('Flight-control time is not aligned to a physics tick')
        if tick < self.last_tick:
            raise ValueError('Flight-control time moved backwards')
        if tick > self.next_tick:
            raise ValueError('A scheduled flight-control update was skipped')
        self.last_tick = tick
        if tick == self.next_tick:
            self.next_tick += self.stride
            return True
        return False

class NativeUAVFlightController:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.rotor = NativeRotorDynamics(model, data)
        self.root = self.rotor.root
        from . import flight_law
        self.legacy = flight_law
        self.controller = self.legacy.MuJoCoCFNTSMGFAController(legacy_args())
        self.control_period_s = .004
        self.control_clock = PhysicsTickClock(model.opt.timestep, self.control_period_s, float(data.time))
        self.controller.dt = self.control_period_s
        self.controller.mass = float(model.body_subtreemass[self.root])
        self.geometry = FourBarGeometry(json.loads((CONFIG/'hardware_spec.json').read_text()))
        self.drive = json.loads((CONFIG/'fold_drive.json').read_text())['drive']
        self.arms = [int(model.joint('arm_hinge_%d' % i).qposadr[0]) for i in range(4)]
        self.fold_actuator = int(model.actuator('single_fold_motor').id)
        self.fold_position = int(model.joint('fold_servo').qposadr[0])
        self._omega_command = np.zeros(4)
        self.observer_inputs = {}
        self.control_trace = {}
        self.control_updates = 0
        self.inverse_inertia = self.legacy.Jinv_fast
        self._bind_observers()

    def _bind_observers(self):
        original_position = self.controller.feo_pos.update
        original_attitude = self.controller.feo_att.update

        def position(t, output, rate, analytic_contribution, dt):
            value = self.observer_inputs['position_acceleration']
            return original_position(t, output, rate, value, dt)

        def attitude(t, output, rate, analytic_contribution, dt):
            value = self.observer_inputs['attitude_acceleration']
            return original_attitude(t, output, rate, value, dt)

        self.controller.feo_pos.update = position
        self.controller.feo_att.update = attitude

    def state17(self):
        state = np.zeros(17)
        old = self.legacy
        state[old.IDX_P] = self.data.xpos[self.root]
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                self.root, velocity, 0)
        state[old.IDX_V] = velocity[3:]
        # BODY/local is the inertial-principal frame (ximat), not the CAD
        # body frame (xmat). Full tensors make this distinction observable.
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY,
                                self.root, velocity, 1)
        state[old.IDX_OMEGA] = velocity[:3]
        state[old.IDX_ETA] = old.rotmat_to_euler_zyx(self.data.xmat[self.root].reshape(3, 3))
        state[old.IDX_MOTOR] = abs(self.rotor.relative_speeds(self.data))
        state[old.IDX_ALPHA] = math.degrees(self.geometry.alpha_deploy+float(np.mean(self.data.qpos[self.arms])))
        return state

    def allocation(self):
        return self.rotor.allocation_body(self.data)[[2, 3, 4, 5]]

    def known_control_wrench(self, effort):
        return self.rotor.frame_control_wrench(self.data, effort)

    def fold_servo_target(self, alpha_degrees):
        alpha = math.radians(alpha_degrees)
        if not math.isfinite(alpha) or not self.geometry.alpha_fold <= alpha <= self.geometry.alpha_deploy:
            raise ValueError('Fold command outside the registered mechanism range')
        theta = self.geometry.inverse(alpha)
        return -self.geometry.ratio*(theta-self.geometry.theta_deploy)

    def apply(self, reference, alpha_target_deg=45., wind_world=(0., 0., 0.)):
        """After mj_step1, before mj_step2; caller clears external force once.

        Reference order is position, Euler attitude, velocity, Euler rate,
        acceleration, Euler acceleration, as in the retained controller.
        The docking FSM's returned fold command is diagnostic only: the mission
        supplies a separate mechanical command, as in PhysicalUAV.apply.
        """
        if len(reference) != 6 or any(np.asarray(v).shape != (3,) or not np.isfinite(v).all() for v in reference):
            raise ValueError('Six finite reference 3-vectors required')
        target = self.fold_servo_target(alpha_target_deg)
        stamp = float(self.data.time)
        if self.control_clock.take_update(stamp):
            state = self.state17()
            effort = self.data.actuator_force[self.rotor.actuators].copy()
            known = self.known_control_wrench(effort)
            alpha = state[self.legacy.IDX_ALPHA]
            self.observer_inputs = dict(
                position_acceleration=known['force_world_N']/self.controller.mass+self.model.opt.gravity,
                attitude_acceleration=self.inverse_inertia(alpha)@known['torque_body_Nm'],
                previous_native_motor_effort_Nm=effort)
            analytic, fsm_alpha, extras = self.controller.compute(stamp, state, reference)
            allocation = self.allocation()
            self._omega_command = self.legacy.remap_controller_omega_to_physical_allocator(analytic, alpha, allocation)
            self.control_trace = dict(
                desired_wrench=self.legacy.M_fast(alpha)@(self.legacy.K_F*analytic**2),
                allocated_steady_wrench=allocation@(self.legacy.K_F*self._omega_command**2),
                position_error=extras['e_pos'], attitude_reference_error=extras['e_att'],
                observer_position=extras['d_hat_pos'], observer_attitude=extras['d_hat_att'],
                ignored_legacy_fsm_fold_command_deg=float(fsm_alpha), controller_time_s=stamp)
            self.control_updates += 1
        self.data.ctrl[self.fold_actuator] = np.clip(
            self.drive['kp_Nm_rad']*(target-self.data.qpos[self.fold_position])/self.drive['torque_cap_Nm'], -1., 1.)
        return dict(self.rotor.apply(self.data, self._omega_command, wind_world),
                    command_rad_s=self._omega_command.copy(), **self.control_trace)

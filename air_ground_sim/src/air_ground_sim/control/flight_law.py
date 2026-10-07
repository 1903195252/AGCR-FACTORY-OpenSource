from __future__ import annotations
import math
import numpy as np
import mujoco
from typing import Optional
import argparse
from .sim_core import *
from .observers import GFAFEO, MissionFSM, signed_pow, accel_to_thrust_attitude

def rotmat_to_euler_zyx(R: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw for the same ZYX convention used by sim_core."""
    pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)

class MuJoCoCFNTSMGFAController:
    """MuJoCo-calibrated CFNTSM + GFA-FEO controller for STEP freejoint dynamics."""

    def __init__(self, args: argparse.Namespace):
        self.method = args.method
        self.mass = M_UAV
        self.dt = 1e-3
        self.fsm = MissionFSM()
        self.feo_pos = GFAFEO(
            f_g=args.fg,
            use_internal_model=args.method != "M3",
            ell1=1.2,
            ell2=1.6,
            ell3=0.45,
            L=5.0 * args.feo_gain_scale,
            eps1=0.12,
            gamma_n=4.0 * args.feo_gain_scale,
            mu_v=6.0,
        )
        self.feo_att = GFAFEO(
            f_g=args.fg,
            use_internal_model=args.method != "M3",
            ell1=1.2,
            ell2=1.6,
            ell3=0.45,
            L=6.5 * args.feo_gain_scale,
            eps1=0.12,
            gamma_n=4.0 * args.feo_gain_scale,
            mu_v=6.0,
        )
        self.feo_pos.sigma_im = 0.5 * args.feo_leak_scale
        self.feo_att.sigma_im = 0.5 * args.feo_leak_scale
        self.feo_pos._d_hat_max = 4.0 * args.dhat_max_scale
        self.feo_att._d_hat_max = 8.0 * args.dhat_max_scale
        self.feo_pos_comp_scale = args.feo_pos_comp_scale
        self.feo_att_comp_scale = args.feo_att_comp_scale
        self.kp_pos = np.array([1.4, 1.4, 14.0]) * args.pos_gain_scale
        self.kd_pos = np.array([2.8, 2.8, 7.5]) * args.pos_gain_scale
        self.ks_pos = np.array([0.18, 0.18, 0.72]) * args.pos_gain_scale
        self.kp_att = np.array([115.0, 115.0, 42.0]) * args.att_gain_scale
        self.kd_att = np.array([17.0, 17.0, 6.5]) * args.att_gain_scale
        self.ks_att = np.array([0.22, 0.22, 0.08]) * args.att_gain_scale
        self.beta_pos = 0.22
        self.beta_att = 0.45
        self.p_q = 9.0 / 7.0
        self.rho = 0.7
        self._prev_eta_des: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None

    def compute(self, t: float, x_meas: np.ndarray, gait_full) -> tuple:
        target_pos, target_eta, target_vel, target_eta_dot = gait_full[:4]
        target_acc = gait_full[4] if len(gait_full) >= 6 else np.zeros(3)
        e_pos = x_meas[IDX_P] - target_pos
        e_vel = x_meas[IDX_V] - target_vel
        e_att_ref = x_meas[IDX_ETA] - target_eta
        phase = self.fsm.update(
            t,
            e_pos[:2],
            e_pos[2],
            e_att_ref,
            x_meas[IDX_V][2] - target_vel[2],
            x_meas[IDX_ALPHA],
            self.dt,
        )

        d_hat_pos = self.feo_pos.d_hat
        s_pos = e_pos + self.beta_pos * signed_pow(e_vel, self.p_q)
        accel_fb = (
            -self.kp_pos * e_pos
            - self.kd_pos * e_vel
            - self.ks_pos * signed_pow(s_pos, self.rho)
            - self.feo_pos_comp_scale * d_hat_pos
        )
        accel_fb[:2] = np.clip(accel_fb[:2], -3.0, 3.0)
        accel_fb[2] = float(np.clip(accel_fb[2], -5.0, 12.0))
        accel_total = target_acc + accel_fb
        F_total, phi_des, theta_des = accel_to_thrust_attitude(accel_total, target_eta[2], self.mass)
        eta_des = np.array([phi_des + target_eta[0], theta_des + target_eta[1], target_eta[2]])

        e_att = x_meas[IDX_ETA] - eta_des
        if self._prev_eta_des is not None and self._prev_t is not None and t > self._prev_t:
            eta_des_dot = (eta_des - self._prev_eta_des) / max(t - self._prev_t, 1e-6)
            eta_des_dot = np.clip(eta_des_dot, -20.0, 20.0)
        else:
            eta_des_dot = target_eta_dot.copy()
        self._prev_eta_des = eta_des.copy()
        self._prev_t = t
        omega_ref = euler_rate_to_body_rate(eta_des[0], eta_des[1]) @ eta_des_dot
        e_rate = x_meas[IDX_OMEGA] - omega_ref
        d_hat_att = self.feo_att.d_hat
        s_att = e_att + self.beta_att * signed_pow(e_rate, self.p_q)
        J = J_fast(x_meas[IDX_ALPHA])
        Jinv = Jinv_fast(x_meas[IDX_ALPHA])
        ang_acc_cmd = (
            -self.kp_att * e_att
            - self.kd_att * e_rate
            - self.ks_att * signed_pow(s_att, self.rho)
            - self.feo_att_comp_scale * d_hat_att
        )
        tau_cmd = J @ ang_acc_cmd
        tau_cmd[0] = float(np.clip(tau_cmd[0], -1.8, 1.8))
        tau_cmd[1] = float(np.clip(tau_cmd[1], -1.8, 1.8))
        tau_cmd[2] = float(np.clip(tau_cmd[2], -0.22, 0.22))

        f_actual = K_F * x_meas[IDX_MOTOR] ** 2
        R_body = R_ZYX(x_meas[IDX_ETA][0], x_meas[IDX_ETA][1], x_meas[IDX_ETA][2])
        g_u_pos = (float(np.sum(f_actual)) / self.mass) * R_body @ np.array([0.0, 0.0, 1.0])
        g_u_pos = g_u_pos - np.array([0.0, 0.0, G_ACC])
        self.feo_pos.update(t, e_pos, e_vel, g_u_pos, self.dt)
        u_actual = M_fast(x_meas[IDX_ALPHA]) @ f_actual
        self.feo_att.update(t, e_att, e_rate, Jinv @ u_actual[1:4], self.dt)

        u_cmd = np.array([np.clip(F_total, 0.1, self.mass * G_ACC * 4.0), *tau_cmd])
        f_motors = np.linalg.pinv(M_fast(x_meas[IDX_ALPHA]), rcond=1e-4) @ u_cmd
        f_motors = np.clip(f_motors, 0.0, 30.0)
        if np.sum(f_motors) > 1e-6:
            f_motors *= u_cmd[0] / np.sum(f_motors)
        omega_cmd = np.sqrt(np.clip(f_motors / K_F, 0.0, 2500.0**2))
        extras = {
            "phase": phase,
            "lambda": 0.0,
            "d_hat_pos": d_hat_pos.copy(),
            "d_hat_att": d_hat_att.copy(),
            "e_pos": e_pos.copy(),
            "e_att": e_att_ref.copy(),
        }
        return omega_cmd, self.fsm.get_alpha_cmd(), extras

def remap_controller_omega_to_physical_allocator(
    omega_cmd: np.ndarray,
    alpha_deg: float,
    B_physical: np.ndarray,
    residual_wrench: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Preserve the controller wrench but allocate through MuJoCo rotor sites."""
    thrust_paper = K_F * np.asarray(omega_cmd, dtype=float) ** 2
    desired_wrench = M_fast(alpha_deg) @ thrust_paper
    if residual_wrench is not None:
        desired_wrench = desired_wrench + np.asarray(residual_wrench, dtype=float)
    thrust_physical = np.linalg.pinv(B_physical, rcond=1e-4) @ desired_wrench
    thrust_physical = np.clip(thrust_physical, 0.0, 30.0)
    total_des = float(np.clip(desired_wrench[0], 0.0, 120.0))
    thrust_sum = float(np.sum(thrust_physical))
    if thrust_sum > 1e-6 and total_des > 1e-6:
        thrust_physical *= total_des / thrust_sum
        thrust_physical = np.clip(thrust_physical, 0.0, 30.0)
    return np.sqrt(np.clip(thrust_physical / K_F, 0.0, 2500.0**2))

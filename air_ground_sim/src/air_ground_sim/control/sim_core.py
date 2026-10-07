#!/usr/bin/env python3
"""
sim_core.py  —  6-DoF foldable-quadrotor dynamics (paper §2–§3)

Physical model:
  - Freudenstein four-bar fold linkage  (§2.1)
  - Fold-dependent inertia J(α) and allocation B(α)  (§2.2–2.3)
  - 1st-order motor dynamics τ_m = 20 ms
  - Ground-effect model (Cheeseman–Bennett)
  - Dog-body gait model at f_g = 2.5 Hz  (§3.5)
  - RK4 integration at dt = 1 ms (default)

State vector  x ∈ R^17:
  [px,py,pz, vx,vy,vz, φ,θ,ψ, ωx,ωy,ωz, Ω1..Ω4, α]
"""

import numpy as np
from numpy.linalg import norm, inv

# ================================================================
#  Physical constants  (Table 1 in paper)
# ================================================================
M_UAV   = 2.50           # kg   quadrotor mass
M_DOG   = 2.50           # kg   walking-robot mass (after lock)
G_ACC   = 9.81           # m/s²
M_ARM   = 0.18           # kg   each arm assembly
K_F     = 1.5e-5         # N·s²  thrust coefficient
K_TAU   = 2.4e-7         # N·m·s²  torque coefficient
C_TAU   = K_TAU / K_F    # ≈ 0.016 m  torque / thrust ratio
J_ROTOR = 3.8e-5         # kg·m²  rotor inertia
K_D     = 2.5e-3         # N·m·s  aero-drag coeff on body
TAU_M   = 0.020          # s     motor time constant
R_P     = 0.065          # m     propeller radius (65 mm)

# Fold angle limits (degrees)
ALPHA_DEPLOY = 45.0
ALPHA_FOLD   = -76.5      # deg  collision-limited fold angle

# Sensor noise std
SIGMA_P     = 0.0005     # m   (Vicon / OptiTrack motion capture)
SIGMA_ETA   = np.radians(0.1)
SIGMA_OMEGA = np.radians(0.5)

# Gait nominal
V_NOM   = 0.50           # m/s  nominal walk speed
F_G_NOM = 2.50           # Hz   gait frequency = v / l_leg (l_leg ≈ 0.20 m)
A_HEAVE = 0.005          # m    heave amplitude (5 mm at 2f_g)
A_ROLL  = np.radians(1.5)  # rad  roll amplitude (f_g/2)
A_PITCH = np.radians(2.0)  # rad  pitch amplitude (f_g)

# Ground effect (Cheeseman–Bennett)
def K_GE(h):
    """Ground-effect thrust multiplier.  h = height above ground [m]."""
    rp = R_P
    if h > 4 * rp:
        return 1.0
    r_ratio = rp / max(h, 0.01)
    return 1.0 / (1.0 - 0.25 * r_ratio**2)

# ================================================================
#  Freudenstein four-bar linkage  (§2.1)
# ================================================================
# Geometry (mm -> m internally; we store motor positions in mm then convert)
_A_F  = 64.0     # mm  fixed pivot offset
_L_A  = 95.0     # mm  arm length
_L_C  = 26.45    # mm  coupler
_THETA_BASE = np.array([45.0, 135.0, 225.0, 315.0])  # deg: arm base angles

def motor_pos_mm(alpha_deg, arm_idx):
    """Motor position in body-XY [mm] for a given fold angle α (deg)."""
    theta = _THETA_BASE[arm_idx]
    rad_a = np.radians(alpha_deg)
    rad_t = np.radians(theta)
    x = (_A_F + _L_A * np.cos(rad_a)) * np.cos(rad_t)
    y = (_A_F + _L_A * np.cos(rad_a)) * np.sin(rad_t)
    return np.array([x, y])

def motor_positions_m(alpha_deg):
    """4×2 motor positions in body frame [m]."""
    pos = np.zeros((4, 2))
    for k in range(4):
        pos[k] = motor_pos_mm(alpha_deg, k) * 1e-3
    return pos

# ================================================================
#  Inertia & allocation  (§2.2–2.3)
# ================================================================
def J_body_base():
    """Central body inertia (plate model)."""
    m_plate = M_UAV - 4 * M_ARM
    a = 0.075   # half-side [m]
    Ixx = m_plate * (2 * a)**2 / 12
    return np.diag([Ixx, Ixx, 2 * Ixx])

def J_alpha(alpha_deg):
    """Total inertia tensor at fold angle α [3×3]."""
    J = J_body_base().copy()
    for k in range(4):
        r_mm = motor_pos_mm(alpha_deg, k)
        r = np.array([r_mm[0] * 1e-3, r_mm[1] * 1e-3, 0.0])
        r2 = np.dot(r, r)
        J += M_ARM * (r2 * np.eye(3) - np.outer(r, r))
    return J

def alloc_B(alpha_deg):
    """3×4 torque allocation matrix B(α)."""
    sigma = [-1, +1, -1, +1]
    B = np.zeros((3, 4))
    for k in range(4):
        r_mm = motor_pos_mm(alpha_deg, k)
        mx, my = r_mm * 1e-3
        B[0, k] = my          # roll  moment arm
        B[1, k] = -mx         # pitch moment arm
        B[2, k] = sigma[k] * C_TAU
    return B

def full_M(alpha_deg):
    """4×4 allocation matrix M(α) = [[1,1,1,1]; B(α)]."""
    B = alloc_B(alpha_deg)
    return np.vstack([np.ones((1, 4)), B])

# ================================================================
#  Pre-computed lookup tables (for speed in inner loop)
# ================================================================
_LUT_N = 500
_LUT_ALPHA = np.linspace(ALPHA_FOLD, ALPHA_DEPLOY, _LUT_N)
_LUT_J     = np.array([J_alpha(a) for a in _LUT_ALPHA])
_LUT_JINV  = np.array([inv(J_alpha(a)) for a in _LUT_ALPHA])
_LUT_B     = np.array([alloc_B(a) for a in _LUT_ALPHA])
_LUT_FULLM = np.array([full_M(a) for a in _LUT_ALPHA])

def _lut_idx(alpha_deg):
    return int(np.clip(
        (alpha_deg - ALPHA_FOLD) / (ALPHA_DEPLOY - ALPHA_FOLD) * (_LUT_N - 1),
        0, _LUT_N - 1))

def J_fast(alpha_deg):
    return _LUT_J[_lut_idx(alpha_deg)]

def Jinv_fast(alpha_deg):
    return _LUT_JINV[_lut_idx(alpha_deg)]

def B_fast(alpha_deg):
    return _LUT_B[_lut_idx(alpha_deg)]

def M_fast(alpha_deg):
    return _LUT_FULLM[_lut_idx(alpha_deg)]

# ================================================================
#  Rotation matrix & Euler conversions
# ================================================================
def R_ZYX(phi, theta, psi):
    """Body→world rotation (ZYX Euler)."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([
        [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [ -st,            ct*sp,             ct*cp   ]])

def body_rate_to_euler_rate(phi, theta):
    """W⁻¹ matrix: ω_body → η̇ Euler."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    ct_safe = ct if abs(ct) > 1e-6 else np.sign(ct + 1e-10) * 1e-6
    return np.array([
        [1, sp*st/ct_safe, cp*st/ct_safe],
        [0, cp,            -sp          ],
        [0, sp/ct_safe,    cp/ct_safe   ]])

def euler_rate_to_body_rate(phi, theta):
    """W matrix: η̇ Euler → ω_body."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [1,  0,   -st  ],
        [0,  cp,  sp*ct],
        [0, -sp,  cp*ct]])

# ================================================================
#  State vector indices
# ================================================================
IDX_P     = slice(0, 3)
IDX_V     = slice(3, 6)
IDX_ETA   = slice(6, 9)
IDX_OMEGA = slice(9, 12)
IDX_MOTOR = slice(12, 16)
IDX_ALPHA = 16
STATE_DIM = 17

# ================================================================
#  Gait Model (§3.5)
# ================================================================
class GaitModel:
    """Dog-body periodic motion: heave + roll + pitch oscillations."""

    def __init__(self, v_walk=V_NOM, f_g=F_G_NOM, heading_psi=0.0,
                 tau_advance=TAU_M):
        self.v = v_walk
        self.f_g = f_g
        self.psi0 = heading_psi
        self.A_heave = A_HEAVE
        self.A_roll  = A_ROLL
        self.A_pitch = A_PITCH
        self.tau_adv  = tau_advance   # phase-advance for feedforward (s)

    def state(self, t):
        """Returns (pos[3], euler[3], vel[3], euler_rate[3])."""
        w = 2 * np.pi * self.f_g
        cpsi, spsi = np.cos(self.psi0), np.sin(self.psi0)

        px = self.v * t * cpsi
        py = self.v * t * spsi
        pz = 0.35 + self.A_heave * np.sin(2 * w * t)  # heave at 2f_g

        phi   = self.A_roll  * np.sin(0.5 * w * t)     # roll at f_g/2
        theta = self.A_pitch * np.sin(w * t)            # pitch at f_g
        psi   = self.psi0

        vx = self.v * cpsi
        vy = self.v * spsi
        vz = self.A_heave * 2 * w * np.cos(2 * w * t)

        phi_dot   = self.A_roll  * 0.5 * w * np.cos(0.5 * w * t)
        theta_dot = self.A_pitch * w * np.cos(w * t)
        psi_dot   = 0.0

        return (np.array([px, py, pz]),
                np.array([phi, theta, psi]),
                np.array([vx, vy, vz]),
                np.array([phi_dot, theta_dot, psi_dot]))

    def accel(self, t):
        """Returns (pos_accel[3], euler_accel[3]).

        Feedforward acceleration is evaluated at t + tau_advance to
        compensate motor-lag-induced thrust phase delay.
        """
        w = 2 * np.pi * self.f_g
        ta = t + self.tau_adv      # phase-advanced time
        az = -self.A_heave * (2 * w)**2 * np.sin(2 * w * ta)
        phi_ddot   = -self.A_roll  * (0.5 * w)**2 * np.sin(0.5 * w * ta)
        theta_ddot = -self.A_pitch * w**2 * np.sin(w * ta)
        return (np.array([0.0, 0.0, az]),
                np.array([phi_ddot, theta_ddot, 0.0]))

# ================================================================
#  6-DoF Dynamics
# ================================================================
class QuadDynamics:
    """Full 6-DoF foldable-quadrotor dynamics with motor lag."""

    def __init__(self, mass=M_UAV, wind_vel=None):
        self.m = mass
        self.wind = wind_vel if wind_vel is not None else np.zeros(3)
        self.alpha_rate = 30.0  # deg/s fold speed
        # External disturbance injection (callable: t -> (F_ext[3], tau_ext[3]))
        self.ext_disturbance = None

    def derivatives(self, t, x, motor_cmds, alpha_cmd):
        """Compute dx/dt [17]."""
        pos   = x[IDX_P]
        vel   = x[IDX_V]
        eta   = x[IDX_ETA]
        omega = x[IDX_OMEGA]
        Omega = x[IDX_MOTOR]
        alpha = x[IDX_ALPHA]
        phi, theta, psi = eta

        R = R_ZYX(phi, theta, psi)

        # Motor dynamics (1st-order lag)
        dOmega = (np.asarray(motor_cmds) - Omega) / TAU_M

        # Fold dynamics
        alpha_err = alpha_cmd - alpha
        dalpha = np.clip(alpha_err / 0.1, -self.alpha_rate, self.alpha_rate)

        # Per-motor thrust
        thrusts = K_F * Omega**2
        F_total = np.sum(thrusts)

        # Ground effect
        h = max(pos[2], 0.02)
        Kge = K_GE(h) if h < 4 * R_P else 1.0
        F_ge = F_total * Kge

        # External disturbance forces and torques
        F_ext = np.zeros(3)
        tau_ext = np.zeros(3)
        if self.ext_disturbance is not None:
            F_ext, tau_ext = self.ext_disturbance(t)

        # Translational dynamics
        F_body_z  = np.array([0.0, 0.0, F_ge])
        F_world   = R @ F_body_z
        F_grav    = np.array([0.0, 0.0, -self.m * G_ACC])
        v_rel     = vel - self.wind
        F_drag    = -0.02 * 1.225 * norm(v_rel) * v_rel
        accel     = (F_world + F_grav + F_drag + F_ext) / self.m

        # Rotational dynamics
        J    = J_fast(alpha)
        Jinv = Jinv_fast(alpha)
        B    = B_fast(alpha)

        tau_ctrl = B @ thrusts
        sigma = np.array([-1, +1, -1, +1])
        tau_gyro = J_ROTOR * np.cross(omega, np.array([0, 0, 1])) * np.sum(sigma * Omega)
        tau_drag = -K_D * omega
        tau_total = tau_ctrl + tau_gyro + tau_drag + tau_ext
        domega = Jinv @ (tau_total - np.cross(omega, J @ omega))

        # Euler rate
        theta_safe = np.clip(theta, -1.4, 1.4)
        W_inv = body_rate_to_euler_rate(phi, theta_safe)
        deta = W_inv @ omega

        dx = np.zeros(STATE_DIM)
        dx[IDX_P]     = vel
        dx[IDX_V]     = accel
        dx[IDX_ETA]   = deta
        dx[IDX_OMEGA] = domega
        dx[IDX_MOTOR] = dOmega
        dx[IDX_ALPHA] = dalpha
        return dx

# ================================================================
#  Sensor model
# ================================================================
class SensorModel:
    def __init__(self, rng=None):
        self.rng = rng or np.random.default_rng()

    def measure(self, x):
        xm = x.copy()
        xm[IDX_P]     += self.rng.normal(0, SIGMA_P, 3)
        xm[IDX_ETA]   += self.rng.normal(0, SIGMA_ETA, 3)
        xm[IDX_OMEGA] += self.rng.normal(0, SIGMA_OMEGA, 3)
        return xm

# ================================================================
#  RK4 integrator
# ================================================================
def rk4_step(dyn, t, x, motor_cmds, alpha_cmd, dt):
    k1 = dyn.derivatives(t,        x,             motor_cmds, alpha_cmd)
    k2 = dyn.derivatives(t+dt/2,   x + dt/2*k1,  motor_cmds, alpha_cmd)
    k3 = dyn.derivatives(t+dt/2,   x + dt/2*k2,  motor_cmds, alpha_cmd)
    k4 = dyn.derivatives(t+dt,     x + dt*k3,     motor_cmds, alpha_cmd)
    x_new = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    x_new[IDX_MOTOR] = np.clip(x_new[IDX_MOTOR], 0, 2500)
    x_new[IDX_ALPHA] = np.clip(x_new[IDX_ALPHA], ALPHA_FOLD, ALPHA_DEPLOY)
    return x_new

# ================================================================
#  Simulation runner
# ================================================================
class Simulation:
    def __init__(self, dt=1e-3, T=20.0, mass=M_UAV,
                 wind_speed=2.0, wind_dir=0.0, seed=None):
        self.dt = dt
        self.T  = T
        self.N  = int(T / dt)
        self.rng = np.random.default_rng(seed)
        wind = wind_speed * np.array([np.cos(wind_dir), np.sin(wind_dir), 0])
        self.dynamics = QuadDynamics(mass=mass, wind_vel=wind)
        self.sensor   = SensorModel(rng=self.rng)
        self.gait     = GaitModel()

    def initial_state(self, pos=None, vel=None, alpha=ALPHA_DEPLOY):
        x0 = np.zeros(STATE_DIM)
        x0[IDX_P] = pos if pos is not None else np.array([0, 0, 2.0])
        if vel is not None:
            x0[IDX_V] = vel
        Omega_hover = np.sqrt(self.dynamics.m * G_ACC / (4 * K_F))
        x0[IDX_MOTOR] = Omega_hover
        x0[IDX_ALPHA] = alpha
        return x0

    def run(self, controller, x0=None):
        if x0 is None:
            x0 = self.initial_state()
        t_arr = np.linspace(0, self.T, self.N + 1)
        x_arr = np.zeros((self.N + 1, STATE_DIM))
        u_arr = np.zeros((self.N + 1, 4))
        x_arr[0] = x0
        extras_list = []

        for i in range(self.N):
            t = t_arr[i]
            x = x_arr[i]
            xm = self.sensor.measure(x)
            gs = self.gait.state(t)
            ga = self.gait.accel(t)
            gait_full = gs + ga   # 6-tuple

            result = controller.compute(t, xm, gait_full)
            motor_cmds, alpha_cmd, extras = result
            u_arr[i] = motor_cmds
            extras_list.append(extras)

            x_arr[i+1] = rk4_step(self.dynamics, t, x,
                                   motor_cmds, alpha_cmd, self.dt)
            if x_arr[i+1, 2] < 0:
                x_arr[i+1, 2] = 0
                x_arr[i+1, 5] = max(x_arr[i+1, 5], 0)

        return {'t': t_arr, 'x': x_arr, 'u': u_arr, 'extras': extras_list}

# ================================================================
#  Metrics
# ================================================================
def compute_metrics(result, gait):
    """Standard performance metrics from simulation."""
    t, x = result['t'], result['x']
    pos_err, att_err = [], []
    for i, ti in enumerate(t):
        gp, ge, _, _ = gait.state(ti)
        pos_err.append(x[i, IDX_P] - gp)
        att_err.append(x[i, IDX_ETA] - ge)
    pos_err = np.array(pos_err)
    att_err = np.array(att_err)
    N2 = len(t) // 2
    pos_rmse_mm = np.sqrt(np.mean(np.sum(pos_err[N2:]**2, axis=1))) * 1000
    att_rmse_deg = np.degrees(
        np.sqrt(np.mean(np.sum(att_err[N2:, :2]**2, axis=1))))
    peak_att_deg = np.degrees(
        np.max(np.sqrt(np.sum(att_err[:, :2]**2, axis=1))))
    # Final z error
    gp_end, _, _, _ = gait.state(t[-1])
    z_final_mm = (x[-1, 2] - gp_end[2]) * 1000
    return {
        'pos_rmse_mm': pos_rmse_mm,
        'att_rmse_deg': att_rmse_deg,
        'peak_att_deg': peak_att_deg,
        'z_final_mm': z_final_mm,
    }


if __name__ == '__main__':
    Omega_h = np.sqrt(M_UAV * G_ACC / (4 * K_F))

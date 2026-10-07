from __future__ import annotations
import math
import numpy as np
import mujoco
from numpy.linalg import norm
from .sim_core import *

def signed_pow(x, gamma):
    """⌈x⌉^γ = |x|^γ·sign(x), with smooth linear near zero."""
    eps = 0.01
    ax = np.abs(x)
    if np.isscalar(x):
        if ax < eps:
            return eps**(gamma - 1) * x
        return ax**gamma * np.sign(x)
    out = np.empty_like(x, dtype=float)
    mask = ax < eps
    out[mask] = eps**(gamma - 1) * x[mask]
    out[~mask] = ax[~mask]**gamma * np.sign(x[~mask])
    return out

class LPF3:
    """First-order low-pass filter for 3-channel signals.

    Butterworth-equivalent discrete filter:  y[k] = α·x[k] + (1−α)·y[k−1]
    where α = dt / (dt + 1/(2π·f_c)).
    """

    def __init__(self, fc, dt=1e-3):
        tau = 1.0 / (2 * np.pi * fc)
        self.alpha = dt / (dt + tau)
        self._y = None

    def reset(self, y0=None):
        self._y = y0.copy() if y0 is not None else None

    def __call__(self, x):
        if self._y is None:
            self._y = x.copy()
            return self._y.copy()
        self._y = self.alpha * x + (1 - self.alpha) * self._y
        return self._y.copy()

class GFAFEO:
    """Gait-Frequency-Aware Finite-time Extended Observer.

    Structure (per channel j, eq:feo):
        ż₁ = z₂ − ℓ₁·L  ·⌈ẽ⌉^(1−ε₁)
        ż₂ = g·u + z₃ + d_IM − ℓ₂·L² ·⌈ẽ⌉^(1−2ε₁) + vel_corr
        ż₃ =                  − ℓ₃·L³ ·⌈ẽ⌉^(1−3ε₁) + d_IM

    where ẽ = z₁ − y_meas, g·u is the known control contribution.
    Output: d̂ = z₃ + d_IM.
    """

    def __init__(self, f_g=F_G_NOM, use_internal_model=True,
                 ell1=3.0, ell2=3.0, ell3=1.0,
                 L=20.0, eps1=0.25,
                 gamma_n=50.0, mu_v=5.0,
                 kappa_L=1.8, kappa_gamma=1.8):
        self.f_g = f_g
        self.use_im = use_internal_model
        self.ell1 = ell1
        self.ell2 = ell2
        self.ell3 = ell3
        self.L = L
        self.L_nom = L
        self.eps1 = eps1
        self.gamma_n = gamma_n
        self.gamma_n_nom = gamma_n
        self.mu_v = mu_v
        self.kappa_L = kappa_L
        self.kappa_gamma = kappa_gamma

        # Harmonics: f_g/2, f_g, 2f_g
        self.harmonics = [0.5, 1.0, 2.0]
        self.n_h = len(self.harmonics)

        # Observer state: [z1, z2, z3] per channel
        self.z = np.zeros((3, 3))
        # Internal model Fourier coefficients: [a_n, b_n] per harmonic per channel
        self.im_coeff = np.zeros((3, 2 * self.n_h))  # shape (3, 6)

        self._d_hat_max = 50.0
        self._initialized = False
        # Stored d_hat for use in the NEXT control step
        self._d_hat = np.zeros(3)
        # Sigma-modification: leakage rate for IM coefficient decay
        self.sigma_im = 0.1   # 1/s → time constant ~10s

    def initialize(self, y_meas, y_rate):
        """Set observer state to match current measurements (avoids startup spike)."""
        for ch in range(3):
            self.z[ch, 0] = y_meas[ch]
            self.z[ch, 1] = y_rate[ch]
            self.z[ch, 2] = 0.0
        self.im_coeff[:] = 0.0
        self._d_hat[:] = 0.0
        self._initialized = True

    def reset(self, kappa_hs=1.8):
        """Hot-switch reset: zero disturbance state, boost bandwidth."""
        self.z[:, 2] = 0.0       # zero d_hat, keep position/rate estimates
        self.im_coeff[:] = 0.0   # IM coefficients must relearn
        self._d_hat[:] = 0.0
        self.L = self.L_nom * kappa_hs
        self.gamma_n = self.gamma_n_nom * kappa_hs

    def set_enhanced(self, on=True):
        if on:
            self.L = self.L_nom * self.kappa_L
            self.gamma_n = self.gamma_n_nom * self.kappa_gamma
        else:
            self.L = self.L_nom
            self.gamma_n = self.gamma_n_nom

    @property
    def d_hat(self):
        """Last computed disturbance estimate (for use before update)."""
        return self._d_hat.copy()

    def update(self, t, y_meas, y_rate, g_u, dt):
        """One observer step.

        Args:
            t: current time
            y_meas: measured output [3] (error: pos or attitude)
            y_rate: measured derivative [3] (error rate)
            g_u: known control contribution g(α)·u [3]
            dt: time step
        Returns:
            d_hat: estimated disturbance [3]
        """
        # Auto-initialize on first call
        if not self._initialized:
            self.initialize(y_meas, y_rate)

        L = self.L
        e1 = self.eps1
        d_hat = np.zeros(3)

        for ch in range(3):
            e_obs = self.z[ch, 0] - y_meas[ch]   # ẽ = ẑ₁ - y

            # Internal model contribution (eq:IM)
            d_IM = 0.0
            if self.use_im:
                for j, n in enumerate(self.harmonics):
                    wj = 2 * np.pi * n * self.f_g
                    d_IM += (self.im_coeff[ch, j] * np.sin(wj * t)
                             + self.im_coeff[ch, self.n_h + j] * np.cos(wj * t))

            # Velocity error correction (eq:vel_corr)
            vel_corr = -self.mu_v * (self.z[ch, 1] - y_rate[ch])

            # L-scaled FEO dynamics (eq:feo)
            dz1 = self.z[ch, 1] - self.ell1 * L * signed_pow(e_obs, 1 - e1)
            dz2 = (g_u[ch] + self.z[ch, 2] + d_IM
                   - self.ell2 * L**2 * signed_pow(e_obs, 1 - 2*e1)
                   + vel_corr)
            dz3 = -self.ell3 * L**3 * signed_pow(e_obs, 1 - 3*e1) + d_IM

            self.z[ch, 0] += dz1 * dt
            self.z[ch, 1] += dz2 * dt
            self.z[ch, 2] += dz3 * dt
            self.z[ch, 2] = np.clip(self.z[ch, 2], -self._d_hat_max, self._d_hat_max)

            # IM adaptation with σ-modification (eq:IM_adapt)
            if self.use_im:
                for j, n in enumerate(self.harmonics):
                    wj = 2 * np.pi * n * self.f_g
                    self.im_coeff[ch, j] += (-self.gamma_n * e_obs * np.sin(wj * t)
                                              - self.sigma_im * self.im_coeff[ch, j]) * dt
                    self.im_coeff[ch, self.n_h + j] += (-self.gamma_n * e_obs * np.cos(wj * t)
                                                         - self.sigma_im * self.im_coeff[ch, self.n_h + j]) * dt
                # Clip IM coefficients to prevent runaway
                self.im_coeff[ch] = np.clip(self.im_coeff[ch], -self._d_hat_max, self._d_hat_max)

            d_hat[ch] = self.z[ch, 2] + d_IM

        self._d_hat = np.clip(d_hat, -self._d_hat_max, self._d_hat_max)
        return self._d_hat.copy()

class MissionFSM:
    APPROACH = 0; ALIGN = 1; DESCEND = 2; DOCK = 3
    LOCK = 4; STOW = 5; TAKEOFF = 6; CRUISE = 7
    NAMES = ['APPROACH','ALIGN','DESCEND','DOCK',
             'LOCK','STOW','TAKEOFF','CRUISE']

    def __init__(self):
        self.phase = self.APPROACH
        self.phase_time = 0.0
        self.locked = False
        self.t_last_switch = 0.0

    def update(self, t, pos_err_xy, z_err, att_err, vz, alpha, dt):
        old = self.phase
        self.phase_time += dt
        if self.phase == self.APPROACH:
            if norm(pos_err_xy) < 0.5 and abs(z_err) < 1.0:
                self.phase = self.ALIGN
        elif self.phase == self.ALIGN:
            if norm(pos_err_xy) < 0.05 and abs(z_err) < 0.3:
                self.phase = self.DESCEND
        elif self.phase == self.DESCEND:
            if norm(pos_err_xy) < 0.03 and abs(z_err) < 0.05:
                self.phase = self.DOCK
        elif self.phase == self.DOCK:
            if (norm(pos_err_xy) < 0.005
                and np.all(np.abs(att_err[:2]) < np.radians(2.0))
                and abs(vz) < 0.1):
                self.phase = self.LOCK
        elif self.phase == self.LOCK:
            if self.phase_time > 0.5:
                self.locked = True
                self.phase = self.STOW
        elif self.phase == self.STOW:
            if alpha <= ALPHA_FOLD + 1.0:
                self.phase = self.TAKEOFF
        elif self.phase == self.TAKEOFF:
            if abs(z_err) < 0.05 and abs(vz) < 0.2:
                self.phase = self.CRUISE
        if self.phase != old:
            self.t_last_switch = t
            self.phase_time = 0.0
        return self.phase

    def get_alpha_cmd(self):
        if self.phase <= self.ALIGN:
            return ALPHA_DEPLOY
        elif self.phase <= self.DOCK:
            progress = min(self.phase_time / 6.0, 1.0)
            return ALPHA_DEPLOY + progress * (ALPHA_FOLD - ALPHA_DEPLOY)
        elif self.phase <= self.STOW:
            return ALPHA_FOLD
        elif self.phase == self.TAKEOFF:
            progress = min(self.phase_time / 4.0, 1.0)
            return ALPHA_FOLD + progress * (ALPHA_DEPLOY - ALPHA_FOLD)
        else:
            return ALPHA_DEPLOY

def accel_to_thrust_attitude(accel, psi, mass):
    """Convert desired acceleration [3] → (F, phi_des, theta_des).

    Uses yaw-rotated body-frame decomposition.
    """
    ax, ay, az = accel
    az_g = az + G_ACC

    # Total thrust = m * |a + g|
    F = mass * np.sqrt(ax**2 + ay**2 + az_g**2)
    F = np.clip(F, 0.5, mass * G_ACC * 4)

    # Yaw rotation to body frame
    cp, sp = np.cos(psi), np.sin(psi)
    ax_b =  ax * cp + ay * sp
    ay_b =  ax * sp - ay * cp
    az_safe = az_g + 1e-6

    phi_des   = np.arctan2(ay_b, az_safe)
    theta_des = np.arctan2(ax_b, np.sqrt(ay_b**2 + az_safe**2))

    phi_des   = np.clip(phi_des, -0.5, 0.5)
    theta_des = np.clip(theta_des, -0.5, 0.5)

    return F, phi_des, theta_des

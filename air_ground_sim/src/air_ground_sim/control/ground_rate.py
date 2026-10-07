"""Bounded chassis-rate feedback ahead of the existing wheel torque servos.

The output is an equivalent differential-wheel rate command, not an applied
body wrench. Original torque limits, collision and friction remain unchanged.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class YawFeedbackConfig:
    proportional: float = 3.
    integral: float = 4.
    drive_limit_rps: float = 2.
    drive_slew_rps2: float = 4.


class YawRateFeedback:
    def __init__(self, config=YawFeedbackConfig()):
        self.config = config
        if any(not math.isfinite(v) or v <= 0 for v in vars(config).values()):
            raise ValueError('Positive finite yaw feedback parameters required')
        self.reset()

    def reset(self):
        self.integral = self.drive = 0.

    def update(self, requested, measured, dt):
        if not all(math.isfinite(v) for v in (requested, measured, dt)) or not 0 < dt <= .1:
            raise ValueError('Finite rates and a positive sub-100ms integration interval required')
        cfg, error = self.config, requested - measured
        candidate = self.integral + cfg.integral * dt * error
        raw = requested + cfg.proportional * error + candidate
        # Conditional integration prevents winding up against the drive bound.
        if not ((raw > cfg.drive_limit_rps and error > 0) or
                (raw < -cfg.drive_limit_rps and error < 0)):
            self.integral = max(-cfg.drive_limit_rps, min(cfg.drive_limit_rps, candidate))
        target = max(-cfg.drive_limit_rps, min(cfg.drive_limit_rps,
                     requested + cfg.proportional * error + self.integral))
        delta = cfg.drive_slew_rps2 * dt
        self.drive += max(-delta, min(delta, target - self.drive))
        return self.drive

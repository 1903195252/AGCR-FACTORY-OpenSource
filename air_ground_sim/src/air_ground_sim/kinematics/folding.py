from __future__ import annotations
import math
import numpy as np
import mujoco
def rotation(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]])

class FourBarGeometry:
    def __init__(self, specification):
        self.spec = specification
        uav, fold = specification["uav"], specification["uav"]["folding"]
        self.pivot = np.array([uav["pivot_square_side_m"] / 2] * 2)
        self.crank = fold["crank_radius_m"]
        self.rod = fold["coupler_length_m"]
        self.output = fold["output_link_length_m"]
        self.beta = -math.asin(fold["pin_offset_m"] / self.output)
        self.ratio = fold["main_gear_teeth"] / fold["pinion_teeth"]
        self.alpha_deploy = math.radians(fold["deployed_arm_deg"])
        self.alpha_fold = math.radians(fold["folded_arm_deg"])
        self.theta_deploy = self.inverse(self.alpha_deploy)
        self.theta_fold = self.inverse(self.alpha_fold)

    def configuration(self, theta):
        C = self.crank * np.array([math.cos(theta), math.sin(theta)])
        delta = self.pivot - C
        distance = np.linalg.norm(delta)
        if not abs(self.rod - self.output) < distance < self.rod + self.output:
            raise ValueError("Four-bar assembly is infeasible or at a toggle")
        a = (self.rod**2 - self.output**2 + distance**2) / (2 * distance)
        h2 = self.rod**2 - a**2
        if h2 <= 0:
            raise ValueError("Four-bar toggle")
        D = C + a * delta / distance + math.sqrt(h2) * np.array([delta[1], -delta[0]]) / distance
        alpha = math.atan2(*(D - self.pivot)[::-1]) - self.beta
        return C, D, alpha

    def inverse(self, alpha):
        # Source calculation identifies the monotonically descending branch.
        low, high = math.radians(53.), math.radians(120.)
        if not self.configuration(high)[2] <= alpha <= self.configuration(low)[2]:
            raise ValueError("Requested fold angle outside the verified assembly branch")
        for _ in range(64):
            middle = (low + high) / 2
            if self.configuration(middle)[2] > alpha:
                low = middle
            else:
                high = middle
        return (low + high) / 2

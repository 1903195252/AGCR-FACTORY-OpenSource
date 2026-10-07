"""C2-continuous quintic references. No planning, privileged maps or learned models."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass
class Segment:
    start: float
    duration: float
    coefficients: np.ndarray

    def sample(self, stamp):
        t = float(np.clip(stamp-self.start, 0., self.duration))
        powers = np.array([1.,t,t*t,t**3,t**4,t**5])
        p = powers @ self.coefficients
        if stamp >= self.start+self.duration:
            return p, np.zeros_like(p), np.zeros_like(p)
        v = np.array([0.,1.,2*t,3*t*t,4*t**3,5*t**4]) @ self.coefficients
        a = np.array([0.,0.,2.,6*t,12*t*t,20*t**3]) @ self.coefficients
        return p,v,a


class References:
    def __init__(self, air, ground):
        self.segments = {}
        for robot,value in [('air',air),('ground',ground)]:
            array=np.asarray(value,dtype=float)
            self.segments[robot]=Segment(0.,1.,np.vstack((array,np.zeros((5,len(array))))))

    def reference(self, robot, stamp):
        if robot not in self.segments: raise ValueError('robot must be air or ground')
        return tuple(x.copy() for x in self.segments[robot].sample(stamp))

    def goto(self, robot, target, now, duration):
        p,v,a=self.reference(robot,now)
        goal=np.asarray(target,dtype=float).copy()
        if goal.shape!=p.shape or not np.isfinite(goal).all(): raise ValueError('Invalid reference shape/value')
        if not math.isfinite(duration) or duration<=0: raise ValueError('Positive finite duration required')
        goal[-1]=p[-1]+math.atan2(math.sin(goal[-1]-p[-1]),math.cos(goal[-1]-p[-1]))
        T=float(duration)
        c=np.zeros((6,len(p)));c[:3]=[p,v,a/2]
        matrix=np.array([[T**3,T**4,T**5],[3*T*T,4*T**3,5*T**4],[6*T,12*T*T,20*T**3]])
        rhs=np.stack((goal-p-v*T-a*T*T/2,-v-a*T,-a))
        c[3:]=np.linalg.solve(matrix,rhs)
        self.segments[robot]=Segment(float(now),T,c)
        return now+T

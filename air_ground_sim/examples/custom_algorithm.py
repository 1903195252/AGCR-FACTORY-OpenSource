"""Replace next_targets() with your planner; the simulator knows nothing about it."""
from air_ground_sim import Simulator
import numpy as np


def next_targets(state, observations):
    # Minimal interface illustration, not an obstacle-avoidance algorithm.
    air=state['air']['position_m'].copy();air[2]=1.5
    ground=state['ground']['position_m'][:2]+[.2,0.]
    return np.r_[air,0.],np.r_[ground,0.]


def main():
    with Simulator('factory_industrial') as sim:
        state=sim.state()
        observations=sim.observe(width=160,height=120,rings=4,azimuth_samples=60)
        air,ground=next_targets(state,observations)
        sim.command(air=air,ground=ground,duration=4.)
        for _ in range(round(2./sim.timestep)):
            state=sim.step()
        print(state['time_s'],state['air'],state['ground'])


if __name__=='__main__':main()

#We optimize circuit c, where we optimize the width of a total of four transistors. 
#Code for the Reinforced Learning Environment
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import numpy as np
import matplotlib.pyplot as plt
import sys
import PySpice
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from PySpice.Spice.Library import SpiceLibrary

from mpl_toolkits.mplot3d import Axes3D
from PySpice.Probe.WaveForm import WaveForm
import csv
from datetime import datetime
from circuitC_analysis_1102 import simulate


class CustomFunctionEnv(gym.Env):
    def __init__(self):
        super(CustomFunctionEnv, self).__init__()
        # Action space: move in 2D (dx1, dx2)
        self.xmin = 180  # 180nm : Smallest possible transistor size
        self.xmax = 2250  # 2250nm : Largest possible transistor size
        
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(4,), dtype=np.float32) #Limit action from -0.1 to 0.1

        # Observation space: different ranges for x1 and x2
        self.observation_space = spaces.Box(low=np.array([0,0,0,0]), high=np.array([1,1,1,1]), dtype=np.float32) #Limit exploration space from 0 to 1


        # Initial state
        self.state = self._random_point_in_range()

    def _random_point_in_range(self):
        """Generate a random point within the defined range for x1 and x2."""
        random_point = np.random.uniform(low=np.array([0,0,0,0]), high=np.array([1,1,1,1]))
        random_state = [random_point[0],random_point[1],random_point[2],random_point[3]]
        return random_state
    
    def _normalize(self, x):
        """Normalize a value from the original range to 0-1 range."""
        return (x - self.xmin) / (self.xmax - self.xmin)

    def _denormalize(self, x_norm):
        """Denormalize a value from 0-1 range to the original range (180nm to 2250nm)."""
        return x_norm * (self.xmax - self.xmin) + self.xmin
    
    def FOM(self, w1,w2,w3,w4):
        """We created a single goal metric called FOM, which is a performance formula that represents optimization, where higher FOM means better optimization."""
        fom = simulate(w1 * 1e-9, w2 * 1e-9, w3 * 1e-9,w4* 1e-9) 

        return fom
    
    def step(self, state, action):
        # Update state based on action
        x1, x2, x3 ,x4= state

        denorm_x1,denorm_x2,denorm_x3,denorm_x4= self._denormalize(x1),self._denormalize(x2),self._denormalize(x3),self._denormalize(x4)

        old = self.FOM(denorm_x1,denorm_x2,denorm_x3,denorm_x4)

        # Apply action
        next_state = [x + y for x, y in zip(state,action)]
        
        # Ensure state is within bounds
        next_state[0] = np.clip(next_state[0], 0,1)  
        next_state[1] = np.clip(next_state[1], 0,1)
        next_state[2] = np.clip(next_state[2], 0,1)
        next_state[3] = np.clip(next_state[3], 0,1)
        # Calculate new FoM and reward
        x1_new, x2_new, x3_new ,x4_new= next_state
        denorm_x1_new,denorm_x2_new,denorm_x3_new,denorm_x4_new= self._denormalize(x1_new),self._denormalize(x2_new),self._denormalize(x3_new),self._denormalize(x4_new)

        new = self.FOM(denorm_x1_new,denorm_x2_new,denorm_x3_new,denorm_x4_new)

        print(f"Old FoM: {old}, New FoM: {new}")
        diff = new - old  # Reward is the change in FoM
        if diff>0:
            reward = diff
        else:
            reward = 1000*diff

        # In this example, the episode does not end naturally (so we set done = False)
        terminated = False
        truncated = False

        # Info can be used for debugging or analysis (optional)
        info = old

        return next_state, reward, terminated, truncated, info



    def reset(self, seed=None, options=None):
        # Reset the state to a random point within the range and handle the seed
        super().reset(seed=seed)
        np.random.seed(seed)  # Set the seed for reproducibility
        self.state = self._random_point_in_range()  # Ensure the initial state is clipped within the observation space
        return self.state, {}

    def render(self, mode='human'):
        # Visualization is optional and not necessary for the basic functionality
        pass

    def close(self):
        pass

    def random_action(self, min_vals=[-0.1,-0.1,-0.1,-0.1], max_vals=[0.1,0.1,0.1,0.1]):
        random_act = np.array([np.random.rand() * (max_val - min_val) + min_val for min_val, max_val in zip(min_vals, max_vals)])
        rand_action = [random_act[0],random_act[1],random_act[2],random_act[3]]
        return rand_action
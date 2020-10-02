'''
The code in this file is largely copied from the DQN_lab jupyter notebook
provided as an assignment for the Reinforcement Learning course at the
University of Amsterdam, and modified where necessary.
'''

import random
import numpy as np

class BairdsCounterExample:
    def __init__(self):
        self.n_states = 7

    def reset(self):
        self.current = np.random.choice(np.arange(0, 6))
        # Return state as one-hot encoding
        state = np.zeros(7)
        state[self.current-1] = 1.0
        return state

    def step(self, action):
        if action == 1:
            self.current = 7
        elif action == 0:
            self.current = np.random.choice(np.arange(0, 6))
        done = False
        # Create one-hot encoding of the state
        state = np.zeros(7)
        state[self.current-1] = 1.0
        return state, 0, done, ""

    def seed(self, seed):
        random.seed(seed)

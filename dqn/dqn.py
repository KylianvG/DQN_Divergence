'''
The code in this file is largely copied from the DQN_lab jupyter notebook
provided as an assignment for the Reinforcement Learning course at the
University of Amsterdam, and modified where necessary.
'''

import random
from torch import nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, num_hidden=128, num_in=4, num_out=2):
        nn.Module.__init__(self)
        self.l1 = nn.Linear(num_in, num_hidden)
        self.l2 = nn.Linear(num_hidden, num_out)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x)))

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        # If memory is full, remove the first added item
        if len(self) > self.capacity:
            self.memory.pop(0)

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self)))

    def __len__(self):
        return len(self.memory)

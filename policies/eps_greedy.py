'''
The code in this file is largely copied from the DQN_lab jupyter notebook
provided as an assignment for the Reinforcement Learning course at the
University of Amsterdam, and modified where necessary.
'''

import random
import torch

class EpsilonGreedyPolicy(object):
    """
    A simple epsilon greedy policy.
    """
    def __init__(self, Q, epsilon, its=1000, choices=[0,1]):
        self.Q = Q
        self.epsilon = epsilon
        self.min_eps = epsilon
        self.decrease_its = 1000
        self.choices = choices

    def sample_action(self, obs):
        """
        This method takes a state as input and returns an action sampled from this policy.

        Args:
            obs: current state

        Returns:
            An action (int).
        """
        r = random.random()
        # Explore
        if r < self.epsilon:
            a = random.choice(self.choices)
        # Exploit
        else:
            with torch.no_grad():
                # Added .eval() and .train() because nn.BatchNorm1d requires it.
                self.Q.eval()
                out = self.Q(torch.from_numpy(obs).unsqueeze(0).float())
                self.Q.train()
                a = torch.argmax(out).item()
        return a

    def set_epsilon(self, it):
        self.epsilon = self._calc_epsilon(it)

    def _calc_epsilon(self, it):
        epsilon = max(self.min_eps, 1 - it * ((1-self.min_eps) / self.decrease_its))
        return epsilon

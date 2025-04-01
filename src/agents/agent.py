import numpy as np
import random

class DroneDQNAgent:
    def __init__(self, obs_size, action_size, device):
        self.obs_size = obs_size
        self.action_size = action_size
        self.device = device
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def act(self, observation):
        """Select an action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            # Placeholder: pretend we always pick action 0
            return 0

    def step(self, state, action, reward, next_state, done):
        """Store experience (not implemented here)"""
        pass

    def learn(self):
        """Perform a learning step (not implemented here)"""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

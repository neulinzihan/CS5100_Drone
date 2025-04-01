import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class DroneDQNAgent:
    """
    A simple DQN agent for a single drone. 
    Observations: (row, col, coverage_frac) 
    Actions: 5 (0=up,1=down,2=left,3=right,4=stay)
    
    We store transitions and do a basic DQN update.
    """

    def __init__(self,
                 obs_size=3,
                 action_size=5,
                 lr=1e-3,
                 gamma=0.99,
                 epsilon=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995,
                 buffer_size=10000,
                 batch_size=64,
                 device='cpu'):
        self.obs_size = obs_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        self.replay_buffer = deque(maxlen=self.buffer_size)
        
        # Build network
        self.policy_net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
    
    def act(self, obs):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size-1)
        else:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(obs_t)
            return int(torch.argmax(q_values, dim=1).item())
    
    def step(self, obs, action, reward, next_obs, done):
        """
        Store a transition in the replay buffer.
        """
        self.replay_buffer.append((obs, action, reward, next_obs, done))
    
    def learn(self):
        """
        Sample from replay and do a one-step DQN update.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_batch = zip(*batch)
        
        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        act_tensor = torch.LongTensor(act_batch).to(self.device).unsqueeze(-1)
        rew_tensor = torch.FloatTensor(rew_batch).to(self.device).unsqueeze(-1)
        next_obs_tensor = torch.FloatTensor(next_obs_batch).to(self.device)
        done_tensor = torch.FloatTensor(done_batch).to(self.device).unsqueeze(-1)
        
        # current Q
        current_q = self.policy_net(obs_tensor).gather(1, act_tensor)
        
        # next Q
        with torch.no_grad():
            next_q = self.policy_net(next_obs_tensor).max(dim=1, keepdim=True)[0]
        
        target_q = rew_tensor + self.gamma * next_q * (1 - done_tensor)
        
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

""" A Deep Q network  """

import random

from collections import deque

import gym
import torch
from torch.autograd import Variable


class DQN(object):
    """A NN approach to Q learning"""

    def __init__(self, env, lr=0.01, discount=0.95, epsilon=1):
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_disount = 0.95
        self.env = env
        self.num_states = env.observation_space.n
        self.num_actions = env.action_space.n
        self.memory = deque(maxlen=2000)


    def neural_network(self):
        """A neural net with 2 hidden layers"""
        N, D_in, H, D_out = 32, self.num_states, 24, self.num_actions
        self.model = torch.nn.sequenctial(
                torch.nn.Linear(D_in, H),
                torch.nn.ReLU(),
                torch.nn.Linear(H),
                torch.nn.ReLU(),
                torch.nn.Linear(H, D_out)
                )
        self.loss_fn = torch.nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def remember(self, state, action, reward, n_state, status):
        self.memory.append((state, action, reward, n_state, status))


    def eps_greedy(self, state):
        if  torch.rand(1) > self.epsilon:
            return torch.max(self.model(state))[1] # returns max, indice
        else:
            return env.action_space.sample()

    def reply(self, batch_size=32):
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, n_state, status:





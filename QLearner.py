"""A Qlearner"""

import os
import sys

import gym
from gym import wrappers

import numpy as np

class QLearner(object):
    """Q learner"""

    def __init__(self, env, l_rate=0.02, discount=0.95, epsilon=1):
        self.env = env
        self.learning_rate = l_rate
        self.discount = discount
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.n_obs = env.observation_space.n
        self.n_actions = env.action_space.n
        self.q_mat = np.random.rand(self.n_obs, self.n_actions)
        self.rand = 0
        self.nrand = 0

    def eps_greedy(self, state):
        """Returns action that gives maximum value or a random action.
        """
        if np.random.rand() > self.epsilon:
            self.nrand += 1
            return np.argmax(self.q_mat[state])
        else:
            self.rand += 1
            return self.env.action_space.sample()

    def q_update(self, state, action, reward, next_state):
        """Update q matrix using Bellman's eqn"""
        q_val = self.q_mat[state][action]
        new_q_val = reward + self.discount * max(self.q_mat[next_state])
        delta_q = new_q_val - q_val
        self.q_mat[state][action] += self.learning_rate * delta_q

def main():
    """Entry point"""
    name = "FrozenLake8x8-v0"
    episodes = sys.argv[1]
    env = gym.make(name)
    temp_path = os.path.join("/tmp/", name)
    env = wrappers.Monitor(env, temp_path, force=True)
    learner = QLearner(env)
    for _ in range(episodes):
        state = env.reset()
        iterations = 0
        while True:
            action = learner.eps_greedy(state)
            n_state, reward, done, _ = env.step(action)
            if done:
                reward = 1 if reward > 0 else -1
                learner.q_update(state, action, reward, n_state)
                if reward > 0:
                    env.render()
                break
            reward = 0
            learner.q_update(state, action, reward, n_state)
            state = n_state
            iterations += 1
        # eps greedy annealing
        if learner.epsilon > learner.epsilon_min:
            learner.epsilon *= learner.epsilon_decay
    print("Random actions : {}, Non Random actions: {}".format(learner.rand,
                                                               learner.nrand))
    env.close()
    return temp_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: {} {}".format(sys.argv[0], "<episodes>"))
    saved_path = main()
    # gym.upload(saved_path, api_key="sk_FgkcTtZuR2GGwN4TNabgg")

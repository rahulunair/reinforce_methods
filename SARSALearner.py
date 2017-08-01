"""A Qlearner"""

import os

import gym
from gym import wrappers

import matplotlib.pyplot as plt
import numpy as np


class SARSALearner(object):
    """Q learner"""

    def __init__(self, env, l_rate=0.2, discount=0.95, epsilon=0.1):
        self.env = env
        self.learning_rate = l_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.95
        self.n_obs = env.observation_space.n
        self.n_actions = env.action_space.n
        self.q_mat = np.zeros([self.n_obs, self.n_actions])
        self.rand = 0
        self.nrand = 0
        self.rewards = 0


    def eps_greedy(self, state):
            """Returns action that gives maximum value or a random action.
            """
            if np.random.rand() > self.epsilon:
                self.nrand += 1
                return np.argmax(self.q_mat[state])
            else:
                self.rand += 1
                return self.env.action_space.sample()


    def q_update(self, state, action, reward, next_state, next_action):
        """Update q matrix using Bellman's eqn"""
        q_val = self.q_mat[state][action]
        new_q_val = reward + self.discount * self.q_mat[next_state][next_action]
        delta_q = new_q_val - q_val
        self.q_mat[state][action] += self.learning_rate * delta_q


def main():
    """Entry point"""
    name = "FrozenLake8x8-v0"
    episodes = 1000
    env = gym.make(name)
    temp_path = os.path.join("/tmp/", name)
    env = wrappers.Monitor(env, temp_path, force=True)
    learner = SARSALearner(env)
    total_rewards = 0
    for _ in range(episodes):
        state = env.reset()
        while True:
            action = learner.eps_greedy(state)
            n_state, reward, done, _ = env.step(action)
            n_action = learner.eps_greedy(n_state)
            if done:
                reward = 1 if reward > 0 else -1
                learner.q_update(state, action, reward, n_state, n_action)
                total_rewards += reward
                break
            state = n_state
            action = n_action
        if learner.epsilon > learner.epsilon_min:
            learner.epsilon *= learner.epsilon_decay
    print("Random actions : {}, Non Random actions: {}".format(learner.rand,
                                                               learner.nrand))
    stats(total_rewards, episodes)
    env.close()


def stats(rewards, episodes):
    print("Total episodes: {}".format(episodes))
    print("Total rewards: {}".format(rewards))
    print("Average rewards: {:.2f}".format(rewards/episodes))


if __name__ == "__main__":
    main()
    # gym.upload(, api_key="sk_FgkcTtZuR2GGwN4TNabgg")

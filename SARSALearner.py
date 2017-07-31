"""A Qlearner"""

import os

import gym
from gym import wrappers

import matplotlib.pyplot as plt
import numpy as np


class SARSALearner(object):
    """Q learner"""

    def __init__(self, env, l_rate=0.2, discount=0.99, epsilon=0.3):
        self.env = env
        self.learning_rate = l_rate
        self.discount = discount
        self.epsilon = np.linspace(epsilon, epsilon/10, 1000)
        self.n_obs = env.observation_space.n
        self.n_actions = env.action_space.n
        self.q_mat = np.zeros([self.n_obs, self.n_actions])
        self.rand = 0
        self.nrand = 0
        self.rewards = 0

    def eps_greedy(self, state, iteration):
        """returns action that gives maximum value or a random action.
        Here I have implemented a crude annealing, dividing by 2 * episodes"""

        # eps = self.epsilon[iteration]
        eps = 0
        if np.random.rand() < eps:
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
    episodes = 10000
    env = gym.make(name)
    temp_path = os.path.join("/tmp/", name)
    env = wrappers.Monitor(env, temp_path, force=True)
    learner = SARSALearner(env)
    total_rewards = 0
    for _ in range(episodes):
        state = env.reset()
        iterations = 0
        while True:
            action = learner.eps_greedy(state, iterations)
            n_state, reward, done, _ = env.step(action)
            n_action = learner.eps_greedy(n_state, iterations)
            if done:
                reward = 1 if reward > 0 else -1
                learner.q_update(state, action, reward, n_state, n_action)
                total_rewards += reward
                break
            state = n_state
            action = n_action
            iterations += 1
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

"""A Qlearner"""

import os

import gym
from gym import wrappers

import numpy as np

class QLearner(object):
    """Q learner"""

    def __init__(self, env, l_rate=0.1, discount=0.9, epsilon=1):
        self.env = env
        self.learning_rate = l_rate
        self.discount = discount
        self.epsilon = np.linspace(epsilon, epsilon/10, 1000)
        self.n_obs = env.observation_space.n
        self.n_actions = env.action_space.n
        self.q_mat = np.random.rand(self.n_obs, self.n_actions)
        self.rand = 0
        self.nrand = 0

    def eps_greedy(self, state, iteration):
        """returns action that gives maximum value or a random action.
        Here I have implemented a crude annealing, dividing by 2 * episodes"""

        eps = self.epsilon[iteration]
        if np.random.rand() <  eps:
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
    env = gym.make(name)
    temp_path = os.path.join("/tmp/", name)
    env = wrappers.Monitor(env, temp_path, force=True)
    q_learner = QLearner(env)
    for _ in range(10000):
        state = env.reset()
        iterations = 0
        while True:
            action = q_learner.eps_greedy(state, iterations)
            n_state, reward, done, _ = env.step(action)
            if done:
                reward = 1 if reward > 0 else -1
                q_learner.q_update(state, action, reward, n_state)
                if reward > 0:
                    env.render()
                break
            reward = 0
            q_learner.q_update(state, action, reward, n_state)
            state = n_state
            iterations += 1
    print("Random actions : {}, Non Random actions: {}".format(q_learner.rand,
                                                               q_learner.nrand))
    return temp_path


if __name__ == "__main__":
    saved_path = main()
    gym.upload(saved_path, api_key="sk_FgkcTtZuR2GGwN4TNabgg")

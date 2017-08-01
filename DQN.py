""" A Deep Q network  """
import sys
import random

from collections import deque

import gym
from gym import wrappers
import numpy as np
import keras


class DQN(object):
    """A NN approach to Q learning"""

    def __init__(self, env, lr=0.01, discount=0.95, epsilon=1):
        self.lr = lr
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.95
        self.env = env
        try:
            self.num_states = env.observation_space.n
        except BaseException:
            self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.n
        self.memory = deque(maxlen=2000)
        self.neural_network()

    def neural_network(self):
        """A neural net with 2 hidden layers"""
        D_in, H, D_out = self.num_states, 32, self.num_actions
        print("Neural net nodes: ", D_in, H, D_out)
        optimizer = keras.optimizers.Adam(lr=self.lr)
        self.model = keras.models.Sequential()
        self.model.add(
            keras.layers.Dense(
                H,
                input_dim=D_in,
                activation="relu"))
        self.model.add(keras.layers.Dense(H, activation="relu"))
        self.model.add(keras.layers.Dense(D_out, activation="linear"))
        self.model.compile(loss="mse", optimizer=optimizer)

    def save(self, state, action, reward, n_state, status):
        self.memory.append((state, action, reward, n_state, status))

    def eps_greedy(self, state):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.model.predict(state)[0])
        else:
            return self.env.action_space.sample()

    def train_model(self, batch_size=32):
        """Sample "SARS`" from memory and train the NN.
        if `done`, set target for action as reward,
        else set target for action as the discounted Q value.
        """
        print("Learning...")
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, n_state, done in batch:
            # q vals for (state, actions)
            target = self.model.predict(state)  # get Q values for (s,a)
            if done:
                target[0][action] = reward
            else:
                next_q = self.model.predict(n_state)[0]
                target[0][action] = reward + self.discount * np.max(next_q)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    @staticmethod
    def reshape_data(data, size):
        return np.reshape(data, [1, size])


def main():
    """Entry point"""
    batch_size = 32
    if len(sys.argv) != 2:
        sys.exit("Usage: {} {}".format(sys.argv[0], "<episodes>"))
    episodes = int(sys.argv[1])
    env = gym.make("CartPole-v0")
    env = wrappers.Monitor(env, "/tmp/Freeze")
    learner = DQN(env)
    learner.neural_network()  # build model and set to self.model
    for _ in range(episodes):
        state = env.reset()
        state = DQN.reshape_data(state, learner.num_states)
        while True:
            action = learner.eps_greedy(state)
            n_state, reward, done, _ = env.step(action)
            n_state = DQN.reshape_data(n_state, learner.num_states)
            reward = 1 if reward > 0 else -1
            learner.save(state, action, reward, n_state, done)
            if done:
                if reward > 1:
                    env.render()
                break
        if len(learner.memory) > batch_size:
            learner.train_model(batch_size)



if __name__ == "__main__":
    main()

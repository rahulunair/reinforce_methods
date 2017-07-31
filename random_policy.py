"""Figures out the best policy by randomly trying random weights"""
import sys

import gym


def main(episodes, env_id):
    env = gym.make(env_id)
    total_reward = 0
    for _ in range(episodes):
        env.reset()
        action = env.action_space.sample()
        while True:
            _, reward, done, _ = env.step(action)
            if done:
                total_reward += reward
                break
    print("Episodes: ", episodes)
    print("Total rewards: ", total_reward)
    print("Average rewards: {:.2f}".format(total_reward/episodes))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: random_policy <env_id> <episodes>")
    env_id, episodes  = sys.argv[1:]
    main(int(episodes), env_id)

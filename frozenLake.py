# -*- coding: utf-8 -*-

import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0")


def q_learning(env, episodes, lr, epsilon, gamma):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    r_list = []

    for i in tqdm(range(num_episodes)):
        s = env.reset()
        rewards = 0
        d = False
        for j in range(100):
            a = np.argmax(Q[s])
            if np.random.rand(1) < epsilon:
                a = env.action_space.sample()
            s1, r, d, _ = env.step(a)
            Q[s, a] = Q[s, a] + lr * (r + gamma * np.max(Q[s1, :]) - Q[s, a])
            rewards += r
            s = s1
            if d:
                break
        r_list.append(rewards)

    return Q, r_list


def get_strategy(Q):
    Pi = np.zeros(env.observation_space.n)
    for i in range(len(Pi)):
        Pi[i] = np.argmax(Q[i])
    return Pi


def test_strategy(Pi, episodes):
    r_list = []
    j_list = []
    for i in tqdm(range(episodes)):
        s = env.reset()
        rewards = 0
        d = False
        j = 0
        while j < 100:
            j += 1
            a = Pi[s]
            s1, r, d, _ = env.step(a)
            rewards += r
            s = s1
            if d:
                break
        r_list.append(rewards)
        j_list.append(j)

    return r_list, j_list


if __name__ == '__main__':
    num_episodes = 10000
    Q, r_list = q_learning(env, num_episodes, .1, .3, .95)
    Pi = get_strategy(Q)

    Pi_show = Pi.reshape((4, 4))
    print("Training score:", np.average(r_list))
    print("Strategy:")
    print(Pi_show)

    r_list, j_list = test_strategy(Pi, 100)
    print("Testing score:", np.average(r_list))
    print("Avg step:", np.average(j_list))
    plt.plot(np.cumsum(r_list) / (np.arange(100)+1))
    plt.show()


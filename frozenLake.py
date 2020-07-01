# -*- coding: utf-8 -*-

import gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0")


def get_action_eps_greedy(Q, s, epsilon, nA):
    policy_s = np.ones(nA) * epsilon / nA
    best_a = np.argmax(Q[s])
    policy_s[best_a] = 1 - epsilon + (epsilon / nA)
    action = np.random.choice(np.arange(nA), p=policy_s)
    return action


def q_learning(env, episodes, lr, epsilon, gamma):
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    r_list = []

    for i in tqdm(range(num_episodes)):
        s = env.reset()
        rewards = 0
        d = False
        j = 0
        for j in range(100):
            a = get_action_eps_greedy(Q, s, 1 - i / num_episodes, env.action_space.n)
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


if __name__ == '__main__':
    num_episodes = 10000
    Q, r_list = q_learning(env, num_episodes, .8, 1, .9)
    Pi = get_strategy(Q)

    Pi_show = Pi.reshape((4, 4))

    print("Score over time:", str(sum(r_list) / num_episodes))
    print("Strategy:")
    print(Pi_show)
    plt.plot(np.cumsum(r_list) / (np.arange(num_episodes)+1))
    plt.show()


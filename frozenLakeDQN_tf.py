# -*- coding: utf-8 -*-

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make("FrozenLake-v0")

tf.reset_default_graph()

inputs1 = tf.placeholder(shape=(1, 16), dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16, 4], 0, 0.01))
Q_out = tf.matmul(inputs1, W)
predict = tf.argmax(Q_out, 1)

next_Q = tf.placeholder(shape=[1, 4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(next_Q - Q_out))
trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
update_model = trainer.minimize(loss)


init = tf.initialize_all_variables()

gamma = .99
e = .3
num_episodes = 2000
j_list_train = []
r_list_train = []
j_list_test = []
r_list_test = []
Pi = np.zeros(16)
with tf.Session() as sess:
    sess.run(init)
    for i in tqdm(range(num_episodes)):
        s = env.reset()
        r_all = 0
        d = False
        j = 0
        while j < 100:
            j += 1
            a, all_Q = sess.run([predict, Q_out], feed_dict={inputs1: np.identity(16)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            s1, r, d, _ = env.step(a[0])
            Q1 = sess.run(Q_out, feed_dict={inputs1: np.identity(16)[s1:s1+1]})
            max_Q1 = np.max(Q1)
            target_Q = all_Q
            target_Q[0, a[0]] = r + gamma * max_Q1
            _, W1 = sess.run([update_model, W], feed_dict={inputs1: np.identity(16)[s:s+1], next_Q: target_Q})
            r_all += r
            s = s1
            if d:
                e = 1/((i/100) + 5)
                break
        r_list_train.append(r_all)
        j_list_train.append(j)

    for s in range(16):
        a = sess.run(predict, feed_dict={inputs1: np.identity(16)[s:s+1]})
        Pi[s] = a


for i in range(100):
    s = env.reset()
    r_all = 0
    d = False
    j = 0
    while j < 100:
        j += 1
        a = Pi[s]
        s1, r, d, _ = env.step(a)
        r_all += r
        s = s1
        if d:
            break
    r_list_test.append(r_all)
    j_list_test.append(j)

print("Training score:", np.average(r_list_train))
print("Testing score:", np.average(r_list_test))
print("Strategy:")
print(Pi.reshape((4, 4)))
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(121)
ax.set_title('train avg rewards')
ax.plot(np.cumsum(r_list_train)/(np.arange(len(r_list_train))+1))
ax = fig.add_subplot(122)
ax.set_title('test avg rewards')
ax.plot(np.cumsum(r_list_test)/(np.arange(len(r_list_test))+1))
plt.show()


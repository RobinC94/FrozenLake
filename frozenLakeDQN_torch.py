# -*- coding: utf-8 -*-

import gym
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


env = gym.make('FrozenLake-v0')

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(device)
dtype = torch.float

model = torch.nn.Sequential(
    torch.nn.Linear(16, 4, bias=False),
)
model = model.cuda()


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.uniform_(m.weight, 0, 0.01)


model.apply(weight_init)

loss_fn = torch.nn.MSELoss(reduction='sum')
lr = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
gamma = 0.95
e = 0.3
num_episodes = 2000
r_list_train = []
r_list_test = []

for i in tqdm(range(num_episodes)):
    s = env.reset()
    r_all  = 0

    for j in range(100):
        input_s = torch.tensor(np.identity(16)[s:s+1], device=device, dtype=dtype)
        q_pred = model(input_s)
        a = int(torch.argmax(q_pred).cpu().numpy())
        if np.random.rand(1) < e:
            a = env.action_space.sample()
        s1, r, d, _ = env.step(a)
        input_s1 = torch.tensor(np.identity(16)[s1:s1+1], device=device, dtype=dtype)
        q1 = model(input_s1)
        max_q1 = torch.max(q1).detach().cpu().numpy()
        target_q = q_pred.clone().detach()
        target_q[0, a] = r + gamma * max_q1
        loss = loss_fn(target_q, q_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        s = s1
        r_all += r
        if d:
            e = 1 / (i/100 + 5)
            break
    r_list_train.append(r_all)

Pi = np.zeros(16)
for s in range(16):
    input_s = torch.tensor(np.identity(16)[s:s+1], device=device, dtype=dtype)
    Pi[s] = torch.argmax(model(input_s)).cpu().numpy()

for i in range(100):
    s = env.reset()
    r_all = 0
    for j in range(100):
        a = Pi[s]
        s1, r, d, _ = env.step(a)
        s = s1
        r_all += r
        if d:
            break
    r_list_test.append(r_all)

print("Training avg score:", np.average(r_list_train))
print("Testing avg score:", np.average(r_list_test))
print("Strategy:")
print(Pi.reshape(4, 4))
fig = plt.figure(figsize=(15, 8))
ax = fig.add_subplot(121)
ax.plot(np.cumsum(r_list_train) / (np.arange(num_episodes)+1))
ax.set_title('train avg rewards')
ax = fig.add_subplot(122)
ax.plot(np.cumsum(r_list_test) / (np.arange(100)+1))
ax.set_title('test avg rewards')
plt.show()


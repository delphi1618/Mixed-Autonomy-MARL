

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline



import matplotlib.pyplot as plt
import glob
import io
import base64

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Memory:
    def __init__(self, Adj, action_dim):
        road_num = np.shape(Adj)[0] - 1
        cell_num = 4 # Need to generalize
        self.action_level = action_dim
        self.road_num = road_num
        self.cell_num = cell_num
        agent_num = road_num*cell_num
        self.agent_num = agent_num

        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(agent_num)]
        self.reward = []
        self.done = [[] for _ in range(agent_num)]

    def get(self):
        actions = torch.tensor(self.actions)
        observations = self.observations

        pi = []
        for i in range(self.agent_num):
            # print(i)
            pi.append(torch.cat(self.pi[i]).view(len(self.pi[i]), self.action_level))

        reward = torch.tensor(self.reward)
        done = self.done

        return actions, observations, pi, reward, done

    def clear(self):
        self.actions = []
        self.observations = []
        self.pi = [[] for _ in range(self.agent_num)]
        self.reward = []
        self.done = [[] for _ in range(self.agent_num)]

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(state_dim, 64).to(device)
        self.fc2 = nn.Linear(64, 64).to(device)
        self.fc3 = nn.Linear(64, action_dim).to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x)).to(device)
        x = F.relu(self.fc2(x)).to(device)
        return F.softmax(self.fc3(x), dim=-1).to(device)

class Critic2(nn.Module):
    def __init__(self, agent_num, state_dim, action_dim):
        super(Critic2, self).__init__()

        input_dim = 1 + state_dim * agent_num + agent_num

        self.fc1 = nn.Linear(input_dim, 64).to(device)
        self.fc2 = nn.Linear(64, 64).to(device)
        self.fc3 = nn.Linear(64, action_dim).to(device)

    def forward(self, x):
        x = F.relu(self.fc1(x)).to(device)
        x = F.relu(self.fc2(x)).to(device)
        return self.fc3(x).to(device)

class COMA(object):
    def __init__(self, Adj, source, cell_num, action_level, lr_c, lr_a, gamma, target_update_steps):
        self.count = 0
        self.action_level = action_level
        self.map = Adj
        self.mapT = np.transpose(Adj)
        road_num = np.shape(Adj)[0] - 1
        self.road_num = road_num
        self.cell_num = cell_num
        self.agent_num = road_num * (cell_num+1)
        self.state_dim = 8
        self.actors = [Actor(8, action_level) for _ in range(self.agent_num)]
        self.linkN = np.zeros((road_num, 1))
        self.source = source
        for i in range(road_num):
            linkList = np.array(np.nonzero(self.map[i, :]))
            lengthList = linkList.shape[1]
            self.linkN[i, 0] = lengthList
            self.actors[i*(cell_num+1)+cell_num] = Actor(self.state_dim, action_level)
    
        self.memory = Memory(self.map, self.action_level)
        self.critic = Critic2(self.agent_num, self.state_dim, self.action_level) # Need Modify
        self.critic_target = Critic2(self.agent_num, self.state_dim, self.action_level) # Need Modify
        self.gamma = gamma
        self.c_loss_fn = torch.nn.MSELoss()
        self.action = np.zeros((road_num, cell_num+1))
        self.actors_optimizer = [torch.optim.Adam(self.actors[k].parameters(), lr=lr_a) for k in range(agent_num)]
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.target_update_steps = target_update_steps

    def get_actions(self, inHV, inAV, outHV, outAV, cHV, cAV, velHV, velAV):
        actions = []
        obsa = []

        for k in range(self.agent_num):
            j = int(k % (self.cell_num+1))
            i = int(np.floor(k/(self.cell_num+1)))
            flag = 0
            if j == self.cell_num:
                j = j-1
                flag = 1
            # obs_arr = [inHV(i, j), inAV(i, j), outHV(i, j), outAV(i, j), cHV(i, j), cAV(i, j), velHV(i, j), velAV(i, j)]
            obs_arr = [inHV[i, j], inAV[i, j], outHV[i, j], outAV[i, j], cHV[i, j], cAV[i, j], velHV[i, j], velAV[i, j]]
            obs_arr2 = np.array(obs_arr)
            obs = torch.tensor(obs_arr2).float()
            obsa.append(obs_arr)
            
            
            if flag == 1:
                p_dist = self.actors[k](obs)
                p_act = Categorical(p_dist).sample()
                self.action[i, j] = p_act
                self.memory.pi[k].append(p_dist)
                actions.append(p_act.item())
            else:
                v_dist = self.actors[k](obs)
                v_act = Categorical(v_dist).sample()
                self.action[i, j] = v_act
                self.memory.pi[k].append(v_dist)
                # print(self.memory.pi[k])
                actions.append(v_act.item())

        # observations_arr = [inHV, inAV, outHV, outAV, cHV, cAV, velHV, velAV]
        # observations = torch.tensor(observations_arr)
        obsa2 = np.array(obsa)
        observations = torch.tensor(obsa2)

        self.memory.observations.append(observations)
        self.memory.actions.append(actions)

        return actions


    def train(self, ):
        # o1_list, a1_list, pi_a1_list, o2_list, a2_list, pi_a2_list, r_list
        actor_optimizer = self.actors_optimizer
        critic_optimizer = self.critic_optimizer
        actions, observations, pi, reward, done = self.memory.get()

        for i in range(self.agent_num):
            # train actor
            # print(i)

            input_critic = self.build_input_critic(i, observations, actions)
            Q_target = self.critic_target(input_critic).detach()

            action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)

            baseline = torch.sum(pi[i] * Q_target, dim=1).detach()
            # print(len(baseline))
            Q_taken_target = torch.gather(Q_target, dim=1, index=action_taken).squeeze()
            advantage = Q_taken_target - baseline

            log_pi = torch.log(torch.gather(pi[i], dim=1, index=action_taken).squeeze())

            actor_loss = - torch.mean(advantage * log_pi)

            actor_optimizer[i].zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[i].parameters(), 5)
            actor_optimizer[i].step()

            # train critic

            Q = self.critic(input_critic)

            action_taken = actions.type(torch.long)[:, i].reshape(-1, 1)
            Q_taken = torch.gather(Q, dim=1, index=action_taken).squeeze()

            # TD(0)
            # print(reward[:, i])
            r = torch.zeros(len(reward[:, i]))
            for t in range(len(reward[:, i])):
                # print(len(Q_taken_target))
                if done[t]:
                    r[t] = reward[:, i][t]
                else:
                    r[t] = reward[:, i][t] + self.gamma * Q_taken_target[t]

            critic_loss = torch.mean((r - Q_taken) ** 2)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5)
            critic_optimizer.step()

        if self.count == self.target_update_steps:
            self.critic_target.load_state_dict(COMA.critic.state_dict())
            self.count = 0
        else:
            self.count += 1

        self.memory.clear()

    def build_input_critic(self, agent_id, observations, actions):
        batch_size = len(observations)

        ids = (torch.ones(batch_size) * agent_id).view(-1, 1)

        observations = torch.cat(observations).view(batch_size, self.state_dim * self.agent_num)
        input_critic = torch.cat([observations.type(torch.float32), actions.type(torch.float32)], dim=-1)
        input_critic = torch.cat([ids, input_critic], dim=-1)

        return input_critic

class environment:
    def __init__(self, Adj, source, cell_num, times):
        self.map = Adj
        self.mapT = np.transpose(Adj)
        road_num = np.shape(Adj)[0] - 1
        self.road_num = road_num
        self.cell_num = cell_num
        self.times = times
        self.graph = np.array( [ [Cell(i, j, times) for i in range(cell_num)] for j in range(road_num) ], dtype=object)
        self.outAV = np.zeros((road_num, cell_num))
        self.outHV = np.zeros((road_num, cell_num))
        self.inAV = np.zeros((road_num, cell_num))
        self.inHV = np.zeros((road_num, cell_num))
        self.cAV = np.zeros((road_num, cell_num))
        self.cHV = np.zeros((road_num, cell_num))
        self.traffic = np.zeros((road_num, cell_num))
        self.autorate = np.zeros((road_num, cell_num))
        self.velAV = np.zeros((road_num, cell_num))
        self.velHV = np.zeros((road_num, cell_num))
        self.delay = np.zeros((road_num, cell_num))
        self.link = np.zeros((road_num, 5))
        self.linkin = np.zeros((road_num, 5))
        self.source = source
        self.sourceAV = 100*np.random.rand(source.size)
        self.sourceHV = 100*np.random.rand(source.size)
        self.counter = 0
        for i in range(road_num):
            linkList = np.array(np.nonzero(self.map[i, :]))
            lengthList = linkList.shape[1]
            self.link[i, 0] = lengthList
            self.link[i, 1:lengthList+1] = linkList
            linkListT = np.array(np.nonzero(self.mapT[i, :]))
            lengthListT = linkListT.shape[1]
            self.linkin[i, 0] = lengthListT
            self.linkin[i, 1:lengthListT+1] = linkListT 
    
    def setNum(self, x, y, attr):
        for i in range(attr.shape[0]):
            if attr[i] >= 0:
                if i == 0:
                    self.graph[x, y].capacity = attr[i]
                elif i == 1:
                    self.graph[x, y].maxspeed = attr[i]
                elif i == 2:
                    self.graph[x, y].length = attr[i]
                elif i == 3:
                    self.graph[x, y].traffic = attr[i]
                elif i == 4:
                    self.graph[x, y].autorate = attr[i]
                elif i == 5:
                    self.graph[x, y].velocity_AV = attr[i]
                elif i == 6:
                    self.graph[x, y].velocity_HV = attr[i]

    def step(self, v_act):
        # Require deal with the case out of boundary
        for i in range(np.shape(self.graph)[0]):
            for j in range(np.shape(self.graph)[1]):
                self.graph[i, j].routeHV
        for i in range(np.shape(self.graph)[0]):
            for j in range(np.shape(self.graph)[1]):
                self.graph[i, j].velocity_AV = v_act[i*(np.shape(self.graph)[1]+1)+j]
                self.velHV[i, j] = self.graph[i, j].speedHV()
        
        for i in range(np.shape(self.graph)[0]):
            for j in range(np.shape(self.graph)[1]):
                self.velAV[i, j] = v_act[i*(np.shape(self.graph)[1]+1)+j]
        
        for i in range(np.shape(self.graph)[0]):
            for j in range(np.shape(self.graph)[1]):
                
                self.outAV[i, j] = self.graph[i, j].outflowAV()
                self.outHV[i, j] = self.graph[i, j].outflowHV()
                
                
        # self.sourceAV = sourceAV
        # self.sourceHV = sourceHV
        for i in range(self.road_num):
            for j in range(self.cell_num):
                self.traffic[i, j], self.autorate[i, j] = self.graph[i, j].get_state()
                self.cAV[i, j] = self.traffic[i, j] * self.autorate[i, j]
                self.cHV[i, j] = self.traffic[i, j] - self.cAV[i, j]

        reward = self.calReward(v_act)
        self.counter += 1
        if self.counter >= 1000:
            done_n = True
        else:
            done_n = False
        

        return self.inHV, self.inAV, self.outHV, self.outAV, self.cHV, self.cAV, self.velHV, self.velAV, reward, done_n

    def get_obs(self,):
        
        return self.inHV, self.inAV, self.outHV, self.outAV, self.cHV, self.cAV, self.velHV, self.velAV

    def reset(self,):
        iH = self.inHV
        iA = self.inAV
        oH = self.outHV
        oA = self.outAV
        cH = self.cHV
        cA = self.cAV
        vH = self.velHV
        vA = self.velAV
        self.graph = np.array( [ [Cell(i, j, self.times) for i in range(cell_num)] for j in range(road_num) ], dtype=object)
        self.outAV = np.zeros((self.road_num, self.cell_num))
        self.outHV = np.zeros((self.road_num, self.cell_num))
        self.inAV = np.zeros((self.road_num, self.cell_num))
        self.inHV = np.zeros((self.road_num, self.cell_num))
        self.cAV = np.zeros((self.road_num, self.cell_num))
        self.cHV = np.zeros((self.road_num, self.cell_num))
        self.traffic = np.zeros((self.road_num, self.cell_num))
        self.autorate = np.zeros((self.road_num, self.cell_num))
        self.velAV = np.zeros((self.road_num, self.cell_num))
        self.velHV = np.zeros((self.road_num, self.cell_num))
        self.delay = np.zeros((self.road_num, self.cell_num))
        self.link = np.zeros((self.road_num, 5))
        self.linkin = np.zeros((self.road_num, 5))
        self.source = self.source
        self.sourceAV = 100*np.random.rand(self.source.size)
        self.sourceHV = 100*np.random.rand(self.source.size)
        self.counter = 0
        for i in range(road_num):
            linkList = np.array(np.nonzero(self.map[i, :]))
            lengthList = linkList.shape[1]
            self.link[i, 0] = lengthList
            self.link[i, 1:lengthList+1] = linkList
            linkListT = np.array(np.nonzero(self.mapT[i, :]))
            lengthListT = linkListT.shape[1]
            self.linkin[i, 0] = lengthListT
            self.linkin[i, 1:lengthListT+1] = linkListT
        return iH, iA, oH, oA, cH, cA, vH, vA

    def calReward(self, v_act):
        # Reward
        rew = []
        for i in range(self.road_num):
            for j in range(self.cell_num):
                self.traffic[i, j], self.autorate[i, j] = self.graph[i, j].get_state()
                self.cAV[i, j] = self.traffic[i, j] * self.autorate[i, j]
                self.cHV[i, j] = self.traffic[i, j] - self.cAV[i, j]
                rew.append(-1*(self.cAV[i, j]*v_act[i*self.cell_num+j] + self.cHV[i, j]*self.velHV[i, j]))
                if j == self.cell_num - 1:
                    rew.append(-1*(self.cAV[i, j]*v_act[i*self.cell_num+j] + self.cHV[i, j]*self.velHV[i, j]))

        return rew

    def updateCell(self, p_act, p_HV):
        for i in range(np.shape(self.graph)[0]):
            for j in range(np.shape(self.graph)[1]):
                if j == 0:
                    if i in self.source:
                        source_idx = np.where(self.source == i)
                        self.graph[i, j].updateState(self.sourceAV[source_idx], self.sourceHV[source_idx])
                        self.inAV[i, j] = self.sourceAV[source_idx]
                        self.inHV[i, j] = self.sourceHV[source_idx]
                    else:
                        if self.linkin[i, 0] == 1:
                            self.graph[i, j].updateState(self.outAV[self.linkin[i, 1], self.cell_num-1], self.outHV[self.linkin[i, 1], self.cell_num-1])
                            self.inAV[i, j] = self.outAV[self.linkin[i, 1], self.cell_num-1]
                            self.inHV[i, j] = self.outHV[self.linkin[i, 1], self.cell_num-1]
                        elif self.linkin[i, 0] > 1:
                            # Need to check
                            inNum = self.linkin[i, 0]
                            inList = self.linkin[i, 1: inNum]
                            flowAV = 0
                            flowHV = 0
                            for k in range(inNum):
                                road = inList[k]
                                routeN = np.where(self.link[road, :])
                                flowAV += self.outAV[inList[k], self.cell_num-1]*p_act[inList[k], routeN]
                                flowHV += self.outHV[inList[k], self.cell_num-1]*p_HV[inList[k], routeN]
                            self.graph[i, j].updateState(flowAV, flowHV)
                            self.inAV[i, j] = flowAV
                            self.inHV[i, j] = flowHV
                            
                else: 
                    self.graph[i, j].updateState(self.outAV[i, j-1], self.outHV[i, j-1])
                    self.inAV[i, j] = self.outAV[i, j-1]
                    self.inHV[i, j] = self.outHV[i, j-1]

class Cell:
    def __init__(self, x, y, t = 10, type = None, incell = None, outcell = None):
        self.x = x
        self.y = y
        self.t = t
        self.capacity = 20
        self.maxspeed = 30
        self.length = 400
        self.traffic = 400
        self.autorate = 0.5
        self.velocity_space = 16
        self.outAV = 0
        self.outHV = 0
        if type == "intersection":
            self.routing_space = 4
        else:
            self.routing_space = 1
        
        self.velocity_AV = 10
        self.velocity_HV = 9
        self.rp_AV = 0.25 * np.ones((4, 1))
        self.rp_HV = 0.25 * np.ones((4, 1))

    def outflowAV(self):
        # Generate the number of AVs going out
        # print(type(self.t))
        self.outAV = self.traffic*self.autorate*self.velocity_AV*self.t/self.length
        return self.outAV

    def outflowHV(self):
        # Generate the number of HVs going out
        self.outHV = self.traffic*(1-self.autorate)*self.velocity_HV*self.t/self.length
        return self.outHV

    def speedHV(self):
        # Determine HV speed according to AV rate and AV speed
        uni_low = self.velocity_AV - (1-self.autorate)*0.25*self.maxspeed
        uni_high = self.velocity_AV + (1-self.autorate)*0.25*self.maxspeed
        if uni_low <= 1:
            uni_low = 1
        if uni_high >= self.maxspeed:
            uni_high = self.maxspeed
        if uni_high <= uni_low:
            uni_low = uni_high
        self.velocity_HV = np.random.uniform(uni_low, uni_high)

        return self.velocity_HV

    def routeHV(self):
        None

    def get_state(self):
        return self.traffic, self.autorate

    def updateState(self, inAV, inHV):
        oldAV = self.traffic*self.autorate
        oldHV = self.traffic - oldAV
        outAV = self.outflowAV
        outHV = self.outflowHV
        newAV = oldAV + inAV - outAV
        newHV = oldHV + inHV - outHV
        self.traffic = newAV + newHV
        self.autorate = newAV/self.traffic
        return None


def genGraph(cell_num, road_num, times):
    Graph = np.array( [ [Cell(i, j, times) for i in range(cell_num)] for j in range(road_num) ], dtype=object)
    return Graph

def routingHV(Map):
    # Probability proportion to the remaining delay

    return None

import matplotlib.pyplot as plt
import numpy as np

def genAdj(value):
    if value >= 0:
        Adj = np.zeros((10, 10))
        Adj[0, 1] = 1
        Adj[0, 2] = 1
        Adj[1, 3] = 1
        Adj[1, 5] = 1
        Adj[2, 4] = 1
        Adj[2, 6] = 1
        Adj[3, 6] = 1
        Adj[4, 5] = 1
        Adj[5, 9] = 1
        Adj[6, 9] = 1
        Adj[7, 3] = 1
        Adj[7, 5] = 1
        Adj[8, 4] = 1
        Adj[8, 6] = 1
    else:
        Adj = 0

    return Adj

gamma = 0.95
lr_a = 0.0001
lr_c = 0.005
target_update_steps = 10
road_num = 9
cell_num = 3
agent_num = road_num * (cell_num + 1)
time_interval = 10
action_level = 16
Adj = genAdj(road_num+1)
source = np.array((0, 7, 8))
roadmap = environment(Adj, source, cell_num, time_interval)
obs = []
# print(Adj.shape[0])
COMA = COMA(Adj, source, cell_num, action_level, lr_c, lr_a, gamma, target_update_steps)
max_epi_iter = 1000
max_MC_iter = 200
train_curve = []
# print(np.shape(roadmap.graph))

episode_reward = 0
episodes_reward = []

n_episodes = 1000
episode = 0
inHV, inAV, outHV, outAV, cHV, cAV, velHV, velAV = roadmap.get_obs()

# print(Adj)

# !pip install ipdb

while episode < n_episodes:
    actions = COMA.get_actions(inHV, inAV, outHV, outAV, cHV, cAV, velHV, velAV)
    inHV, inAV, outHV, outAV, cHV, cAV, velHV, velAV, reward, done_n = roadmap.step(actions)

    COMA.memory.reward.append(reward)
    for i in range(agent_num):
        COMA.memory.done.append(done_n)

    episode_reward += sum(reward)

    # obs = next_obs

    if done_n:
        episodes_reward.append(episode_reward)
        episode_reward = 0

        episode += 1

        obs = roadmap.reset()

        if episode % 10 == 0:
            COMA.train()

        if episode % 10 == 0:
            print(f"episode: {episode}, average reward: {sum(episodes_reward[-100:]) / 100}")

def moving_average(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')

plt.plot(moving_average(episodes_reward, 100))
plt.title('Learning curve')
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()






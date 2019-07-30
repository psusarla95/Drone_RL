import gym
import gym_uav
import numpy as np
from Source.miscfun.geometry import *
from Source.dqn_agent_old import DQN
import torch
import torch.optim as optim
import torch.nn as nn

def Generate_BeamDir(N):
    min_ang = 0#-math.pi/2
    max_ang = np.pi#math.pi/2
    step_size = (max_ang-min_ang)/N

    BeamSet = []#np.zeros(N)#np.fft.fft(np.eye(N))

    #B_i = (i)pi/(N-1), forall 0 <= i <= N-1; 0< beta < pi/(N-1)
    val = min_ang
    for i in range(N):
        BeamSet.append(val + (i+1)*step_size)#(i+1)*(max_ang-min_ang)/(N)

    return np.array(BeamSet) #eval(strBeamSet_list)#np.ndarray.tolist(BeamSet)

beamset = Generate_BeamDir(8)
env = gym.make('uav-v0')

#if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obs = env.reset()
inp_state = torch.tensor(obs, dtype=torch.float32).to(device)
print("init state: {0}".format(inp_state.shape))

#print(torch.cat((inp_state, inp_state, inp_state)))
n_actions = env.act_space.n
n_inputs = inp_state.shape[1]


policy_net = DQN(n_inputs, n_actions).to(device)
print(policy_net(inp_state))
print(policy_net(inp_state).max(1)[0])
print(env.act_space)
#optimizer = optim.RMSprop(policy_net.parameters())

#print(policy_net.state_dict())

#ue = np.array(sph2cart(1.5345, 0, 20.0))
#print(ue)
action = 3
a = torch.tensor([[action]], dtype=torch.long)
print("tensor action, item: {0}".format(a))
next_obs, rwd, done, _ = env.step(action)
print("next state: {0}, rwd: {1}, done: {2}".format(next_obs, rwd, done))




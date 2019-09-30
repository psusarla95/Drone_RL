import gym
import gym_uav
import torch
import numpy as np


"""
This class is mainly for doing some pre-processing, converting numpy arrays from env to tensors and viceversa

"""

class EnvManager():
    def __init__(self, device, env_name, seed):
        self.device = device
        self.env = gym.make(env_name)
        self.env.seed(seed)
        self.done = False

    def reset(self, rate_thr, measure):
        self.state = self.env.reset(rate_thr, measure)
        return torch.tensor(self.state, device=self.device, dtype=torch.float32).unsqueeze(0)

    def close(self):
        self.env.close()

    def step(self, action_tensor):

        action = action_tensor.item()
        next_state, reward, done, _ = self.env.step(action)

        next_state_tensor = torch.tensor(next_state, device=self.device, dtype=torch.float32).unsqueeze(0)
        reward_tensor = torch.tensor([reward], device=self.device)
        done_tensor = torch.tensor([done], device=self.device)

        return next_state_tensor, reward_tensor, done_tensor, _

    def num_actions_available(self):
        return self.env.act_space.n

    def check_boundaries(self):
        flag = False

        ue_x = self.env.state[0]
        ue_y = self.env.state[1]
        ue_xdest = self.env.ue_xdest[0]
        ue_ydest = self.env.ue_ydest[0]
        ue_xmin = np.min(self.env.ue_xloc)
        ue_ymin = np.min(self.env.ue_yloc)

        if (ue_x > ue_xdest) or (ue_x < ue_xmin):
            flag = True
        elif (ue_y > ue_ydest) or (ue_y < ue_ymin):
            flag = True
        else:
            flag = False

        return flag
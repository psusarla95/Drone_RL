import gym
import gym_uav
import torch



"""
This class is mainly for doing some pre-processing, converting numpy arrays from env to tensors and viceversa

"""

class EnvManager():
    def __init__(self, device, env_name, seed):
        self.device = device
        self.env = gym.make(env_name)
        self.env.seed(seed)
        self.done = False

    def reset(self):
        self.state = self.env.reset()
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



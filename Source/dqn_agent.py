import numpy as np
import random
import math
from collections import namedtuple, deque

from Source.NN_model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)      #replay buffer size
BATCH_SIZE = 60             #minibatch size
GAMMA = 0.99                #discount factor
TAU = 1e-3                  #for soft update of target parameters
LR = 5e-3                   #learning rate
UPDATE_EVERY = 10            #how often to update the network



class Agent():
    "Interacts with and learns from the environment"

    def __init__(self, strategy, action_size, seed, device):
        """
        Initialize and agent object

        :param state_size (float): dimension of each state
        :param action_size (int): dimension of each action
        :param seed (int): random seed
        """
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.strategy = strategy
        self.current_step = 0
        self.device = device

        #Q network
        self.qnetwork_local = QNetwork(action_size, seed).to(self.device)
        self.qnetwork_target = QNetwork(action_size, seed).to(self.device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.qnetwork_target.eval()

        # set biases to all zeros
        #self.qnetwork_local.model.fc1.bias.data.fill_(0)

        # set initial random weights from std normal distribution
        #policy_net.model.fc1.weight.data.normal_(std=0.01)
        #print(policy_net.model)

        #Replay Memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, self.device)

        #Initialize tstep (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        #Save the experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        #Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step +1) % UPDATE_EVERY

        if self.t_step == 0:
            #If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                return self.learn(experiences, GAMMA)
            return None
        return None

    def act(self, state):
        """
        Returns actions for given states based on current policy
        :param state (array_like): current_state
        :param eps (float): epsilon, for epsilon-greedy action selection
        :return: Action
        """
        eps = self.strategy.get_exploration_rate(self.current_step)
        self.current_step +=1

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        if eps > random.random():
            return random.randrange(self.action_size) #explore
        else:
            self.qnetwork_local.eval()
            with torch.no_grad():
                action = self.qnetwork_local(state).argmax(dim=1).item()  #exploit
                self.qnetwork_local.train()
                return action

        #print("[Agent] state: {}".format(state))
        #self.qnetwork_local.eval()
        #with torch.no_grad():
        #    action_values = self.qnetwork_local(state)
        #self.qnetwork_local.train()

        #epsilon-greedy action selection
        #if random.random() > eps:
        #    return np.argmax(action_values.cpu().data.numpy()), np.max(action_values.cpu().data.numpy())
        #else:
        #    action = random.choice(np.arange(self.action_size))
            #print("[Agent] random action: ", action)
        #    action_val = action_values.cpu().data.numpy()[0][action]
        #    return action, action_val

    def learn(self, experiences, gamma):
        """
        Perform Backpropoagation and Update value parameters using given batch of experience tuples
        :param experiences (Tuple(torch.tensor)): batch of (s,a,r,s', done) tuples
        :param gamma (float): discount factor
        :return: None
        """
        states, actions, rewards, next_states, dones = experiences
        #print("[Agent] States: {}".format(states))
        self.qnetwork_local.train()

        #Computing max predicted Q values (for next states) from target network model
        Q_targets_next =self.qnetwork_target.forward(next_states).detach().max(1)[0].unsqueeze(1)
        #print("[Agent] Q_targets_next.shape: {}".format(Q_targets_next.shape))
        #Compute best Q targets for the current policy
        Q_targets = rewards + (gamma*Q_targets_next*(1-dones))

        #Compute Q values for current states, actions pairs
        Q_expected = self.qnetwork_local.forward(states).gather(1, actions)
        #print("[Agent] Q_expected.shape: {}".format(Q_expected.shape))
        #compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #------ Update Target Network-------#
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)
        return loss

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for local_param, target_param in zip(local_model.parameters(), target_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples

    Note:
        This finite buffer size adds noise to the outputs on function approx.
        larger the buffer size, more is the experience and less the number of episodes needed for training
    """

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """
        Initialize a ReplayBuffer object
        :param action_size (int): dimension of each action
        :param buffer_size (int): maximum size of the buffer
        :param batch_size (int): size of each training batch
        :param seed (int): random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience =namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed =random.seed(seed)
        self.device = device

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experience from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        "Return the current size of replay memory"
        return len(self.memory)



class EpsilonGreedyStrategy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end)*math.exp(-1. * current_step* self.decay)
        #return max(self.end, (self.decay)**current_step)
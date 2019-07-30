import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """ Actor (Policy) Model- Value based method"""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        hidden_layers = [32,32]
        drop_p =0.5

        #add the first hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])

        #add a variable number of hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.hidden_layers.apply(self._init_weights)

        #create an output layer
        self.output =nn.Linear(hidden_layers[-1], action_size)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        #self.dropout = nn.Dropout(drop_p)

    def forward(self, state):
        """
        Build a network that maps state->action values
        :param state: input observation to the network
        :return: probability action outputs from the network architecture
        """

        #forward through each layer in"hidden layer",with ReLU activation unit between them
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
            #state = self.dropout(state)
        state = self.output(state)
        return state

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
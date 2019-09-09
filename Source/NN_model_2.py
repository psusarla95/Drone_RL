import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """ Actor (Policy) Model- Value based method"""

    def __init__(self, action_size, seed):
        """Initialize parameters and build model
        Params
        ======
            state_size (int): Dimension of each state [(1,6) in UAV-v2 environment]
            action_size (int): Dimension formed by overall actions
            seed (int): Random seed
        """
        super(QNetwork,self).__init__()
        self.seed = torch.manual_seed(seed)

        #self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3,stride=1, padding=1)
        #nn.init.xavier_uniform_(self.conv1.weight)
        #nn.init.zeros_(self.conv1.bias)
        #self.bn1 = nn.BatchNorm1d(num_features=8)

        self.fc1 = nn.Linear(in_features=2,out_features=128)
        #nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(in_features=128, out_features=128)
        #nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)

        self.output = nn.Linear(in_features=128, out_features=action_size)
        #nn.init.xavier_uniform_(self.output.weight)
        nn.init.kaiming_normal_(self.output.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.output.bias)

    def forward(self, inp_tensor):
        """
        Feed forward function of designed neural network
        :param inp_tensor: Size([BATCH_SIZE, 1,6])
        :return: probability action outputs from the network architecture
        """

        #(1) input layer
        t = inp_tensor

        #(2) hidden conv layer
        #t = self.conv1(t)
        #t = self.bn1(t)
        #t = F.relu(t)

        #t= t.reshape(-1, 8*7)
        #(3) hidden linear layer
        t = self.fc1(t)
        t = F.relu(t)

        #(4) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        out = self.output(t)
        return out






import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()

        #self.pipe = nn.Sequential(
            #nn.Conv1d(in_channels=n_inputs, out_channels=5, kernel_size=2, stride=1, padding=1),
            #nn.BatchNorm1d(5),

        #    nn.Linear(n_inputs, n_outputs),
        #    nn.ReLU(),
        #    nn.Linear(n_outputs, n_outputs),
        #    nn.ReLU(),
        #    nn.Linear(n_outputs, n_outputs),
        #    nn.ReLU()
            #nn.Softmax(dim=0)
            #nn.Conv1d(in_channels=5, out_channels=5, kernel_size=2, stride=1, padding=1),
            #nn.BatchNorm1d(5),
            #nn.ReLU(),
            #nn.Conv1d(in_channels=5, out_channels=n_outputs, kernel_size=2, stride=1, padding=1),
            #nn.Sigmoid()
        #)

        self.model = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(input_size, hidden_size[0])),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(hidden_size[0], hidden_size[0])),
            ('relu2', nn.ReLU()),
            ('logits', nn.Linear(hidden_size[0], output_size)),
            # ('softmax', nn.Softmax(dim=1))
            #('relu3', nn.ReLU())
        ]))

    #Called with either one element to determin next action or a batch during
    #optimization. Returns tensor([[left0exp, right0exp]...])
    def forward(self, x):
        output = self.model(x)
        #print(output)
        return output#.view(output.size(0), -1)#.view(-1, 1).squeeze(dim=1)



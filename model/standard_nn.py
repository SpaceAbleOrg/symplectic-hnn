# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

# Originally written for the project and by:
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
from utils import choose_nonlinearity


class MLP(torch.nn.Module):
    """ Just a salt-of-the-earth multi-layer perceptron neural network. """

    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim, bias=False)

        for l in [self.linear1, self.linear2, self.linear3]:
            torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = choose_nonlinearity(nonlinearity)

    def forward(self, x):
        h = self.nonlinearity(self.linear1(x))
        h = self.nonlinearity(self.linear2(h))
        return self.linear3(h)

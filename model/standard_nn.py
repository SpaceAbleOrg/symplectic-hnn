# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

# Originally written for the project and by:
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch
from utils import choose_nonlinearity


class MLP(torch.nn.Module):
    """ Just a salt-of-the-earth multi-layer perceptron neural network. """

    def __init__(self, hidden_layers, input_dim, hidden_dim, output_dim, nonlinearity='tanh'):
        super(MLP, self).__init__()
        # As a subclass of nn.Module, all class attributes that subclass Parameter will automatically
        # be added to the parameters_list of any instance of this class. This is necessary to easily access
        # these parameters with the optimizer (which is initialized simply with model.parameters()).

        self.hidden_layers = hidden_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.nonlinearity = choose_nonlinearity(nonlinearity)

        # OLD HARDCODED
        #self.layer0 = torch.nn.Linear(input_dim, hidden_dim)
        #self.layer1 = torch.nn.Linear(hidden_dim, hidden_dim)
        #self.layer2 = torch.nn.Linear(hidden_dim, output_dim, bias=False)
        # for l in [self.layer0, self.layer1, self.layer2]:
        #    torch.nn.init.orthogonal_(l.weight)  # use a principled initialization

        # Set and get attributes with a variable name, to properly register them as parameters (see comment above)
        self.__setattr__('layer0', torch.nn.Linear(input_dim, hidden_dim))  # INPUT LAYER
        for i in range(self.hidden_layers):
            self.__setattr__(f'layer{i+1}', torch.nn.Linear(hidden_dim, hidden_dim))  # HIDDEN LAYER

        # Remove bias as the Hamiltonian is only determined up to a constant
        N = self.hidden_layers + 1
        self.__setattr__(f'layer{N}', torch.nn.Linear(hidden_dim, output_dim, bias=False))  # OUTPUT LAYER

        # INITIALIZE ALL LAYERS
        for i in range(self.hidden_layers + 2):
            torch.nn.init.orthogonal_(self.__getattr__(f'layer{i}').weight)  # use a principled initialization

    def forward(self, x):
        for i in range(self.hidden_layers + 1):
            x = self.nonlinearity(self.__getattr__(f'layer{i}')(x))
        N = self.hidden_layers + 1
        x = self.__getattr__(f'layer{N}')(x)
        return x

        # x = self.nonlinearity(self.layer0(x))
        # x = self.nonlinearity(self.layer1(x))
        # x = self.layer2(x)
        # return x


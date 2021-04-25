# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

# Originally written for the project and by:
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch

from model.standard_nn import MLP
from utils import symplectic_form, save_path


def get_hnn(args):
    # Create the standard MLP with args.dim inputs and one single scalar output, the Hamiltonian
    nn_model = MLP(hidden_layers=args.hidden_layers, input_dim=args.dim, hidden_dim=args.hidden_dim, output_dim=1,
                   nonlinearity=args.nonlinearity)
    # Use this model to create a Hamiltonian Neural Network, which knows how to differentiate the Hamiltonian
    model = HNN(nn_model)

    return model


def load_model(args):
    saved_dict = torch.load(save_path(args))
    args = saved_dict['args']  # Loads all other arguments as saved initially when the model was trained

    # Create a model using the same args and load its state_dict
    model = get_hnn(args)
    model.load_state_dict(saved_dict['model'])

    return model, args


class HNN(torch.nn.Module):
    """ Learn arbitrary Hamiltonian vector fields that are the symplectic derivative of a scalar function H """
    def __init__(self, differentiable_model):
        super().__init__()
        self.differentiable_model = differentiable_model

        # Symplectic form J in matrix form in canonical coords
        self.J = symplectic_form(differentiable_model.input_dim)

    def forward(self, x):
        return self.differentiable_model(x)

    def time_derivative(self, x):
        """ Calculates the Hamiltonian vector field from the predicted Hamiltonian (self.forward) """
        H = self.forward(x)  # traditional forward pass, predicts Hamiltonian

        gradH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]  # gradient using autograd

        # Do batch matrix-vector multiplication, since gradH contains batch_size many 2n-vectors
        # (Recall that J is antisymmetric and orthogonal: J^T = -J = J^(-1).)
        vector_field = torch.einsum('ij,aj->ai', self.J.T, gradH)
        # An alternative string for this batch operation would be 'ij,...j->...i' to not specify the extra dims.

        return vector_field

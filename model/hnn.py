# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

# Originally written for the project and by:
# Hamiltonian Neural Networks | 2019
# Sam Greydanus, Misko Dzamba, Jason Yosinski

import torch

from utils import symplectic_form


class HNN(torch.nn.Module):
    """ Learn arbitrary Hamiltonian vector fields that are the symplectic derivative of a scalar function H """
    def __init__(self, differentiable_model):
        super(HNN, self).__init__()
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

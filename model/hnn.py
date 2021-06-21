import torch

from model.args import UpdatableNamespace
from model.standard_nn import MLP
from model.data import symplectic_form
from utils import save_path


class HNN(torch.nn.Module):
    """ This class allows to learn arbitrary Hamiltonian vector fields that are the symplectic derivative of a
        scalar function H. It is built as a wrapper around any other `torch.nn.Module` which has to be passed
        to this class' constructor as `differentiable_model`. It notably defines the new `derivative` method which
        calculates the gradient of the model's `output.sum()` multiplied on the left by the symplectic form J^{-1}. """

    @staticmethod
    def create(args):
        """ This helper method allows to create the typical HNN where the base `differentiable_model`
            is a standard MLP and has output dimension 1 only (i.e. directly tries to predict H and nothing more).
            The created model will be randomly initialized. """
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')

        # Create the standard MLP with args.dim inputs and one single scalar output, the Hamiltonian
        nn_model = MLP(hidden_layers=args.hidden_layers, input_dim=args.dim, hidden_dim=args.hidden_dim, output_dim=1,
                       nonlinearity=args.nonlinearity)
        # Use this model to create a Hamiltonian Neural Network, which knows how to differentiate the Hamiltonian
        model = HNN(nn_model)

        return model.to('cuda') if torch.cuda.is_available() else model

    @staticmethod
    def load(args, cpu=False):
        """ This helper method allows to load the `state_dict` of an already trained model, re-instantiating the
            HNN object and returning it together with exact set of `args` that had been used to training. """
        if cpu:
            saved_dict = torch.load(save_path(args), map_location=torch.device('cpu'))
        else:
            saved_dict = torch.load(save_path(args))

        args = UpdatableNamespace.get(saved_dict['args'])  # Loads all other arguments as saved initially during training

        # Create a model using the same args and load its state_dict
        model = HNN.create(args)
        model.load_state_dict(saved_dict['model'])

        return model, args

    def __init__(self, differentiable_model):
        super().__init__()
        self.differentiable_model = differentiable_model

        # Symplectic form J in matrix form in canonical coords
        self.J = symplectic_form(differentiable_model.input_dim)

    def forward(self, x):
        # squeeze the last dimension in [batch_size, 1] since H is a scalar
        return self.differentiable_model(x).squeeze(-1)

    def derivative(self, x):
        """ Calculates the Hamiltonian vector field from the predicted Hamiltonian (self.forward) as the symplectic
            gradient, that is the matrix-vector product J^{-1} · \grad H."""
        H = self.forward(x)  # traditional forward pass, predicts Hamiltonian

        gradH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]  # gradient using autograd

        # Do batch matrix-vector multiplication, since gradH contains batch_size many 2n-vectors
        # (Recall that J is antisymmetric and orthogonal: J^T = -J = J^(-1).)
        vector_field = torch.einsum('ij,aj->ai', self.J.T, gradH)
        # An alternative string for this batch operation would be 'ij,...j->...i' to not specify the extra dims.

        return vector_field


class CorrectedHNN(HNN):
    """ This subclass of HNN is compatible with the `SymplecticOneStepScheme.corrected` method (if defined) to
        obtain the Hamiltonian as given by the modified equation, upto the order that the
        `SymplecticOneStepScheme.corrected` is implemented for.

        This class can notably be instantiated with a trained model to make a post-training correction to the learned
        modified Hamiltonian. But, alternatively, one may also train directly with the modified equation (to be tested).
    """

    def __init__(self, differentiable_model, scheme, h):
        super().__init__(differentiable_model)
        self.scheme = scheme
        self.h = h

    @staticmethod
    def get(hnn, scheme, h):
        return CorrectedHNN(hnn.differentiable_model, scheme, h)

    def forward(self, x):
        hamiltonian = super().forward(x)
        hamiltonian_corrected = self.scheme.corrected(hamiltonian, x, self.h)
        return hamiltonian_corrected

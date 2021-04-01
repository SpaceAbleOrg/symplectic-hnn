# Symplectic Hamiltonian Neural Networs | 2021
# Marco David

import torch
import numpy as np
from abc import ABC, abstractmethod

from utils import L2_loss


class Loss(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def scheme(self, x, t):
        pass

    def loss(self, model, x, t, return_dist=False):
        """
            Generic template to implement different loss functions. Standardized format for the parameters as follows.

        Parameters
        ----------
        model: An HNN model which defines both the 'forward' and 'time_derivative' methods.
        x: The input data in the shape [nb of samples; nb of data points per sample; dim of coords]
        t: The times associated to each data point, in the shape [nb of samples; nb of data points per sample; 1].
        return_dist: Optional. Default=False. Boolean value specifying whether the loss distribution over all data
                        points is returned, or simply its mean.

        Returns
        -------
        A torch loss object which can be either read out or used to optimize the model with any optimizer.
        """
        # Note: The loss can be written in two ways which are equivalent up to multiplication by the time step h.
        # Here, it was chosen to calculate the finite differences of real data and compare with the NN predictions
        # directly – instead of integrating the NN predictions to obtain alternate data points (\tilde y_{n+1}).
        # For implicit methods this has the distinct advantage of not having to solve anything by fixed point iteration.

        # STEP 1 – CALCULATE FINITE DIFFERENCES OF TRAINING DATA
        h = torch.tensor(np.diff(t, axis=1)).float()  # t[:, 1:] - t[:, :-1]

        # Calculate finite differences as approximation of the derivatives
        # -> The division of two tensors with non-equal nb of axes requires the extra dimension to be the first one,
        # -> not the last like one would intuitively expect. Hence we need to permute axes forth and back.
        diff = (x[:, 1:] - x[:, :-1]).permute(2, 0, 1)
        dxdt_hat = (diff / h).permute(1, 2, 0)

        # STEP 2 – APPLY IMPLEMENTED SCHEME TO THE HAMILTONIAN NEURAL NETWORK PREDICTIONS
        dxdt = self.scheme(model, x)

        return L2_loss(dxdt, dxdt_hat, mean=not return_dist)

    def __call__(self, *args, **kwargs):
        self.loss(self, *args, **kwargs)


class OneStepLoss(ABC, Loss):
    @abstractmethod
    def argument(self, x):
        pass

    def scheme(self, model, x):
        x = self.argument(x)

        # Flatten into the HNN shape [X, dim of coords] to call the neural network in vectorized form.
        x_flat = x.flatten(0, 1)
        dxdt_flat = model.time_derivative(x_flat)

        # Unflatten to compare with data predictions.
        dxdt = dxdt_flat.view(x.shape)

        return dxdt


class EulerSympLoss(OneStepLoss):
    def argument(self, x):
        P, Q = torch.split(x, self.args.dim//2)
        return torch.stack((P[:, 1:], Q[:, :-1]), dim=-1)


class MidpointLoss(OneStepLoss):
    def argument(self, x):
        return (x[:, 1:] + x[:, :-1]) / 2

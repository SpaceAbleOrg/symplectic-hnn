# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

import warnings

import torch
import numpy as np
from abc import ABC, abstractmethod

from utils import choose_helper


def L2_loss(u, v, mean=True):
    return (u - v).pow(2).mean() if mean else (u - v).pow(2)


class OneStepScheme(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def argument(self, yn, ynplusone):
        pass


class ForwardEuler(OneStepScheme):
    def argument(self, yn, ynplusone):
        return yn


class SymplecticOneStepScheme(OneStepScheme, ABC):
    @abstractmethod
    def corrected(self, hamiltonian, x, h, order):
        pass


class SymplecticEuler(SymplecticOneStepScheme):
    def argument(self, yn, ynplusone):
        pn, qn = torch.split(yn, self.args.dim // 2, dim=-1)
        pnplusone, qnplusone = torch.split(ynplusone, self.args.dim // 2, dim=-1)

        return torch.cat((pnplusone, qn), dim=-1)

    def corrected(self, hamiltonian, x, h, order=1):
        MAX = 2
        if order > MAX:
            raise NotImplementedError(f"Higher order corrections (> {MAX}) are not (yet) implemented.")

        # The Hamiltonian is a scalar, so it has shape [batch_size, 1] – the next line removes the superfluous dimension
        hamiltonian = hamiltonian.squeeze(-1)

        # The sum is over the remaining 'batch' dimension which allows vectorization of the network
        dH = torch.autograd.grad(hamiltonian.sum(), x, create_graph=True)[0]
        dH_p, dH_q = torch.split(dH, self.args.dim // 2, dim=-1)

        # Calculate the first-order correction to the Hamiltonian for the SympEuler scheme
        # The einsum realizes a batch dot product between dH_p and dH_q
        correction = hamiltonian - h/2 * torch.einsum('ai,ai->a', dH_p, dH_q)

        # if order > 1:
        #    print(dH.shape, dH)
        #    print(x.shape, x)
        #    ddH = torch.autograd.grad(dH, x, create_graph=True, grad_outputs=torch.ones_like(dH))[0]
        #    print(ddH.shape)
        #    print(ddH[0, 0])
        #    raise RuntimeError("STOP. Not yet implemented.")

        #    t1 = t2 = t3 = 0
        #    correction = correction - h**2 / 6 * (t1 + t2 + t3)

        return correction


class ImplicitMidpoint(SymplecticOneStepScheme):
    def argument(self, yn, ynplusone):
        return (yn + ynplusone) / 2

    def corrected(self, hamiltonian, x, h, order=1):
        if order > 1:
            raise NotImplementedError("Higher order corrections are not (yet) implemented.")

        warnings.warn("The midpoint rule has no first order correction. Calling its `corrected` method is not yet"
                      "supported and simply is the identity function for now.", RuntimeWarning)
        return hamiltonian


def choose_scheme(name):
    schemes = {'euler-forw': ForwardEuler,
               'euler-symp': SymplecticEuler,
               'midpoint': ImplicitMidpoint
               }

    return choose_helper(schemes, name, choose_what="Integration scheme")


class Loss(ABC):
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def prediction(self, x, t):
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
        dxdt = self.prediction(model, x)

        return L2_loss(dxdt, dxdt_hat, mean=not return_dist)

    # Allow to call the loss object directly for evaluation
    __call__ = loss


class OneStepLoss(Loss):
    def __init__(self, args):
        super().__init__(args)
        self.scheme = choose_scheme(self.args.loss_type)(self.args)

    def prediction(self, model, x):
        xplusone = x[:, 1:]
        xn = x[:, :-1]

        x = self.scheme.argument(xn, xplusone)

        # Flatten into the HNN shape [X, dim of coords] to call the neural network in vectorized form.
        x_flat = x.flatten(0, 1)
        dxdt_flat = model.time_derivative(x_flat)

        # Unflatten to compare with data predictions.
        dxdt = dxdt_flat.view(x.shape)

        return dxdt

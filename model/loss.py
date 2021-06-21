import warnings

import torch
import numpy as np
from abc import ABC, abstractmethod

from utils import choose_helper


def L2_loss(u, v, mean=True):
    """ Calculates the squared (!) L2 loss between two elements, returning either the full error distribution
        or simply the mean, according to the kwarg `mean`. """
    return (u - v).pow(2).mean() if mean else (u - v).pow(2)


class OneStepScheme(ABC):
    """ This is an abstract base class for a certain class of one-step integration scheme, explicitly depending only
        on y_0 or implicitly depending both on y_0 and y_1. The schemes can only define an argument
        s(y_0, y_1) which then yields the full scheme as y_1 = y_0 + h f(s). (The data points y_1 are given, too,
        because we are solving an inverse problem: Learn the equation from observation data.)

        Its subclasses will be used during training of HNNs, notably in the loss functions defined by subclasses of
        the `Loss` abstract base class.. """
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def argument(self, y0, y1):
        pass


class ForwardEuler(OneStepScheme):
    """ Implements the forward Euler scheme, by simply evaluating the vector field at y_n. """
    def argument(self, y0, y1):
        return y0


class SymplecticOneStepScheme(OneStepScheme, ABC):
    """ Provides an abstract base class for one-step schemes that are additionally symplectic schemes, since there
        exists a Hamiltonian modified equation for those schemes. Requires those schemes to also implement this
        modified equation through the `corrected` method to correct the modified Hamiltonian that is learned
        during training of an SHNN with such a scheme in the loss function. """
    @abstractmethod
    def corrected(self, hamiltonian, x, h, order):
        pass


class SymplecticEuler(SymplecticOneStepScheme):
    """ Implements the symplectic Euler scheme by splitting the variables y = (p, q) into two vectors of half
        the length and matching them accordingly to yield s(y_0, y_1). """
    def argument(self, y0, y1):
        p0, q0 = torch.split(y0, self.args.dim // 2, dim=-1)
        p1, q1 = torch.split(y1, self.args.dim // 2, dim=-1)

        return torch.cat((p1, q0), dim=-1)

    def corrected(self, hamiltonian, x, h, order=1):
        MAX = 2
        if order > MAX:
            raise NotImplementedError(f"Higher order corrections (> {MAX}) are not (yet) implemented.")

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
    """ Implements the symplectic implicit midpoint rule. """
    def argument(self, y0, y1):
        return (y0 + y1) / 2

    def corrected(self, hamiltonian, x, h, order=1):
        MAX = 1
        if order > MAX:
            raise NotImplementedError(f"Higher order corrections (> {MAX}) are not (yet) implemented.")

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
    """ Abstract base class which allows to implement different loss functions, which are accesible via the
        method `loss`. To subclass, implement the `prediction` method, to which the real data will be compared
        in the already implemented function `loss`. """
    def __init__(self, args):
        self.args = args

    @abstractmethod
    def prediction(self, model, x):
        """
            This method needs to be implemented to define a loss function. It should calculate the predicted
            symplectic gradient J^{-1} · \grad \tilde H of the given model at the given point x.

        Parameters
        ----------
        model: The HNN (or other model with a `derivative`  method) to be evaluated.
        x: The batch of data points (y_0, y_1) of position vectors, specifying where to evaluate the model.

        Returns
        -------
        The prediction of the symplectic gradient of the model at the given point(s).
        """
        pass

    def loss(self, model, x, t, return_dist=False):
        """
            Generic template to implement different loss functions. Standardized format for the parameters as follows.

        Parameters
        ----------
        model: An HNN model which defines both the 'forward' and 'derivative' methods.
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
        # For implicit methods both have the advantage of not having to solve anything by fixed point iteration.

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
    """ This `Loss` subclass implements the `prediction` method generically for any `OneStepScheme` instance by
        simply evaluating the given model at the argument specified by the scheme as s(y_0, y_1). """
    def __init__(self, args):
        super().__init__(args)
        self.scheme = choose_scheme(self.args.loss_type)(self.args)

    def prediction(self, model, x):
        xplusone = x[:, 1:]
        xn = x[:, :-1]

        x = self.scheme.argument(xn, xplusone)

        # Flatten into the HNN shape [X, dim of coords] to call the neural network in vectorized form.
        x_flat = x.flatten(0, 1)
        dxdt_flat = model.derivative(x_flat)

        # Unflatten to compare with data predictions.
        dxdt = dxdt_flat.view(x.shape)

        return dxdt

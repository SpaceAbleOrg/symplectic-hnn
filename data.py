# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

import torch
import numpy as np

import autograd
from abc import ABC, abstractmethod
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split


def symplectic_form(n, canonical_coords=True):
    if canonical_coords:
        Id = torch.eye(n)
        J = torch.cat([Id[n // 2:], -Id[:n // 2]])
    else:
        '''Constructs the Levi-Civita permutation tensor'''
        J = torch.ones(n, n)  # matrix of ones
        J *= 1 - torch.eye(n)  # clear diagonals
        J[::2] *= -1  # pattern of signs
        J[:, ::2] *= -1

        for i in range(n):  # make asymmetric
            for j in range(i + 1, n):
                J[i, j] *= -1
    return J


class HamiltonianDataSet(ABC):
    def __init__(self, h, noise):
        self.h = h
        self.noise = noise

    @staticmethod
    @abstractmethod
    def dimension():
        pass

    @abstractmethod
    def hamiltonian(self, p, q, t=None):
        pass

    def dynamics_fn(self, t, coords):
        gradH = autograd.grad(self.hamiltonian)(*np.split(coords, 2), t=t)
        J = symplectic_form(gradH.shape[0])
        return J.T @ gradH

    @abstractmethod
    def get_initial_value(self):
        pass

    def get_trajectory(self, t_span=(0, 3), rtol=1e-6, **kwargs):
        t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) / self.h))

        y0 = self.get_initial_value()
        ivp_solution = solve_ivp(fun=self.dynamics_fn, t_span=t_span, y0=y0, t_eval=t_eval, rtol=rtol, **kwargs)

        y = ivp_solution['y']
        y += np.random.randn(*y.shape) * self.noise

        return y, t_eval

    def get_dataset(self, seed=0, samples=50, test_split=0.2, **kwargs):
        data = {'meta': locals()}
        np.random.seed(seed)

        # Randomly sample inputs
        ys, ts = [], []
        for _ in range(samples):
            y, t = self.get_trajectory(**kwargs)
            ys.append(y.T)
            ts.append(t)

        data['coords'] = np.array(ys).squeeze()
        # Also add t to the data (although all rows will usually be the same)
        data['t'] = np.array(ts).squeeze()

        # Make a train/test split
        t_train, t_test, y_train, y_test = train_test_split(data['t'], data['coords'], test_size=test_split)
        data.update({'t': t_train, 'test_t': t_test, 'coords': y_train, 'test_coords': y_test})

        return data

    def get_analytic_field(self, xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
        field = {'meta': locals()}

        # Meshgrid to get lattice
        b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
        ys = np.stack([b.flatten(), a.flatten()])

        # Get vector field on lattice
        dydt = [self.dynamics_fn(None, y) for y in ys.T]
        dydt = np.stack(dydt).T

        field['x'] = ys.T
        field['dx'] = dydt.T
        return field


class HarmonicOscillator(HamiltonianDataSet):
    """ Implements the Hamiltonian of an ideal harmonic oscillator in N dimensions, e.g. a spring-mass system. """

    @staticmethod
    def dimension():
        return 2

    def hamiltonian(self, p, q, t=None):
        H = 1/2 * (p ** 2 + q ** 2)
        return H
        # if isinstance(H, np.ndarray):
        #     H = H.sum()

    def get_initial_value(self):
        """ Create a random initial point between (-1, -1) and (1, 1). """
        y0 = 2 * np.random.rand(self.dimension()) - 1
        return y0

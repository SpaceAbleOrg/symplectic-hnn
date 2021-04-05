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
        Id = np.eye(n)
        J = np.concatenate([Id[n // 2:], -Id[:n // 2]])
    else:
        '''Constructs the Levi-Civita permutation tensor'''
        J = np.ones((n, n)) - np.eye(n)  # matrix of ones, without diagonal
        J[::2] *= -1  # pattern of signs
        J[:, ::2] *= -1

        for i in range(n):  # make asymmetric
            for j in range(i + 1, n):
                J[i, j] *= -1
    return torch.tensor(J, dtype=torch.float32)


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

    def bundled_hamiltonian(self, coords, t=None):
        return self.hamiltonian(*np.split(coords, 2), t=t)

    def dynamics_fn(self, t, coords):
        gradH = autograd.grad(self.bundled_hamiltonian)(coords, t=t)
        J = symplectic_form(gradH.shape[0])
        return J.T @ gradH

    @staticmethod
    @abstractmethod
    def random_initial_value():
        pass

    def get_trajectory(self, t_span=(0, 3), rtol=1e-6, **kwargs):
        t_eval = np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) / self.h))

        y0 = self.random_initial_value()
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
    """ Implements the Hamiltonian of an ideal harmonic oscillator in 1 dimension, e.g. a spring-mass system. """

    @staticmethod
    def dimension():
        """ Returns 2 for the full system's dimensionality: one q position coordinate, one p momentum coordinate. """
        return 2

    def hamiltonian(self, p, q, t=None):
        H = 1/2 * (p ** 2 + q ** 2)
        return H

    @staticmethod
    def random_initial_value():
        # Create a random initial point between (-1, -1) and (1, 1).
        y0 = 2 * np.random.rand(HarmonicOscillator.dimension()) - 1
        # Ensure that the norm is at least 0.1
        y0 = y0 / np.sum(y0**2) * (0.1 + np.random.rand())
        return y0


class NonlinearPendulum(HamiltonianDataSet):
    """ Implements the Hamiltonian of an ideal non-linear pendulum in 1 dimension. """

    @staticmethod
    def dimension():
        """ Returns 2 for the full system's dimensionality: one q position coordinate, one p momentum coordinate. """
        return 2

    def hamiltonian(self, p, q, t=None):
        H = 1/2 * p ** 2 + (1 - np.cos(q))
        return H

    @staticmethod
    def random_initial_value():
        """ Start at a random initial point between (-π/2, +π/2) rad, with initial momentum zero. """
        theta = np.pi * (np.random.rand() - 1/2)
        return np.array([0, theta])

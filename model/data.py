# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

import torch
import numpy as np

import autograd
from abc import ABC, abstractmethod
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split

from utils import choose_helper


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


def get_t_eval(t_span, h):
    return np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) / h) + 1)


def choose_data(name):
    datasets = {'spring': HarmonicOscillator,
                'pendulum': NonlinearPendulum,
                'fput': FermiPastaUlamTsingou,  # FPUT = Fermi-Pasta-Ulam-Tsingou (see GNI book)
                'twobody': TwoBody
                }

    return choose_helper(datasets, name, choose_what="Data set name")


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
        """ To implement the Hamiltonian function H(p, q) of the respective system. All functions must be from
            the `autograd.numpy` library to automatically allow generation of training data sets. """
        pass

    def bundled_hamiltonian(self, coords, t=None):
        return self.hamiltonian(*np.split(coords, 2), t=t).squeeze()

    def dynamics_fn(self, t, coords):
        gradH = autograd.grad(self.bundled_hamiltonian)(coords, t=t)
        J = symplectic_form(self.dimension()).cpu().numpy()
        return J.T @ gradH  # does NOT return a torch.Tensor but a numpy.ndarray

    @staticmethod
    @abstractmethod
    def phase_space_boundaries():
        """ Using these values, cls.random_initial_point() creates a random initial point
            from the interval [cmin, cmax]^dim where dim=cls.dimension(). """
        # To implement as: return cmin, cmax
        pass

    @classmethod
    def random_initial_value(cls, draw_omega_m=False):
        """ Return a random initial value which will be used to generate individual trajectories for the dataset.
            The space this is drawn from is denoted $\Omega_d$, i.e. the 'data generation' region of phase space.

            By default this is simply drawn uniformly from the interval provided by `cls.phase_space_boundaries()`
            raised to `cls.dimension()` (i.e. one value from this interval for each dimension of the problem). """
        cmin, cmax = cls.phase_space_boundaries()
        d = cmax - cmin
        m = (cmax + cmin) / 2

        if draw_omega_m:
            # If measuring from \Omega_m, divide by the (2n)th root of 2, so that the region $\Omega_m$ has half
            # the hyper volume of the region \Omega_d
            d /= 2**(1/cls.dimension())

        return (d * np.random.rand(cls.dimension())) + m - d/2

    @staticmethod
    @abstractmethod
    def static_initial_value():
        """ Return a canonical (constant) initial value for plotting an individual trajectory to exemplify
            the flow predicted by the (Symplectic) HNN; obtained by integrating the HNN's gradient vector field. """
        pass

    def get_trajectory(self, t_span=(0, 3), rtol=1e-9, y0=None, **kwargs):
        t_eval = get_t_eval(t_span, self.h)

        if y0 is None:
            y0 = self.random_initial_value()
        ivp_solution = solve_ivp(fun=self.dynamics_fn,
                                 t_span=t_span, y0=y0, t_eval=t_eval, rtol=rtol, method='RK45', **kwargs)

        y = ivp_solution.y.T
        if self.noise > 0:
            y += np.random.randn(*y.shape) * self.noise

        return y, t_eval

    def get_dataset(self, seed=0, samples=1500, test_split=0.2, print_args=None, **kwargs):
        data = {'meta': locals()}
        np.random.seed(seed)

        # Randomly sample inputs
        ys, ts = [], []
        for i in range(samples):
            if print_args and print_args.verbose and i % print_args.print_every == 0:
                print(f"Generating sample {i}...", end='\r')
            y, t = self.get_trajectory(t_span=(0, self.h), **kwargs)
            ys.append(y)
            ts.append(t)

        if print_args.verbose:
            print()  # to not overwrite the previous lines because they ended in \r

        coords = np.array(ys).squeeze()
        # Also add t to the data (although all rows will usually be the same)
        t = np.array(ts).squeeze()

        # Make a train/test split
        t_train, t_test, y_train, y_test = train_test_split(t, coords, test_size=test_split, shuffle=True)
        data.update({'t': t_train, 'test_t': t_test, 'coords': y_train, 'test_coords': y_test})

        return data

    def get_analytic_field(self, gridsize=20):
        field = {'meta': locals()}

        # Meshgrid to get lattice
        cmin, cmax = self.plot_boundaries()
        b, a = np.meshgrid(np.linspace(cmin, cmax, gridsize), np.linspace(cmin, cmax, gridsize))
        ys = np.stack([b.flatten(), a.flatten()])

        # Get vector field on lattice
        dydt = [self.dynamics_fn(None, y) for y in ys.T]
        dydt = np.stack(dydt).T

        field['x'] = ys.T
        field['y'] = dydt.T
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
    def phase_space_boundaries():
        return -2, 2

    @staticmethod
    def static_initial_value():
        return np.array([0., 1.])


class NonlinearPendulum(HamiltonianDataSet):
    """ Implements the Hamiltonian of an ideal non-linear pendulum in 1 dimension. """

    @staticmethod
    def dimension():
        """ Returns 2 for the full system's dimensionality: one q position coordinate, one p momentum coordinate. """
        return 2

    def hamiltonian(self, p, q, t=None):
        H = 1/2 * p ** 2 + (1 - autograd.numpy.cos(q))
        return H

    @staticmethod
    def phase_space_boundaries():
        return -2*np.pi, 2*np.pi

    @staticmethod
    def static_initial_value():
        return np.array([0.2, 2.5])


class TwoBody(HamiltonianDataSet):
    """ Implements the Hamiltonian of the two-body problem in 2 spatial dimensions. """

    @staticmethod
    def dimension():
        """ Returns 8 for the full system's dimensionality: two q position coordinates, two p momentum coordinates. """
        return 8

    def hamiltonian(self, p, q, t=None):
        H = 1/2 * autograd.numpy.sum(p ** 2) + 1 / autograd.numpy.sum((q[0:2] - q[2:4]) ** 2)
        return H

    @staticmethod
    def phase_space_boundaries():
        return -1.5, 1.5

    @classmethod
    def random_initial_value(cls, draw_omega_m=False):
        """ Start at a random initial point and initial momentum, in the interval provided by
            `TwoBody.phase_space_boundaries()` for each dimension of the problem (see `TwoBody.dimension()`).

            However, ensure that the distance of the two bodies squared is not smaller than a certain threshold
            in order to avoid the poles of the Hamiltonian. Should a provided value provided by the superclass
            implementation violate this condition, redraw. """
        threshold = 1e-1

        # The fact that this problem is defined for dimension 8 is hardcoded here.
        y = super().random_initial_value()  # = (p1, p2, q1, q2) which are all respectively of dimension 2

        q1, q2 = y[4:6], y[6:8]
        if ((q1 - q2) ** 2).sum() < threshold:
            return cls.random_initial_value()  # redraw
        else:
            return y

    @staticmethod
    def static_initial_value():
        p1, p2 = np.array([0.8, 0]), np.array([-1.2, 0])
        q1, q2 = np.array([0, 1]), np.array([0, -1])
        return np.concatenate((p1, p2, q1, q2))


class FermiPastaUlamTsingou(HamiltonianDataSet):
    """ Implements the Hamiltonian of the Fermi-Pasta-Ulam-Tsingou problem for m=3 (see GNI Section I.5.1). """

    @staticmethod
    def dimension():
        return 12

    def hamiltonian(self, p, q, t=None, omega=1):
        H = 1/2 * autograd.numpy.sum(p ** 2)\
            + omega**2 / 4 * autograd.numpy.sum((q[1::2] - q[::2]) ** 2)\
            + q[0] ** 4 + q[-1] ** 4 + autograd.numpy.sum((q[2::2] - q[1:-1:2]) ** 4)  # non-linear springs
        return H

    @staticmethod
    def phase_space_boundaries():
        # The "true" RK45 solution has p and q oscillating between -2 and +2,
        # these limits ensure that this hypercube is well covered.
        return -2.1, 2.1

    @staticmethod
    def static_initial_value(**kwargs):
        omega = kwargs['omega']
        sq2 = np.sqrt(2)

        p0 = np.array([0, sq2, 0, 0, 0, 0])
        q0 = np.array([(1 - 1/omega)/sq2, (1 + 1/omega)/sq2, 0, 0, 0, 0])
        return np.concatenate((p0, q0))

import torch
import numpy as np

import autograd
from abc import ABC, abstractmethod
from scipy.integrate import solve_ivp
from sklearn.model_selection import train_test_split

from utils import choose_helper


def symplectic_form(n, canonical_coords=True):
    """ Returns a `torch.Tensor` containing the canonical symplectic n x  n matrix. If `canonical_coords=True` (default)
        this matrix will be in 2x2 block form containing ±Id or 0 as usual of sizes n//2 each, corresponding to an
        arrangement y = (p1, ..., pn, q1, ..., qn). Otherwise, the matrix will be in alternating checkerboard pattern
        like the Levi-Civita permutation tensor, corresponding to y = (p1, q1, ..., pn, qn). """
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
                'double-pendulum': DoublePendulum,
                'fput': FermiPastaUlamTsingou,  # FPUT = Fermi-Pasta-Ulam-Tsingou (see GNI book)
                'twobody': TwoBody
                }

    return choose_helper(datasets, name, choose_what="Data set name")


class HamiltonianDataSet(ABC):
    """ Abstract base class for any data set that can be generated from a Hamiltonian (i.e. a scalar function of
        position and momentum variables). """
    def __init__(self, h, noise):
        self.h = h
        self.noise = noise

    @staticmethod
    @abstractmethod
    def dimension():
        """ Abstract method that requires any specific data set to return its full dimension 2n, i.e. the number
            of different q and p variables that are used in total. """
        pass

    @classmethod
    @abstractmethod
    def hamiltonian(cls, p, q, t=None):
        """ Abstract method that requires any specific data set to implement the Hamiltonian function H(p, q)
            defining the respective physical  system. All functions must be from the `autograd.numpy` library to
            automatically allow generation of training data sets. """
        pass

    @classmethod
    def bundled_hamiltonian(cls, coords, t=None):
        """ Wrapper function for the Hamiltonian, that allows to call H(y) directly, where y = (p, q). """
        return cls.hamiltonian(*np.split(coords, 2), t=t).squeeze()

    def dynamics_fn(self, t, coords):
        """ This method provides the Hamiltonian vector field J^{-1} · \grad H of the implemented Hamiltonian
            through use of the `autograd` library to calculate the gradient automagically.
            Note that this method returns a `numpy.ndarray` instead of a `torch.Tensor`. """
        gradH = autograd.grad(self.bundled_hamiltonian)(coords, t=t)
        J = symplectic_form(self.dimension()).cpu().numpy()
        return J.T @ gradH  # does NOT return a torch.Tensor but a numpy.ndarray

    @staticmethod
    @abstractmethod
    def phase_space_boundaries():
        """ Abstract method that requires any specific data set to specify a tuple of two numbers (a, b) defining the
            subregion \Omega_d in phase space R^2n, where `cls.dimension() == 2n`. Using these values, the method
            `cls.random_initial_point` creates a random initial point from the interval [a, b]^2n. """
        # To implement as: return cmin, cmax
        pass

    @classmethod
    def random_initial_value(cls, draw_omega_m=False):
        """ This methods returns a random initial value from the subregion $\Omega_d$ of phase space (i.e. the
            'data generation' region of phase space). These initial values are for example used to generate individual
            trajectories of the system for the dataset.

            By default this is simply drawn uniformly from the interval provided by `cls.phase_space_boundaries()`
            in each dimension of the problem. """
        cmin, cmax = cls.phase_space_boundaries()
        d = cmax - cmin
        m = (cmax + cmin) / 2

        if draw_omega_m:
            # VERSION 1
            # If measuring from \Omega_m, divide by the (2n)th root of fm (fm=2 by default),
            # so that the region $\Omega_m$ has 1/fm the hyper volume of the region \Omega_d
            # fm = 2
            # d /= fm**(1/cls.dimension())
            # This does not work well for dimensions 8 (twobody problem) or higher, because 2^(1/8) = 1.1

            # VERISON 2
            # Simply divide all by the square root of 2.
            d /= np.sqrt(2)

        return (d * np.random.rand(cls.dimension())) + m - d / 2

    @staticmethod
    @abstractmethod
    def static_initial_value():
        """ Abstract method that requires any specific data set to define a canonical (constant) initial value for
            plotting an individual trajectory in order to exemplify the flow predicted by a (Symplectic) HNN. This
            trajectory will be obtained by integrating the HNN's symplectic vector field. """
        pass

    def get_trajectory(self, t_span=(0, 3), rtol=1e-9, y0=None, t_eval=None, **kwargs):
        """ Based on the specified Hamiltonian, this method generates a single trajectory obtained by integrating
            the true Hamiltonian vector field with the RK45 scheme and a relative tolerance of `rtol` (default 1e-9).
            The solution is integrated over the interval `t_span` and evaluated at the points in `t_eval`.

            Returns an array y containing the points y(t_i) and the given / generated list `t_eval` of all t_i.
        """
        if t_eval is None:
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
        """ Wrapper method for the `get_trajectory` method, which generates N=`samples` (default 1500) trajectories
            and accumulates them. Additionally does a train / test split before returning a dict of train/test
            data points (y_0, y_1) as well as the times (t_0, t_1) used to generate these values. """
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
        """ Based on the specified Hamiltonian, this method calculates the analytic Hamiltonian (symplectic) vector
            field on a 2D meshgrid defined in both dimensions by the interval of `cls.phase_space_boundaries`,
            and with the given `gridsize` (default 20). Returns a dictionary with the flattened and stacked
            meshgrid (key 'x') and the vector field that was calculated (key 'y').

            WARNING: This method only works for systems of dimension 2.
        """
        field = {'meta': locals()}

        # Meshgrid to get lattice
        cmin, cmax = self.phase_space_boundaries()
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
        return 2

    @classmethod
    def hamiltonian(cls, p, q, t=None):
        H = 1 / 2 * (p ** 2 + q ** 2)
        return H

    @staticmethod
    def phase_space_boundaries():
        return -2, 2

    @staticmethod
    def static_initial_value():
        return np.array([0., 1.5])


class NonlinearPendulum(HamiltonianDataSet):
    """ Implements the Hamiltonian of an ideal non-linear pendulum in 1 dimension. """

    @staticmethod
    def dimension():
        return 2

    @classmethod
    def hamiltonian(cls, p, q, t=None):
        H = 1 / 2 * p ** 2 + (1 - autograd.numpy.cos(q))
        return H

    @staticmethod
    def phase_space_boundaries():
        return -np.pi, np.pi

    @classmethod
    def random_initial_value(cls, draw_omega_m=False, bound_energy=False):
        y = super().random_initial_value(draw_omega_m=draw_omega_m)

        if not bound_energy:
            return y

        # else check if the y has enough energy to do a full spin (i.e. H > 2, with an extra threshold of 0.1 here)
        if cls.bundled_hamiltonian(y) > 2 - 0.1:
            return cls.random_initial_value(draw_omega_m=draw_omega_m, bound_energy=True)
        else:
            return y

    @staticmethod
    def static_initial_value():
        return np.array([0., 3])


class DoublePendulum(HamiltonianDataSet):
    """ Implements the Hamiltonian of a double pendulum. """

    @staticmethod
    def dimension():
        return 4

    @classmethod
    def hamiltonian(cls, p, q, t=None):
        an = autograd.numpy
        qdiff = q[0] - q[1]
        return (1 / 2 * p[0] ** 2 + p[1] ** 2 - p[0] * p[1] * an.cos(qdiff)) / (1 + an.sin(qdiff) ** 2) \
               - 2 * an.cos(q[0]) - an.cos(q[1])

    @staticmethod
    def phase_space_boundaries():
        return -np.pi, np.pi

    @staticmethod
    def static_initial_value():
        return np.array([0, 0, np.pi/2, np.pi/2])


class TwoBody(HamiltonianDataSet):
    """ Implements the Hamiltonian of the two-body problem in 2 spatial dimensions. """

    @staticmethod
    def dimension():
        return 8

    @classmethod
    def hamiltonian(cls, p, q, t=None):
        # Regularize the possible pole from 1 / |q_1 - q_2| by adding a small epsilon to the denominator.
        epsilon = 1e-1
        H = 1 / 2 * autograd.numpy.sum(p ** 2) + 1 / (
                autograd.numpy.sum(autograd.numpy.abs((q[0:2] - q[2:4]))) + epsilon)
        return H

    @staticmethod
    def phase_space_boundaries():
        return -1.5, 1.5

    @classmethod
    def random_initial_value(cls, draw_omega_m=False):
        y = super().random_initial_value()  # = (p1, p2, q1, q2) which are all respectively of dimension 2

        # If the Hamiltonian is not regularized, ensure that generated initial data is far from any poles,
        #   i.e. ensure that the distance of the two bodies squared is not smaller than a certain threshold-
        #   Should a given value violate this condition,redraw. The fact that this problem is defined for
        #   dimension 8 is hardcoded here.

        # threshold = 1e-1
        # q1, q2 = y[4:6], y[6:8]
        # if ((q1 - q2) ** 2).sum() < threshold:
        #    return cls.random_initial_value()  # redraw
        # else:
        #    return y

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

    @classmethod
    def hamiltonian(cls, p, q, t=None, omega=1):
        H = 1 / 2 * autograd.numpy.sum(p ** 2) \
            + omega ** 2 / 4 * autograd.numpy.sum((q[1::2] - q[::2]) ** 2) \
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
        q0 = np.array([(1 - 1 / omega) / sq2, (1 + 1 / omega) / sq2, 0, 0, 0, 0])
        return np.concatenate((p0, q0))

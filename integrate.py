# Symplectic Hamiltonian Neural Networks | 2021
# Marco David

import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point

from model.loss import choose_scheme


def get_predicted_vector_field(model, args, gridsize=20):
    """ Calculates the vector field predicted by an HNN by evaluating the network's prediction on a meshgrid. """
    cmin, cmax = args.data_class.plot_boundaries()

    # Mesh grid to get lattice
    b, a = np.meshgrid(np.linspace(cmin, cmax, gridsize), np.linspace(cmin, cmax, gridsize))
    xs = np.stack([b.flatten(), a.flatten()]).T

    # Run model
    mesh_x = torch.tensor(xs, requires_grad=True, dtype=torch.float32)
    mesh_dx = model.time_derivative(mesh_x)
    return {'x': xs, 'y': mesh_dx.data.numpy()}


def integrate_model_rk45(model, t_span, y0, fun=None, **kwargs):
    def default_fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32)
        x = x.view(1, np.size(np_x))  # batch size of 1
        dx = model.time_derivative(x).data.numpy().reshape(-1)
        return dx

    # Shortcut syntax, not pythonic: fun = fun or default_fun
    if fun is None:
        fun = default_fun

    # Note carefully the .y.T at the end of this call, to obtain the trajectory, in standard format (wrt this project)
    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs).y.T


def integrate_model_custom(model, t_span, y0, args):
    dim = args.dim  # assert dim == np.size(y0)
    scheme = choose_scheme(args.loss_type)(args)

    def iter_fn(y_var, yn, h):
        y_var = torch.tensor(y_var, requires_grad=True, dtype=torch.float32).view(1, dim)
        yn = torch.tensor(yn, requires_grad=True, dtype=torch.float32).view(1, dim)

        y_arg = scheme.argument(yn, y_var)

        return (yn + h * model.time_derivative(y_arg)).detach().numpy().squeeze()

    y = y0
    ys = [y0]
    t = t_span[0]
    ts = [t]

    while t <= t_span[1]:
        # Alternative to scipy's fixed_point: Iterate the function by hand, say 10 times for an error h^10.
        #yn = y
        #for i in range(10):
        #    y = iter_fn(y, yn, args.h)

        # Kwarg method='iteration' possible, too, without accelerated convergence
        # Kwarg xtol=1e-8 normally not attainable with <500 iterations
        y = fixed_point(iter_fn, y, args=(y, args.h), xtol=1e-4)

        ys.append(y)
        t += args.h
        ts.append(t)

    return np.array(ys), np.array(ts)
    # return np.array(ys)


def phase_space_helper(axes, boundary, title, aspect_equal=False):
    """ Sets up a phase space plot with the correct axis labels, boundaries and title. """
    axes.set_xlabel("$p$", fontsize=14)
    axes.set_ylabel("$q$", rotation=0, fontsize=14)
    axes.set_title(title, pad=10)

    if aspect_equal:
        axes.set_aspect('equal')

    if boundary:
        axes.set_xlim(boundary)
        axes.set_ylim(boundary)


def plot_dataset(axes, train_data, test_data, args, title):
    """ Plots a dataset in phase space, explicitly coloring training data blue and testing data red. """
    axes.plot(train_data[:, 0], train_data[:, 1], 'X', color='blue')
    axes.plot(test_data[:, 0], test_data[:, 1], 'X', color='red')

    phase_space_helper(axes, args.data_class.plot_boundaries(), title)


def phase_space_plot(axes, field, traj, title, args, LS=100, LW=1):
    """ Plots a trajectory in phase space, together with a vector field in the background. """
    # LS for number of differently colored plot segments. LW for line width.
    axes.quiver(field['x'][:, 0], field['x'][:, 1], field['y'][:, 0], field['y'][:, 1],
                cmap='gray_r', scale=30, width=6e-3, color=(.5, .5, .5))

    for i, l in enumerate(np.array_split(traj, LS)):
        color = (float(i) / LS, 0, 1 - float(i) / LS)
        axes.plot(l[:, 0], l[:, 1], color=color, linewidth=LW)

    phase_space_helper(axes, args.data_class.plot_boundaries(), title)


def plot_helper(axes, x, y, title=None):
    axes.plot(x, y)

    if title:
        axes.set_title(title)

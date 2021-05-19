# Symplectic Hamiltonian Neural Networks | 2021
# Marco David and Florian Méhats

import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point

from model.loss import choose_scheme
from model.hnn import HNN, CorrectedHNN

from train import setup


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


def hamiltonian_error_grid(model, data_loader):
    dim = data_loader.dimension()
    n = dim//2

    cmin, cmax = data_loader.plot_boundaries()
    p, q = np.linspace(cmin, cmax), np.linspace(cmin, cmax)
    P, Q = np.meshgrid(*([p]*n), *([q]*n))

    y_flat = np.stack((P, Q), axis=-1).reshape(-1, 2)  # reshape (50, 50, 2) to (2500, 2)
    H_flat = model.forward(torch.tensor(y_flat, dtype=torch.float32, requires_grad=True)).detach().cpu().numpy()
    H = H_flat.reshape(*P.shape)

    # Calculate the correct Hamiltonian on the grid
    H_grid = np.array([data_loader.bundled_hamiltonian(y) for y in y_flat]).reshape(*P.shape)

    # Calculate the global constant up to which the model predicts H
    zero_tensor = torch.tensor(np.zeros(dim), dtype=torch.float32, requires_grad=True).view(1, dim)
    H0_pred = model.forward(zero_tensor).detach().cpu().numpy()

    # Return the meshgrid space and the Hamiltonian error
    # VERSION 1
    return P, Q, H - H0_pred - H_grid

    # VERSION 2
    #diff = H - H_grid
    #return P, Q, diff - diff.mean()


def hamiltonian_error_sampled(model, data_loader, N=2000):
    # Create an array y0 of N samples in the shape (N, dim) where dim is the dim of the specific problem, i.e.
    dim = data_loader.dimension()
    y0_list = [data_loader.random_initial_value() for _ in range(N)]
    y0 = torch.tensor(np.array(y0_list), dtype=torch.float32, requires_grad=True)

    # Also create a zero tensor of shape (1, dim)
    zero_tensor = torch.tensor(np.zeros(dim), dtype=torch.float32, requires_grad=True).view(1, dim)

    H_pred = model.forward(y0).detach().cpu().numpy()
    H0 = model.forward(zero_tensor).detach().cpu().numpy()
    H_exact = np.array([data_loader.bundled_hamiltonian(y) for y in y0_list])

    # VERSION 1
    return H_pred - H0 - H_exact

    # VERSION 2
    #diff = H_pred - H_exact
    #return diff - diff.mean()


def calc_herr(base_args, hs, corrected=False, save_dir_prefix='/results/experiment-'):
    errors = []
    for h in hs:
        args = base_args | {'h': h}

        # Only requires name, loss-type, h, noise (to locate the .tar file)
        args = setup(args, save_dir_prefix=save_dir_prefix)

        # Loads the model and (re)loads all arguments as initially saved after training
        model, args = HNN.load(args, cpu=True)

        if corrected:
            scheme = choose_scheme(args.loss_type)(args)
            model = CorrectedHNN.get(model, scheme, h)

        # Get the data loader to compare to the true values
        data_loader = args.data_class(args.h, args.noise)

        # Hamiltonian Error on the meshgrid spanned by P and Q
        # P, Q, H_err = hamiltonian_error_grid(model, data_loader)
        # H_err = H_err.flatten()  # We are not interested in the structure for now.

        # Hamiltonian Error based on random samples in phase space
        H_err = hamiltonian_error_sampled(model, data_loader)

        # Calculate mean and standard deviation (TODO: could also change to quartiles)
        mean, std = np.abs(H_err).mean(), np.abs(H_err).std()
        errors.append((mean, std))

    return np.array(errors)

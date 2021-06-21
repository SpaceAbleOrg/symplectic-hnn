import torch
import pickle
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fixed_point
from scipy.stats import sem

from model.loss import choose_scheme
from model.hnn import HNN, CorrectedHNN
from model.data import get_t_eval

from train import setup


def integrate_model_rk45(model, t_span, t_eval, y0, fun=None, **kwargs):
    """ Integrates a given model using scipy's RK45 method, starting from the initial value `y0` over the given
        `t_span`. Returns the values y(t) obtained for each t value in `t_eval`. """
    def default_fun(t, np_x):
        x = torch.tensor(np_x, requires_grad=True, dtype=torch.float32)
        x = x.view(1, np.size(np_x))  # batch size of 1
        dx = model.derivative(x).data.numpy().reshape(-1)
        return dx

    # Shortcut syntax, not pythonic: fun = fun or default_fun
    if fun is None:
        fun = default_fun

    # Note carefully the .y.T at the end of this call, to obtain the trajectory, in standard format (wrt this project)
    return solve_ivp(fun=fun, t_span=t_span, t_eval=t_eval, y0=y0, **kwargs).y.T, t_eval


def integrate_model_custom(model, t_span, y0, args):
    """ Integrates the given model starting from the initial point y0, using the integration scheme determined by
        `args.loss_type` with a time step `args.h`. That means, this method uses the same scheme and step as used during
        the training of the model. """
    dim = args.dim  # assert dim == np.size(y0)
    scheme = choose_scheme(args.loss_type)(args)

    def iter_fn(y_var, yn, h):
        y_var = torch.tensor(y_var, requires_grad=True, dtype=torch.float32).view(1, dim)
        yn = torch.tensor(yn, requires_grad=True, dtype=torch.float32).view(1, dim)

        y_arg = scheme.argument(yn, y_var)

        return (yn + h * model.derivative(y_arg)).detach().numpy().squeeze()

    y = y0
    ys = [y0]
    t = t_span[0]
    ts = [t]

    while t <= t_span[1]:
        # Alternative to scipy's fixed_point: Iterate the function by hand, say 10 times for an error h^10.
        # yn = y
        # for i in range(10):
        #     y = iter_fn(y, yn, args.h)

        # Kwarg method='iteration' possible, too, without accelerated convergence
        y = fixed_point(iter_fn, y, args=(y, args.h), xtol=1e-4)

        ys.append(y)
        t += args.h
        ts.append(t)

    return np.array(ys), np.array(ts)


def integrate_trajectory(model, args, t_span, y0, same_method, t_eval=None):
    """ This method wraps the two methods ```integrate_model_custom``` and ```integrate_model_rk45```. Thus, it allows
        to obtain some predicted trajectory from a model and its arguments; either by exactly following the predicted
        vector field using RK45 (i.e. the learned modified vector field, in the case of a symplectic method)
        or by using the same method as used during training (which can even yield the exact solution of the true
        Hamiltonian, see the theory in the article). """
    if t_eval is None:
        t_eval = get_t_eval(t_span, args.h)

    if same_method:
        # Integrate with the same method and same time step used for training
        y, t = integrate_model_custom(model, t_span, y0, args)
    else:
        # Use RK45 with rtol of 1e-9 to have RK45 effectively yield the true flow of the learned vector field
        y, t = integrate_model_rk45(model, t_span, t_eval, y0, rtol=1e-9, method='RK45')

    return y, t


def hamiltonian_error_grid(model, data_loader, N=50):
    """ This function computes a 2D meshgrid of the region \Omega_d in phase space and calculates the error between
        the predicted and true Hamiltonians at each point on this grid. It returns the grid and the error arrays which
        allow to easily plot the results as a contour plot.

        WARNING: This function doesn't generalize well to dimensions > 2."""

    dim = data_loader.dimension()
    n = dim//2

    cmin, cmax = data_loader.phase_space_boundaries()
    p, q = np.linspace(cmin, cmax, N), np.linspace(cmin, cmax, N)
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
    # VERSION 1 – ATTENTION! This adds the error at y=0 everywhere, do not use!
    #return P, Q, H - H0_pred - H_grid

    # VERSION 2 – This is the correct way to do it, removing the average added constant
    diff = H - H_grid
    return P, Q, diff - diff.mean()


def hamiltonian_error_sampled(model, data_loader, omega_m, N=2000):
    """ This function computes the error between the predicted and true Hamiltonians in the region \Omega_d in phase
        space by randomly sampling N points in this region. This function returns the full distribution of errors
        which can be used to do a statistical analysis, afterwards.

        This function works in any dimension."""

    # Create an array y0 of N samples in the shape (N, dim) where dim is the dim of the specific problem, i.e.
    dim = data_loader.dimension()
    # draw_omega_m ensures that we don't draw values from the boundary of \Omega_d, where the model isn't well trained
    y0_list = [data_loader.random_initial_value(draw_omega_m=omega_m) for _ in range(N)]
    y0 = torch.tensor(np.array(y0_list), dtype=torch.float32, requires_grad=True)

    # Also create a zero tensor of shape (1, dim)
    zero_tensor = torch.tensor(np.zeros(dim), dtype=torch.float32, requires_grad=True).view(1, dim)

    H_pred = model.forward(y0).detach().cpu().numpy()
    H0 = model.forward(zero_tensor).detach().cpu().numpy()
    H_exact = np.array([data_loader.bundled_hamiltonian(y) for y in y0_list])

    # VERSION 1 – DO NOT USE
    #return H_pred - H0 - H_exact

    # VERSION 2 – This is the correct way to do it.
    diff = H_pred - H_exact
    return diff - diff.mean()


def load_model(args, corrected, save_dir_prefix):
    """ This function wraps the `HNN.load` function. """
    # Only requires name, loss-type, h, noise (to locate the .tar file)
    args = setup(args, save_dir_prefix=save_dir_prefix)

    # Loads the model and (re)loads all arguments as initially saved after training
    model, args = HNN.load(args, cpu=True)

    if corrected:
        scheme = choose_scheme(args.loss_type)(args)
        model = CorrectedHNN.get(model, scheme, args.h)

    return model, args


def calc_herr(base_args, hs, omega_m=True, corrected=False, save_dir_prefix='/results/experiment-'):
    """ This function wraps the function `hamiltonian_error_sampled` by computing the mean and the quartiles
        of the distribution of errors in phase space. Moreover, it does do for an arbitrary number of different values
        of h, loading a new trained model each time.

        Additionally, it allows to instantiate a CorrectedHNN using the kwarg `corrected`. """
    errors = []
    for h in hs:
        args = base_args | {'h': h}

        model, args = load_model(args, corrected, save_dir_prefix)

        # Get the data loader to compare to the true values
        data_loader = args.data_class(args.h, args.noise)

        # Hamiltonian Error on the meshgrid spanned by P and Q
        # P, Q, H_err = hamiltonian_error_grid(model, data_loader)
        # H_err = H_err.flatten()  # We are not interested in the structure for now.

        # Hamiltonian Error based on random samples in phase space, in absolute value
        H_err = np.abs(hamiltonian_error_sampled(model, data_loader, omega_m))

        # VERSION 1 – Calculate mean and standard deviation
        # mean, std = H_err.mean(), H_err.std()
        # errors.append((mean, std))

        # VERSION 2 – Calculate mean and standard error
        #mean, stderr = H_err.mean(), sem(H_err)
        #errors.append((mean, stderr))

        # VERSION 3 – Calculate mean and quartiles
        mean = H_err.mean()
        q1, q3 = np.percentile(H_err, 25), np.percentile(H_err, 75)
        #print(f"h={h}: mean={mean}, stdev={H_err.std()}, q1={q1}, q3={q3}")
        errors.append((mean, q1, q3))

    return np.array(errors)


def calc_mse(args, t_span, N=30, same_method=False, corrected=False, t_eval=None, save_dir_prefix='/results/experiment-'):
    """ This function calculates the mean squared error (MSE) in the coordinates of a predicted trajectoy, with respect
        to the true trajectory (obtained by integrating the real Hamiltonian system with RK45 and low error tolerance).
        It does so for N randomly chosen initial values (drawn from Omega_m), and returns the mean error over all
         these samples, together with the standard error of the mean.

        The results are not returned but rather pickled into a file, which can be loaded with `load_mse`. """
    model, new_args = load_model(args, corrected, save_dir_prefix)

    data_loader = new_args.data_class(new_args.h, new_args.noise)

    errs = []
    for i in range(N):
        if new_args.name == 'pendulum':
            y0 = data_loader.random_initial_value(draw_omega_m=True, bound_energy=True)
        else:
            y0 = data_loader.random_initial_value(draw_omega_m=True)

        true_y, _ = data_loader.get_trajectory(t_span=t_span, y0=y0, t_eval=t_eval)
        y, _ = integrate_trajectory(model, new_args, t_span, y0, same_method, t_eval=t_eval)

        errs.append(np.sum((y - true_y) ** 2, axis=1))

    # Perform the mean (and stderr of the mean) over all N trajectories, but not over time
    errs = np.array(errs)
    mse = np.mean(errs, axis=0)
    stderr = sem(errs, axis=0)

    # Save results
    with open(args.save_dir + f"/mse-{args.name}-{args.loss_type}{'-corrected' if corrected else ''}-h{args.h}.pck", "wb") as file:
        pickle.dump({'error_distribution': errs, 'mse': mse, 'stderr': stderr, 'N': N,
                     't_span': t_span, 't_eval': t_eval}, file)


def load_mse(args, corrected=False):
    """ This function loads and returns the MSE in coordinates (and its standard error) as saved by `calc_mse`. """
    with open(args.save_dir + f"/mse-{args.name}-{args.loss_type}{'-corrected' if corrected else ''}-h{args.h}.pck", "rb") as file:
        return pickle.load(file)
        # return stored_dict['mse'], stored_dict['stderr']

# Symplectic Hamiltonian Neural Networs | 2021
# Marco David

import torch
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from utils import setup_args, save_path
from model.standard_nn import MLP
from model.hnn import HNN
from model.data import get_t_eval


def load_model(args):
    # Create the standard MLP with args.dim inputs and one single scalar output, the Hamiltonian
    nn_model = MLP(args.dim, args.dim_hidden_layer, output_dim=1, nonlinearity=args.nonlinearity)
    # Use this model to create a Hamiltonian Neural Network, which knows how to differentiate the Hamiltonian
    model = HNN(nn_model)
    # Load saved state using the standard save_path
    model.load_state_dict(torch.load(save_path(args)))

    return model


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


def integrate_model(model, t_span, y0, fun=None, **kwargs):
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


def integrate_euler_symp(model, args, t_span, y0):
    y = y0
    ys = [y0]
    t = t_span[0]
    ts = [t]

    size = np.size(y0)

    # y_tens = torch.tensor(y, requires_grad=True, dtype=torch.float32).view(1, size)
    # dy_tens = model.time_derivative(y_tens).data.numpy().reshape(-1)

    while t <= t_span[1]:
        fp = 0
        y = y + args.h * fp
        ys.append(y)
        t += args.h

    return ys


def integrate_midpoint(model, args, t_span, y0):
    pass


def phase_space_helper(axes, boundary, title):
    """ Sets up a phase space plot with the correct axis labels, boundaries and title. """
    axes.set_xlabel("$p$", fontsize=14)
    axes.set_ylabel("$q$", rotation=0, fontsize=14)
    axes.set_title(title, pad=10)

    if boundary:
        axes.set_xlim(boundary)
        axes.set_ylim(boundary)


def dataset_plot(axes, train_data, test_data, args, title):
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


def final_plot(model, args, t_span=(0, 300)):
    t_eval = get_t_eval(t_span, args.h)
    kwargs = {'t_eval': t_eval, 'rtol': 1e-6, 'method': 'RK45'}

    # INTEGRATE MODEL
    static_y0 = args.data_class.static_initial_value()
    pred_field = get_predicted_vector_field(model, args)
    pred_traj = integrate_model(model, t_span, static_y0, **kwargs)

    # Calculate the Hamiltonian along the trajectory
    H = model.forward(torch.tensor(pred_traj, dtype=torch.float32)).data.numpy()

    # TRUE TRAJECTORY FOR REFERENCE
    data_loader = args.data_class(args.h, args.noise)
    exact_field = data_loader.get_analytic_field()
    exact_traj, _ = data_loader.get_trajectory(t_span=t_span, y0=static_y0)
    # dataset = data_loader.get_dataset(seed=args.seed, samples=3000, test_split=0.05)  # (3000, 2, 2)
    # dataset_plot(ax[2], dataset['coords'][:, 0], dataset['test_coords'][:, 0], args, "Dataset for ...")

    # Calculate the initial Hamiltonian = Hamiltonian at all times of true trajectory
    Hy0 = data_loader.bundled_hamiltonian(static_y0)

    # Calculate the respective errors
    traj_error = np.linalg.norm(pred_traj - exact_traj, axis=1)
    H_error = np.abs(H - Hy0)

    # === BEGIN PLOTTING ===
    fig = plt.figure(figsize=(25, 6), facecolor='white', dpi=300)
    ax = [fig.add_subplot(1, 4, i + 1, frameon=True) for i in range(4)]  # kwarg useful sometimes: aspect='equal'

    title_true = f"True Trajectory\n (Integrated with RK45)"
    phase_space_plot(ax[0], exact_field, exact_traj, title_true, args)

    title_pred = f"Symplectic HNN: $h = {args.h}, t_f = {t_span[1]}$\n Trained with {args.loss_type}, Integrated with RK45"
    phase_space_plot(ax[1], pred_field, pred_traj, title_pred, args)

    N = len(t_eval)
    lim = N//3
    title_both = f"$p$ Coordinate vs. Time \n (Note: for more clarity, $t_f = {t_eval[lim]})"

    axes = ax[2]
    axes.plot(t_eval[:lim], exact_traj[:lim, 0])
    axes.plot(t_eval[:lim], pred_traj[:lim, 0])
    axes.set_title(title_both)

    title_trajerror = f"Deviation (norm of error) of the two trajectories \n (Over full timespan, $t_f = {t_span[1]}$)"
    plot_helper(ax[3], t_eval, traj_error, title_trajerror)

    # TODO Eventually fix the scientific notation for the axis scale, see this question:
    #       https://stackoverflow.com/questions/42656139/set-scientific-notation-with-fixed-exponent-and-significant-digits-for-multiple
    #title3 = r"Deviation of the Hamiltonian: $|H(y(t)) - H(y_0)|$"
    #plot_helper(ax[3], t_eval, H_error, title3)

    # Old code to investigate individual
    #ax[2].plot(t_eval, exact_traj[:, 0], color='blue')
    #ax[2].plot(t_eval, pred_traj[:, 0], color='red')

    # SAVE FIGURE USING USUAL PATH
    plt.savefig(save_path(args, ext='pdf'))


if __name__ == "__main__":
    args = setup_args()

    model = load_model(args)

    final_plot(model, args)

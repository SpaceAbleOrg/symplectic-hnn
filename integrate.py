# Symplectic Hamiltonian Neural Networs | 2021
# Marco David

import torch
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from utils import setup_args, save_path
from model.standard_nn import MLP
from model.hnn import HNN


def load_model(args):
    # Create the standard MLP with args.dim inputs and one single scalar output, the Hamiltonian
    nn_model = MLP(args.dim, args.dim_hidden_layer, output_dim=1, nonlinearity=args.nonlinearity)
    # Use this model to create a Hamiltonian Neural Network, which knows how to differentiate the Hamiltonian
    model = HNN(nn_model)
    # Load saved state using the standard save_path
    model.load_state_dict(torch.load(save_path(args)))

    return model


def get_predicted_vector_field(model, xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2, gridsize=20):
    # Mesh grid to get lattice
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
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

    fun = default_fun if fun is None else fun
    return solve_ivp(fun=fun, t_span=t_span, y0=y0, **kwargs)


def _plot_field_and_trajectory(axes, field, traj, title, LS=100, LW=1):
    """ LS for number of line segments. LW for line width. """
    axes.quiver(field['x'][:, 0], field['x'][:, 1], field['y'][:, 0], field['y'][:, 1],
                cmap='gray_r', scale=30, width=6e-3, color=(.5, .5, .5))

    for i, l in enumerate(np.array_split(traj, LS)):
        color = (float(i) / LS, 0, 1 - float(i) / LS)
        axes.plot(l[:, 0], l[:, 1], color=color, linewidth=LW)

    axes.set_xlabel("$p$", fontsize=14)
    axes.set_ylabel("$q$", rotation=0, fontsize=14)
    axes.set_title(title, pad=10)


def final_plot(model, args, t_span=(0, 1000)):
    kwargs = {'t_eval': np.linspace(t_span[0], t_span[1], int((t_span[1] - t_span[0]) / args.h)),
              'rtol': 1e-6, 'method': 'RK45'}

    pred_field = get_predicted_vector_field(model)
    pred_traj = integrate_model(model, t_span, args.init_value, **kwargs).y.T

    # BEGIN PLOTTING
    fig = plt.figure(figsize=(24, 6), facecolor='white', dpi=300)
    ax = [fig.add_subplot(1, 4, i + 1, frameon=True) for i in range(4)]

    _plot_field_and_trajectory(ax[0], pred_field, pred_traj, "title")

    plt.savefig(args.save_dir + '/' + args.name + '.pdf')


if __name__ == "__main__":
    args = setup_args()

    model = load_model(args)

    final_plot(model, args)

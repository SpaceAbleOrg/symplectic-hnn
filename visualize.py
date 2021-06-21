import torch
import numpy as np

from model.args import get_args


# Configure MPL parameters (taken from https://github.com/jbmouret/matplotlib_for_papers)
golden_ratio = (5**.5 - 1) / 2
params = {
    # Use the golden ratio to make plots aesthetically pleasing
    'figure.figsize': [5, 5*golden_ratio],
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document, titles slightly larger
    # In the end, most plots will anyhow be shrunk to fit onto a US Letter / DIN A4 page
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
}

default_args = get_args(lenient=True) | {'name': 'pendulum'}  # returns the default arguments, with name overwritten

print_method = {'euler-forw': "forw. Euler (HNN)",
                'euler-symp': "symp. Euler",
                'midpoint': "implicit midpoint"}
print_name = {'spring': "Harmonic Oscillator",
              'pendulum': "Pendulum",
              'double-pendulum': "Double Pendulum",
              'fput': "FPUT Problem",
              'twobody': "Two-body Problem"}


def get_predicted_vector_field(model, args, gridsize=20):
    """ Calculates the vector field predicted by an HNN by evaluating the network's prediction on a meshgrid. """
    cmin, cmax = args.data_class.phase_space_boundaries()

    # Mesh grid to get lattice
    b, a = np.meshgrid(np.linspace(cmin, cmax, gridsize), np.linspace(cmin, cmax, gridsize))
    xs = np.stack([b.flatten(), a.flatten()]).T

    # Run model
    mesh_x = torch.tensor(xs, requires_grad=True, dtype=torch.float32)
    mesh_dx = model.derivative(mesh_x)
    return {'x': xs, 'y': mesh_dx.data.numpy()}


def phase_space_helper(axes, boundary, title, aspect_equal=False):
    """ Sets up a phase space plot with the correct axis labels, boundaries and title. """
    axes.set_xlabel("$p$")  # fontsize=14)
    axes.set_ylabel("$q$", rotation=0)  # fontsize=14)
    axes.set_title(title)  # pad=10)

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

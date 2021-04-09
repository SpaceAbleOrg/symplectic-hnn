# Symplectic Hamiltonian Neural Networs | 2021
# Marco David

import torch
import numpy as np
import matplotlib.pyplot as plt

from integrate import load_model, integrate_model
from utils import setup_args, save_path
from model.data import get_t_eval


def osc_invariants(traj, omega):
    p, q = np.split(traj, 2, axis=-1)
    # I_j = 1/2 * (y_{1,j}^2 + omega^2 x_{1,j}^2)
    Ij = 1 / 2 * ((p[:, 1::2] - p[:, :-1:2]) ** 2 / 2 + omega ** 2 * (q[:, 1::2] - q[:, :-1:2]) ** 2 / 2)
    return Ij


def final_plot(model, args, t_span=(0, 150)):
    t_eval = get_t_eval(t_span, args.h)
    kwargs = {'t_eval': t_eval, 'rtol': 1e-6, 'method': 'RK45'}

    # INTEGRATE MODEL
    # TODO how to customize OMEGA ???
    omega = 50
    static_y0 = args.data_class.static_initial_value(omega=omega)
    pred_traj = integrate_model(model, t_span, static_y0, **kwargs)

    # Calculate the Hamiltonian along the trajectory
    H_pred = model.forward(torch.tensor(pred_traj, dtype=torch.float32)).data.numpy()

    # Calculate the oscillation energies along the trajectory
    Ij_pred = osc_invariants(pred_traj, omega)

    # TRUE TRAJECTORY FOR REFERENCE
    data_loader = args.data_class(args.h, args.noise)
    exact_traj, _ = data_loader.get_trajectory(t_span=t_span, y0=static_y0)

    # Calculate the initial Hamiltonian and the Hamiltonian at all times of true trajectory
    # Also calculate the oscillatory invariants along the 'exact' trajectory
    Hy0 = data_loader.bundled_hamiltonian(static_y0)
    H_exact = np.array([data_loader.bundled_hamiltonian(y) for y in exact_traj])
    Ij_exact = osc_invariants(exact_traj, omega)

    # Calculate the respective errors
    #traj_error = np.linalg.norm(pred_traj - exact_traj, axis=1)
    #H_error = np.abs(H - Hy0)

    # === BEGIN PLOTTING ===
    fig = plt.figure(figsize=(25, 6), facecolor='white', dpi=300)
    #ax = [fig.add_subplot(1, 4, i + 1, frameon=True) for i in range(4)]  # kwarg useful sometimes: aspect='equal'

    ax1 = fig.add_subplot(1, 2, 1, frameon=True)
    for i in range(3):
        ax1.plot(t_eval, Ij_pred[:, i], label=f'$I_{i+1}$')
    ax1.plot(t_eval, H_pred, label='$H$')
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2, frameon=True)
    for i in range(3):
        ax2.plot(t_eval, Ij_exact[:, i], label=f'$I_{i+1}$')
    ax2.plot(t_eval, H_exact, label='$H$')
    ax2.legend()

    # SAVE FIGURE USING USUAL PATH
    plt.savefig(save_path(args, ext='pdf'))


if __name__ == "__main__":
    args = setup_args()

    model = load_model(args)

    final_plot(model, args)

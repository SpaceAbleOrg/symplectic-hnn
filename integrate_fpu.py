# Symplectic Hamiltonian Neural Networs | 2021
# Marco David

import torch
import numpy as np
import matplotlib.pyplot as plt

from integrate import integrate_model
from utils import setup_args, save_path
from model.hnn import get_hnn
from model.data import get_t_eval


def pq_to_xy0(traj):
    p, q = np.split(traj, 2, axis=-1)
    y0 = (p[:, 1::2] + p[:, :-1:2]) / np.sqrt(2)
    x0 = (q[:, 1::2] + q[:, :-1:2]) / np.sqrt(2)
    return x0, y0


def pq_to_xy1(traj):
    p, q = np.split(traj, 2, axis=-1)
    y1 = (p[:, 1::2] - p[:, :-1:2]) / np.sqrt(2)
    x1 = (q[:, 1::2] - q[:, :-1:2]) / np.sqrt(2)
    return x1, y1


def osc_invariants(traj, omega):
    x1, y1 = pq_to_xy1(traj)
    # I_j = 1/2 * (y_{1,j}^2 + omega^2 x_{1,j}^2)
    Ij = 1/2 * (y1 ** 2 + omega ** 2 * x1 ** 2)
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
    xplts, yplts = 1, 4
    #ax = [fig.add_subplot(1, 4, i + 1, frameon=True) for i in range(4)]  # kwarg useful sometimes: aspect='equal'

    ax1 = fig.add_subplot(xplts, yplts, 1, frameon=True)
    for i in range(3):
        ax1.plot(t_eval, Ij_pred[:, i], label=f'$I_{i+1}$')
    ax1.plot(t_eval, H_pred, label='$H$')
    ax1.set_title("SHNN Prediction")
    ax1.legend()

    ax2 = fig.add_subplot(xplts, yplts, 3, frameon=True)
    for i in range(3):
        ax2.plot(t_eval, Ij_exact[:, i], label=f'$I_{i+1}$')
    ax2.plot(t_eval, H_exact, label='$H$')
    ax2.set_title("True Solution (RK45)")
    ax2.legend()

    ax3 = fig.add_subplot(xplts, yplts, 4, frameon=True)
    for i in range(6):
        ax3.plot(t_eval, exact_traj[:, i], label=f'$p_{i+1}$')
        ax3.plot(t_eval, exact_traj[:, 6+i], label=f'$q_{i+1}$')
    ax3.set_title("True Solution (RK45)")
    ax3.legend()

    ax4 = fig.add_subplot(xplts, yplts, 2, frameon=True)
    for i in range(6):
        ax4.plot(t_eval, pred_traj[:, i], label=f'$p_{i+1}$')
        ax4.plot(t_eval, pred_traj[:, 6+i], label=f'$q_{i+1}$')
    ax4.set_title("SHNN Prediction")
    ax4.legend()
    # for i in range(3):
    #     x1, y1 = pq_to_xy1(exact_traj)
    #     x0, y0 = pq_to_xy0(exact_traj)
    #     ax4.plot(t_eval, x0[:, i], label=f'$x0_{i+1}$')
    #     ax4.plot(t_eval, y0[:, i], label=f'$y0_{i+1}$')
    #     ax4.plot(t_eval, x1[:, i], label=f'$x1_{i+1}$')
    #     ax4.plot(t_eval, y1[:, i], label=f'$y1_{i+1}$')

    # SAVE FIGURE USING USUAL PATH
    plt.savefig(save_path(args, ext='pdf'))


if __name__ == "__main__":
    args = setup_args()

    model = get_hnn(args)

    # Load saved state using the standard save_path
    model.load_state_dict(torch.load(save_path(args)))

    final_plot(model, args)
